# train_optim.py
"""
FTRAIN - Advanced Training Optimization Utilities (v2.0)
========================================================

Designed for:
- LoRA / DoRA / full fine-tuning of causal LMs
- small and large models (multi-device aware)
- fp16 (GradScaler), bf16 and fp32 training
- gradient accumulation with correct token weighting
- unstable-training detection and rollback
- adaptive learning-rate selection

This module intentionally does NOT own the training loop.
It provides optimization primitives that the trainer calls.

Every public name and signature from v1 still works. What changed in v2.0
-------------------------------------------------------------------------
Bug fixes
  * ``optimizer_step``: with an AMP GradScaler, a non-finite step used to
    return *before* ``scaler.step()/update()``, so the scaler never shrank its
    scale after an overflow. The scaler now always runs; a disabled scaler no
    longer steps on NaN gradients.
  * ``LRFinder``: the model snapshot cloned the *entire* state dict to CPU
    (gigabytes for an 8B model, although only LoRA weights change). It now
    snapshots trainable parameters only. The optimizer snapshot used to alias
    live state tensors and is now a real copy.
  * ``LRFinder``: the smoothed loss was anchored to the first batch for ~50
    steps (no bias correction), the moving average was misaligned with the LR
    axis, and a loader shorter than ``n`` silently produced a sweep that never
    left the tiny-LR region. All three are fixed.
  * ``LRFinder`` used to overwrite every param group with one absolute LR,
    flattening layer-wise LR ratios. Group ratios are now preserved.
  * ``EMALoss`` was permanently poisoned by a single NaN/Inf loss.
  * ``move_batch_to_device`` ignored HF ``BatchEncoding`` and crashed on
    namedtuples.
  * ``OptimConfig`` was never consumed by anything; it now drives the
    ``build_*`` factories and ``run_train_step``.

New capabilities
  * ``run_train_step``: one optimizer step over a window of micro-batches with
    *token-weighted* loss scaling (the correct global token mean), non-finite
    handling, GradScaler support and optional anomaly detection.
  * ``InstabilityGuard``: consecutive-skip / loss-spike detection with
    last-known-good rollback of trainable weights.
  * Schedulers: ``linear``, ``constant`` and ``wsd`` next to the cosine ones,
    plus ``build_scheduler`` and ``scale_learning_rate`` (a persistent LR
    change that survives ``scheduler.step()``).
  * ``build_optimizer``: fused AdamW auto-detection, 8-bit AdamW with a safe
    fallback, and a ``param_group_fn`` hook for layer-wise LR groups.
  * Batched (foreach) gradient norm / health checks, ``describe_param_groups``
    and richer statistics (throughput, EMA std, recent-loss window).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import abc as _abc
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


# ============================================================
# Configuration
# ============================================================

_VALID_SCHEDULERS = ("cosine", "cosine_restarts", "linear", "constant", "wsd")
_VALID_OPTIMIZERS = (
    "adamw",
    "adamw_torch",
    "adamw_fused",
    "adam",
    "adamw_8bit",
    "paged_adamw_8bit",
)


@dataclass
class OptimConfig:
    """Single source of truth for optimization settings.

    The first fifteen fields are the original v1 fields, in their original
    order, so positional construction keeps working. Everything after them is
    new and defaulted.
    """

    # Learning rate
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6

    # Warmup
    warmup_ratio: float = 0.03
    warmup_steps: int = 0

    # Scheduler
    scheduler: str = "cosine"
    num_cycles: float = 0.5

    # Gradient handling
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Stability
    detect_anomaly: bool = False
    skip_nonfinite: bool = True

    # EMA
    use_ema_loss: bool = True
    ema_decay: float = 0.95

    # LR finder
    lr_find_start: float = 1e-7
    lr_find_end: float = 1.0
    lr_find_steps: int = 100

    # ---- new in v2.0 ----
    # Scheduler extras
    restart_interval: int = 50
    restart_amplitude_decay: float = 0.15
    wsd_decay_ratio: float = 0.1

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # Accumulation budget
    target_batch_tokens: int = 8192
    max_accumulation_multiplier: int = 4

    # Instability guard
    max_consecutive_skips: int = 20
    spike_sigma: float = 6.0

    # Statistics
    recent_window: int = 50

    # LR finder extras
    lr_find_smooth_beta: float = 0.98
    lr_find_divergence_factor: float = 4.0

    def warmup_steps_for(self, total_steps: int) -> int:
        """Absolute warmup steps: explicit ``warmup_steps`` wins over the ratio."""
        total_steps = max(1, int(total_steps))
        if self.warmup_steps > 0:
            return min(int(self.warmup_steps), total_steps)
        if self.warmup_ratio <= 0:
            return 0
        return max(1, int(self.warmup_ratio * total_steps))

    def validate(self) -> "OptimConfig":
        """Raise ``ValueError`` listing every problem; return ``self`` if clean."""
        problems: List[str] = []

        if not self.learning_rate > 0:
            problems.append("learning_rate must be > 0")
        if self.min_learning_rate < 0:
            problems.append("min_learning_rate must be >= 0")
        if self.min_learning_rate > self.learning_rate:
            problems.append("min_learning_rate must be <= learning_rate")
        if not 0.0 <= self.warmup_ratio < 1.0:
            problems.append("warmup_ratio must be in [0, 1)")
        if self.warmup_steps < 0:
            problems.append("warmup_steps must be >= 0")
        if str(self.scheduler).lower() not in _VALID_SCHEDULERS:
            problems.append(f"scheduler must be one of {_VALID_SCHEDULERS}")
        if str(self.optimizer).lower() not in _VALID_OPTIMIZERS:
            problems.append(f"optimizer must be one of {_VALID_OPTIMIZERS}")
        if self.gradient_accumulation_steps < 1:
            problems.append("gradient_accumulation_steps must be >= 1")
        if self.max_grad_norm is not None and self.max_grad_norm < 0:
            problems.append("max_grad_norm must be >= 0 (0 or None disables clipping)")
        if not 0.0 <= self.ema_decay < 1.0:
            problems.append("ema_decay must be in [0, 1)")
        if not 0.0 < self.lr_find_start < self.lr_find_end:
            problems.append("need 0 < lr_find_start < lr_find_end")
        if self.lr_find_steps < 2:
            problems.append("lr_find_steps must be >= 2")
        if self.restart_interval < 1:
            problems.append("restart_interval must be >= 1")
        if not 0.0 < self.wsd_decay_ratio <= 1.0:
            problems.append("wsd_decay_ratio must be in (0, 1]")
        if self.weight_decay < 0:
            problems.append("weight_decay must be >= 0")
        if self.target_batch_tokens < 1:
            problems.append("target_batch_tokens must be >= 1")
        if self.max_accumulation_multiplier < 1:
            problems.append("max_accumulation_multiplier must be >= 1")
        if self.max_consecutive_skips < 1:
            problems.append("max_consecutive_skips must be >= 1")
        if self.spike_sigma <= 0:
            problems.append("spike_sigma must be > 0")
        if self.recent_window < 1:
            problems.append("recent_window must be >= 1")
        if not 0.0 <= self.lr_find_smooth_beta < 1.0:
            problems.append("lr_find_smooth_beta must be in [0, 1)")
        if self.lr_find_divergence_factor <= 1.0:
            problems.append("lr_find_divergence_factor must be > 1")

        if problems:
            raise ValueError("Invalid OptimConfig: " + "; ".join(problems))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimConfig":
        """Build from a dict, ignoring unknown keys (old/new config files)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ============================================================
# Internal helpers
# ============================================================

def _infer_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _autocast_ctx(device, dtype):
    if dtype is None:
        return nullcontext()
    device_type = getattr(device, "type", str(device)).split(":")[0]
    return torch.autocast(device_type=device_type, dtype=dtype)


def _scaler_enabled(scaler) -> bool:
    if scaler is None:
        return False
    is_enabled = getattr(scaler, "is_enabled", None)
    return bool(is_enabled()) if callable(is_enabled) else True


def _extract_loss(output) -> torch.Tensor:
    """Pull the loss out of a model output (HF ModelOutput, dict, tuple, tensor)."""
    if torch.is_tensor(output):
        return output
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, _abc.Mapping):
        loss = output.get("loss")
    if loss is None and isinstance(output, (tuple, list)) and output:
        loss = output[0]
    if loss is None:
        raise ValueError(
            "Model output has no loss. Pass labels in the batch, or supply a "
            "forward_fn that returns a loss."
        )
    return loss


def _clone_to_cpu(obj: Any) -> Any:
    """Deep copy of a nested structure with every tensor cloned onto the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _per_tensor_norms(tensors: Sequence[torch.Tensor], norm_type: float = 2.0) -> List[torch.Tensor]:
    """Per-tensor norms in the original order, computed with fused foreach
    kernels per (device, dtype) group instead of one kernel launch per tensor."""
    if not tensors:
        return []

    norms: List[Optional[torch.Tensor]] = [None] * len(tensors)
    groups: Dict[Tuple[Any, Any], List[int]] = defaultdict(list)
    for i, t in enumerate(tensors):
        groups[(t.device, t.dtype)].append(i)

    foreach = getattr(torch, "_foreach_norm", None)
    for indices in groups.values():
        chunk = [tensors[i] for i in indices]
        try:
            if foreach is None:
                raise NotImplementedError
            result = foreach(chunk, norm_type)
        except (RuntimeError, TypeError, NotImplementedError):
            result = [torch.linalg.vector_norm(t, norm_type) for t in chunk]
        for i, r in zip(indices, result):
            norms[i] = r
    return norms  # type: ignore[return-value]


def _combine_norms(norms: Sequence[torch.Tensor], norm_type: float = 2.0) -> torch.Tensor:
    device = norms[0].device
    stacked = torch.stack([n.to(device=device, dtype=torch.float32) for n in norms])
    if math.isinf(norm_type):
        return stacked.max()
    return torch.linalg.vector_norm(stacked, norm_type)


# ============================================================
# Batch / loss utilities
# ============================================================

def move_batch_to_device(batch: Any, device, non_blocking: bool = True) -> Any:
    """
    Recursively move tensors inside a batch to the target device.

    Handles tensors, dicts / any Mapping, lists, tuples, namedtuples and
    objects exposing ``.to(device)`` (e.g. HF ``BatchEncoding``).
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)

    if not isinstance(batch, (dict, list, tuple)):
        to_fn = getattr(batch, "to", None)
        if callable(to_fn):
            try:
                return to_fn(device)
            except Exception:  # fall through to the structural handlers
                pass

    if isinstance(batch, _abc.Mapping):
        return {
            k: move_batch_to_device(v, device, non_blocking)
            for k, v in batch.items()
        }

    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return type(batch)(*[move_batch_to_device(v, device, non_blocking) for v in batch])

    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device, non_blocking) for v in batch)

    return batch


def is_finite_loss(loss) -> bool:
    """
    Safely check whether a loss (tensor or number) is finite.
    """
    if loss is None:
        return False
    if torch.is_tensor(loss):
        return bool(torch.isfinite(loss.detach()).all().item())
    try:
        return math.isfinite(float(loss))
    except (TypeError, ValueError):
        return False


def count_batch_tokens(batch: Any, pad_token_id: Optional[int] = None) -> int:
    """Real (non-padding) tokens in a batch: attention_mask if present,
    otherwise input_ids (optionally excluding ``pad_token_id``)."""
    if not isinstance(batch, _abc.Mapping):
        return 0
    mask = batch.get("attention_mask")
    if torch.is_tensor(mask):
        return int(mask.sum().item())
    ids = batch.get("input_ids")
    if torch.is_tensor(ids):
        if pad_token_id is not None:
            return int((ids != pad_token_id).sum().item())
        return int(ids.numel())
    return 0


def count_label_tokens(batch: Any, ignore_index: int = -100, shift: bool = True) -> int:
    """Tokens that actually contribute to a causal-LM loss.

    ``shift=True`` mirrors HF causal LMs, which score ``labels[..., 1:]``.
    """
    if not isinstance(batch, _abc.Mapping):
        return 0
    labels = batch.get("labels")
    if not torch.is_tensor(labels):
        return 0
    if shift:
        labels = labels[..., 1:]
    return int((labels != ignore_index).sum().item())


def scale_loss_for_accumulation(
    loss: torch.Tensor,
    micro_tokens: Optional[int] = None,
    window_tokens: Optional[int] = None,
    accumulation_steps: int = 1,
) -> torch.Tensor:
    """Scale a micro-batch loss before ``backward()``.

    Dividing every micro-batch loss by ``accumulation_steps`` over-weights
    short micro-batches: the accumulated gradient is not the gradient of the
    global token-mean loss. With ``micro_tokens`` and ``window_tokens`` the
    loss is weighted by its token share instead, which is exact.
    """
    if micro_tokens is not None and window_tokens is not None and window_tokens > 0:
        return loss * (micro_tokens / window_tokens)
    return loss / max(1, int(accumulation_steps))


# ============================================================
# Gradient utilities
# ============================================================

def get_grad_norm(model, norm_type: float = 2.0) -> float:
    """
    Total gradient norm (L2 by default) across all parameters with grads.

    Uses batched foreach kernels and a single device sync. Returns ``inf`` if
    any gradient is non-finite and 0.0 if there are no gradients. ``model`` may
    also be any iterable of parameters.
    """
    params = model.parameters() if hasattr(model, "parameters") else model
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0

    total = _combine_norms(_per_tensor_norms(grads, norm_type), norm_type)
    value = float(total.item())
    return value if math.isfinite(value) else float("inf")


def clip_gradients(model, max_grad_norm: float = 1.0) -> float:
    """
    Clip gradients and return the pre-clipping norm (``inf`` if non-finite).

    ``max_grad_norm`` of ``None`` or <= 0 disables clipping but still returns
    the norm.
    """
    if max_grad_norm is None or max_grad_norm <= 0:
        return get_grad_norm(model)

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
    return norm if math.isfinite(norm) else float("inf")


def gradient_health(model) -> Dict[str, Any]:
    """
    Batched gradient diagnostics.

    Keys from v1 are unchanged (``gradient_tensors``, ``finite``, ``nonzero``,
    ``finite_ratio``). v2 adds ``nan_tensors``, ``inf_tensors``, ``total_norm``,
    ``max_norm`` and ``worst_param`` (the tensor with the largest norm, which
    points at the layer that is exploding).
    """
    named = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    if not named:
        return {
            "gradient_tensors": 0,
            "finite": 0,
            "nonzero": 0,
            "finite_ratio": 1.0,
            "nan_tensors": 0,
            "inf_tensors": 0,
            "total_norm": 0.0,
            "max_norm": 0.0,
            "worst_param": None,
        }

    names = [n for n, _ in named]
    norms = _per_tensor_norms([g for _, g in named], 2.0)
    device = norms[0].device
    values = torch.stack([n.to(device=device, dtype=torch.float32) for n in norms]).cpu().tolist()

    total = len(values)
    finite = sum(1 for v in values if math.isfinite(v))
    nan = sum(1 for v in values if math.isnan(v))
    inf = sum(1 for v in values if math.isinf(v))
    nonzero = sum(1 for v in values if v != 0.0)

    finite_values = [(v, n) for v, n in zip(values, names) if math.isfinite(v)]
    if finite_values:
        max_norm, worst = max(finite_values, key=lambda item: item[0])
        total_norm = math.sqrt(sum(v * v for v, _ in finite_values))
    else:
        max_norm, worst, total_norm = 0.0, None, 0.0

    # A non-finite tensor is always the worst offender.
    bad = [n for v, n in zip(values, names) if not math.isfinite(v)]
    if bad:
        worst = bad[0]

    return {
        "gradient_tensors": total,
        "finite": finite,
        "nonzero": nonzero,
        "finite_ratio": finite / total if total else 1.0,
        "nan_tensors": nan,
        "inf_tensors": inf,
        "total_norm": total_norm if not bad else float("inf"),
        "max_norm": max_norm,
        "worst_param": worst,
    }


# ============================================================
# EMA Loss Tracker
# ============================================================

class EMALoss:
    """
    Exponential moving-average loss tracker (mean and variance).

    Individual batch losses are very noisy in language-model training. Non-finite
    values are ignored (and counted in ``skipped``) instead of permanently
    poisoning the average.
    """

    def __init__(self, decay: float = 0.95):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = float(decay)
        self.value: Optional[float] = None
        self.variance: float = 0.0
        self.count: int = 0
        self.skipped: int = 0

    def update(self, loss) -> Optional[float]:
        try:
            x = float(loss)
        except (TypeError, ValueError):
            self.skipped += 1
            return self.value

        if not math.isfinite(x):
            self.skipped += 1
            return self.value

        if self.value is None:
            self.value = x
            self.variance = 0.0
        else:
            alpha = 1.0 - self.decay
            diff = x - self.value
            increment = alpha * diff
            self.value += increment
            # Exponentially weighted variance (West / Finch recurrence).
            self.variance = self.decay * (self.variance + diff * increment)

        self.count += 1
        return self.value

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def reset(self) -> None:
        self.value = None
        self.variance = 0.0
        self.count = 0
        self.skipped = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "value": self.value,
            "variance": self.variance,
            "count": self.count,
            "skipped": self.skipped,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.value = state.get("value")
        self.variance = float(state.get("variance", 0.0))
        self.count = int(state.get("count", 0))
        self.skipped = int(state.get("skipped", 0))


# ============================================================
# Trainable-weight snapshots
# ============================================================

def snapshot_trainable_params(model) -> Dict[str, torch.Tensor]:
    """CPU copy of every trainable parameter.

    For LoRA/DoRA this is the adapter only (tens of MB), not the frozen base
    model, which makes snapshots cheap even for 8B-parameter models.
    """
    return {
        name: p.detach().to("cpu", copy=True)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def restore_trainable_params(model, snapshot: Dict[str, torch.Tensor], strict: bool = True) -> None:
    """Copy a snapshot from :func:`snapshot_trainable_params` back in place."""
    params = dict(model.named_parameters())
    missing = [name for name in snapshot if name not in params]
    if strict and missing:
        raise KeyError(f"Parameters missing from model: {missing[:5]}")

    with torch.no_grad():
        for name, saved in snapshot.items():
            p = params.get(name)
            if p is not None:
                p.copy_(saved.to(device=p.device, dtype=p.dtype))


# ============================================================
# Learning Rate Finder
# ============================================================

class LRFinder:
    """
    Robust learning-rate range test.

    * Snapshots/restores trainable weights only, the optimizer state (as a real
      copy), the GradScaler state and every group's LR.
    * Exponential LR sweep that preserves the *relative* LR of every param
      group, so layer-wise LR decay / gate boosts survive the sweep. The
      returned suggestion is expressed for the first ("reference") group.
    * Bias-corrected loss smoothing, divergence and NaN/Inf detection.
    * Cycles the loader if it has fewer than ``n`` batches.
    * Optional ``scaler`` and ``autocast_dtype`` so the sweep runs exactly like
      real training.
    """

    def __init__(
        self,
        model,
        opt,
        dev,
        start=1e-7,
        end=1.0,
        n=100,
        smooth_beta=0.98,
        divergence_factor=4.0,
        max_grad_norm=1.0,
        *,
        scaler=None,
        autocast_dtype=None,
    ):
        if not 0 < float(start) < float(end):
            raise ValueError("need 0 < start < end")
        if int(n) < 2:
            raise ValueError("n must be >= 2")

        self.model = model
        self.opt = opt
        self.dev = dev

        self.start = float(start)
        self.end = float(end)
        self.n = int(n)

        self.smooth_beta = float(smooth_beta)
        self.divergence_factor = float(divergence_factor)
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler
        self.autocast_dtype = autocast_dtype

        self.lrs: List[float] = []
        self.losses: List[float] = []      # smoothed (bias-corrected)
        self.raw_losses: List[float] = []
        self.stop_reason: Optional[str] = None
        self.skipped_steps = 0
        self._scales: List[float] = []

    # ---- LR plumbing -------------------------------------------------

    def _set_lr(self, lr: float) -> None:
        """Set the reference LR; every other group keeps its original ratio."""
        for group, scale in zip(self.opt.param_groups, self._scales):
            group["lr"] = lr * scale

    def _get_lr(self) -> float:
        if not self.opt.param_groups:
            return self.start
        return float(self.opt.param_groups[0]["lr"])

    def _batches(self, loader) -> Iterator[Any]:
        """Iterate the loader repeatedly so a short loader still yields ``n`` steps."""
        while True:
            produced = False
            for batch in loader:
                produced = True
                yield batch
            if not produced:
                return

    # ---- the sweep ---------------------------------------------------

    def range_test(self, loader, restore: bool = True) -> float:
        """Run the sweep and return a suggested LR.

        ``restore=True`` puts weights, optimizer state and scaler state back
        exactly as they were. Group LRs are always restored.
        """
        self.lrs.clear()
        self.losses.clear()
        self.raw_losses.clear()
        self.stop_reason = None
        self.skipped_steps = 0

        was_training = self.model.training
        original_lrs = [float(g["lr"]) for g in self.opt.param_groups]
        reference = next((b for b in original_lrs if b > 0), 1.0)
        self._scales = [b / reference for b in original_lrs]

        weights = snapshot_trainable_params(self.model) if restore else None
        opt_state = _clone_to_cpu(self.opt.state_dict()) if restore else None
        scaler_state = (
            copy.deepcopy(self.scaler.state_dict())
            if restore and self.scaler is not None and hasattr(self.scaler, "state_dict")
            else None
        )

        mult = (self.end / self.start) ** (1.0 / max(1, self.n - 1))
        lr = self.start
        self._set_lr(lr)

        beta = self.smooth_beta
        running = 0.0
        best = float("inf")
        scaler_active = _scaler_enabled(self.scaler)

        self.model.train()

        try:
            for step, batch in zip(range(self.n), self._batches(loader)):
                batch = move_batch_to_device(batch, self.dev)
                self.opt.zero_grad(set_to_none=True)

                with _autocast_ctx(self.dev, self.autocast_dtype):
                    loss = _extract_loss(self.model(**batch))

                if not is_finite_loss(loss):
                    self.stop_reason = "non-finite loss"
                    logger.warning("LR finder stopped: non-finite loss at step %d", step)
                    break

                raw = float(loss.detach().item())

                # Bias-corrected EMA (Adam-style): responsive from step 0.
                running = beta * running + (1.0 - beta) * raw
                smoothed = running / (1.0 - beta ** (step + 1))

                self.lrs.append(lr)
                self.raw_losses.append(raw)
                self.losses.append(smoothed)

                (self.scaler.scale(loss) if scaler_active else loss).backward()
                _, stepped = optimizer_step(
                    self.model, self.opt, self.scaler, self.max_grad_norm
                )
                if not stepped:
                    self.skipped_steps += 1

                best = min(best, smoothed)

                if step > 5 and smoothed > self.divergence_factor * best:
                    self.stop_reason = "diverged"
                    logger.info("LR finder stopped due to divergence.")
                    break

                lr *= mult
                self._set_lr(lr)

            if self.stop_reason is None:
                self.stop_reason = "completed"

        finally:
            if restore:
                if weights is not None:
                    restore_trainable_params(self.model, weights)
                if opt_state is not None:
                    self.opt.load_state_dict(opt_state)
                if scaler_state is not None:
                    self.scaler.load_state_dict(scaler_state)

            # Never leave the optimizer at the last (huge) swept LR.
            for group, original in zip(self.opt.param_groups, original_lrs):
                group["lr"] = original

            self.opt.zero_grad(set_to_none=True)
            self.model.train() if was_training else self.model.eval()

        return self.suggest()

    # ---- analysis ----------------------------------------------------

    def plot_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lrs, smoothed_losses) as arrays, ready for any plotting library."""
        return np.asarray(self.lrs, dtype=np.float64), np.asarray(self.losses, dtype=np.float64)

    def _curve(self) -> Tuple[np.ndarray, np.ndarray]:
        losses = np.asarray(self.losses, dtype=np.float64)
        lrs = np.asarray(self.lrs, dtype=np.float64)
        keep = np.isfinite(losses) & np.isfinite(lrs) & (lrs > 0)
        return lrs[keep], losses[keep]

    def candidates(self) -> Dict[str, float]:
        """All suggestion heuristics, before the safety factor.

        ``steepest``  LR of steepest loss descent (before the loss minimum)
        ``valley``    2/3 of the way down the longest descending stretch
        ``min_loss``  LR at the smoothed-loss minimum divided by 10
        ``auto``      median of the three (robust to one bad heuristic)
        """
        lrs, losses = self._curve()
        if len(losses) < 5:
            return {}

        window = int(min(7, max(3, len(losses) // 15)))
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(losses, (pad, pad), mode="edge")
        smooth = np.convolve(padded, np.ones(window) / window, mode="valid")

        log_lr = np.log10(lrs)
        if np.ptp(log_lr) == 0:
            return {}

        gradient = np.gradient(smooth, log_lr)

        min_idx = int(np.argmin(smooth))
        steep_idx = int(np.argmin(gradient[: max(2, min_idx + 1)]))

        descending = np.diff(smooth) < 0
        best_len = best_start = run_len = run_start = 0
        for i, is_down in enumerate(descending):
            if is_down:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len > best_len:
                    best_len, best_start = run_len, run_start
            else:
                run_len = 0
        valley_idx = (
            min(best_start + int(round(best_len * 2 / 3)), len(lrs) - 1)
            if best_len >= 2
            else steep_idx
        )

        steepest = float(lrs[steep_idx])
        valley = float(lrs[valley_idx])
        min_loss = float(lrs[min_idx]) / 10.0
        auto = float(np.median([steepest, valley, min_loss]))
        return {"steepest": steepest, "valley": valley, "min_loss": min_loss, "auto": auto}

    def suggest(self, method: str = "auto", safety_factor: float = 1.0) -> float:
        """Suggested LR for the reference group.

        ``suggest("steepest", safety_factor=0.1)`` reproduces the v1 heuristic.
        The result is clipped to ``[start, end]``.
        """
        lrs, losses = self._curve()

        if len(losses) == 0:
            return 1e-4
        if len(losses) < 5:
            return float(lrs[int(np.argmin(losses))])

        candidates = self.candidates()
        if not candidates:
            return float(lrs[int(np.argmin(losses))])
        if method not in candidates:
            raise ValueError(f"method must be one of {sorted(candidates)}")

        suggested = float(
            np.clip(candidates[method] * float(safety_factor), self.start, self.end)
        )
        logger.info("LR Finder suggestion (%s): %.3e", method, suggested)
        return suggested

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": len(self.lrs),
            "stop_reason": self.stop_reason,
            "skipped_steps": self.skipped_steps,
            "candidates": self.candidates(),
            "suggested": self.suggest(),
        }


# ============================================================
# Adaptive Gradient Accumulation
# ============================================================

def adaptive_accumulation(
    current_accumulation,
    batch_tokens,
    target_tokens=8192,
    maximum_multiplier=4,
):
    """
    Choose accumulation steps from a token budget.

    Example:
        batch_tokens = 2048, target_tokens = 8192  ->  accumulation = 4

    ``batch_tokens`` should be the real (non-padding) tokens per micro-batch;
    see :func:`count_batch_tokens`.
    """
    current = max(1, int(current_accumulation))

    if batch_tokens is None or batch_tokens <= 0:
        return current

    desired = math.ceil(target_tokens / batch_tokens)
    maximum = max(1, current * max(1, int(maximum_multiplier)))
    return max(1, min(desired, maximum))


# ============================================================
# Schedulers
# ============================================================
# All schedulers return LambdaLR. LambdaLR multiplies each group's own base LR
# by the returned factor, so every scheduler below preserves per-group LR
# ratios (layer-wise LR decay etc.) automatically.

def _ratio(lr: float, base_lr: float) -> float:
    return lr / base_lr if base_lr > 0 else 0.0


def _warmup_factor(step: int, warmup_steps: int) -> float:
    return float(step + 1) / warmup_steps


def cosine_scheduler(
    optimizer,
    base_lr,
    min_lr,
    warmup_steps,
    total_steps,
    num_cycles=0.5,
):
    """
    Stable warmup + cosine decay scheduler.

    ``num_cycles=0.5`` is a single half-cosine from ``base_lr`` down to
    ``min_lr``.
    """
    base_lr = float(base_lr)
    min_lr = min(float(min_lr), base_lr)
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(2.0 * math.pi * num_cycles * progress))
        return _ratio(min_lr + (base_lr - min_lr) * cosine, base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cosine_restart_scheduler(
    opt,
    base,
    mn,
    warm,
    tot,
    ri,
    amplitude_decay=0.15,
):
    """
    Warmup + cosine schedule with warm restarts every ``ri`` steps.

    Each restart peaks at ``1 / (1 + amplitude_decay * cycle)`` of the base LR
    (v1 hard-coded 0.15; pass 0 for constant-amplitude restarts). ``tot`` is
    accepted for API symmetry with the other schedulers.
    """
    base = float(base)
    mn = min(float(mn), base)
    warm = max(0, int(warm))
    tot = max(1, int(tot))  # noqa: F841  (kept for API symmetry)
    ri = max(1, int(ri))
    amplitude_decay = max(0.0, float(amplitude_decay))

    def lam(step):
        if step < warm:
            return _warmup_factor(step, warm)

        decay_step = step - warm
        cycle = decay_step // ri
        progress = (decay_step % ri) / ri

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        amplitude = 1.0 / (1.0 + amplitude_decay * cycle)
        return _ratio(mn + (base - mn) * cosine * amplitude, base)

    return torch.optim.lr_scheduler.LambdaLR(opt, lam)


def linear_scheduler(optimizer, base_lr, min_lr, warmup_steps, total_steps):
    """Warmup then a straight line from ``base_lr`` down to ``min_lr``."""
    base_lr = float(base_lr)
    min_lr = min(float(min_lr), base_lr)
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return _ratio(base_lr + (min_lr - base_lr) * progress, base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def constant_scheduler(optimizer, warmup_steps=0):
    """Warmup then a constant LR."""
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def wsd_scheduler(
    optimizer,
    base_lr,
    min_lr,
    warmup_steps,
    total_steps,
    decay_ratio=0.1,
):
    """
    Warmup - Stable - Decay.

    Holds ``base_lr`` until the final ``decay_ratio`` of training, then decays
    linearly to ``min_lr``. Unlike cosine, the LR during the long stable phase
    does not depend on ``total_steps``, so resumed or extended runs (e.g. after
    a dropped notebook session) stay on the same trajectory.
    """
    base_lr = float(base_lr)
    min_lr = min(float(min_lr), base_lr)
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    decay_steps = max(1, int(round(float(decay_ratio) * total_steps)))
    decay_start = max(warmup_steps, total_steps - decay_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        progress = min(1.0, max(0.0, progress))
        return _ratio(base_lr + (min_lr - base_lr) * progress, base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, cfg: OptimConfig, total_steps: int):
    """Create the scheduler named by ``cfg.scheduler``."""
    cfg.validate()
    total_steps = max(1, int(total_steps))
    warmup = cfg.warmup_steps_for(total_steps)
    name = str(cfg.scheduler).lower()

    if name == "cosine":
        return cosine_scheduler(
            optimizer, cfg.learning_rate, cfg.min_learning_rate,
            warmup, total_steps, cfg.num_cycles,
        )
    if name == "cosine_restarts":
        return cosine_restart_scheduler(
            optimizer, cfg.learning_rate, cfg.min_learning_rate,
            warmup, total_steps, cfg.restart_interval,
            amplitude_decay=cfg.restart_amplitude_decay,
        )
    if name == "linear":
        return linear_scheduler(
            optimizer, cfg.learning_rate, cfg.min_learning_rate, warmup, total_steps,
        )
    if name == "constant":
        return constant_scheduler(optimizer, warmup)
    if name == "wsd":
        return wsd_scheduler(
            optimizer, cfg.learning_rate, cfg.min_learning_rate,
            warmup, total_steps, cfg.wsd_decay_ratio,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def get_learning_rates(optimizer) -> List[float]:
    """Current LR of every param group."""
    return [float(g["lr"]) for g in optimizer.param_groups]


def scale_learning_rate(optimizer, factor: float, scheduler=None) -> None:
    """Persistently multiply the LR by ``factor``.

    Setting ``group["lr"]`` alone is undone by the next ``scheduler.step()``,
    because LambdaLR recomputes LRs from its own ``base_lrs``. This scales the
    live LR, ``initial_lr`` and the scheduler's ``base_lrs`` together.
    """
    factor = float(factor)
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * factor
        if "initial_lr" in group:
            group["initial_lr"] = group["initial_lr"] * factor
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [b * factor for b in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [lr * factor for lr in scheduler._last_lr]


# ============================================================
# Optimizer Factory
# ============================================================

def _build_param_groups(
    model,
    learning_rate: float,
    weight_decay: float,
    param_group_fn: Optional[Callable[[str, torch.nn.Parameter], Any]],
    no_decay_keywords: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, float, bool], List[torch.nn.Parameter]] = {}
    group_order: Dict[str, int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        group_name, multiplier = "default", 1.0
        if param_group_fn is not None:
            result = param_group_fn(name, param)
            if isinstance(result, str):
                group_name = result
            elif result is not None:
                group_name, multiplier = result[0], float(result[1])

        group_order.setdefault(group_name, len(group_order))

        lowered = name.lower()
        no_decay = param.ndim < 2 or any(k in lowered for k in no_decay_keywords)
        buckets.setdefault((group_name, multiplier, no_decay), []).append(param)

    if not buckets:
        raise ValueError("build_optimizer: model has no trainable parameters")

    groups = []
    for (group_name, multiplier, no_decay) in sorted(
        buckets, key=lambda k: (group_order[k[0]], k[1], k[2])
    ):
        groups.append(
            {
                "params": buckets[(group_name, multiplier, no_decay)],
                "lr": learning_rate * multiplier,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"{group_name}/{'no_decay' if no_decay else 'decay'}",
                "lr_mult": multiplier,
            }
        )
    return groups


def _can_use_fused(groups: Sequence[Dict[str, Any]]) -> bool:
    if not torch.cuda.is_available():
        return False
    return all(
        p.is_cuda and p.is_floating_point()
        for g in groups
        for p in g["params"]
    )


def build_optimizer(
    model,
    learning_rate=2e-4,
    weight_decay=0.01,
    optimizer_name="adamw",
    betas=(0.9, 0.95),
    eps=1e-8,
    *,
    param_group_fn: Optional[Callable[[str, torch.nn.Parameter], Any]] = None,
    no_decay_keywords: Sequence[str] = ("bias", "norm", "ln_"),
    fused: Optional[bool] = None,
):
    """
    Create a robust optimizer with no weight decay on biases / norms / 1-D
    parameters.

    ``optimizer_name``: ``adamw`` | ``adamw_torch`` | ``adamw_fused`` | ``adam``
    | ``adamw_8bit`` | ``paged_adamw_8bit`` (bitsandbytes; falls back to torch
    AdamW with a warning if unavailable).

    ``param_group_fn(name, param)`` may return ``None``, a group name, or
    ``(group_name, lr_multiplier)`` so the trainer can build layer-wise LR
    groups without this module knowing anything about layer numbering. Each
    group is still split into decay / no-decay parts and carries ``name`` and
    ``lr_mult`` keys (see :func:`describe_param_groups`).

    ``fused=None`` auto-enables fused AdamW when every trainable parameter is a
    CUDA float tensor, falling back to the standard kernel if that fails.
    """
    name = str(optimizer_name).lower()
    if name not in _VALID_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    groups = _build_param_groups(
        model, learning_rate, weight_decay, param_group_fn, no_decay_keywords
    )

    if name in ("adamw_8bit", "paged_adamw_8bit"):
        try:
            import bitsandbytes as bnb  # type: ignore

            cls = bnb.optim.PagedAdamW8bit if name.startswith("paged") else bnb.optim.AdamW8bit
            return cls(groups, lr=learning_rate, betas=betas, eps=eps)
        except Exception as exc:  # ImportError, missing CUDA libs, ...
            logger.warning(
                "bitsandbytes optimizer '%s' unavailable (%s); "
                "falling back to torch AdamW.",
                name, exc,
            )
            name = "adamw"

    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW

    if fused is None:
        use_fused = name == "adamw_fused" or _can_use_fused(groups)
    else:
        use_fused = bool(fused)

    if use_fused:
        try:
            return cls(groups, lr=learning_rate, betas=betas, eps=eps, fused=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Fused optimizer unavailable (%s); using the standard kernel.", exc)

    return cls(groups, lr=learning_rate, betas=betas, eps=eps)


def build_optimizer_from_config(model, cfg: OptimConfig, **kwargs):
    """``build_optimizer`` driven by an :class:`OptimConfig`."""
    cfg.validate()
    return build_optimizer(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        **kwargs,
    )


def describe_param_groups(optimizer) -> List[Dict[str, Any]]:
    """Structured summary of every param group.

    Handy for catching mis-named or mis-assigned groups (a recurring source of
    silent bugs in layer-wise LR setups).
    """
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        params = list(group["params"])
        summary.append(
            {
                "index": index,
                "name": group.get("name", f"group_{index}"),
                "num_tensors": len(params),
                "num_params": sum(p.numel() for p in params),
                "lr": float(group.get("lr", float("nan"))),
                "initial_lr": group.get("initial_lr"),
                "weight_decay": group.get("weight_decay"),
                "lr_mult": group.get("lr_mult"),
            }
        )
    return summary


def format_param_groups(optimizer) -> str:
    """Plain-text table of :func:`describe_param_groups`."""
    rows = describe_param_groups(optimizer)
    header = f"{'#':>2}  {'name':<28} {'tensors':>8} {'params':>14} {'lr':>10} {'wd':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        wd = r["weight_decay"]
        lines.append(
            f"{r['index']:>2}  {str(r['name'])[:28]:<28} {r['num_tensors']:>8} "
            f"{r['num_params']:>14,} {r['lr']:>10.3e} "
            f"{'-' if wd is None else format(wd, '.3g'):>7}"
        )
    return "\n".join(lines)


# ============================================================
# Optimizer Step
# ============================================================

def optimizer_step(
    model,
    optimizer,
    scaler=None,
    max_grad_norm=1.0,
    *,
    skip_nonfinite: bool = True,
):
    """
    Safe optimizer step. Returns ``(grad_norm, stepped)``.

    * With an enabled GradScaler the scaler always runs ``step()`` and
      ``update()``. ``GradScaler.step`` itself skips the underlying optimizer
      step when it recorded an inf/NaN during ``unscale_``, and ``update``
      is what shrinks the scale afterwards. Returning early (as v1 did) froze
      the scaler after the first overflow.
    * Without a scaler (or with a disabled one) a non-finite gradient norm skips
      the update when ``skip_nonfinite`` is true.
    * Gradients are always cleared afterwards.
    """
    active_scaler = _scaler_enabled(scaler)

    if active_scaler:
        scaler.unscale_(optimizer)

    grad_norm = clip_gradients(model, max_grad_norm)
    finite = math.isfinite(grad_norm)

    if not finite:
        logger.warning("Non-finite gradient norm (%s); this update will be skipped.", grad_norm)

    if active_scaler:
        scaler.step(optimizer)
        scaler.update()
        stepped = finite
    elif finite or not skip_nonfinite:
        optimizer.step()
        stepped = True
    else:
        stepped = False

    optimizer.zero_grad(set_to_none=True)
    return grad_norm, stepped


# ============================================================
# One full optimizer step over a window of micro-batches
# ============================================================

@dataclass
class StepResult:
    loss: float                      # token-weighted mean loss (nan if the window was discarded)
    grad_norm: Optional[float]       # pre-clip norm; None if no update was attempted
    stepped: bool                    # True if the optimizer actually updated the weights
    skipped_nonfinite: bool          # True if the window/update was dropped as non-finite
    tokens: int = 0                  # real tokens processed in the window
    micro_batches: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_forward(model, batch):
    return model(**batch)


def run_train_step(
    model,
    batches: Sequence[Any],
    optimizer,
    cfg: Optional[OptimConfig] = None,
    *,
    scaler=None,
    device=None,
    autocast_dtype=None,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    token_counter: Optional[Callable[[Any], int]] = None,
    max_grad_norm: Optional[float] = None,
    skip_nonfinite: Optional[bool] = None,
    detect_anomaly: Optional[bool] = None,
) -> StepResult:
    """
    One optimizer step over ``batches`` (a window of micro-batches).

    * The loss is weighted by each micro-batch's share of the window's tokens,
      so the accumulated gradient is exactly the gradient of the global token
      mean (plain ``loss / accumulation_steps`` is biased when lengths vary).
      Falls back to equal weights if token counts are unavailable.
    * A non-finite loss discards the whole window (``skip_nonfinite``).
    * Works with GradScaler, autocast, and custom losses via ``forward_fn``
      (e.g. DPO), which may return a loss tensor or an object with ``.loss``.
    * ``cfg`` supplies ``max_grad_norm``, ``skip_nonfinite`` and
      ``detect_anomaly``; keyword arguments override it.

    The model is expected to already be in train mode. Typical loop::

        cfg = OptimConfig(gradient_accumulation_steps=4).validate()
        optimizer = build_optimizer_from_config(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, total_steps)
        stats, guard = build_training_stats(cfg), build_instability_guard(model, cfg)

        for _ in range(total_steps):
            window = list(itertools.islice(batch_iter, cfg.gradient_accumulation_steps))
            result = run_train_step(model, window, optimizer, cfg, scaler=scaler)
            if result.stepped:
                scheduler.step()
            info = stats.update(result.loss, result.tokens)
            if guard.record(result.loss, result.stepped) == "abort":
                guard.rollback(optimizer, scheduler, lr_scale=0.5)
    """
    if not batches:
        raise ValueError("run_train_step needs at least one micro-batch")

    cfg = cfg if cfg is not None else OptimConfig()
    clip = cfg.max_grad_norm if max_grad_norm is None else max_grad_norm
    skip = cfg.skip_nonfinite if skip_nonfinite is None else skip_nonfinite
    anomaly = cfg.detect_anomaly if detect_anomaly is None else detect_anomaly

    forward = forward_fn or _default_forward
    counter = token_counter or count_label_tokens
    dev = device if device is not None else _infer_device(model)

    counts = [max(0, int(counter(b))) for b in batches]
    total_label_tokens = sum(counts)
    if total_label_tokens > 0:
        weights = [c / total_label_tokens for c in counts]
    else:
        weights = [1.0 / len(batches)] * len(batches)

    tokens_processed = sum(count_batch_tokens(b) for b in batches)
    active_scaler = _scaler_enabled(scaler)

    loss_value = 0.0
    nonfinite = False

    with (torch.autograd.set_detect_anomaly(True) if anomaly else nullcontext()):
        for batch, weight in zip(batches, weights):
            moved = move_batch_to_device(batch, dev)

            with _autocast_ctx(dev, autocast_dtype):
                loss = _extract_loss(forward(model, moved))

            if not is_finite_loss(loss):
                nonfinite = True
                if skip:
                    break

            scaled = loss * weight
            (scaler.scale(scaled) if active_scaler else scaled).backward()
            loss_value += float(loss.detach().item()) * weight

    if nonfinite and skip:
        optimizer.zero_grad(set_to_none=True)
        return StepResult(
            loss=float("nan"),
            grad_norm=None,
            stepped=False,
            skipped_nonfinite=True,
            tokens=tokens_processed,
            micro_batches=len(batches),
        )

    grad_norm, stepped = optimizer_step(
        model, optimizer, scaler, clip, skip_nonfinite=skip
    )
    return StepResult(
        loss=loss_value,
        grad_norm=grad_norm,
        stepped=stepped,
        skipped_nonfinite=not stepped,
        tokens=tokens_processed,
        micro_batches=len(batches),
    )


# ============================================================
# Instability detection and recovery
# ============================================================

class InstabilityGuard:
    """
    Tracks skipped steps and loss spikes; can roll trainable weights back to the
    last known-good snapshot.

    ``record()`` returns one of:
      ``"ok"``       healthy step
      ``"skipped"``  the update was skipped (non-finite loss/gradients)
      ``"spike"``    loss jumped ``spike_sigma`` deviations above its EMA
      ``"abort"``    ``max_consecutive_skips`` skipped steps in a row

    The guard reports; the trainer decides. A typical reaction to ``"abort"`` or
    ``"spike"`` is ``rollback(optimizer, scheduler, lr_scale=0.5)``.
    """

    def __init__(
        self,
        model=None,
        max_consecutive_skips: int = 20,
        spike_sigma: float = 6.0,
        spike_min_steps: int = 20,
        snapshot_every: int = 0,
        ema_decay: float = 0.98,
    ):
        self.model = model
        self.max_consecutive_skips = max(1, int(max_consecutive_skips))
        self.spike_sigma = float(spike_sigma)
        self.spike_min_steps = max(1, int(spike_min_steps))
        self.snapshot_every = max(0, int(snapshot_every))

        self.ema = EMALoss(ema_decay)
        self.consecutive_skips = 0
        self.total_skips = 0
        self.spikes = 0
        self.healthy_steps = 0
        self._steps_since_snapshot = 0
        self._snapshot: Optional[Dict[str, torch.Tensor]] = None

    @property
    def has_snapshot(self) -> bool:
        return self._snapshot is not None

    def snapshot(self) -> None:
        """Save the current trainable weights as the last known-good state."""
        if self.model is None:
            raise ValueError("InstabilityGuard needs a model to snapshot")
        self._snapshot = snapshot_trainable_params(self.model)
        self._steps_since_snapshot = 0

    def record(self, loss, stepped: bool = True) -> str:
        finite = is_finite_loss(loss)

        if not stepped or not finite:
            self.consecutive_skips += 1
            self.total_skips += 1
            if self.consecutive_skips >= self.max_consecutive_skips:
                return "abort"
            return "skipped"

        self.consecutive_skips = 0
        value = float(loss)

        status = "ok"
        if self.ema.count >= self.spike_min_steps and self.ema.value is not None:
            mean = self.ema.value
            floor = 0.05 * abs(mean)  # avoid hair-trigger on very smooth losses
            threshold = mean + self.spike_sigma * max(self.ema.std, floor)
            if value > threshold:
                status = "spike"
                self.spikes += 1
                value = threshold  # winsorize so a spike doesn't inflate the baseline

        self.ema.update(value)

        if status == "ok":
            self.healthy_steps += 1
            self._steps_since_snapshot += 1
            if (
                self.model is not None
                and self.snapshot_every > 0
                and self._steps_since_snapshot >= self.snapshot_every
            ):
                self.snapshot()

        return status

    def rollback(
        self,
        optimizer=None,
        scheduler=None,
        lr_scale: Optional[float] = None,
        reset_optimizer_state: bool = False,
    ) -> bool:
        """Restore the last snapshot. Returns False if there is none.

        ``lr_scale`` persistently scales the LR (see :func:`scale_learning_rate`);
        ``reset_optimizer_state`` clears Adam moments that may hold garbage from
        the bad steps.
        """
        if self.model is None or self._snapshot is None:
            return False

        restore_trainable_params(self.model, self._snapshot)

        if optimizer is not None:
            if reset_optimizer_state:
                optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            if lr_scale is not None:
                scale_learning_rate(optimizer, lr_scale, scheduler)

        self.consecutive_skips = 0
        return True

    def reset(self) -> None:
        self.ema.reset()
        self.consecutive_skips = 0
        self.total_skips = 0
        self.spikes = 0
        self.healthy_steps = 0
        self._steps_since_snapshot = 0


def build_instability_guard(model, cfg: OptimConfig, snapshot_every: int = 0) -> InstabilityGuard:
    cfg.validate()
    return InstabilityGuard(
        model,
        max_consecutive_skips=cfg.max_consecutive_skips,
        spike_sigma=cfg.spike_sigma,
        snapshot_every=snapshot_every,
    )


# ============================================================
# Training Statistics
# ============================================================

class TrainingStats:
    """Running loss / throughput statistics.

    ``update()`` keeps the v1 keys (``step``, ``loss``, ``ema_loss``, ``tokens``)
    and adds ``ema_std``, ``recent_loss``, ``tokens_per_second``,
    ``recent_tokens_per_second`` and ``skipped``. ``recent_history`` is a list
    of the latest finite losses, convenient for sparklines.
    """

    def __init__(
        self,
        ema_decay=0.95,
        recent_window: int = 50,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._clock = clock or time.perf_counter
        window = max(1, int(recent_window))

        self.steps = 0
        self.tokens = 0
        self.total_loss = 0.0
        self.finite_steps = 0
        self.skipped = 0

        self.ema = EMALoss(ema_decay)
        self.recent: Deque[float] = deque(maxlen=window)
        self._t0 = self._clock()
        self._marks: Deque[Tuple[float, int]] = deque(maxlen=window)
        self._marks.append((self._t0, 0))

    def update(self, loss, tokens=0) -> Dict[str, Any]:
        try:
            value = float(loss)
        except (TypeError, ValueError):
            value = float("nan")

        self.steps += 1
        self.tokens += int(tokens)

        if math.isfinite(value):
            self.finite_steps += 1
            self.total_loss += value
            self.recent.append(value)
        else:
            self.skipped += 1

        ema = self.ema.update(value)
        self._marks.append((self._clock(), self.tokens))

        return {
            "step": self.steps,
            "loss": value,
            "ema_loss": ema,
            "ema_std": self.ema.std,
            "recent_loss": self.recent_mean,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens_per_second,
            "recent_tokens_per_second": self.recent_tokens_per_second,
            "skipped": self.skipped,
        }

    @property
    def mean_loss(self) -> float:
        if self.finite_steps == 0:
            return float("nan")
        return self.total_loss / self.finite_steps

    @property
    def recent_mean(self) -> float:
        if not self.recent:
            return float("nan")
        return sum(self.recent) / len(self.recent)

    @property
    def recent_history(self) -> List[float]:
        return list(self.recent)

    @property
    def elapsed(self) -> float:
        return max(0.0, self._clock() - self._t0)

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.elapsed
        return self.tokens / elapsed if elapsed > 0 else 0.0

    @property
    def recent_tokens_per_second(self) -> float:
        """Throughput over the last ``recent_window`` updates (excludes
        one-off warmup / compile time that skews the overall average)."""
        if len(self._marks) < 2:
            return 0.0
        t_old, tok_old = self._marks[0]
        t_new, tok_new = self._marks[-1]
        dt = t_new - t_old
        return (tok_new - tok_old) / dt if dt > 0 else 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "total_loss": self.total_loss,
            "finite_steps": self.finite_steps,
            "skipped": self.skipped,
            "ema": self.ema.state_dict(),
            "recent": list(self.recent),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.steps = int(state.get("steps", 0))
        self.tokens = int(state.get("tokens", 0))
        self.total_loss = float(state.get("total_loss", 0.0))
        self.finite_steps = int(state.get("finite_steps", 0))
        self.skipped = int(state.get("skipped", 0))
        self.ema.load_state_dict(state.get("ema", {}))
        self.recent.clear()
        self.recent.extend(state.get("recent", []))


def build_training_stats(cfg: OptimConfig, **kwargs) -> TrainingStats:
    return TrainingStats(ema_decay=cfg.ema_decay, recent_window=cfg.recent_window, **kwargs)


def build_lr_finder(model, optimizer, device, cfg: OptimConfig, **kwargs) -> LRFinder:
    """``LRFinder`` configured from an :class:`OptimConfig`."""
    cfg.validate()
    return LRFinder(
        model,
        optimizer,
        device,
        start=cfg.lr_find_start,
        end=cfg.lr_find_end,
        n=cfg.lr_find_steps,
        smooth_beta=cfg.lr_find_smooth_beta,
        divergence_factor=cfg.lr_find_divergence_factor,
        max_grad_norm=cfg.max_grad_norm,
        **kwargs,
    )


# ============================================================
# Parameter Statistics
# ============================================================

def trainable_parameter_stats(model) -> Dict[str, Any]:
    """Parameter counts. v1 keys unchanged; v2 adds ``trainable_bytes`` and
    ``lora_params`` (parameters whose name contains ``lora``)."""
    trainable = 0
    total = 0
    trainable_bytes = 0
    lora = 0

    for name, p in model.named_parameters():
        count = p.numel()
        total += count
        if p.requires_grad:
            trainable += count
            trainable_bytes += count * p.element_size()
        if "lora" in name.lower():
            lora += count

    percentage = 100.0 * trainable / total if total > 0 else 0.0

    return {
        "trainable": trainable,
        "total": total,
        "percentage": percentage,
        "trainable_bytes": trainable_bytes,
        "lora_params": lora,
    }


# ============================================================
# Memory Utilities
# ============================================================

def gpu_memory_stats() -> Dict[str, Dict[str, float]]:
    """Per-GPU memory in GB. v1 keys unchanged; v2 adds ``free_gb`` / ``total_gb``
    (device-wide, from the driver) when available."""
    if not torch.cuda.is_available():
        return {}

    result: Dict[str, Dict[str, float]] = {}
    gib = 1024 ** 3

    for i in range(torch.cuda.device_count()):
        stats = {
            "allocated_gb": torch.cuda.memory_allocated(i) / gib,
            "reserved_gb": torch.cuda.memory_reserved(i) / gib,
            "peak_gb": torch.cuda.max_memory_allocated(i) / gib,
        }
        try:
            free, total = torch.cuda.mem_get_info(i)
            stats["free_gb"] = free / gib
            stats["total_gb"] = total / gib
        except Exception:
            pass
        result[f"gpu_{i}"] = stats

    return result


def reset_gpu_peak_memory() -> None:
    """Reset peak-memory counters (call before a phase you want to measure)."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


# ============================================================
# Public exports
# ============================================================

__all__ = [
    # v1 API
    "OptimConfig",
    "LRFinder",
    "EMALoss",
    "TrainingStats",
    "adaptive_accumulation",
    "cosine_scheduler",
    "cosine_restart_scheduler",
    "build_optimizer",
    "optimizer_step",
    "clip_gradients",
    "get_grad_norm",
    "gradient_health",
    "trainable_parameter_stats",
    "gpu_memory_stats",
    "move_batch_to_device",
    "is_finite_loss",
    # v2 additions
    "StepResult",
    "run_train_step",
    "InstabilityGuard",
    "linear_scheduler",
    "constant_scheduler",
    "wsd_scheduler",
    "build_scheduler",
    "build_optimizer_from_config",
    "build_lr_finder",
    "build_training_stats",
    "build_instability_guard",
    "describe_param_groups",
    "format_param_groups",
    "get_learning_rates",
    "scale_learning_rate",
    "snapshot_trainable_params",
    "restore_trainable_params",
    "count_batch_tokens",
    "count_label_tokens",
    "scale_loss_for_accumulation",
    "reset_gpu_peak_memory",
]
