
"""
FTRAIN Merge Algorithms
=======================

Deep, defensive, memory-conscious model-merging utilities for FTRAIN.

Public API
----------
compute_fisher
dare_merge
task_arithmetic
ties_merge_state_dict

Additional API
--------------
MergeConfig
MergeDiagnostics
weighted_merge
fisher_merge
dare_merge_state_dict
merge_state_dicts
calculate_delta
detect_tensor_health

Design goals
------------
- Diagonal Fisher information estimation.
- DARE merging.
- Task Arithmetic.
- TIES merging.
- Multi-model delta handling.
- Optional Fisher-weighted merging.
- Numerical safety and NaN/Inf protection.
- Device and dtype management.
- Shape validation.
- Deterministic seeded operations.
- CPU-offloaded Fisher storage.
- Memory-conscious TIES implementation.
- Per-tensor diagnostics.
- Delta clipping and candidate stabilization.
- Architecture-aware compatibility hooks.
- Preservation of integer/bool structural tensors.
- Backward compatibility with the original public functions.

Important
---------
These algorithms operate on already-corresponding tensors/keys.

Different model architectures must be aligned by a higher-level architecture
mapping layer (for example FTRAIN CBA) before invoking tensor-level arithmetic.

CBA can decide:
    - which tensors correspond,
    - which source model should dominate,
    - whether conflict is high,
    - whether a tensor needs alignment,
    - and which merge strategy should be selected.

This module then performs the mathematical merge safely.

No function here claims that a weight-space merge is behaviorally superior.
Post-merge benchmark/evaluation remains necessary.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


TensorDict = Dict[str, torch.Tensor]


__all__ = [
    # Configuration / diagnostics
    "MergeConfig",
    "MergeDiagnostics",

    # Fisher
    "compute_fisher",

    # Tensor operations
    "dare_merge",
    "task_arithmetic",
    "weighted_merge",
    "fisher_merge",

    # State-dict operations
    "ties_merge_state_dict",
    "dare_merge_state_dict",
    "merge_state_dicts",

    # Utilities
    "calculate_delta",
    "detect_tensor_health",
]


# =============================================================================
# Constants
# =============================================================================

_EPS = 1e-8
_DEFAULT_NUM_SAMPLES = 50

_DEFAULT_EXPLOSION_RATIO = 5.0
_DEFAULT_DELTA_NORM_RATIO = 2.5
_DEFAULT_DARE_DROP_RATE = 0.9
_DEFAULT_TIES_DENSITY = 0.2

_CONFLICT_ACTIONS = {
    "ties",
    "dare",
}

_SENSITIVE_NAME_HINTS = (
    "embed_tokens",
    "tok_embeddings",
    "word_embeddings",
    "lm_head",
    "output_projection",
    "layernorm",
    "layer_norm",
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class MergeConfig:
    """
    Centralized configuration for safe merge operations.

    The existing public functions do not require this class, but FTRAIN can
    construct one and feed its values into the individual algorithms.
    """

    strict_shapes: bool = True
    preserve_unmatched: bool = True

    max_delta_norm_ratio: float = _DEFAULT_DELTA_NORM_RATIO
    output_explosion_ratio: float = _DEFAULT_EXPLOSION_RATIO

    ties_density: float = _DEFAULT_TIES_DENSITY
    dare_drop_rate: float = _DEFAULT_DARE_DROP_RATE

    sanitize_nonfinite: bool = True

    protect_sensitive_tensors: bool = True
    sensitive_mix_floor: float = 0.15

    def validate(self) -> None:
        if self.max_delta_norm_ratio <= 0:
            raise ValueError(
                "max_delta_norm_ratio must be > 0."
            )

        if self.output_explosion_ratio <= 0:
            raise ValueError(
                "output_explosion_ratio must be > 0."
            )

        if not 0 <= self.ties_density <= 1:
            raise ValueError(
                "ties_density must be in [0, 1]."
            )

        if not 0 <= self.dare_drop_rate <= 1:
            raise ValueError(
                "dare_drop_rate must be in [0, 1]."
            )

        if not 0 <= self.sensitive_mix_floor <= 1:
            raise ValueError(
                "sensitive_mix_floor must be in [0, 1]."
            )


@dataclass
class MergeDiagnostics:
    """
    Captain-friendly diagnostics for a state-dict merge.
    """

    operation: str

    total_keys: int = 0
    merged_keys: int = 0
    preserved_keys: int = 0
    skipped_keys: int = 0
    shape_mismatch_keys: int = 0
    nonfloating_keys: int = 0
    repaired_keys: int = 0

    max_output_ratio: float = 1.0
    min_output_ratio: float = 1.0
    mean_output_ratio: float = 1.0

    max_delta_ratio: float = 0.0
    nonfinite_outputs: int = 0

    warnings: List[str] = field(default_factory=list)
    per_key: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "total_keys": self.total_keys,
            "merged_keys": self.merged_keys,
            "preserved_keys": self.preserved_keys,
            "skipped_keys": self.skipped_keys,
            "shape_mismatch_keys": self.shape_mismatch_keys,
            "nonfloating_keys": self.nonfloating_keys,
            "repaired_keys": self.repaired_keys,
            "max_output_ratio": self.max_output_ratio,
            "min_output_ratio": self.min_output_ratio,
            "mean_output_ratio": self.mean_output_ratio,
            "max_delta_ratio": self.max_delta_ratio,
            "nonfinite_outputs": self.nonfinite_outputs,
            "warnings": list(self.warnings),
            "per_key": dict(self.per_key),
        }


# =============================================================================
# General helpers
# =============================================================================

def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _is_mergeable_tensor(tensor: torch.Tensor) -> bool:
    """
    Floating-point and complex tensors support merge arithmetic.

    Integer/bool tensors are treated as structural metadata and are preserved
    from the target/base model rather than arithmetically interpolated.
    """
    return (
        isinstance(tensor, torch.Tensor)
        and (
            tensor.is_floating_point()
            or tensor.is_complex()
        )
    )


def _validate_probability(
    value: float,
    name: str,
) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a number, got {value!r}."
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1, got {value}."
        )

    return value


def _validate_positive(
    value: float,
    name: str,
) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a number, got {value!r}."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero, got {value}."
        )

    return value


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _same_shape(
    a: torch.Tensor,
    b: torch.Tensor,
) -> bool:
    return tuple(a.shape) == tuple(b.shape)


def _require_same_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    operation: str,
) -> None:
    if not _same_shape(a, b):
        raise ValueError(
            f"{operation}: tensor shape mismatch: "
            f"{tuple(a.shape)} vs {tuple(b.shape)}."
        )


def _require_floating(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if not _is_mergeable_tensor(tensor):
        raise TypeError(
            f"{name} must be floating-point or complex; "
            f"got dtype={tensor.dtype}, shape={tuple(tensor.shape)}."
        )


def _safe_norm(
    tensor: torch.Tensor,
) -> float:
    if tensor.numel() == 0:
        return 0.0

    try:
        return float(
            tensor.detach()
            .float()
            .norm()
            .item()
        )
    except (RuntimeError, ValueError):
        return float("inf")


def _finite_or_zero(
    tensor: torch.Tensor,
) -> torch.Tensor:
    if not _is_mergeable_tensor(tensor):
        return tensor

    return torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _copy_preserving_meta(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return tensor.detach().clone()


def _is_sensitive_key(
    key: str,
) -> bool:
    lowered = str(key).lower()
    return any(
        hint in lowered
        for hint in _SENSITIVE_NAME_HINTS
    )


def _compatible_baseline(
    tensor: torch.Tensor,
    baseline: Optional[torch.Tensor],
) -> bool:
    return (
        baseline is not None
        and isinstance(baseline, torch.Tensor)
        and _same_shape(tensor, baseline)
        and _is_mergeable_tensor(tensor)
        and _is_mergeable_tensor(baseline)
    )


def _repair_output(
    tensor: torch.Tensor,
    *,
    clamp_abs: Optional[float] = None,
) -> torch.Tensor:
    if not _is_mergeable_tensor(tensor):
        return tensor

    result = torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if clamp_abs is not None:
        result = result.clamp(
            min=-float(clamp_abs),
            max=float(clamp_abs),
        )

    return result


def _normalize_weights(
    weights: Sequence[float],
) -> List[float]:
    values = []

    for weight in weights:
        weight = _safe_float(
            weight,
            default=0.0,
        )

        if weight < 0:
            raise ValueError(
                "Merge weights cannot be negative."
            )

        values.append(weight)

    total = sum(values)

    if total <= _EPS:
        return [
            1.0 / max(1, len(values))
            for _ in values
        ]

    return [
        value / total
        for value in values
    ]


def _effective_sensitive_alpha(
    alpha: float,
    *,
    sensitive: bool,
    floor: float,
) -> float:
    """
    Keep a sensitive parameter from being completely overwritten.

    alpha represents the contribution of the incoming/source tensor.
    """
    alpha = max(0.0, min(1.0, float(alpha)))

    if not sensitive:
        return alpha

    return min(
        alpha,
        1.0 - float(floor),
    )


# =============================================================================
# Device and autocast helpers
# =============================================================================

def _resolve_device(
    device: Union[str, torch.device],
) -> torch.device:
    resolved = torch.device(device)

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    if resolved.type == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or
            not torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested, but MPS is not available."
            )

    return resolved


def _supports_amp(
    device: torch.device,
) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_available()

    if device.type == "cpu":
        return hasattr(torch, "amp")

    if device.type == "mps":
        return hasattr(torch, "amp")

    return False


def _choose_amp_dtype(
    device: torch.device,
) -> torch.dtype:
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass

        return torch.float16

    if device.type == "cpu":
        return torch.bfloat16

    if device.type == "mps":
        return torch.float16

    return torch.float32


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    if not enabled or not _supports_amp(device):
        return torch.autocast(
            device_type="cpu",
            enabled=False,
        )

    dtype = _choose_amp_dtype(device)

    try:
        return torch.amp.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=True,
        )
    except (AttributeError, TypeError):
        try:
            return torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=True,
            )
        except Exception:
            return torch.autocast(
                device_type="cpu",
                enabled=False,
            )


def _move_batch_to_device(
    batch: Any,
    device: torch.device,
) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

    if isinstance(batch, Mapping):
        return {
            key: _move_batch_to_device(
                value,
                device,
            )
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(
            _move_batch_to_device(
                value,
                device,
            )
            for value in batch
        )

    if isinstance(batch, list):
        return [
            _move_batch_to_device(
                value,
                device,
            )
            for value in batch
        ]

    return batch


def _extract_loss(
    outputs: Any,
) -> Optional[torch.Tensor]:
    if hasattr(outputs, "loss"):
        loss = outputs.loss
    elif isinstance(outputs, Mapping):
        loss = outputs.get("loss")
    elif isinstance(outputs, (tuple, list)) and outputs:
        loss = outputs[0]
    else:
        loss = None

    if not isinstance(loss, torch.Tensor):
        return None

    if loss.numel() != 1:
        return None

    return loss


def _batch_size_from_batch(
    batch: Any,
) -> int:
    if isinstance(batch, Mapping):
        for key in (
            "input_ids",
            "labels",
            "attention_mask",
        ):
            value = batch.get(key)

            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
            ):
                return max(1, int(value.shape[0]))

        for value in batch.values():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
            ):
                return max(1, int(value.shape[0]))

    if (
        isinstance(batch, torch.Tensor)
        and batch.ndim >= 1
    ):
        return max(1, int(batch.shape[0]))

    return 1


# =============================================================================
# Tensor health / diagnostics
# =============================================================================

def detect_tensor_health(
    tensor: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Return numerical-health diagnostics without mutating the tensor.
    """
    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "tensor must be a torch.Tensor."
        )

    total = int(tensor.numel())

    if (
        total == 0
        or
        not _is_mergeable_tensor(tensor)
    ):
        return {
            "finite": True,
            "finite_ratio": 1.0,
            "norm": 0.0,
            "baseline_norm": 0.0,
            "norm_ratio": 1.0,
            "max_abs": 0.0,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }

    finite_count = int(
        torch.isfinite(tensor).sum().item()
    )

    finite_ratio = (
        finite_count / total
    )

    value = tensor.detach().float()

    result = {
        "finite": finite_ratio == 1.0,
        "finite_ratio": finite_ratio,
        "norm": _safe_norm(tensor),
        "baseline_norm": 0.0,
        "norm_ratio": 1.0,
        "max_abs": _safe_float(
            value.abs().max()
        ),
        "mean": _safe_float(
            value.mean()
        ),
        "std": _safe_float(
            value.std(
                unbiased=False
            )
        ) if total > 1 else 0.0,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }

    if (
        baseline is not None
        and
        _compatible_baseline(
            tensor,
            baseline,
        )
    ):
        base_norm = _safe_norm(
            baseline
        )

        result["baseline_norm"] = base_norm

        if (
            base_norm > _EPS
            and
            math.isfinite(base_norm)
        ):
            result["norm_ratio"] = (
                result["norm"]
                /
                max(
                    base_norm,
                    _EPS,
                )
            )

    return result


# =============================================================================
# Fisher Information
# =============================================================================

def _safe_grad_square(
    grad: torch.Tensor,
) -> torch.Tensor:
    value = grad.detach().float()

    value = torch.nan_to_num(
        value,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return value.square()


def compute_fisher(
    model: nn.Module,
    loader: Iterator,
    device: Union[str, torch.device] = "cuda",
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    use_amp: bool = True,
    *,
    trainable_only: bool = False,
    offload_to_cpu: bool = True,
    max_grad_norm: Optional[float] = None,
    clear_cache_every: int = 0,
    loss_reduction: str = "mean",
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Compute a diagonal Fisher information approximation.

    The returned tensor for each parameter is the estimated mean squared
    gradient.

    Fisher storage is FP32 and may be CPU-offloaded to avoid consuming GPU
    memory needed by the model itself.
    """
    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be > 0."
        )

    if max_grad_norm is not None:
        max_grad_norm = _validate_positive(
            max_grad_norm,
            "max_grad_norm",
        )

    loss_reduction = str(
        loss_reduction
    ).lower()

    if loss_reduction not in {
        "mean",
        "sum",
    }:
        raise ValueError(
            "loss_reduction must be 'mean' or 'sum'."
        )

    resolved_device = _resolve_device(
        device
    )

    was_training = model.training

    fisher: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue

        if (
            trainable_only
            and
            not param.requires_grad
        ):
            continue

        storage_device = (
            torch.device("cpu")
            if offload_to_cpu
            else param.device
        )

        fisher[name] = torch.zeros(
            param.shape,
            dtype=torch.float32,
            device=storage_device,
        )

    if not fisher:
        logger.warning(
            "No floating-point parameters selected for Fisher."
        )
        return None

    samples_processed = 0
    batches_processed = 0
    invalid_batches = 0

    try:
        model.eval()

        amp_enabled = bool(
            use_amp
            and
            _supports_amp(
                resolved_device
            )
        )

        for batch in loader:
            if samples_processed >= num_samples:
                break

            batch_size = _batch_size_from_batch(
                batch
            )

            remaining = (
                num_samples
                -
                samples_processed
            )

            effective_batch_size = min(
                batch_size,
                remaining,
            )

            batch_device = (
                _move_batch_to_device(
                    batch,
                    resolved_device,
                )
            )

            model.zero_grad(
                set_to_none=True
            )

            try:
                with _autocast_context(
                    resolved_device,
                    amp_enabled,
                ):
                    outputs = model(
                        **batch_device
                    )

                    loss = _extract_loss(
                        outputs
                    )

                if loss is None:
                    invalid_batches += 1
                    continue

                if not bool(
                    torch.isfinite(
                        loss.detach()
                    ).item()
                ):
                    invalid_batches += 1
                    continue

                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm,
                    )

                with torch.no_grad():
                    for name, param in (
                        model.named_parameters()
                    ):
                        if name not in fisher:
                            continue

                        grad = param.grad

                        if grad is None:
                            continue

                        grad_sq = _safe_grad_square(
                            grad
                        )

                        if offload_to_cpu:
                            grad_sq = grad_sq.cpu()

                        fisher[name].add_(
                            grad_sq,
                            alpha=float(
                                effective_batch_size
                            ),
                        )

                samples_processed += (
                    effective_batch_size
                )

                batches_processed += 1

            except RuntimeError as exc:
                invalid_batches += 1

                logger.warning(
                    "Skipping Fisher batch due to runtime error: %s",
                    exc,
                )

            finally:
                model.zero_grad(
                    set_to_none=True
                )

            if (
                clear_cache_every > 0
                and
                batches_processed > 0
                and
                batches_processed
                %
                clear_cache_every
                == 0
                and
                resolved_device.type == "cuda"
            ):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        if samples_processed <= 0:
            logger.error(
                "Fisher processed zero valid samples."
            )
            return None

        denominator = float(
            max(
                1,
                samples_processed,
            )
        )

        with torch.no_grad():
            for name in fisher:
                fisher[name].div_(
                    denominator
                )

                fisher[name].nan_to_num_(
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        logger.info(
            "Fisher complete: %d samples, %d batches, "
            "%d invalid batches, %d tensors.",
            samples_processed,
            batches_processed,
            invalid_batches,
            len(fisher),
        )

        return fisher

    except Exception as exc:
        logger.exception(
            "Fisher computation failed: %s",
            exc,
        )
        return None

    finally:
        model.zero_grad(
            set_to_none=True
        )

        if was_training:
            model.train()


# =============================================================================
# DARE tensor merge
# =============================================================================

def dare_merge(
    da: torch.Tensor,
    db: torch.Tensor,
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    DARE-style merge between two tensors.

    ``da`` is treated as the base tensor and ``db`` as the incoming tensor.

    For 0 < drop_rate < 1:
        delta = db - da
        retained_delta = random_mask(delta)
        result = da + retained_delta

    When rescale=True, the retained delta is divided by the keep probability
    so its expected magnitude is approximately preserved.

    The random generator is local and deterministic when ``seed`` is supplied.
    """
    _require_floating(
        da,
        name="da",
    )
    _require_floating(
        db,
        name="db",
    )

    _require_same_shape(
        da,
        db,
        operation="DARE merge",
    )

    drop_rate = _validate_probability(
        drop_rate,
        "drop_rate",
    )

    if drop_rate == 0.0:
        return db.detach().clone()

    if drop_rate == 1.0:
        return da.detach().clone()

    keep_probability = 1.0 - drop_rate

    generator = None

    if seed is not None:
        generator = torch.Generator(
            device=da.device
        )
        generator.manual_seed(
            int(seed)
        )

    with torch.no_grad():
        base = da.float()
        target = db.float()

        delta = (
            target
            -
            base
        )

        random_values = torch.rand(
            delta.shape,
            device=delta.device,
            generator=generator,
        )

        mask = (
            random_values < keep_probability
        ).to(
            dtype=delta.dtype
        )

        if rescale:
            mask.div_(
                keep_probability
            )

        result = (
            base
            +
            delta * mask
        )

        result = _repair_output(
            result
        )

        return result.to(
            device=da.device,
            dtype=da.dtype,
        )


# =============================================================================
# DARE state-dict merge
# =============================================================================

def dare_merge_state_dict(
    model_a: Mapping[str, torch.Tensor],
    model_b: Mapping[str, torch.Tensor],
    *,
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None,
    strict_shapes: bool = True,
    preserve_unmatched: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Apply DARE independently to every corresponding state-dict tensor.
    """
    drop_rate = _validate_probability(
        drop_rate,
        "drop_rate",
    )

    output: Dict[str, torch.Tensor] = {}

    names = list(model_a.keys())

    if preserve_unmatched:
        names.extend(
            key
            for key in model_b.keys()
            if key not in model_a
        )

    # Deterministic name order.
    names = sorted(
        set(names)
    )

    for index, name in enumerate(names):
        a = model_a.get(name)
        b = model_b.get(name)

        if a is None:
            if preserve_unmatched and b is not None:
                output[name] = b.detach().clone()
            continue

        if b is None:
            if preserve_unmatched:
                output[name] = a.detach().clone()
            continue

        if (
            not _is_mergeable_tensor(a)
            or
            not _is_mergeable_tensor(b)
        ):
            output[name] = a.detach().clone()
            continue

        if not _same_shape(a, b):
            if strict_shapes:
                raise ValueError(
                    f"DARE state merge: shape mismatch for {name!r}: "
                    f"{tuple(a.shape)} vs {tuple(b.shape)}."
                )

            output[name] = a.detach().clone()
            continue

        tensor_seed = (
            None
            if seed is None
            else int(seed) + index
        )

        output[name] = dare_merge(
            a,
            b,
            drop_rate=drop_rate,
            rescale=rescale,
            seed=tensor_seed,
        )

    return output


# =============================================================================
# Task Arithmetic helpers
# =============================================================================

def calculate_delta(
    model: Mapping[str, torch.Tensor],
    base: Mapping[str, torch.Tensor],
    *,
    strict_shapes: bool = True,
    preserve_unmatched_base: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute ``model - base`` for all compatible floating tensors.

    Integer/bool tensors are excluded because task arithmetic is intended for
    learned parameter deltas.
    """
    result: Dict[str, torch.Tensor] = {}

    for name, base_tensor in base.items():
        if not torch.is_tensor(base_tensor):
            continue

        tensor = model.get(name)

        if tensor is None:
            if preserve_unmatched_base:
                result[name] = torch.zeros_like(
                    base_tensor
                )
            continue

        if not _is_mergeable_tensor(
            base_tensor
        ):
            continue

        _require_floating(
            tensor,
            name=f"model[{name!r}]",
        )

        if not _same_shape(
            tensor,
            base_tensor,
        ):
            if strict_shapes:
                raise ValueError(
                    f"calculate_delta: shape mismatch for {name!r}: "
                    f"{tuple(tensor.shape)} vs "
                    f"{tuple(base_tensor.shape)}."
                )
            continue

        delta = (
            tensor.float()
            -
            base_tensor.float()
        )

        result[name] = torch.nan_to_num(
            delta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return result


def _clip_delta_norm(
    delta: torch.Tensor,
    base: torch.Tensor,
    max_norm_ratio: float,
) -> torch.Tensor:
    max_norm_ratio = _validate_positive(
        max_norm_ratio,
        "max_norm_ratio",
    )

    base_norm = _safe_norm(
        base
    )

    delta_norm = _safe_norm(
        delta
    )

    if (
        not math.isfinite(base_norm)
        or
        not math.isfinite(delta_norm)
    ):
        return torch.zeros_like(
            delta
        )

    if delta_norm <= _EPS:
        return delta

    if base_norm <= _EPS:
        return delta

    allowed = (
        base_norm
        *
        max_norm_ratio
    )

    if delta_norm <= allowed:
        return delta

    return (
        delta
        *
        (
            allowed
            /
            max(
                delta_norm,
                _EPS,
            )
        )
    )


# =============================================================================
# Task Arithmetic
# =============================================================================

def task_arithmetic(
    ma: Mapping[str, torch.Tensor],
    mb: Mapping[str, torch.Tensor],
    base: Mapping[str, torch.Tensor],
    scaling: float = 0.5,
    max_norm_ratio: float = 2.5,
    *,
    strict_shapes: bool = True,
    preserve_unmatched_base: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Merge two task deltas against a base model.

        delta_a = A - Base
        delta_b = B - Base

        result = Base + scaling * (delta_a + delta_b)

    ``max_norm_ratio`` limits the combined delta relative to the base tensor,
    reducing the chance of catastrophic movement.
    """
    scaling = _safe_float(
        scaling,
        default=float("nan"),
    )

    if not math.isfinite(scaling):
        raise ValueError(
            "scaling must be finite."
        )

    max_norm_ratio = _validate_positive(
        max_norm_ratio,
        "max_norm_ratio",
    )

    merged: Dict[str, torch.Tensor] = {}

    for name, base_tensor in base.items():
        if not torch.is_tensor(base_tensor):
            continue

        tensor_a = ma.get(name)
        tensor_b = mb.get(name)

        if (
            tensor_a is None
            and
            tensor_b is None
        ):
            if preserve_unmatched_base:
                merged[name] = (
                    base_tensor.detach().clone()
                )
            continue

        if not _is_mergeable_tensor(
            base_tensor
        ):
            merged[name] = (
                base_tensor.detach().clone()
            )
            continue

        if tensor_a is not None:
            _require_floating(
                tensor_a,
                name=f"ma[{name!r}]",
            )

            if strict_shapes:
                _require_same_shape(
                    tensor_a,
                    base_tensor,
                    operation=f"Task Arithmetic ({name})",
                )
            elif not _same_shape(
                tensor_a,
                base_tensor,
            ):
                tensor_a = None

        if tensor_b is not None:
            _require_floating(
                tensor_b,
                name=f"mb[{name!r}]",
            )

            if strict_shapes:
                _require_same_shape(
                    tensor_b,
                    base_tensor,
                    operation=f"Task Arithmetic ({name})",
                )
            elif not _same_shape(
                tensor_b,
                base_tensor,
            ):
                tensor_b = None

        base_fp32 = (
            base_tensor
            .detach()
            .float()
        )

        delta_a = (
            tensor_a.detach().float()
            -
            base_fp32
            if tensor_a is not None
            else torch.zeros_like(
                base_fp32
            )
        )

        delta_b = (
            tensor_b.detach().float()
            -
            base_fp32
            if tensor_b is not None
            else torch.zeros_like(
                base_fp32
            )
        )

        combined = (
            float(scaling)
            *
            (
                delta_a
                +
                delta_b
            )
        )

        combined = torch.nan_to_num(
            combined,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        combined = _clip_delta_norm(
            combined,
            base_fp32,
            max_norm_ratio,
        )

        result = (
            base_fp32
            +
            combined
        )

        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        merged[name] = result.to(
            device=base_tensor.device,
            dtype=base_tensor.dtype,
        )

    return merged


# =============================================================================
# Weighted merge
# =============================================================================

def weighted_merge(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    reference: Optional[torch.Tensor] = None,
    normalize_weights: bool = True,
    max_output_ratio: Optional[float] = None,
    sanitize: bool = True,
) -> torch.Tensor:
    """
    Memory-conscious weighted average of two or more tensors.

    Unlike ``torch.stack(tensors)``, this accumulates into one FP32 tensor and
    avoids creating a [N, ...] temporary stack.

    ``reference`` can be supplied to detect/cap output norm growth.
    """
    if not tensors:
        raise ValueError(
            "weighted_merge requires at least one tensor."
        )

    weights = list(weights)

    if len(weights) != len(tensors):
        raise ValueError(
            "weights length must match tensors length."
        )

    for index, tensor in enumerate(tensors):
        _require_floating(
            tensor,
            name=f"tensors[{index}]",
        )

    first = tensors[0]

    for index, tensor in enumerate(tensors[1:], start=1):
        _require_same_shape(
            first,
            tensor,
            operation=f"Weighted merge tensor {index}",
        )

    final_weights = (
        _normalize_weights(weights)
        if normalize_weights
        else [
            float(weight)
            for weight in weights
        ]
    )

    accumulator = torch.zeros_like(
        first,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for tensor, weight in zip(
            tensors,
            final_weights,
        ):
            accumulator.add_(
                tensor.detach().float(),
                alpha=float(weight),
            )

        if sanitize:
            accumulator = _repair_output(
                accumulator
            )

        if (
            reference is not None
            and
            max_output_ratio is not None
        ):
            _require_same_shape(
                accumulator,
                reference,
                operation="Weighted merge reference",
            )

            ref_norm = _safe_norm(
                reference
            )

            out_norm = _safe_norm(
                accumulator
            )

            ratio = (
                out_norm
                /
                max(
                    ref_norm,
                    _EPS,
                )
            )

            if (
                ref_norm > _EPS
                and
                math.isfinite(ratio)
                and
                ratio > float(max_output_ratio)
            ):
                accumulator = (
                    accumulator
                    *
                    (
                        ref_norm
                        *
                        float(max_output_ratio)
                        /
                        max(
                            out_norm,
                            _EPS,
                        )
                    )
                )

        return accumulator.to(
            device=first.device,
            dtype=first.dtype,
        )


# =============================================================================
# Fisher-weighted merge
# =============================================================================

def fisher_merge(
    tensors: Sequence[torch.Tensor],
    fisher: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-8,
    reference: Optional[torch.Tensor] = None,
    max_output_ratio: Optional[float] = None,
) -> torch.Tensor:
    """
    Fisher-weighted parameter merge.

    For each tensor position:

        merged = sum(F_i * theta_i) / sum(F_i)

    This gives a parameter more influence when its estimated diagonal Fisher
    importance is larger.

    Important:
        Fisher tensors must correspond exactly to `tensors`.
    """
    if not tensors:
        raise ValueError(
            "fisher_merge requires at least one tensor."
        )

    if len(tensors) != len(fisher):
        raise ValueError(
            "tensors and fisher must have the same length."
        )

    first = tensors[0]

    for index, tensor in enumerate(tensors):
        _require_floating(
            tensor,
            name=f"tensors[{index}]",
        )

        _require_same_shape(
            first,
            tensor,
            operation=f"Fisher merge tensor {index}",
        )

    weighted_sum = torch.zeros_like(
        first,
        dtype=torch.float32,
    )

    importance_sum = torch.zeros_like(
        first,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for index, (
            tensor,
            fisher_tensor,
        ) in enumerate(
            zip(
                tensors,
                fisher,
            )
        ):
            if not isinstance(
                fisher_tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    f"fisher[{index}] must be a tensor."
                )

            _require_same_shape(
                tensor,
                fisher_tensor,
                operation=f"Fisher merge fisher[{index}]",
            )

            importance = (
                fisher_tensor
                .detach()
                .float()
            )

            importance = torch.nan_to_num(
                importance,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(
                float(epsilon)
            )

            weighted_sum.add_(
                tensor.detach().float()
                *
                importance,
            )

            importance_sum.add_(
                importance
            )

        importance_sum.clamp_min_(
            float(epsilon)
        )

        output = (
            weighted_sum
            /
            importance_sum
        )

        output = _repair_output(
            output
        )

        if (
            reference is not None
            and
            max_output_ratio is not None
        ):
            ref_norm = _safe_norm(
                reference
            )

            out_norm = _safe_norm(
                output
            )

            if (
                ref_norm > _EPS
                and
                out_norm
                /
                ref_norm
                >
                float(max_output_ratio)
            ):
                scale = (
                    ref_norm
                    *
                    float(max_output_ratio)
                    /
                    max(
                        out_norm,
                        _EPS,
                    )
                )

                output.mul_(
                    scale
                )

        return output.to(
            device=first.device,
            dtype=first.dtype,
        )


# =============================================================================
# TIES implementation
# =============================================================================

def _topk_mask(
    tensor: torch.Tensor,
    density: float,
) -> torch.Tensor:
    density = _validate_probability(
        density,
        "density",
    )

    if tensor.numel() == 0:
        return torch.zeros_like(
            tensor,
            dtype=torch.bool,
        )

    if density <= 0:
        return torch.zeros_like(
            tensor,
            dtype=torch.bool,
        )

    if density >= 1:
        return torch.ones_like(
            tensor,
            dtype=torch.bool,
        )

    flat = tensor.detach().float().abs()

    k = max(
        1,
        min(
            flat.numel(),
            int(
                math.ceil(
                    flat.numel()
                    *
                    density
                )
            ),
        ),
    )

    if k >= flat.numel():
        return torch.ones_like(
            tensor,
            dtype=torch.bool,
        )

    threshold = torch.topk(
        flat.reshape(-1),
        k=k,
        largest=True,
        sorted=False,
    ).values.min()

    return (
        tensor.abs()
        >=
        threshold
    )


def _sign_consensus(
    deltas: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not deltas:
        raise ValueError(
            "TIES requires at least one delta."
        )

    positive_score = torch.zeros_like(
        deltas[0],
        dtype=torch.float32,
    )

    negative_score = torch.zeros_like(
        deltas[0],
        dtype=torch.float32,
    )

    for delta in deltas:
        value = (
            delta.detach()
            .float()
        )

        positive_score.add_(
            torch.where(
                value > 0,
                value.abs(),
                torch.zeros_like(
                    value
                ),
            )
        )

        negative_score.add_(
            torch.where(
                value < 0,
                value.abs(),
                torch.zeros_like(
                    value
                ),
            )
        )

    elected = torch.zeros_like(
        positive_score
    )

    elected[
        positive_score > negative_score
    ] = 1.0

    elected[
        negative_score > positive_score
    ] = -1.0

    return elected


def _ties_merge_deltas(
    deltas: Sequence[torch.Tensor],
    density: float,
    scaling: float,
) -> torch.Tensor:
    if not deltas:
        raise ValueError(
            "TIES requires at least one delta."
        )

    density = _validate_probability(
        density,
        "density",
    )

    if not math.isfinite(
        float(scaling)
    ):
        raise ValueError(
            "scaling must be finite."
        )

    trimmed = []

    for delta in deltas:
        mask = _topk_mask(
            delta,
            density,
        )

        trimmed.append(
            torch.where(
                mask,
                delta,
                torch.zeros_like(
                    delta
                ),
            )
        )

    elected_sign = _sign_consensus(
        trimmed
    )

    total = torch.zeros_like(
        trimmed[0],
        dtype=torch.float32,
    )

    count = torch.zeros_like(
        trimmed[0],
        dtype=torch.float32,
    )

    for delta in trimmed:
        value = (
            delta.detach()
            .float()
        )

        matching = (
            (
                torch.sign(value)
                ==
                elected_sign
            )
            &
            (
                elected_sign
                !=
                0
            )
        )

        total.add_(
            torch.where(
                matching,
                value,
                torch.zeros_like(
                    value
                ),
            )
        )

        count.add_(
            matching.to(
                torch.float32
            )
        )

    count.clamp_min_(
        1.0
    )

    output = (
        total
        /
        count
    )

    output.mul_(
        float(scaling)
    )

    return torch.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def ties_merge_state_dict(
    models: List[Mapping[str, torch.Tensor]],
    base: Mapping[str, torch.Tensor],
    density: float = 0.2,
    scaling: float = 1.0,
    *,
    strict_shapes: bool = True,
    include_unmatched: bool = True,
    max_delta_norm_ratio: Optional[float] = None,
    diagnostics: Optional[MergeDiagnostics] = None,
) -> Dict[str, torch.Tensor]:
    """
    Memory-conscious TIES state-dict merge.

    The implementation intentionally avoids stacking deltas from all models.
    This is important for large models where a temporary tensor with an extra
    model dimension can exceed available RAM/VRAM.
    """
    if not models:
        raise ValueError(
            "TIES requires at least one model."
        )

    density = _validate_probability(
        density,
        "density",
    )

    if not math.isfinite(
        float(scaling)
    ):
        raise ValueError(
            "scaling must be finite."
        )

    output: Dict[str, torch.Tensor] = {}

    if diagnostics is None:
        diagnostics = MergeDiagnostics(
            operation="ties"
        )

    diagnostics.total_keys = len(
        base
    )

    for name, base_tensor in base.items():
        if not isinstance(
            base_tensor,
            torch.Tensor,
        ):
            if include_unmatched:
                output[name] = base_tensor
                diagnostics.preserved_keys += 1
            continue

        if not _is_mergeable_tensor(
            base_tensor
        ):
            if include_unmatched:
                output[name] = (
                    base_tensor.detach().clone()
                )

            diagnostics.nonfloating_keys += 1
            diagnostics.preserved_keys += 1
            continue

        deltas: List[torch.Tensor] = []
        contributors = 0

        for model_index, model_state in enumerate(
            models
        ):
            tensor = model_state.get(
                name
            )

            if tensor is None:
                continue

            if not _is_mergeable_tensor(
                tensor
            ):
                diagnostics.nonfloating_keys += 1
                continue

            if not _same_shape(
                tensor,
                base_tensor,
            ):
                diagnostics.shape_mismatch_keys += 1

                if strict_shapes:
                    raise ValueError(
                        f"TIES merge: shape mismatch for {name!r}: "
                        f"base={tuple(base_tensor.shape)}, "
                        f"model[{model_index}]="
                        f"{tuple(tensor.shape)}."
                    )

                continue

            delta = (
                tensor.detach().float()
                -
                base_tensor.detach().float()
            )

            delta = torch.nan_to_num(
                delta,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            deltas.append(
                delta
            )
            contributors += 1

        if not deltas:
            if include_unmatched:
                output[name] = (
                    base_tensor.detach().clone()
                )
                diagnostics.preserved_keys += 1
            else:
                diagnostics.skipped_keys += 1
            continue

        final_delta = _ties_merge_deltas(
            deltas,
            density=density,
            scaling=scaling,
        )

        if max_delta_norm_ratio is not None:
            final_delta = _clip_delta_norm(
                final_delta,
                base_tensor.float(),
                max_delta_norm_ratio,
            )

        candidate = (
            base_tensor.detach().float()
            +
            final_delta
        )

        candidate = _repair_output(
            candidate
        )

        output_tensor = candidate.to(
            device=base_tensor.device,
            dtype=base_tensor.dtype,
        )

        output[name] = output_tensor
        diagnostics.merged_keys += 1

        base_norm = _safe_norm(
            base_tensor
        )

        output_norm = _safe_norm(
            output_tensor
        )

        ratio = (
            output_norm
            /
            max(
                base_norm,
                _EPS,
            )
        )

        if math.isfinite(ratio):
            diagnostics.max_output_ratio = max(
                diagnostics.max_output_ratio,
                ratio,
            )
            diagnostics.min_output_ratio = min(
                diagnostics.min_output_ratio,
                ratio,
            )

        diagnostics.per_key[name] = {
            "contributors": contributors,
            "output_ratio": ratio,
            "density": density,
            "scaling": float(scaling),
        }

    ratios = [
        value["output_ratio"]
        for value in diagnostics.per_key.values()
        if math.isfinite(
            value["output_ratio"]
        )
    ]

    if ratios:
        diagnostics.mean_output_ratio = (
            sum(ratios)
            /
            len(ratios)
        )

    return output


# =============================================================================
# Generic multi-model state-dict merge
# =============================================================================

def merge_state_dicts(
    models: Sequence[Mapping[str, torch.Tensor]],
    *,
    weights: Optional[Sequence[float]] = None,
    base: Optional[Mapping[str, torch.Tensor]] = None,
    method: str = "weighted",
    density: float = 0.2,
    scaling: float = 1.0,
    drop_rate: float = 0.9,
    seed: Optional[int] = None,
    strict_shapes: bool = True,
    preserve_unmatched: bool = True,
    max_delta_norm_ratio: Optional[float] = 2.5,
    fisher: Optional[
        Sequence[
            Mapping[str, torch.Tensor]
        ]
    ] = None,
) -> Dict[str, torch.Tensor]:
    """
    Unified merge entry point for FTRAIN.

    Supported methods
    -----------------
    weighted:
        Weighted parameter average.

    ties:
        TIES deltas relative to `base`.

    dare:
        Sequential DARE application relative to the first model.

    task_arithmetic:
        Sum task deltas relative to `base`.

    fisher:
        Fisher-weighted parameter merge.

    This function intentionally stays architecture-agnostic.
    """
    if not models:
        raise ValueError(
            "models cannot be empty."
        )

    method = str(
        method
    ).strip().lower()

    if weights is None:
        weights = [
            1.0
            for _ in models
        ]

    if len(weights) != len(models):
        raise ValueError(
            "weights must have the same length as models."
        )

    normalized_weights = _normalize_weights(
        weights
    )

    if method == "ties":
        if base is None:
            raise ValueError(
                "TIES requires base."
            )

        diagnostics = MergeDiagnostics(
            operation="ties"
        )

        return ties_merge_state_dict(
            list(models),
            base,
            density=density,
            scaling=scaling,
            strict_shapes=strict_shapes,
            include_unmatched=preserve_unmatched,
            max_delta_norm_ratio=max_delta_norm_ratio,
            diagnostics=diagnostics,
        )

    if method == "task_arithmetic":
        if base is None:
            raise ValueError(
                "task_arithmetic requires base."
            )

        # Generalize task arithmetic to N models by summing weighted deltas.
        result: Dict[str, torch.Tensor] = {}

        for name, base_tensor in base.items():
            if not isinstance(
                base_tensor,
                torch.Tensor,
            ):
                continue

            if not _is_mergeable_tensor(
                base_tensor
            ):
                result[name] = (
                    base_tensor.detach().clone()
                )
                continue

            accumulator = torch.zeros_like(
                base_tensor,
                dtype=torch.float32,
            )

            found = False

            for model_state, weight in zip(
                models,
                normalized_weights,
            ):
                tensor = model_state.get(
                    name
                )

                if tensor is None:
                    continue

                if not _is_mergeable_tensor(
                    tensor
                ):
                    continue

                if not _same_shape(
                    tensor,
                    base_tensor,
                ):
                    if strict_shapes:
                        raise ValueError(
                            f"task_arithmetic merge: shape mismatch "
                            f"for {name!r}."
                        )
                    continue

                delta = (
                    tensor.detach().float()
                    -
                    base_tensor.detach().float()
                )

                accumulator.add_(
                    delta,
                    alpha=float(weight),
                )

                found = True

            if not found:
                if preserve_unmatched:
                    result[name] = (
                        base_tensor.detach().clone()
                    )
                continue

            if max_delta_norm_ratio is not None:
                accumulator = _clip_delta_norm(
                    accumulator,
                    base_tensor.float(),
                    max_delta_norm_ratio,
                )

            candidate = (
                base_tensor.float()
                +
                float(scaling)
                *
                accumulator
            )

            result[name] = (
                _repair_output(
                    candidate
                )
                .to(
                    device=base_tensor.device,
                    dtype=base_tensor.dtype,
                )
            )

        return result

    if method == "dare":
        result = (
            models[0]
        )

        for index, model_state in enumerate(
            models[1:]
        ):
            pair_seed = (
                None
                if seed is None
                else int(seed) + index
            )

            result = dare_merge_state_dict(
                result,
                model_state,
                drop_rate=drop_rate,
                rescale=True,
                seed=pair_seed,
                strict_shapes=strict_shapes,
                preserve_unmatched=preserve_unmatched,
            )

        return result

    if method == "fisher":
        if fisher is None:
            raise ValueError(
                "Fisher merge requires fisher data."
            )

        if len(fisher) != len(models):
            raise ValueError(
                "fisher must have one state dictionary per model."
            )

        result: Dict[str, torch.Tensor] = {}

        if base is None:
            base = models[0]

        for name, fallback_tensor in base.items():
            tensors_here = []
            fisher_here = []

            for model_state, fisher_state in zip(
                models,
                fisher,
            ):
                tensor = model_state.get(
                    name
                )

                fisher_tensor = fisher_state.get(
                    name
                )

                if (
                    tensor is None
                    or
                    fisher_tensor is None
                ):
                    continue

                if not _is_mergeable_tensor(
                    tensor
                ):
                    continue

                if not _same_shape(
                    tensor,
                    fallback_tensor,
                ):
                    if strict_shapes:
                        raise ValueError(
                            f"fisher merge: shape mismatch for {name!r}."
                        )
                    continue

                if not _same_shape(
                    tensor,
                    fisher_tensor,
                ):
                    raise ValueError(
                        f"fisher merge: Fisher shape mismatch for {name!r}."
                    )

                tensors_here.append(
                    tensor
                )

                fisher_here.append(
                    fisher_tensor
                )

            if not tensors_here:
                if preserve_unmatched:
                    result[name] = (
                        fallback_tensor.detach().clone()
                    )
                continue

            result[name] = fisher_merge(
                tensors_here,
                fisher_here,
                reference=fallback_tensor,
                max_output_ratio=(
                    max_delta_norm_ratio
                    if max_delta_norm_ratio is not None
                    else None
                ),
            )

        return result

    if method == "weighted":
        result: Dict[str, torch.Tensor] = {}

        if base is None:
            base = models[0]

        for name, fallback_tensor in base.items():
            if not isinstance(
                fallback_tensor,
                torch.Tensor,
            ):
                continue

            compatible = []

            compatible_weights = []

            for model_state, weight in zip(
                models,
                normalized_weights,
            ):
                tensor = model_state.get(
                    name
                )

                if tensor is None:
                    continue

                if not _is_mergeable_tensor(
                    tensor
                ):
                    continue

                if not _same_shape(
                    tensor,
                    fallback_tensor,
                ):
                    if strict_shapes:
                        raise ValueError(
                            f"weighted merge: shape mismatch for {name!r}."
                        )
                    continue

                compatible.append(
                    tensor
                )

                compatible_weights.append(
                    weight
                )

            if not compatible:
                if preserve_unmatched:
                    result[name] = (
                        fallback_tensor.detach().clone()
                    )
                continue

            result[name] = weighted_merge(
                compatible,
                compatible_weights,
                reference=fallback_tensor,
                max_output_ratio=(
                    max_delta_norm_ratio
                    if max_delta_norm_ratio is not None
                    else None
                ),
            )

        return result

    raise ValueError(
        f"Unknown merge method {method!r}. "
        "Expected weighted, ties, dare, task_arithmetic, or fisher."
    )


# =============================================================================
# Backward-compatible aliases / final exports
# =============================================================================

__all__ = [
    "MergeConfig",
    "MergeDiagnostics",
    "compute_fisher",
    "dare_merge",
    "task_arithmetic",
    "weighted_merge",
    "fisher_merge",
    "ties_merge_state_dict",
    "dare_merge_state_dict",
    "merge_state_dicts",
    "calculate_delta",
    "detect_tensor_health",
]
