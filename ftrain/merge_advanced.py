"""
FTRAIN Merge Algorithms
=======================

High-performance and defensive merging utilities used by FTRAIN.

Supported functionality
-----------------------
• Diagonal Fisher information estimation
• DARE merging
• Task Arithmetic
• TIES merging
• Multi-model delta handling
• Numerical safety / NaN protection
• Device and dtype management
• Shape validation
• Deterministic seeded operations
• CPU-offloaded Fisher storage
• Architecture-aware compatibility hooks
• Memory-conscious TIES implementation
• Merge diagnostics

Important
---------
These functions intentionally do NOT blindly merge tensors with incompatible
shapes.

Different model architectures can only be merged when a higher-level
architecture mapper has established that two tensors represent compatible
parameters. This module provides the mathematical operations; the architecture
mapping layer should decide which tensors correspond to each other.

Example
-------
    fisher = compute_fisher(
        model,
        loader,
        device="cuda",
        num_samples=100,
    )

    result = dare_merge(
        model_a_weight,
        model_b_weight,
        drop_rate=0.9,
    )

    result = task_arithmetic(
        model_a,
        model_b,
        base,
    )

    result = ties_merge_state_dict(
        [model_a, model_b],
        base,
        density=0.2,
    )
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterator, Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "compute_fisher",
    "dare_merge",
    "task_arithmetic",
    "ties_merge_state_dict",
]


TensorDict = Dict[str, torch.Tensor]


# =============================================================================
# Constants
# =============================================================================

_EPS = 1e-8
_DEFAULT_NUM_SAMPLES = 50


# =============================================================================
# General helpers
# =============================================================================


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _is_mergeable_tensor(tensor: torch.Tensor) -> bool:
    """
    Return whether a tensor is mathematically suitable for floating-point
    merge arithmetic.

    Integer/bool tensors such as token IDs or certain bookkeeping tensors
    should never be merged with arithmetic operations.
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

    if value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero, got {value}."
        )

    return value


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
            f"{name} must be a floating-point or complex tensor; "
            f"got dtype={tensor.dtype}, shape={tuple(tensor.shape)}."
        )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _safe_loss(
    loss: torch.Tensor,
) -> bool:
    if not isinstance(loss, torch.Tensor):
        return False

    if loss.numel() != 1:
        return False

    return bool(torch.isfinite(loss.detach()).item())


def _resolve_device(
    device: Union[str, torch.device],
) -> torch.device:
    """
    Resolve and validate a requested device.

    Unlike the old implementation, this doesn't assume CUDA exists merely
    because the caller supplied a CUDA-looking string.
    """
    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for Fisher computation, but CUDA is not "
            "available."
        )

    if resolved.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested for Fisher computation, but MPS is not "
                "available."
            )

    return resolved


def _supports_amp(device: torch.device) -> bool:
    """
    Determine whether autocast is useful/supported on the selected device.
    """
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
        # BF16 is generally the safest CPU autocast choice when available.
        return torch.bfloat16

    if device.type == "mps":
        return torch.float16

    return torch.float32


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    """
    Return a device-correct autocast context.

    ``torch.amp.autocast`` has different practical support characteristics
    across PyTorch versions/devices, so failures should gracefully fall back
    to a disabled context.
    """
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
    """
    Recursively move tensor-containing batches to a device.

    Supports the common Hugging Face dictionary format as well as nested
    dictionaries/lists/tuples.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(
            device,
            non_blocking=device.type == "cuda",
        )

    if isinstance(batch, Mapping):
        return {
            key: _move_batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(
            _move_batch_to_device(value, device)
            for value in batch
        )

    if isinstance(batch, list):
        return [
            _move_batch_to_device(value, device)
            for value in batch
        ]

    return batch


def _extract_loss(
    outputs: Any,
) -> Optional[torch.Tensor]:
    """
    Extract a scalar loss from Hugging Face-style model outputs.
    """
    if hasattr(outputs, "loss"):
        loss = outputs.loss
    elif isinstance(outputs, Mapping):
        loss = outputs.get("loss")
    elif isinstance(outputs, (tuple, list)) and outputs:
        loss = outputs[0]
    else:
        loss = None

    if loss is None:
        return None

    if not isinstance(loss, torch.Tensor):
        return None

    if loss.numel() != 1:
        return None

    return loss


def _batch_size_from_batch(
    batch: Any,
) -> int:
    """
    Infer batch size from common HF batches.
    """
    if isinstance(batch, Mapping):
        for key in (
            "input_ids",
            "labels",
            "attention_mask",
        ):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])

        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])

    if isinstance(batch, torch.Tensor) and batch.ndim >= 1:
        return int(batch.shape[0])

    return 1


def _safe_grad_square(
    grad: torch.Tensor,
) -> torch.Tensor:
    """
    Compute squared gradients in FP32 with finite-value sanitization.
    """
    result = grad.detach().float()

    # Avoid NaN/Inf contamination of the Fisher matrix.
    result = torch.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return result.square()


# =============================================================================
# Fisher Information
# =============================================================================


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
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Compute a diagonal Fisher information approximation.

    Parameters
    ----------
    model:
        Model used for calibration.

    loader:
        Iterable yielding batches compatible with ``model(**batch)``.

    device:
        Device used for the forward/backward computation.

    num_samples:
        Maximum number of examples to process.

    use_amp:
        Use automatic mixed precision when supported.

    trainable_only:
        If True, only parameters with ``requires_grad=True`` are included.

        If False, all floating-point parameters are considered. This is
        usually more appropriate for model-merging Fisher estimation.

    offload_to_cpu:
        Store Fisher diagonals on CPU to avoid consuming GPU memory.

    max_grad_norm:
        Optional gradient clipping before squaring gradients.

    clear_cache_every:
        If > 0, optionally call ``torch.cuda.empty_cache()`` after this many
        processed batches. Usually unnecessary; useful for very large models.

    Returns
    -------
    Optional[Dict[str, torch.Tensor]]
        Per-parameter diagonal Fisher estimates, or None if computation fails.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be torch.nn.Module, got {type(model).__name__}."
        )

    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be > 0, got {num_samples}."
        )

    if max_grad_norm is not None:
        max_grad_norm = _validate_positive(
            max_grad_norm,
            "max_grad_norm",
        )

    resolved_device = _resolve_device(device)

    was_training = model.training

    # -------------------------------------------------------------------------
    # Build Fisher storage.
    # -------------------------------------------------------------------------

    fisher: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue

        if trainable_only and not param.requires_grad:
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
            "Fisher computation skipped: no floating-point parameters "
            "were selected."
        )
        return None

    # -------------------------------------------------------------------------
    # Save model state.
    # -------------------------------------------------------------------------

    try:
        model.eval()

        samples_processed = 0
        batches_processed = 0
        invalid_batches = 0

        amp_enabled = bool(
            use_amp
            and _supports_amp(resolved_device)
        )

        for batch in loader:
            if samples_processed >= num_samples:
                break

            batch_size = max(
                1,
                _batch_size_from_batch(batch),
            )

            # Don't process more samples than requested.
            remaining = num_samples - samples_processed

            batch_device = _move_batch_to_device(
                batch,
                resolved_device,
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
                    logger.debug(
                        "Fisher skipped batch: model did not return a scalar "
                        "loss."
                    )
                    continue

                if not _safe_loss(loss):
                    invalid_batches += 1
                    logger.debug(
                        "Fisher skipped non-finite loss."
                    )
                    continue

                # If a batch contains more examples than requested, scale the
                # contribution so the final average still represents the
                # requested number of examples reasonably.
                effective_batch_size = min(
                    batch_size,
                    remaining,
                )

                # Backpropagate the normal batch loss.
                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm,
                    )

                # -------------------------------------------------------------
                # Accumulate diagonal Fisher.
                # -------------------------------------------------------------

                with torch.no_grad():
                    for name, param in model.named_parameters():
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
                            alpha=float(effective_batch_size),
                        )

                samples_processed += effective_batch_size
                batches_processed += 1

            except RuntimeError as exc:
                # A single problematic calibration batch should not necessarily
                # destroy an entire merge operation.
                invalid_batches += 1

                logger.warning(
                    "Fisher skipped a batch because of runtime error: %s",
                    exc,
                )

                model.zero_grad(
                    set_to_none=True
                )

            finally:
                model.zero_grad(
                    set_to_none=True
                )

            if (
                clear_cache_every > 0
                and batches_processed > 0
                and batches_processed % clear_cache_every == 0
                and resolved_device.type == "cuda"
            ):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        if samples_processed <= 0:
            logger.error(
                "Fisher computation processed zero valid samples."
            )
            return None

        # ---------------------------------------------------------------------
        # Normalize Fisher estimates.
        # ---------------------------------------------------------------------

        denominator = float(
            max(1, samples_processed)
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
            "Fisher computation completed: "
            "%d samples, %d batches, %d invalid batches, "
            "%d parameter tensors.",
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
# DARE
# =============================================================================


def dare_merge(
    da: torch.Tensor,
    db: torch.Tensor,
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    DARE-style delta merge.

    ``da`` is treated as the base tensor and ``db`` as the target tensor.

    The operation is:

        delta = db - da

        mask ~ Bernoulli(1 - drop_rate)

        result = da + mask * delta

    When ``rescale=True`` retained deltas are divided by the keep probability,
    preserving their expected magnitude.

    Edge cases
    ----------
    drop_rate == 0:
        Returns ``db`` exactly.

    drop_rate == 1:
        Returns ``da`` exactly.

    Notes
    -----
    DARE requires equal-shaped tensors. Cross-architecture parameter mapping
    must happen before this function.
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
        return db.clone()

    if drop_rate == 1.0:
        return da.clone()

    keep_probability = 1.0 - drop_rate

    if seed is not None:
        generator = torch.Generator(
            device=da.device
        )
        generator.manual_seed(
            int(seed)
        )
    else:
        generator = None

    with torch.no_grad():
        base = da.float()
        target = db.float()

        delta = target - base

        # Generate a compact byte mask rather than a float tensor. This can
        # substantially reduce temporary memory for large model weights.
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

        result = base + delta * mask

        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return result.to(
            dtype=da.dtype
        )


# =============================================================================
# Task Arithmetic
# =============================================================================


def _clip_delta_norm(
    delta: torch.Tensor,
    base: torch.Tensor,
    max_norm_ratio: float,
) -> torch.Tensor:
    """
    Limit delta magnitude relative to the base tensor.

    This is more stable than clipping the final candidate norm because it
    directly controls how much the merge can move away from the base.
    """
    if max_norm_ratio <= 0.0:
        return torch.zeros_like(delta)

    base_norm = torch.linalg.vector_norm(
        base.float()
    )

    delta_norm = torch.linalg.vector_norm(
        delta.float()
    )

    if not torch.isfinite(base_norm):
        return torch.zeros_like(delta)

    if not torch.isfinite(delta_norm):
        return torch.zeros_like(delta)

    if delta_norm <= _EPS:
        return delta

    allowed = base_norm * max_norm_ratio

    if allowed <= _EPS:
        return torch.zeros_like(delta)

    if delta_norm > allowed:
        delta = delta * (
            allowed / delta_norm
        )

    return delta


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
    Merge model deltas using Task Arithmetic.

    For compatible parameters:

        delta_a = A - Base
        delta_b = B - Base

        merged = Base + scaling * (delta_a + delta_b)

    Parameters absent from both models are preserved from ``base`` when
    ``preserve_unmatched_base=True``.

    ``strict_shapes=True`` protects against accidentally combining unrelated
    tensors with the same parameter name but incompatible dimensions.
    """
    try:
        scaling = float(scaling)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"scaling must be numeric, got {scaling!r}."
        ) from exc

    if not math.isfinite(scaling):
        raise ValueError(
            "scaling must be finite."
        )

    max_norm_ratio = _validate_positive(
        max_norm_ratio,
        "max_norm_ratio",
    )

    merged: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name, base_tensor in base.items():
            if not isinstance(base_tensor, torch.Tensor):
                continue

            # Non-floating tensors should generally be copied from the base
            # rather than mathematically merged.
            if not _is_mergeable_tensor(base_tensor):
                merged[name] = base_tensor.clone()
                continue

            tensor_a = ma.get(name)
            tensor_b = mb.get(name)

            if tensor_a is None and tensor_b is None:
                if preserve_unmatched_base:
                    merged[name] = base_tensor.clone()

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

            base_fp32 = base_tensor.float()

            delta_a = (
                tensor_a.float() - base_fp32
                if tensor_a is not None
                else torch.zeros_like(base_fp32)
            )

            delta_b = (
                tensor_b.float() - base_fp32
                if tensor_b is not None
                else torch.zeros_like(base_fp32)
            )

            combined_delta = (
                scaling * (
                    delta_a + delta_b
                )
            )

            combined_delta = torch.nan_to_num(
                combined_delta,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            combined_delta = _clip_delta_norm(
                combined_delta,
                base_fp32,
                max_norm_ratio,
            )

            candidate = (
                base_fp32 + combined_delta
            )

            candidate = torch.nan_to_num(
                candidate,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            merged[name] = candidate.to(
                dtype=base_tensor.dtype
            )

    return merged


# =============================================================================
# TIES helpers
# =============================================================================


def _topk_mask(
    tensor: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """
    Return a boolean mask retaining approximately ``density`` of values.

    Works safely for scalar and tiny tensors.
    """
    density = _validate_probability(
        density,
        "density",
    )

    if tensor.numel() == 0:
        return torch.zeros_like(
            tensor,
            dtype=torch.bool,
        )

    if density >= 1.0:
        return torch.ones_like(
            tensor,
            dtype=torch.bool,
        )

    if density <= 0.0:
        return torch.zeros_like(
            tensor,
            dtype=torch.bool,
        )

    flat = tensor.abs().reshape(-1)

    k = max(
        1,
        min(
            flat.numel(),
            int(math.ceil(
                flat.numel() * density
            )),
        ),
    )

    if k >= flat.numel():
        return torch.ones_like(
            tensor,
            dtype=torch.bool,
        )

    threshold = torch.topk(
        flat,
        k=k,
        largest=True,
        sorted=False,
    ).values.min()

    return (
        tensor.abs() >= threshold
    )


def _sign_consensus(
    deltas: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Determine the elected TIES sign for each parameter position.

    The sign with the strongest aggregate magnitude wins.

    This is more robust than simply counting +1/-1 because a tiny positive
    delta should not necessarily defeat a huge negative delta.
    """
    if not deltas:
        raise ValueError(
            "Cannot compute sign consensus from zero deltas."
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
        d = delta.float()

        positive_score.add_(
            torch.where(
                d > 0,
                d.abs(),
                torch.zeros_like(d),
            )
        )

        negative_score.add_(
            torch.where(
                d < 0,
                d.abs(),
                torch.zeros_like(d),
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
    """
    Core TIES operation on already computed deltas.
    """
    if not deltas:
        raise ValueError(
            "TIES requires at least one delta."
        )

    density = _validate_probability(
        density,
        "density",
    )

    if not math.isfinite(float(scaling)):
        raise ValueError(
            "scaling must be finite."
        )

    # -------------------------------------------------------------------------
    # Trim
    # -------------------------------------------------------------------------

    trimmed: List[torch.Tensor] = []

    for delta in deltas:
        mask = _topk_mask(
            delta,
            density,
        )

        trimmed.append(
            torch.where(
                mask,
                delta,
                torch.zeros_like(delta),
            )
        )

    # -------------------------------------------------------------------------
    # Elect sign.
    # -------------------------------------------------------------------------

    elected_sign = _sign_consensus(
        trimmed
    )

    # -------------------------------------------------------------------------
    # Disjoint merge.
    #
    # Instead of constructing a huge [num_models, ...] stack, accumulate
    # matching deltas incrementally. This significantly reduces peak memory
    # for large models.
    # -------------------------------------------------------------------------

    total = torch.zeros_like(
        trimmed[0],
        dtype=torch.float32,
    )

    count = torch.zeros_like(
        trimmed[0],
        dtype=torch.float32,
    )

    for delta in trimmed:
        d = delta.float()

        matching = (
            torch.sign(d) == elected_sign
        ) & (
            elected_sign != 0
        )

        total.add_(
            torch.where(
                matching,
                d,
                torch.zeros_like(d),
            )
        )

        count.add_(
            matching.to(torch.float32)
        )

    count.clamp_min_(
        1.0
    )

    merged_delta = (
        total / count
    ) * float(scaling)

    return torch.nan_to_num(
        merged_delta,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# =============================================================================
# TIES State Dict Merge
# =============================================================================


def ties_merge_state_dict(
    models: List[Mapping[str, torch.Tensor]],
    base: Mapping[str, torch.Tensor],
    density: float = 0.2,
    scaling: float = 1.0,
    *,
    strict_shapes: bool = True,
    include_unmatched: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    TIES merge one or more models against a base model.

    Parameters
    ----------
    models:
        Model state dictionaries.

    base:
        Base state dictionary.

    density:
        Fraction of each model's delta retained during the trimming stage.

    scaling:
        Final delta multiplier.

    strict_shapes:
        Raise when a matching parameter name has an incompatible shape.

    include_unmatched:
        Preserve base parameters when no compatible model delta exists.

    Notes
    -----
    This implementation intentionally avoids stacking all model deltas.

    For a model with billions of parameters, constructing:

        [num_models, *parameter_shape]

    can create a very large temporary allocation. Instead, this version
    processes the deltas incrementally.
    """
    if not models:
        raise ValueError(
            "TIES requires at least one model."
        )

    density = _validate_probability(
        density,
        "density",
    )

    if not math.isfinite(float(scaling)):
        raise ValueError(
            "scaling must be finite."
        )

    merged: Dict[str, torch.Tensor] = {}

    statistics = {
        "parameters": 0,
        "merged": 0,
        "unmatched": 0,
        "shape_mismatch": 0,
        "non_floating": 0,
    }

    with torch.no_grad():
        for name, base_tensor in base.items():
            statistics["parameters"] += 1

            if not isinstance(base_tensor, torch.Tensor):
                if include_unmatched:
                    merged[name] = base_tensor

                statistics["unmatched"] += 1
                continue

            if not _is_mergeable_tensor(base_tensor):
                if include_unmatched:
                    merged[name] = base_tensor.clone()

                statistics["non_floating"] += 1
                continue

            deltas: List[torch.Tensor] = []

            for model_index, model_state in enumerate(models):
                tensor = model_state.get(name)

                if tensor is None:
                    continue

                if not _is_mergeable_tensor(tensor):
                    logger.debug(
                        "TIES ignored non-floating tensor %s from model %d.",
                        name,
                        model_index,
                    )
                    continue

                if not _same_shape(
                    tensor,
                    base_tensor,
                ):
                    statistics["shape_mismatch"] += 1

                    if strict_shapes:
                        raise ValueError(
                            f"TIES merge: shape mismatch for parameter "
                            f"{name!r}: base={tuple(base_tensor.shape)}, "
                            f"model[{model_index}]="
                            f"{tuple(tensor.shape)}."
                        )

                    continue

                delta = (
                    tensor.float()
                    - base_tensor.float()
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

            if not deltas:
                if include_unmatched:
                    merged[name] = base_tensor.clone()

                statistics["unmatched"] += 1
                continue

            final_delta = _ties_merge_deltas(
                deltas,
                density=density,
                scaling=scaling,
            )

            candidate = (
                base_tensor.float()
                + final_delta
            )

            candidate = torch.nan_to_num(
                candidate,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            merged[name] = candidate.to(
                dtype=base_tensor.dtype
            )

            statistics["merged"] += 1

    logger.info(
        "TIES merge complete: "
        "%d/%d parameters merged, "
        "%d unmatched, %d shape mismatches, "
        "%d non-floating.",
        statistics["merged"],
        statistics["parameters"],
        statistics["unmatched"],
        statistics["shape_mismatch"],
        statistics["non_floating"],
    )

    return merged
