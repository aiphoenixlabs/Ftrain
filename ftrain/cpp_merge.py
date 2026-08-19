"""
FTRAIN Fast Merge Kernels
=========================

High-performance tensor merge primitives with:

    • CUDA extension acceleration when available.
    • Safe CPU fallback.
    • TorchScript-free fallback paths.
    • Shape/device validation.
    • Thread-safe lazy extension loading.
    • Configurable kernel compilation.
    • Numerically safer SLERP/Fisher operations.
    • Memory-conscious TIES masking.
    • Clear diagnostics when the CUDA extension cannot be built.

Supported operations
--------------------
    fast_weighted_avg()
    fast_fisher_merge()
    fast_slerp()
    fast_ties()

The CUDA extension is optional. FTRAIN must remain functional when:

    • CUDA is unavailable.
    • NVCC is unavailable.
    • The extension fails to compile.
    • The machine is CPU-only.
    • A supported tensor operation is not implemented by the extension.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

__all__ = [
    "fast_weighted_avg",
    "fast_fisher_merge",
    "fast_slerp",
    "fast_ties",
    "cuda_extension_available",
    "clear_extension_cache",
]


# =============================================================================
# Paths / extension state
# =============================================================================

_HERE = Path(__file__).resolve().parent
_KERNEL = _HERE / "merge_kernel.cu"

_EXTENSION_NAME = "ftrain_merge"

_EXT: Any = None
_EXTENSION_FAILED = False
_EXTENSION_ERROR: Optional[BaseException] = None

_EXTENSION_LOCK = threading.Lock()

# Avoid rebuilding the extension over and over if compilation repeatedly fails.
_EXTENSION_BUILD_ATTEMPTED = False


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_pair(
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    """Validate two tensors used by a binary merge."""
    if not isinstance(a, torch.Tensor):
        raise TypeError(
            f"'a' must be a torch.Tensor, got {type(a).__name__}."
        )

    if not isinstance(b, torch.Tensor):
        raise TypeError(
            f"'b' must be a torch.Tensor, got {type(b).__name__}."
        )

    if a.shape != b.shape:
        raise ValueError(
            "Merge tensors must have identical shapes: "
            f"a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}."
        )

    if a.device != b.device:
        raise ValueError(
            "Merge tensors must be on the same device: "
            f"a.device={a.device}, "
            f"b.device={b.device}."
        )


def _validate_fisher(
    a: torch.Tensor,
    b: torch.Tensor,
    fa: torch.Tensor,
    fb: torch.Tensor,
) -> None:
    """Validate model tensors and their Fisher importance tensors."""
    _validate_pair(a, b)

    if not isinstance(fa, torch.Tensor):
        raise TypeError(
            f"'fa' must be a torch.Tensor, got {type(fa).__name__}."
        )

    if not isinstance(fb, torch.Tensor):
        raise TypeError(
            f"'fb' must be a torch.Tensor, got {type(fb).__name__}."
        )

    if fa.shape != a.shape or fb.shape != b.shape:
        raise ValueError(
            "Fisher tensors must have the same shape as their corresponding "
            "model tensors: "
            f"a={tuple(a.shape)}, fa={tuple(fa.shape)}, "
            f"b={tuple(b.shape)}, fb={tuple(fb.shape)}."
        )

    if fa.device != a.device or fb.device != b.device:
        raise ValueError(
            "Fisher tensors must be on the same device as their model tensors."
        )


def _validate_alpha(alpha: float) -> float:
    """Validate a [0, 1] interpolation parameter."""
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"alpha must be numeric, got {alpha!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"alpha must be finite, got {value!r}."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"alpha must be between 0 and 1, got {value}."
        )

    return value


def _validate_density(density: float) -> float:
    """Validate TIES density."""
    try:
        value = float(density)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"density must be numeric, got {density!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"density must be finite, got {value!r}."
        )

    if not 0.0 < value <= 1.0:
        raise ValueError(
            f"density must be in (0, 1], got {value}."
        )

    return value


# =============================================================================
# Device / extension helpers
# =============================================================================


def _cuda_pair(
    a: torch.Tensor,
    b: torch.Tensor,
) -> bool:
    return (
        a.is_cuda
        and b.is_cuda
        and a.device == b.device
    )


def _cuda_available() -> bool:
    """Return whether CUDA appears usable by PyTorch."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _kernel_exists() -> bool:
    if not _KERNEL.is_file():
        logger.debug(
            "FTRAIN merge CUDA kernel does not exist: %s",
            _KERNEL,
        )
        return False

    return True


# =============================================================================
# CUDA extension loader
# =============================================================================


def _build_cuda_extension() -> Any:
    """
    Compile and load the optional CUDA merge extension.

    Compilation is intentionally performed only when it is actually useful.
    """
    global _EXT
    global _EXTENSION_FAILED
    global _EXTENSION_ERROR
    global _EXTENSION_BUILD_ATTEMPTED

    if _EXT is not None:
        return _EXT

    if _EXTENSION_FAILED:
        return None

    if _EXTENSION_BUILD_ATTEMPTED:
        return None

    with _EXTENSION_LOCK:
        if _EXT is not None:
            return _EXT

        if _EXTENSION_FAILED:
            return None

        if _EXTENSION_BUILD_ATTEMPTED:
            return None

        _EXTENSION_BUILD_ATTEMPTED = True

        if not _cuda_available():
            logger.debug(
                "FTRAIN merge CUDA extension skipped because CUDA is unavailable."
            )
            return None

        if not _kernel_exists():
            _EXTENSION_FAILED = True
            _EXTENSION_ERROR = FileNotFoundError(
                f"Merge kernel not found: {_KERNEL}"
            )
            return None

        # OpenMP flags differ considerably across Windows/Linux/macOS.
        # Keep the compilation flags conservative here and let the CUDA
        # source itself determine most optimization behavior.
        extra_cuda_cflags = [
            "-O3",
        ]

        extra_cflags = [
            "-O3",
        ]

        extra_ldflags = []

        if platform.system() == "Linux":
            # GCC/Clang generally support OpenMP.
            extra_cflags.append("-fopenmp")
            extra_ldflags.append("-fopenmp")

        try:
            logger.info(
                "⚙️ Compiling FTRAIN CUDA merge extension from %s",
                _KERNEL,
            )

            _EXT = load(
                name=_EXTENSION_NAME,
                sources=[str(_KERNEL)],
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                verbose=False,
            )

            logger.info(
                "✅ FTRAIN CUDA merge extension compiled successfully."
            )

            return _EXT

        except Exception as exc:
            _EXTENSION_FAILED = True
            _EXTENSION_ERROR = exc
            _EXT = None

            logger.warning(
                "⚠️ FTRAIN CUDA merge extension unavailable. "
                "Falling back to PyTorch implementations: %s",
                exc,
            )

            logger.debug(
                "Detailed merge extension compilation failure.",
                exc_info=True,
            )

            return None


def _mod() -> Any:
    """
    Backward-compatible internal alias for lazy extension loading.
    """
    return _build_cuda_extension()


def cuda_extension_available() -> bool:
    """Return whether the custom CUDA extension can currently be loaded."""
    return _build_cuda_extension() is not None


def clear_extension_cache() -> None:
    """
    Clear cached extension state.

    Useful for testing or when the runtime environment changes, although a
    process restart is generally preferable after failed native compilation.
    """
    global _EXT
    global _EXTENSION_FAILED
    global _EXTENSION_ERROR
    global _EXTENSION_BUILD_ATTEMPTED

    with _EXTENSION_LOCK:
        _EXT = None
        _EXTENSION_FAILED = False
        _EXTENSION_ERROR = None
        _EXTENSION_BUILD_ATTEMPTED = False


# =============================================================================
# Fallback implementations
# =============================================================================


def _weighted_average_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Numerically stable weighted average.

        alpha * a + (1-alpha) * b
    """
    # Perform arithmetic in FP32 to reduce overflow/underflow risk for
    # fp16/bf16 input while returning the original dtype.
    result = (
        alpha * a.float()
        + (1.0 - alpha) * b.float()
    )

    return result.to(
        dtype=a.dtype
    )


def _fisher_merge_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    fa: torch.Tensor,
    fb: torch.Tensor,
) -> torch.Tensor:
    """
    Fisher-weighted merge.

        (fa*a + fb*b) / (fa+fb)

    Negative or non-finite Fisher values are clipped to zero because they do
    not represent meaningful positive importance weights.
    """
    a32 = a.float()
    b32 = b.float()

    fa32 = torch.nan_to_num(
        fa.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)

    fb32 = torch.nan_to_num(
        fb.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)

    denominator = fa32 + fb32

    numerator = (
        fa32 * a32
        + fb32 * b32
    )

    # Where both Fisher values are effectively zero, there is no evidence
    # favoring either model. Falling back to a simple average is preferable
    # to creating arbitrary huge values through division by epsilon.
    zero_weight = denominator <= 1e-12

    result = torch.where(
        zero_weight,
        0.5 * (a32 + b32),
        numerator / denominator.clamp_min(1e-12),
    )

    return result.to(
        dtype=a.dtype
    )


def _slerp_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Spherical interpolation between two tensors.

    Computation is performed in FP32.
    """
    a32 = a.float()
    b32 = b.float()

    flat_a = a32.reshape(-1)
    flat_b = b32.reshape(-1)

    norm_a = torch.linalg.vector_norm(
        flat_a
    )

    norm_b = torch.linalg.vector_norm(
        flat_b
    )

    # Zero vectors have no defined angular direction, so use linear
    # interpolation.
    if (
        norm_a.item() <= 1e-8
        or norm_b.item() <= 1e-8
    ):
        return (
            alpha * a32
            + (1.0 - alpha) * b32
        ).to(a.dtype)

    cosine = (
        torch.sum(
            flat_a * flat_b
        )
        / (
            norm_a
            * norm_b
        )
    )

    cosine = cosine.clamp(
        -1.0,
        1.0,
    )

    # ``acos`` remains a scalar operation. We intentionally calculate omega
    # as a tensor first; only the final scalar is materialized.
    omega = torch.acos(
        cosine
    )

    sin_omega = torch.sin(
        omega
    )

    if sin_omega.abs().item() <= 1e-6:
        return (
            alpha * a32
            + (1.0 - alpha) * b32
        ).to(a.dtype)

    coeff_a = (
        torch.sin(
            (1.0 - alpha)
            * omega
        )
        / sin_omega
    )

    coeff_b = (
        torch.sin(
            alpha * omega
        )
        / sin_omega
    )

    result = (
        coeff_a * a32
        + coeff_b * b32
    )

    return result.to(
        dtype=a.dtype
    )


def _ties_mask(
    tensor: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """
    Create a TIES magnitude mask.

    The returned mask is boolean rather than float32, reducing memory usage.
    """
    flat = tensor.detach().abs().reshape(-1)

    if flat.numel() == 0:
        return torch.empty_like(
            tensor,
            dtype=torch.bool,
        )

    k = max(
        1,
        min(
            flat.numel(),
            int(
                math.ceil(
                    density
                    * flat.numel()
                )
            ),
        ),
    )

    if k >= flat.numel():
        threshold = flat.min()
    else:
        # kthvalue avoids materializing all top-k values.
        threshold = torch.kthvalue(
            flat,
            flat.numel() - k + 1,
        ).values

    return tensor.detach().abs() >= threshold


def _ties_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """
    Conservative two-model TIES merge.

    Retains values where both tensors:

        • belong to the selected magnitude density,
        • have non-zero signs,
        • agree in sign.

    Otherwise the primary model A is retained.

    This preserves the semantic behavior of the original implementation while
    using boolean masks instead of float masks.
    """
    mask_a = _ties_mask(
        a,
        density,
    )

    mask_b = _ties_mask(
        b,
        density,
    )

    sign_a = torch.sign(a)
    sign_b = torch.sign(b)

    consensus = (
        mask_a
        & mask_b
        & (sign_a == sign_b)
        & (sign_a != 0)
    )

    return torch.where(
        consensus,
        0.5 * (a.float() + b.float()),
        a.float(),
    ).to(a.dtype)


# =============================================================================
# Public weighted merge
# =============================================================================


def fast_weighted_avg(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.5,
    use_cuda: bool = True,
) -> torch.Tensor:
    """
    Fast weighted average.

        result = alpha*a + (1-alpha)*b
    """
    _validate_pair(
        a,
        b,
    )

    alpha = _validate_alpha(
        alpha
    )

    extension = (
        _mod()
        if use_cuda and _cuda_pair(a, b)
        else None
    )

    if extension is not None:
        try:
            return extension.weighted_avg_cuda(
                a.contiguous().float(),
                b.contiguous().float(),
                alpha,
            ).to(
                dtype=a.dtype
            )

        except Exception:
            logger.warning(
                "FTRAIN weighted CUDA kernel failed. "
                "Falling back to PyTorch.",
                exc_info=True,
            )

    # CPU implementation may be available in the extension. It is still
    # guarded because kernel symbol names can vary by build.
    extension = (
        _mod()
        if use_cuda is False
        else extension
    )

    if (
        extension is not None
        and hasattr(
            extension,
            "weighted_avg_cpu",
        )
    ):
        try:
            return extension.weighted_avg_cpu(
                a.contiguous().float(),
                b.contiguous().float(),
                alpha,
            ).to(
                dtype=a.dtype
            )

        except Exception:
            logger.warning(
                "FTRAIN weighted CPU kernel failed. "
                "Falling back to PyTorch.",
                exc_info=True,
            )

    return _weighted_average_fallback(
        a,
        b,
        alpha,
    )


# =============================================================================
# Public Fisher merge
# =============================================================================


def fast_fisher_merge(
    a: torch.Tensor,
    b: torch.Tensor,
    fa: torch.Tensor,
    fb: torch.Tensor,
    use_cuda: bool = True,
) -> torch.Tensor:
    """
    Fast Fisher-weighted model merge.
    """
    _validate_fisher(
        a,
        b,
        fa,
        fb,
    )

    extension = (
        _mod()
        if use_cuda and _cuda_pair(a, b)
        else None
    )

    if extension is not None:
        try:
            return extension.fisher_merge_cuda(
                a.contiguous().float(),
                b.contiguous().float(),
                fa.contiguous().float(),
                fb.contiguous().float(),
            ).to(
                dtype=a.dtype
            )

        except Exception:
            logger.warning(
                "FTRAIN Fisher CUDA kernel failed. "
                "Falling back to PyTorch.",
                exc_info=True,
            )

    if (
        extension is not None
        and hasattr(
            extension,
            "fisher_merge_cpu",
        )
    ):
        try:
            return extension.fisher_merge_cpu(
                a.contiguous().float(),
                b.contiguous().float(),
                fa.contiguous().float(),
                fb.contiguous().float(),
            ).to(
                dtype=a.dtype
            )

        except Exception:
            logger.warning(
                "FTRAIN Fisher CPU kernel failed. "
                "Falling back to PyTorch.",
                exc_info=True,
            )

    return _fisher_merge_fallback(
        a,
        b,
        fa,
        fb,
    )


# =============================================================================
# Public SLERP
# =============================================================================


def fast_slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.5,
    use_cuda: bool = True,
) -> torch.Tensor:
    """
    Fast spherical linear interpolation.

    Falls back to linear interpolation for:

        • zero vectors
        • nearly parallel vectors
        • nearly opposite vectors where numerical stability becomes poor
    """
    _validate_pair(
        a,
        b,
    )

    alpha = _validate_alpha(
        alpha
    )

    a32 = a.float()
    b32 = b.float()

    flat_a = a32.reshape(-1)
    flat_b = b32.reshape(-1)

    norm_a = torch.linalg.vector_norm(
        flat_a
    )

    norm_b = torch.linalg.vector_norm(
        flat_b
    )

    if (
        norm_a.item() <= 1e-8
        or norm_b.item() <= 1e-8
    ):
        return _weighted_average_fallback(
            a,
            b,
            alpha,
        )

    cosine = (
        torch.sum(
            flat_a * flat_b
        )
        / (
            norm_a * norm_b
        )
    ).clamp(
        -1.0,
        1.0,
    )

    omega = torch.acos(
        cosine
    )

    sin_omega = torch.sin(
        omega
    )

    # Native kernel receives scalar omega parameters, so materializing those
    # values is unavoidable for that path.
    if (
        sin_omega.abs().item() <= 1e-6
    ):
        return _weighted_average_fallback(
            a,
            b,
            alpha,
        )

    extension = (
        _mod()
        if use_cuda and _cuda_pair(a, b)
        else None
    )

    if extension is not None:
        try:
            return extension.slerp_merge_cuda(
                a.contiguous().float(),
                b.contiguous().float(),
                alpha,
                float(
                    omega.item()
                ),
                float(
                    sin_omega.item()
                ),
            ).to(
                dtype=a.dtype
            )

        except Exception:
            logger.warning(
                "FTRAIN SLERP CUDA kernel failed. "
                "Falling back to PyTorch.",
                exc_info=True,
            )

    return _slerp_fallback(
        a,
        b,
        alpha,
    )


# =============================================================================
# Public TIES merge
# =============================================================================


def fast_ties(
    a: torch.Tensor,
    b: torch.Tensor,
    density: float = 0.5,
    use_cuda: bool = True,
) -> torch.Tensor:
    """
    Fast two-model TIES merge.

    Density selects the highest-magnitude fraction of each tensor.
    """
    _validate_pair(
        a,
        b,
    )

    density = _validate_density(
        density
    )

    # Build boolean masks first. These are also needed for CPU fallback.
    mask_a = _ties_mask(
        a,
        density,
    )

    mask_b = _ties_mask(
        b,
        density,
    )

    extension = (
        _mod()
        if use_cuda and _cuda_pair(a, b)
        else None
    )

    if extension is not None:
        try:
            # Current CUDA kernel contract expects float masks, so only convert
            # here at the native-extension boundary rather than permanently
            # storing masks as float32.
            return extension.ties_merge_cuda(
                a.contiguous().float(),
                b.contiguous().float(),
                mask_a.float().contiguous(),
                mask_b.float().contiguous(),
            ).to(
                dtype=a.dtype
            )

        except Exception:
            logger.warning(
                "FTRAIN TIES CUDA kernel failed. "
                "Falling back to PyTorch.",
                exc_info=True,
            )

    sign_a = torch.sign(
        a
    )
    sign_b = torch.sign(
        b
    )

    consensus = (
        mask_a
        & mask_b
        & (sign_a == sign_b)
        & (sign_a != 0)
    )

    merged = torch.where(
        consensus,
        (
            a.float()
            + b.float()
        ) * 0.5,
        a.float(),
    )

    return merged.to(
        dtype=a.dtype
    )
