"""
FTRAIN Fused DoRA Kernel
========================

High-performance DoRA normalization implemented with Triton, with a safe
PyTorch fallback.

The operation implemented here is:

    output = normalize(comb, dim=1) * mag

where:

    normalize(x) = x / ||x||

This corresponds to the core magnitude-normalization operation used by DoRA.

Design goals
------------
• Correct for arbitrary M x N tensors.
• Correctly process N > Triton block width.
• Support non-contiguous input tensors by normalizing layout before launch.
• Preserve input dtype for the output.
• Accumulate normalization and gradients in FP32.
• Avoid unnecessary synchronization.
• Provide a PyTorch fallback when Triton/CUDA is unavailable.
• Make autograd behavior explicit and reliable.
• Keep the public API small.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "DoraFusedFunction",
    "dora_fused",
    "dora_fused_available",
]


# =============================================================================
# Triton availability
# =============================================================================

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True

except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def dora_fused_available() -> bool:
    """
    Return True when the Triton CUDA implementation is available.

    This does not compile a kernel. It only checks whether Triton is installed
    and CUDA is available.
    """
    return bool(
        _TRITON_AVAILABLE
        and torch.cuda.is_available()
    )


# =============================================================================
# Constants
# =============================================================================

_EPS = 1e-8


# =============================================================================
# Triton kernels
# =============================================================================

if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128},
                num_warps=8,
                num_stages=2,
            ),
        ],
        key=["M", "N"],
    )
    @triton.jit
    def _dora_forward_kernel(
        comb_ptr,
        mag_ptr,
        out_ptr,
        M,
        N,
        stride_cm,
        stride_cn,
        stride_om,
        stride_on,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Forward DoRA normalization.

        Each program handles a rectangular M x N tile.

        Important:
        The original implementation only loaded rn = arange(BN), which meant
        N > BN was never processed. This implementation explicitly tiles both
        dimensions.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = (
            pid_m * BLOCK_M
            + tl.arange(0, BLOCK_M)
        )

        cols = (
            pid_n * BLOCK_N
            + tl.arange(0, BLOCK_N)
        )

        row_mask = rows < M
        col_mask = cols < N

        mask = (
            row_mask[:, None]
            & col_mask[None, :]
        )

        comb = tl.load(
            comb_ptr
            + rows[:, None] * stride_cm
            + cols[None, :] * stride_cn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Row-wise L2 norm.
        squared = comb * comb

        norm_sq = tl.sum(
            squared,
            axis=1,
        )

        norm = tl.sqrt(
            norm_sq + EPS
        )

        inv_norm = 1.0 / norm

        magnitude = tl.load(
            mag_ptr + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        result = (
            comb
            * inv_norm[:, None]
            * magnitude[None, :]
        )

        tl.store(
            out_ptr
            + rows[:, None] * stride_om
            + cols[None, :] * stride_on,
            result,
            mask=mask,
        )


    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128},
                num_warps=8,
                num_stages=2,
            ),
        ],
        key=["M", "N"],
    )
    @triton.jit
    def _dora_backward_kernel(
        grad_out_ptr,
        comb_ptr,
        mag_ptr,
        grad_comb_ptr,
        grad_mag_ptr,
        M,
        N,
        stride_gm,
        stride_gn,
        stride_cm,
        stride_cn,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Backward DoRA normalization.

        Computes:

            y = x / ||x|| * m

        dy -> dx and dm

        Accumulation is performed in FP32.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = (
            pid_m * BLOCK_M
            + tl.arange(0, BLOCK_M)
        )

        cols = (
            pid_n * BLOCK_N
            + tl.arange(0, BLOCK_N)
        )

        row_mask = rows < M
        col_mask = cols < N

        mask = (
            row_mask[:, None]
            & col_mask[None, :]
        )

        grad_out = tl.load(
            grad_out_ptr
            + rows[:, None] * stride_gm
            + cols[None, :] * stride_gn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        comb = tl.load(
            comb_ptr
            + rows[:, None] * stride_cm
            + cols[None, :] * stride_cn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        magnitude = tl.load(
            mag_ptr + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        # ||x||^-1
        norm_sq = tl.sum(
            comb * comb,
            axis=1,
        )

        norm = tl.sqrt(
            norm_sq + EPS
        )

        inv_norm = 1.0 / norm

        inv_norm_cubed = (
            inv_norm
            * inv_norm
            * inv_norm
        )

        # d(x / ||x||)
        #
        # = g / ||x||
        #   - x * (x · g) / ||x||^3
        dot = tl.sum(
            comb * grad_out,
            axis=1,
        )

        grad_comb = (
            magnitude[None, :]
            * inv_norm[:, None]
            * grad_out
            -
            magnitude[None, :]
            * inv_norm_cubed[:, None]
            * dot[:, None]
            * comb
        )

        tl.store(
            grad_comb_ptr
            + rows[:, None] * stride_cm
            + cols[None, :] * stride_cn,
            grad_comb,
            mask=mask,
        )

        # dm = sum(dy * x / ||x||)
        grad_mag = (
            grad_out
            * comb
            * inv_norm[:, None]
        )

        # Different M programs contribute to the same magnitude vector.
        #
        # Atomic accumulation is required because the magnitude parameter is
        # shared across all rows.
        partial_mag = tl.sum(
            grad_mag,
            axis=0,
        )

        tl.atomic_add(
            grad_mag_ptr + cols,
            partial_mag,
            mask=col_mask,
        )


# =============================================================================
# PyTorch fallback
# =============================================================================


def _torch_forward(
    comb: torch.Tensor,
    mag: torch.Tensor,
) -> torch.Tensor:
    """
    Numerically stable PyTorch implementation.

    Norms are calculated in FP32, then the final result is converted back to
    the input dtype.
    """
    work = comb.float()
    magnitude = mag.float()

    norm = torch.linalg.vector_norm(
        work,
        ord=2,
        dim=1,
        keepdim=True,
    ).clamp_min(_EPS)

    result = (
        work
        / norm
        * magnitude.unsqueeze(0)
    )

    return result.to(
        dtype=comb.dtype
    )


# =============================================================================
# Input validation
# =============================================================================


def _validate_inputs(
    comb: torch.Tensor,
    mag: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and normalize inputs before entering either implementation.
    """

    if not isinstance(
        comb,
        torch.Tensor,
    ):
        raise TypeError(
            "comb must be a torch.Tensor."
        )

    if not isinstance(
        mag,
        torch.Tensor,
    ):
        raise TypeError(
            "mag must be a torch.Tensor."
        )

    if comb.ndim != 2:
        raise ValueError(
            f"comb must be 2-D [M, N], got shape={tuple(comb.shape)}."
        )

    if mag.ndim != 1:
        raise ValueError(
            f"mag must be 1-D [N], got shape={tuple(mag.shape)}."
        )

    M, N = comb.shape

    if mag.numel() != N:
        raise ValueError(
            "Magnitude dimension mismatch: "
            f"comb has N={N}, but mag has {mag.numel()} elements."
        )

    if not comb.is_floating_point():
        raise TypeError(
            f"comb must use a floating-point dtype, got {comb.dtype}."
        )

    if not mag.is_floating_point():
        raise TypeError(
            f"mag must use a floating-point dtype, got {mag.dtype}."
        )

    if comb.device != mag.device:
        raise ValueError(
            "comb and mag must be on the same device. "
            f"Got {comb.device} and {mag.device}."
        )

    if comb.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise TypeError(
            f"Unsupported comb dtype: {comb.dtype}."
        )

    # Triton implementation expects a simple contiguous matrix.
    #
    # Making the tensors contiguous here also eliminates complicated stride
    # corner cases and generally improves memory coalescing.
    if not comb.is_contiguous():
        comb = comb.contiguous()

    if not mag.is_contiguous():
        mag = mag.contiguous()

    return comb, mag


# =============================================================================
# Autograd implementation
# =============================================================================


class DoraFusedFunction(
    torch.autograd.Function
):
    """
    Autograd-enabled fused DoRA normalization.

    Public callers should normally use ``dora_fused`` instead of invoking this
    class directly.
    """

    @staticmethod
    def forward(
        ctx,
        comb: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:

        comb, mag = _validate_inputs(
            comb,
            mag,
        )

        # ---------------------------------------------------------------------
        # Empty tensors
        # ---------------------------------------------------------------------

        if comb.numel() == 0:
            output = torch.empty_like(
                comb
            )

            ctx.save_for_backward(
                comb,
                mag,
            )

            return output

        # ---------------------------------------------------------------------
        # Triton path
        # ---------------------------------------------------------------------

        if (
            _TRITON_AVAILABLE
            and comb.is_cuda
            and mag.is_cuda
        ):
            M, N = comb.shape

            output = torch.empty_like(
                comb
            )

            def grid(
                META,
            ):
                return (
                    triton.cdiv(
                        M,
                        META["BLOCK_M"],
                    ),
                    triton.cdiv(
                        N,
                        META["BLOCK_N"],
                    ),
                )

            _dora_forward_kernel[grid](
                comb,
                mag,
                output,
                M,
                N,
                comb.stride(0),
                comb.stride(1),
                output.stride(0),
                output.stride(1),
                EPS=_EPS,
            )

        # ---------------------------------------------------------------------
        # PyTorch fallback
        # ---------------------------------------------------------------------

        else:
            output = _torch_forward(
                comb,
                mag,
            )

        ctx.save_for_backward(
            comb,
            mag,
        )

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: Optional[torch.Tensor],
    ):
        if grad_output is None:
            return None, None

        comb, mag = ctx.saved_tensors

        # Gradient may arrive non-contiguous from an upstream operation.
        grad_output = grad_output.contiguous()

        # ---------------------------------------------------------------------
        # Triton backward
        # ---------------------------------------------------------------------

        if (
            _TRITON_AVAILABLE
            and comb.is_cuda
            and mag.is_cuda
        ):
            M, N = comb.shape

            grad_comb = torch.empty_like(
                comb
            )

            # Always accumulate magnitude gradients in FP32.
            #
            # This is particularly important for FP16/BF16 training because
            # many rows contribute to the same magnitude parameter.
            grad_mag_fp32 = torch.zeros(
                N,
                dtype=torch.float32,
                device=mag.device,
            )

            def grid(
                META,
            ):
                return (
                    triton.cdiv(
                        M,
                        META["BLOCK_M"],
                    ),
                    triton.cdiv(
                        N,
                        META["BLOCK_N"],
                    ),
                )

            _dora_backward_kernel[grid](
                grad_output,
                comb,
                mag,
                grad_comb,
                grad_mag_fp32,
                M,
                N,
                grad_output.stride(0),
                grad_output.stride(1),
                comb.stride(0),
                comb.stride(1),
                EPS=_EPS,
            )

            grad_mag = grad_mag_fp32.to(
                dtype=mag.dtype
            )

        # ---------------------------------------------------------------------
        # PyTorch fallback
        # ---------------------------------------------------------------------

        else:
            with torch.enable_grad():
                # Recompute using autograd rather than maintaining a second
                # manually derived fallback formula.
                #
                # This makes the fallback easier to verify against PyTorch's
                # reference implementation.
                comb_ref = comb.detach().requires_grad_()
                mag_ref = mag.detach().requires_grad_()

                output = _torch_forward(
                    comb_ref,
                    mag_ref,
                )

                grad_comb, grad_mag = torch.autograd.grad(
                    outputs=output,
                    inputs=(
                        comb_ref,
                        mag_ref,
                    ),
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

        return (
            grad_comb,
            grad_mag,
        )


# =============================================================================
# Public API
# =============================================================================


def dora_fused(
    comb: torch.Tensor,
    mag: torch.Tensor,
) -> torch.Tensor:
    """
    Apply fused DoRA normalization.

    Parameters
    ----------
    comb:
        2-D tensor with shape ``[M, N]``.

    mag:
        1-D magnitude tensor with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape and dtype as ``comb``.

    Notes
    -----
    On CUDA systems with Triton installed, the Triton implementation is used.

    Otherwise, a numerically equivalent PyTorch implementation is used.
    """
    return DoraFusedFunction.apply(
        comb,
        mag,
    )


# =============================================================================
# Optional debug helper
# =============================================================================


def _reference_dora(
    comb: torch.Tensor,
    mag: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation useful for tests.

    This intentionally uses plain PyTorch and is not part of the primary API.
    """
    work = comb.float()

    norm = torch.linalg.vector_norm(
        work,
        dim=1,
        keepdim=True,
    ).clamp_min(_EPS)

    return (
        work
        / norm
        * mag.float().unsqueeze(0)
    ).to(
        comb.dtype
    )
