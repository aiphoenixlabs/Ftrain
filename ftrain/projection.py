from pathlib import Path

code = r'''"""
FTRAIN Projection & Representation Alignment Utilities v1.1
============================================================

Projection helpers used by FTRAIN's architecture-aware merger.

Supported strategies
--------------------
identity
procrustes / svd / orthogonal
cca
lstsq

Design goals
------------
• Never mutate source tensors.
• Work with vectors, matrices and higher-dimensional tensors.
• Return mathematically valid projection shapes.
• Keep dtype/device consistent with the source tensor.
• Avoid silent shape errors during ``a @ P``.
• Prefer stable float32 numerical work for decompositions.
• Gracefully fall back to identity when alignment is not possible.
• Preserve the historical public API:
      identity(a, b)
      procrustes(a, b)
      apply_projection(a, s, b)
      PROJECTIONS
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "_m",
    "identity",
    "procrustes",
    "cca",
    "lstsq",
    "orthogonal",
    "apply_projection",
    "PROJECTIONS",
]


# =============================================================================
# Tensor normalization
# =============================================================================

def _m(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor into a 2-D representation matrix.

    Shapes
    ------
    [D]        -> [1, D]
    [M, N]     -> [M, N]
    [A, B, ...] -> [A, B*...]

    This preserves the first dimension as the row/sequence/output dimension.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(t).__name__}."
        )

    tensor = t.detach()

    if tensor.dim() == 0:
        return tensor.reshape(1, 1)

    if tensor.dim() == 1:
        return tensor.unsqueeze(0)

    if tensor.dim() == 2:
        return tensor

    return tensor.reshape(
        tensor.shape[0],
        -1,
    )


def _feature_dim(t: torch.Tensor) -> int:
    """Return the dimension a projection should act on for ``a @ P``."""
    matrix = _m(t)

    if matrix.dim() != 2:
        return 1

    return int(matrix.shape[-1])


def _safe_float_matrix(
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a decomposition input to stable float32.

    bfloat16/float16 SVD and least-squares can be unsupported or numerically
    fragile on some backends, so projection discovery is performed in fp32.
    """
    return t.detach().to(
        dtype=torch.float32
    )


def _cast_projection(
    projection: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Move/cast a projection to the same device/dtype as the reference."""
    return projection.to(
        device=reference.device,
        dtype=reference.dtype,
    )


# =============================================================================
# Identity projection
# =============================================================================

def identity(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Return an identity projection valid for ``a @ P``.

    IMPORTANT:
    The old implementation used ``a.shape[0]``. For matrix projection this is
    incorrect when ``a`` is [out_features, in_features]. The identity must be
    sized to the INPUT/feature dimension, i.e. ``a.shape[-1]``.
    """
    feature_dim = _feature_dim(a)

    return torch.eye(
        feature_dim,
        dtype=a.dtype,
        device=a.device,
    )


# =============================================================================
# Orthogonal Procrustes
# =============================================================================

def _orthogonal_map(
    source: torch.Tensor,
    target: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Estimate an orthogonal transform P that maps source features toward target.

    The returned matrix is shaped:

        [source_features, target_features]

    when the feature dimensions permit an orthogonal/rectangular map.
    """
    source_f = _safe_float_matrix(source)
    target_f = _safe_float_matrix(target)

    if source_f.dim() != 2 or target_f.dim() != 2:
        return None

    if source_f.shape[0] != target_f.shape[0]:
        return None

    source_features = source_f.shape[1]
    target_features = target_f.shape[1]

    try:
        # Same feature dimension: classic orthogonal Procrustes.
        if source_features == target_features:
            cross = source_f.transpose(0, 1) @ target_f

            u, _, vh = torch.linalg.svd(
                cross,
                full_matrices=False,
            )

            return u @ vh

        # Different feature dimensions:
        # construct a rank-min(source_features, target_features) rectangular
        # orthogonal map.
        rank = min(
            source_features,
            target_features,
        )

        cross = source_f.transpose(0, 1) @ target_f

        u, _, vh = torch.linalg.svd(
            cross,
            full_matrices=False,
        )

        u = u[:, :rank]
        vh = vh[:rank, :]

        return u @ vh

    except (RuntimeError, torch.linalg.LinAlgError):
        return None


def procrustes(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a stable Procrustes projection.

    For equal feature dimensions:
        P = U V^T

    For rectangular feature spaces:
        a reduced orthogonal map is returned.

    When alignment cannot be computed, a shape-valid identity/fallback is
    returned.
    """
    a2 = _m(a)
    b2 = _m(b)

    # The row/sample dimension needs to correspond for direct Procrustes.
    if a2.shape[0] != b2.shape[0]:
        logger.debug(
            "Procrustes skipped: row mismatch %s vs %s.",
            tuple(a2.shape),
            tuple(b2.shape),
        )
        return identity(
            a,
            b,
        )

    projection = _orthogonal_map(
        a2,
        b2,
    )

    if projection is None:
        return identity(
            a,
            b,
        )

    return _cast_projection(
        projection,
        a,
    )


# =============================================================================
# CCA-like alignment
# =============================================================================

def cca(
    a: torch.Tensor,
    b: torch.Tensor,
    regularization: float = 1e-4,
) -> torch.Tensor:
    """
    Stable covariance-based alignment.

    This is intentionally a lightweight CCA-style projection rather than a full
    multi-stage statistical CCA implementation. It estimates whitening-like
    transforms and combines them into a feature-space map.

    Falls back to Procrustes/identity on numerical failure.
    """
    a2 = _safe_float_matrix(
        _m(a)
    )
    b2 = _safe_float_matrix(
        _m(b)
    )

    if a2.shape[0] != b2.shape[0]:
        return identity(
            a,
            b,
        )

    try:
        # Center features.
        a_centered = (
            a2
            - a2.mean(
                dim=0,
                keepdim=True,
            )
        )

        b_centered = (
            b2
            - b2.mean(
                dim=0,
                keepdim=True,
            )
        )

        n = max(
            1,
            a_centered.shape[0] - 1,
        )

        cov_ab = (
            a_centered.transpose(0, 1)
            @ b_centered
            / n
        )

        cov_aa = (
            a_centered.transpose(0, 1)
            @ a_centered
            / n
        )

        cov_bb = (
            b_centered.transpose(0, 1)
            @ b_centered
            / n
        )

        eye_a = torch.eye(
            cov_aa.shape[0],
            device=cov_aa.device,
            dtype=cov_aa.dtype,
        )

        eye_b = torch.eye(
            cov_bb.shape[0],
            device=cov_bb.device,
            dtype=cov_bb.dtype,
        )

        cov_aa = (
            cov_aa
            + regularization * eye_a
        )

        cov_bb = (
            cov_bb
            + regularization * eye_b
        )

        # Symmetric inverse square root for A covariance.
        eval_a, evec_a = torch.linalg.eigh(
            cov_aa
        )

        eval_a = eval_a.clamp_min(
            regularization
        )

        inv_sqrt_a = (
            evec_a
            @ torch.diag(
                eval_a.rsqrt()
            )
            @ evec_a.transpose(0, 1)
        )

        # Symmetric inverse square root for B covariance.
        eval_b, evec_b = torch.linalg.eigh(
            cov_bb
        )

        eval_b = eval_b.clamp_min(
            regularization
        )

        inv_sqrt_b = (
            evec_b
            @ torch.diag(
                eval_b.rsqrt()
            )
            @ evec_b.transpose(0, 1)
        )

        whitened = (
            inv_sqrt_a
            @ cov_ab
            @ inv_sqrt_b
        )

        u, _, vh = torch.linalg.svd(
            whitened,
            full_matrices=False,
        )

        # A -> B style feature transform.
        projection = (
            inv_sqrt_a
            @ u
            @ vh
            @ inv_sqrt_b
        )

        # If numerical dimensions don't support a direct matrix product for
        # the merger, prefer the simpler Procrustes transform.
        if projection.shape[0] != _feature_dim(a):
            return procrustes(
                a,
                b,
            )

        return _cast_projection(
            projection,
            a,
        )

    except (
        RuntimeError,
        torch.linalg.LinAlgError,
    ):
        logger.debug(
            "CCA projection failed; falling back to Procrustes.",
            exc_info=True,
        )

        return procrustes(
            a,
            b,
        )


# =============================================================================
# Least-squares alignment
# =============================================================================

def lstsq(
    a: torch.Tensor,
    b: torch.Tensor,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Estimate a least-squares linear map P for:

        A @ P ≈ B

    This is useful when the two representations are related but not strictly
    orthogonal.

    Returns:
        P shaped [A_features, B_features]

    The matrix can be used directly as ``a @ P`` when dimensions permit.
    """
    a2 = _safe_float_matrix(
        _m(a)
    )
    b2 = _safe_float_matrix(
        _m(b)
    )

    if a2.shape[0] != b2.shape[0]:
        return identity(
            a,
            b,
        )

    try:
        features = a2.shape[1]

        regularizer = (
            torch.eye(
                features,
                dtype=a2.dtype,
                device=a2.device,
            )
            * ridge
        )

        ata = (
            a2.transpose(0, 1)
            @ a2
        )

        atb = (
            a2.transpose(0, 1)
            @ b2
        )

        projection = torch.linalg.solve(
            ata + regularizer,
            atb,
        )

        return _cast_projection(
            projection,
            a,
        )

    except (
        RuntimeError,
        torch.linalg.LinAlgError,
    ):
        logger.debug(
            "Least-squares projection failed; falling back to Procrustes.",
            exc_info=True,
        )

        return procrustes(
            a,
            b,
        )


# =============================================================================
# Explicit aliases
# =============================================================================

def orthogonal(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Explicit alias for Procrustes orthogonal alignment."""
    return procrustes(
        a,
        b,
    )


def svd(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """SVD-based orthogonal alignment."""
    return procrustes(
        a,
        b,
    )


# =============================================================================
# Projection registry
# =============================================================================

PROJECTIONS: Dict[
    str,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
] = {
    "identity": identity,
    "procrustes": procrustes,
    "svd": svd,
    "cca": cca,
    "orthogonal": orthogonal,
    "lstsq": lstsq,
}


# =============================================================================
# Public application helper
# =============================================================================

def apply_projection(
    a: torch.Tensor,
    strategy: str,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a named projection strategy.

    Parameters
    ----------
    a:
        Source/base tensor.

    strategy:
        Projection name.

    b:
        Target/comparison tensor.

    Returns
    -------
    torch.Tensor
        Projection matrix.

    Notes
    -----
    Unknown strategies safely fall back to identity, preserving the historical
    FTRAIN behavior.
    """
    if not isinstance(
        a,
        torch.Tensor,
    ):
        raise TypeError(
            f"a must be torch.Tensor, got {type(a).__name__}."
        )

    if not isinstance(
        b,
        torch.Tensor,
    ):
        raise TypeError(
            f"b must be torch.Tensor, got {type(b).__name__}."
        )

    normalized_strategy = str(
        strategy
    ).strip().lower()

    function = PROJECTIONS.get(
        normalized_strategy,
        identity,
    )

    try:
        projection = function(
            a,
            b,
        )

    except Exception:
        logger.warning(
            "FTRAIN projection '%s' failed; using identity.",
            normalized_strategy,
            exc_info=True,
        )

        projection = identity(
            a,
            b,
        )

    # Final shape safety for the common merger operation:
    #
    #     projected_a = a @ P
    #
    # P's number of rows MUST equal the last dimension of a.
    expected_rows = _feature_dim(
        a
    )

    if (
        projection.ndim != 2
        or projection.shape[0] != expected_rows
    ):
        logger.debug(
            "Projection '%s' returned invalid shape %s for source feature "
            "dimension %d; using identity.",
            normalized_strategy,
            tuple(projection.shape),
            expected_rows,
        )

        projection = identity(
            a,
            b,
        )

    return projection


# =============================================================================
# Lightweight self-test
# =============================================================================

def _self_test() -> Dict[str, Tuple[int, ...]]:
    """
    Lightweight developer test.

    Verifies the most important historical bug:
    identity(a, b) must use a.shape[-1], not a.shape[0].
    """
    a = torch.randn(
        8,
        16,
    )

    b = torch.randn(
        8,
        16,
    )

    expected_shape = (
        16,
        16,
    )

    outputs = {
        "identity": identity(a, b),
        "procrustes": procrustes(a, b),
        "svd": svd(a, b),
        "cca": cca(a, b),
        "orthogonal": orthogonal(a, b),
        "lstsq": lstsq(a, b),
    }

    for name, value in outputs.items():
        if tuple(value.shape) != expected_shape:
            raise AssertionError(
                f"{name} returned {tuple(value.shape)}, "
                f"expected {expected_shape}."
            )

    # Verify the merger operation itself is valid.
    projected = a @ outputs["procrustes"]

    if projected.shape != a.shape:
        raise AssertionError(
            "Projected tensor shape does not match source tensor."
        )

    return {
        name: tuple(value.shape)
        for name, value in outputs.items()
    }


if __name__ == "__main__":
    print(_self_test())
'''
path = Path("/mnt/data/ftrain_projection_FULL_enhanced_v1_1.py")
path.write_text(code, encoding="utf-8")
compile(code, str(path), "exec")
ns = {}
exec(compile(code, str(path), "exec"), ns)
print(f"Created: {path}")
print(f"Lines: {len(code.splitlines())}")
print("Self-test:", ns["_self_test"]())
