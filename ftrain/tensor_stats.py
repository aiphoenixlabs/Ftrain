# ============================================================
# FTRAIN - Deep Tensor State / Statistics Engine
# ============================================================

import math
import logging

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import torch


logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TensorStatsConfig:
    """
    Controls the cost/quality of tensor analysis.

    Expensive operations such as SVD are bounded by matrix limits,
    while cheap scalar reductions can still inspect the full tensor.
    """

    # Maximum number of values used for distribution statistics.
    max_sample_elements: int = 1_000_000

    # Maximum matrix dimensions used for spectral analysis.
    max_matrix_rows: int = 2048
    max_matrix_cols: int = 2048

    # Number of singular values retained.
    max_singular_values: int = 16

    # Power iteration fallback.
    power_iterations: int = 8

    # Exact SVD is only used below this matrix size.
    exact_svd_elements: int = 4_000_000

    # Near-zero detection.
    near_zero_rtol: float = 1e-3
    near_zero_atol: float = 1e-8

    # Quantile calculation.
    compute_quantiles: bool = True
    quantiles: Tuple[float, ...] = (
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    )

    # Reproducible deterministic sampling.
    deterministic_sampling: bool = True


DEFAULT_CONFIG = TensorStatsConfig()


# ============================================================
# TENSOR STATISTICS
# ============================================================

@dataclass
class TensorStats:

    # -------------------------
    # Identity
    # -------------------------

    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str

    # -------------------------
    # Original FTRAIN metrics
    # -------------------------

    mean: float = 0.0
    std: float = 0.0
    l2_norm: float = 0.0
    spectral_norm: float = 0.0
    effective_rank: float = 0.0
    entropy: float = 0.0
    sparsity: float = 0.0
    dead: bool = False

    singular_values: List[float] = field(default_factory=list)

    # -------------------------
    # Device / memory
    # -------------------------

    device: str = ""
    requires_grad: bool = False
    estimated_bytes: int = 0
    sampled_numel: int = 0

    # -------------------------
    # Numerical health
    # -------------------------

    finite_ratio: float = 1.0
    min_value: float = 0.0
    max_value: float = 0.0
    min_abs: float = 0.0
    max_abs: float = 0.0

    mean_abs: float = 0.0
    rms: float = 0.0
    median: float = 0.0

    # -------------------------
    # Sparsity / structure
    # -------------------------

    zero_ratio: float = 0.0
    near_zero_ratio: float = 0.0
    constant: bool = False

    # -------------------------
    # Rank / spectrum
    # -------------------------

    stable_rank: float = 0.0
    effective_rank_ratio: float = 0.0
    spectral_ratio: float = 0.0
    condition_number: float = 0.0
    singular_entropy: float = 0.0

    # -------------------------
    # Distribution shape
    # -------------------------

    skewness: float = 0.0
    kurtosis: float = 0.0

    quantiles: Dict[str, float] = field(default_factory=dict)

    # Matrix representation used during spectral analysis.
    matrix_shape: Tuple[int, int] = (0, 0)

    # Error is kept instead of crashing an entire model scan.
    error: Optional[str] = None


# ============================================================
# TENSOR RELATION
# ============================================================

@dataclass
class TensorRelation:
    """
    Relationship between two corresponding tensors.

    Designed specifically for:
        CBA
        merge conflict analysis
        model comparison
        weight alignment
    """

    name: str
    shape: Tuple[int, ...]

    # Direction / scale
    cosine_similarity: float = 0.0
    norm_a: float = 0.0
    norm_b: float = 0.0
    norm_ratio_b_over_a: float = 0.0

    # Difference
    delta_l2: float = 0.0
    relative_delta_l2: float = 0.0
    mean_abs_delta: float = 0.0
    max_abs_delta: float = 0.0

    # Conflicts
    sign_agreement: float = 1.0
    sign_conflict_ratio: float = 0.0
    strong_conflict_ratio: float = 0.0

    # Shared structure
    activation_overlap: float = 0.0

    # Zero behavior
    exact_zero_a: float = 0.0
    exact_zero_b: float = 0.0

    shape_compatible: bool = True
    error: Optional[str] = None


# ============================================================
# STATE DICT SUMMARY
# ============================================================

@dataclass
class StateDictSummary:

    tensors: int = 0
    trainable_tensors: int = 0

    total_numel: int = 0
    total_bytes: int = 0

    mean_l2_norm: float = 0.0
    mean_spectral_norm: float = 0.0
    mean_sparsity: float = 0.0
    mean_effective_rank_ratio: float = 0.0

    dead_tensors: int = 0
    nonfinite_tensors: int = 0
    high_sparsity_tensors: int = 0

    by_dtype: Dict[str, int] = field(default_factory=dict)


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    x: torch.Tensor,
    default: float = 0.0,
) -> float:

    try:
        return float(x.detach().item())
    except Exception:
        return default


def _float_view(t: torch.Tensor) -> torch.Tensor:
    """
    Converts tensor to a numerically safe floating representation.

    FP16/BF16/int -> FP32
    FP32 -> unchanged
    FP64 -> preserved

    Complex tensors are converted to magnitude.
    """

    x = t.detach()

    if x.is_complex():
        return x.abs().float()

    if x.is_floating_point():

        if x.dtype in (
            torch.float64,
            torch.float32,
        ):
            return x

        return x.float()

    return x.float()


# ============================================================
# DETERMINISTIC SAMPLING
# ============================================================

def _deterministic_indices(
    length: int,
    limit: int,
    device: torch.device,
) -> torch.Tensor:

    if length <= limit:
        return torch.arange(
            length,
            device=device,
        )

    idx = torch.linspace(
        0,
        length - 1,
        steps=limit,
        device=device,
        dtype=torch.float64,
    ).round().long()

    return torch.unique_consecutive(idx)


def _sample_flat(
    x: torch.Tensor,
    max_elements: int,
    deterministic: bool = True,
) -> torch.Tensor:

    flat = x.reshape(-1)

    n = flat.numel()

    if n <= max_elements:
        return flat

    if deterministic:

        idx = _deterministic_indices(
            n,
            max_elements,
            flat.device,
        )

        return flat.index_select(
            0,
            idx,
        )

    idx = torch.randint(
        0,
        n,
        (max_elements,),
        device=flat.device,
    )

    return flat.index_select(
        0,
        idx,
    )


# ============================================================
# MATRIX CONVERSION
# ============================================================

def _to_matrix(
    x: torch.Tensor,
    max_rows: int,
    max_cols: int,
    deterministic: bool = True,
) -> torch.Tensor:
    """
    Converts arbitrary tensors into a 2D representation.

    scalar:
        1 x 1

    vector:
        1 x N

    matrix:
        M x N

    higher rank:
        first dimension x flattened remainder
    """

    if x.ndim == 0:

        matrix = x.reshape(1, 1)

    elif x.ndim == 1:

        matrix = x.reshape(1, -1)

    elif x.ndim == 2:

        matrix = x

    else:

        matrix = x.reshape(
            x.shape[0],
            -1,
        )

    rows, cols = matrix.shape

    # -------------------------
    # Row sampling
    # -------------------------

    if rows > max_rows:

        if deterministic:

            ridx = _deterministic_indices(
                rows,
                max_rows,
                matrix.device,
            )

        else:

            ridx = torch.randperm(
                rows,
                device=matrix.device,
            )[:max_rows]

        matrix = matrix.index_select(
            0,
            ridx,
        )

    # -------------------------
    # Column sampling
    # -------------------------

    if cols > max_cols:

        if deterministic:

            cidx = _deterministic_indices(
                cols,
                max_cols,
                matrix.device,
            )

        else:

            cidx = torch.randperm(
                cols,
                device=matrix.device,
            )[:max_cols]

        matrix = matrix.index_select(
            1,
            cidx,
        )

    return matrix


# ============================================================
# POWER ITERATION
# ============================================================

@torch.inference_mode()
def _power_spectral_norm(
    matrix: torch.Tensor,
    iterations: int = 8,
) -> float:
    """
    Estimates the largest singular value without full SVD.

    Useful fallback for very large matrices.
    """

    if matrix.numel() == 0:
        return 0.0

    matrix = matrix.float()

    rows, cols = matrix.shape

    if rows == 0 or cols == 0:
        return 0.0

    # Exact tiny matrix path.
    if matrix.numel() <= 4096:

        try:

            sv = torch.linalg.svdvals(
                matrix
            )

            if sv.numel() > 0:
                return _safe_float(
                    sv[0]
                )

        except Exception:
            pass

    # Deterministic initial vector.
    v = torch.ones(
        cols,
        device=matrix.device,
        dtype=matrix.dtype,
    )

    v_norm = torch.linalg.vector_norm(v)

    if _safe_float(v_norm) < 1e-12:
        return 0.0

    v = v / v_norm

    # Power iteration.
    for _ in range(
        max(
            1,
            iterations,
        )
    ):

        u = matrix @ v

        u_norm = torch.linalg.vector_norm(
            u
        )

        if _safe_float(u_norm) < 1e-12:
            return 0.0

        u = u / u_norm

        v = matrix.transpose(
            0,
            1,
        ) @ u

        v_norm = torch.linalg.vector_norm(
            v
        )

        if _safe_float(v_norm) < 1e-12:
            return 0.0

        v = v / v_norm

    sigma = torch.linalg.vector_norm(
        matrix @ v
    )

    return _safe_float(
        sigma
    )


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

@torch.inference_mode()
def _singular_analysis(
    matrix: torch.Tensor,
    config: TensorStatsConfig,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    List[float],
]:
    """
    Returns:

        spectral_norm
        effective_rank
        effective_rank_ratio
        normalized_entropy
        condition_number
        top_singular_values
    """

    if matrix.numel() == 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [],
        )

    matrix = matrix.float()

    rows, cols = matrix.shape

    max_rank = max(
        1,
        min(
            rows,
            cols,
        ),
    )

    # --------------------------------------------------------
    # Exact SVD for reasonably sized matrices
    # --------------------------------------------------------

    try:

        if matrix.numel() <= config.exact_svd_elements:

            sv = torch.linalg.svdvals(
                matrix
            )

        else:

            # For large matrices, still use the already-capped
            # matrix, but don't rely on the resulting full spectrum
            # if the matrix is huge.
            #
            # This is intentionally bounded by max_matrix_rows /
            # max_matrix_cols from _to_matrix().
            sv = torch.linalg.svdvals(
                matrix
            )

    except Exception as exc:

        logger.debug(
            "SVD failed: %s",
            exc,
        )

        sigma = _power_spectral_norm(
            matrix,
            config.power_iterations,
        )

        return (
            sigma,
            0.0,
            0.0,
            0.0,
            0.0,
            [],
        )

    # Remove invalid singular values.
    sv = sv[
        torch.isfinite(sv)
    ]

    if sv.numel() == 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [],
        )

    sv = sv.clamp_min(
        1e-12
    )

    # Largest singular value.
    sigma_max = _safe_float(
        sv[0]
    )

    # --------------------------------------------------------
    # Entropy of singular-value distribution
    # --------------------------------------------------------

    probs = (
        sv
        / sv.sum().clamp_min(1e-12)
    )

    entropy_raw = _safe_float(
        -(
            probs
            * probs.log()
        ).sum()
    )

    effective_rank = math.exp(
        min(
            50.0,
            entropy_raw,
        )
    )

    effective_rank_ratio = (
        effective_rank
        / max_rank
    )

    normalized_entropy = (
        entropy_raw
        / math.log(
            max(
                2,
                sv.numel(),
            )
        )
    )

    # --------------------------------------------------------
    # Condition number
    # --------------------------------------------------------

    sigma_min = _safe_float(
        sv[-1]
    )

    condition_number = (
        sigma_max
        / max(
            sigma_min,
            1e-12,
        )
    )

    return (
        sigma_max,
        effective_rank,
        effective_rank_ratio,
        normalized_entropy,
        condition_number,
        sv[
            : config.max_singular_values
        ].tolist(),
    )


# ============================================================
# MAIN TENSOR ANALYZER
# ============================================================

@torch.inference_mode()
def compute_tensor_stats(
    name: str,
    t: torch.Tensor,
    config: Optional[TensorStatsConfig] = None,
) -> TensorStats:
    """
    Deep tensor statistics.

    Backward compatible:

        compute_tensor_stats(
            "layer.weight",
            tensor
        )
    """

    if not isinstance(
        t,
        torch.Tensor,
    ):
        raise TypeError(
            "Expected torch.Tensor, "
            f"got {type(t).__name__}"
        )

    cfg = config or DEFAULT_CONFIG

    stats = TensorStats(
        name=name,
        shape=tuple(
            int(v)
            for v in t.shape
        ),
        numel=int(
            t.numel()
        ),
        dtype=str(
            t.dtype
        ),
        device=str(
            t.device
        ),
        requires_grad=bool(
            t.requires_grad
        ),
        estimated_bytes=int(
            t.numel()
            * t.element_size()
        ),
    )

    # --------------------------------------------------------
    # Empty tensor
    # --------------------------------------------------------

    if t.numel() == 0:

        stats.dead = True

        stats.error = "empty tensor"

        return stats

    try:

        x = _float_view(t)

        # ----------------------------------------------------
        # Exact finite ratio
        # ----------------------------------------------------

        finite_mask = torch.isfinite(
            x
        )

        finite_count = int(
            finite_mask.sum().item()
        )

        stats.finite_ratio = (
            finite_count
            / max(
                1,
                x.numel(),
            )
        )

        if finite_count == 0:

            stats.dead = True

            stats.error = (
                "tensor contains "
                "no finite values"
            )

            return stats

        # ----------------------------------------------------
        # Sample for expensive distribution calculations
        # ----------------------------------------------------

        sampled = _sample_flat(
            x,
            cfg.max_sample_elements,
            cfg.deterministic_sampling,
        ).float()

        stats.sampled_numel = int(
            sampled.numel()
        )

        sampled_finite = torch.isfinite(
            sampled
        )

        values = sampled[
            sampled_finite
        ]

        if values.numel() == 0:

            stats.dead = True

            stats.error = (
                "sample contains "
                "no finite values"
            )

            return stats

        # ----------------------------------------------------
        # Basic distribution
        # ----------------------------------------------------

        stats.mean = _safe_float(
            values.mean()
        )

        stats.std = _safe_float(
            values.std(
                unbiased=False
            )
        )

        stats.mean_abs = _safe_float(
            values.abs().mean()
        )

        stats.rms = _safe_float(
            torch.sqrt(
                (
                    values * values
                ).mean().clamp_min(0.0)
            )
        )

        # IMPORTANT:
        # L2 is computed from the complete tensor, not only
        # from the sample.
        #
        # This preserves the semantic meaning of "l2_norm".
        finite_x = x[
            torch.isfinite(x)
        ]

        stats.l2_norm = _safe_float(
            torch.linalg.vector_norm(
                finite_x
            )
        )

        stats.min_value = _safe_float(
            values.min()
        )

        stats.max_value = _safe_float(
            values.max()
        )

        stats.min_abs = _safe_float(
            values.abs().min()
        )

        stats.max_abs = _safe_float(
            values.abs().max()
        )

        stats.median = _safe_float(
            values.median()
        )

        # ----------------------------------------------------
        # Exact zero ratio
        # ----------------------------------------------------

        stats.zero_ratio = _safe_float(
            (x == 0).float().mean()
        )

        # ----------------------------------------------------
        # Scale-aware near-zero threshold
        # ----------------------------------------------------

        threshold = max(
            cfg.near_zero_atol,
            cfg.near_zero_rtol
            * max(
                1.0,
                stats.rms,
            ),
        )

        stats.near_zero_ratio = _safe_float(
            (
                values.abs()
                <= threshold
            ).float().mean()
        )

        # Keep original sparsity concept.
        stats.sparsity = (
            stats.near_zero_ratio
        )

        # ----------------------------------------------------
        # Constant / dead
        # ----------------------------------------------------

        stats.constant = (
            stats.std
            <= max(
                cfg.near_zero_atol,
                1e-12,
            )
        )

        stats.dead = (
            stats.l2_norm
            < 1e-12
            or stats.max_abs
            < 1e-12
            or stats.finite_ratio
            < 1.0
        )

        # ----------------------------------------------------
        # Higher moments
        # ----------------------------------------------------

        if stats.std > 1e-12:

            z = (
                values
                - stats.mean
            ) / stats.std

            stats.skewness = _safe_float(
                (z ** 3).mean()
            )

            # Excess kurtosis.
            # Normal distribution ~= 0.
            stats.kurtosis = _safe_float(
                (z ** 4).mean()
                - 3.0
            )

        # ----------------------------------------------------
        # Quantiles
        # ----------------------------------------------------

        if cfg.compute_quantiles:

            try:

                q = torch.tensor(
                    cfg.quantiles,
                    dtype=torch.float64,
                    device=values.device,
                )

                q_values = torch.quantile(
                    values.double(),
                    q,
                )

                stats.quantiles = {
                    f"q{int(round(float(p) * 100)):02d}":
                        float(v)

                    for p, v in zip(
                        q.tolist(),
                        q_values.tolist(),
                    )
                }

            except Exception as exc:

                logger.debug(
                    "Quantile calculation failed: %s",
                    exc,
                )

        # ----------------------------------------------------
        # Matrix conversion
        # ----------------------------------------------------

        matrix = _to_matrix(
            x,
            cfg.max_matrix_rows,
            cfg.max_matrix_cols,
            cfg.deterministic_sampling,
        )

        stats.matrix_shape = (
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        )

        # ----------------------------------------------------
        # Spectral analysis
        # ----------------------------------------------------

        if matrix.numel() > 0:

            (
                stats.spectral_norm,
                stats.effective_rank,
                stats.effective_rank_ratio,
                stats.singular_entropy,
                stats.condition_number,
                stats.singular_values,
            ) = _singular_analysis(
                matrix,
                cfg,
            )

            # Fallback if SVD produced nothing useful.
            if stats.spectral_norm <= 0.0:

                stats.spectral_norm = (
                    _power_spectral_norm(
                        matrix,
                        cfg.power_iterations,
                    )
                )

            # ------------------------------------------------
            # Stable rank
            #
            # stable_rank = ||W||_F^2 / ||W||_2^2
            # ------------------------------------------------

            if stats.spectral_norm > 1e-12:

                stats.stable_rank = (
                    stats.l2_norm ** 2
                ) / (
                    stats.spectral_norm ** 2
                )

                stats.spectral_ratio = (
                    stats.spectral_norm
                    / max(
                        stats.l2_norm,
                        1e-12,
                    )
                )

        # Keep original entropy name.
        stats.entropy = (
            stats.singular_entropy
        )

    except Exception as exc:

        stats.error = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.debug(
            "Tensor statistics failed for %s: %s",
            name,
            exc,
        )

    return stats


# ============================================================
# TENSOR COMPARISON
# ============================================================

@torch.inference_mode()
def compare_tensors(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    max_elements: int = 250_000,
    conflict_threshold: float = 0.05,
) -> TensorRelation:
    """
    Compare corresponding tensors.

    Especially useful for:

        model A vs model B
        CBA
        TIES-like conflict analysis
        merge diagnostics
        alignment decisions
    """

    relation = TensorRelation(
        name=name,
        shape=tuple(
            int(v)
            for v in a.shape
        ),
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not isinstance(a, torch.Tensor):

        relation.shape_compatible = False
        relation.error = (
            "a must be a torch.Tensor"
        )

        return relation

    if not isinstance(b, torch.Tensor):

        relation.shape_compatible = False
        relation.error = (
            "b must be a torch.Tensor"
        )

        return relation

    if tuple(a.shape) != tuple(b.shape):

        relation.shape_compatible = False

        relation.error = (
            "shape mismatch: "
            f"{tuple(a.shape)} "
            "vs "
            f"{tuple(b.shape)}"
        )

        return relation

    try:

        # Deterministic sampling on both tensors.
        x = _sample_flat(
            _float_view(a),
            max_elements,
            True,
        )

        y = _sample_flat(
            _float_view(b),
            max_elements,
            True,
        )

        # Only compare finite pairs.
        finite = (
            torch.isfinite(x)
            & torch.isfinite(y)
        )

        x = x[finite]
        y = y[finite]

        if x.numel() == 0:

            relation.error = (
                "no finite overlapping values"
            )

            return relation

        # ----------------------------------------------------
        # Norms
        # ----------------------------------------------------

        relation.norm_a = _safe_float(
            torch.linalg.vector_norm(x)
        )

        relation.norm_b = _safe_float(
            torch.linalg.vector_norm(y)
        )

        relation.norm_ratio_b_over_a = (
            relation.norm_b
            / max(
                relation.norm_a,
                1e-12,
            )
        )

        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        denominator = (
            relation.norm_a
            * relation.norm_b
        )

        if denominator > 1e-12:

            relation.cosine_similarity = (
                _safe_float(
                    (x * y).sum()
                    / denominator
                )
            )

        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        delta = y - x

        relation.delta_l2 = _safe_float(
            torch.linalg.vector_norm(delta)
        )

        relation.relative_delta_l2 = (
            relation.delta_l2
            / max(
                relation.norm_a,
                1e-12,
            )
        )

        relation.mean_abs_delta = _safe_float(
            delta.abs().mean()
        )

        relation.max_abs_delta = _safe_float(
            delta.abs().max()
        )

        # ----------------------------------------------------
        # Strong/non-zero thresholds
        # ----------------------------------------------------

        mean_x = max(
            _safe_float(
                x.abs().mean()
            ),
            1e-12,
        )

        mean_y = max(
            _safe_float(
                y.abs().mean()
            ),
            1e-12,
        )

        eps_x = max(
            conflict_threshold * mean_x,
            1e-12,
        )

        eps_y = max(
            conflict_threshold * mean_y,
            1e-12,
        )

        active_x = (
            x.abs() > eps_x
        )

        active_y = (
            y.abs() > eps_y
        )

        both_active = (
            active_x
            & active_y
        )

        # ----------------------------------------------------
        # Sign agreement
        # ----------------------------------------------------

        if both_active.any().item():

            same_sign = (
                torch.signbit(
                    x[both_active]
                )
                ==
                torch.signbit(
                    y[both_active]
                )
            )

            relation.sign_agreement = (
                _safe_float(
                    same_sign.float().mean()
                )
            )

            relation.sign_conflict_ratio = (
                1.0
                - relation.sign_agreement
            )

        else:

            relation.sign_agreement = 1.0
            relation.sign_conflict_ratio = 0.0

        # ----------------------------------------------------
        # Strong conflict
        #
        # Both values matter and their signs disagree.
        # ----------------------------------------------------

        relation.strong_conflict_ratio = (
            _safe_float(
                (
                    active_x
                    & active_y
                    & ((x * y) < 0)
                ).float().mean()
            )
        )

        # ----------------------------------------------------
        # Structural overlap
        # ----------------------------------------------------

        relation.activation_overlap = (
            _safe_float(
                (
                    active_x
                    & active_y
                ).float().mean()
            )
        )

        # ----------------------------------------------------
        # Zero behavior
        # ----------------------------------------------------

        relation.exact_zero_a = _safe_float(
            (x == 0).float().mean()
        )

        relation.exact_zero_b = _safe_float(
            (y == 0).float().mean()
        )

    except Exception as exc:

        relation.error = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.debug(
            "Tensor comparison failed for %s: %s",
            name,
            exc,
        )

    return relation


# ============================================================
# STATE-DICT SCANNER
# ============================================================

@torch.inference_mode()
def scan_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: Optional[TensorStatsConfig] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, TensorStats]:
    """
    Analyze a complete or selected state_dict.
    """

    wanted = (
        set(names)
        if names is not None
        else None
    )

    result: Dict[
        str,
        TensorStats,
    ] = {}

    for name, tensor in state_dict.items():

        if wanted is not None:
            if name not in wanted:
                continue

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            continue

        result[name] = compute_tensor_stats(
            name,
            tensor,
            config=config,
        )

    return result


# ============================================================
# STATE-DICT COMPARISON
# ============================================================

@torch.inference_mode()
def compare_state_dicts(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    max_elements: int = 250_000,
) -> Dict[str, TensorRelation]:
    """
    Compare all matching tensors in two models.
    """

    result: Dict[
        str,
        TensorRelation,
    ] = {}

    common_names = (
        state_a.keys()
        & state_b.keys()
    )

    for name in common_names:

        a = state_a[name]
        b = state_b[name]

        if not isinstance(
            a,
            torch.Tensor,
        ):
            continue

        if not isinstance(
            b,
            torch.Tensor,
        ):
            continue

        result[name] = compare_tensors(
            name=name,
            a=a,
            b=b,
            max_elements=max_elements,
        )

    return result


# ============================================================
# MODEL SUMMARY
# ============================================================

def summarize_state_dict(
    stats: Mapping[str, TensorStats],
) -> StateDictSummary:
    """
    Aggregate per-tensor statistics.
    """

    summary = StateDictSummary()

    if not stats:
        return summary

    l2_values = []
    spectral_values = []
    sparsity_values = []
    rank_values = []

    for stat in stats.values():

        summary.tensors += 1

        summary.total_numel += (
            stat.numel
        )

        summary.total_bytes += (
            stat.estimated_bytes
        )

        summary.by_dtype[
            stat.dtype
        ] = (
            summary.by_dtype.get(
                stat.dtype,
                0,
            )
            + 1
        )

        if stat.requires_grad:
            summary.trainable_tensors += 1

        if stat.dead:
            summary.dead_tensors += 1

        if stat.finite_ratio < 1.0:
            summary.nonfinite_tensors += 1

        if stat.sparsity >= 0.95:
            summary.high_sparsity_tensors += 1

        if math.isfinite(
            stat.l2_norm
        ):
            l2_values.append(
                stat.l2_norm
            )

        if math.isfinite(
            stat.spectral_norm
        ):
            spectral_values.append(
                stat.spectral_norm
            )

        if math.isfinite(
            stat.sparsity
        ):
            sparsity_values.append(
                stat.sparsity
            )

        if math.isfinite(
            stat.effective_rank_ratio
        ):
            rank_values.append(
                stat.effective_rank_ratio
            )

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    if l2_values:

        summary.mean_l2_norm = (
            sum(l2_values)
            / len(l2_values)
        )

    if spectral_values:

        summary.mean_spectral_norm = (
            sum(spectral_values)
            / len(spectral_values)
        )

    if sparsity_values:

        summary.mean_sparsity = (
            sum(sparsity_values)
            / len(sparsity_values)
        )

    if rank_values:

        summary.mean_effective_rank_ratio = (
            sum(rank_values)
            / len(rank_values)
        )

    return summary


# ============================================================
# JSON-FRIENDLY SERIALIZATION
# ============================================================

def stats_to_dict(
    stats: TensorStats,
) -> Dict[str, Any]:

    return {
        "name": stats.name,
        "shape": list(stats.shape),
        "numel": stats.numel,
        "dtype": stats.dtype,
        "device": stats.device,

        "mean": stats.mean,
        "std": stats.std,
        "l2_norm": stats.l2_norm,

        "spectral_norm": stats.spectral_norm,

        "effective_rank": stats.effective_rank,
        "effective_rank_ratio": (
            stats.effective_rank_ratio
        ),

        "stable_rank": (
            stats.stable_rank
        ),

        "entropy": stats.entropy,
        "singular_entropy": (
            stats.singular_entropy
        ),

        "spectral_ratio": (
            stats.spectral_ratio
        ),

        "condition_number": (
            stats.condition_number
        ),

        "sparsity": stats.sparsity,
        "zero_ratio": stats.zero_ratio,
        "near_zero_ratio": (
            stats.near_zero_ratio
        ),

        "dead": stats.dead,
        "constant": stats.constant,

        "finite_ratio": (
            stats.finite_ratio
        ),

        "min_value": stats.min_value,
        "max_value": stats.max_value,
        "min_abs": stats.min_abs,
        "max_abs": stats.max_abs,

        "mean_abs": stats.mean_abs,
        "rms": stats.rms,
        "median": stats.median,

        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis,

        "singular_values": list(
            stats.singular_values
        ),

        "quantiles": dict(
            stats.quantiles
        ),

        "matrix_shape": list(
            stats.matrix_shape
        ),

        "sampled_numel": (
            stats.sampled_numel
        ),

        "estimated_bytes": (
            stats.estimated_bytes
        ),

        "requires_grad": (
            stats.requires_grad
        ),

        "error": stats.error,
    }


def relation_to_dict(
    relation: TensorRelation,
) -> Dict[str, Any]:

    return {
        "name": relation.name,
        "shape": list(
            relation.shape
        ),

        "cosine_similarity": (
            relation.cosine_similarity
        ),

        "norm_a": relation.norm_a,
        "norm_b": relation.norm_b,

        "norm_ratio_b_over_a": (
            relation.norm_ratio_b_over_a
        ),

        "delta_l2": (
            relation.delta_l2
        ),

        "relative_delta_l2": (
            relation.relative_delta_l2
        ),

        "mean_abs_delta": (
            relation.mean_abs_delta
        ),

        "max_abs_delta": (
            relation.max_abs_delta
        ),

        "sign_agreement": (
            relation.sign_agreement
        ),

        "sign_conflict_ratio": (
            relation.sign_conflict_ratio
        ),

        "strong_conflict_ratio": (
            relation.strong_conflict_ratio
        ),

        "activation_overlap": (
            relation.activation_overlap
        ),

        "exact_zero_a": (
            relation.exact_zero_a
        ),

        "exact_zero_b": (
            relation.exact_zero_b
        ),

        "shape_compatible": (
            relation.shape_compatible
        ),

        "error": relation.error,
    }


# ============================================================
# MODEL HEALTH
# ============================================================

@torch.inference_mode()
def model_tensor_health(
    model: torch.nn.Module,
    config: Optional[TensorStatsConfig] = None,
) -> Dict[str, Any]:
    """
    High-level model diagnostic.

    Useful before:
        training
        merging
        CBA
        checkpoint export
    """

    cfg = config or DEFAULT_CONFIG

    total_params = 0
    trainable_params = 0

    dead_tensors = []
    nonfinite_tensors = []
    high_sparsity_tensors = []

    stats: Dict[
        str,
        TensorStats,
    ] = {}

    for name, param in model.named_parameters():

        total_params += param.numel()

        if param.requires_grad:

            trainable_params += (
                param.numel()
            )

        stat = compute_tensor_stats(
            name,
            param,
            cfg,
        )

        stats[name] = stat

        if stat.dead:
            dead_tensors.append(
                name
            )

        if stat.finite_ratio < 1.0:

            nonfinite_tensors.append(
                name
            )

        if stat.sparsity >= 0.95:

            high_sparsity_tensors.append(
                name
            )

    return {
        "total_params": total_params,

        "trainable_params": (
            trainable_params
        ),

        "trainable_ratio": (
            trainable_params
            / max(
                1,
                total_params,
            )
        ),

        "dead_tensors": (
            dead_tensors
        ),

        "nonfinite_tensors": (
            nonfinite_tensors
        ),

        "high_sparsity_tensors": (
            high_sparsity_tensors
        ),

        "tensor_count": len(stats),

        "stats": stats,
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TensorStatsConfig",
    "TensorStats",
    "TensorRelation",
    "StateDictSummary",

    "compute_tensor_stats",

    "compare_tensors",
    "compare_state_dicts",

    "scan_state_dict",
    "summarize_state_dict",

    "stats_to_dict",
    "relation_to_dict",

    "model_tensor_health",
        ]
