"""
FTRAIN Tensor Similarity & Representation Analysis v1.1
========================================================

Similarity metrics used by FTRAIN's intelligent merger.

Public API
----------
cosine_similarity_weights
cka
neuron_overlap
sv_overlap
similarity_bundle
aggregate_similarity

Design goals
------------
• Stable similarity measurements across different tensor ranks.
• Bounded memory usage for large model tensors.
• Deterministic subsampling.
• Correct handling of shape/row mismatches.
• Numerical safety for fp16/bf16 inputs.
• Avoid O(N^2) memory explosions where practical.
• Graceful fallbacks instead of merger-breaking exceptions.
• Preserve the original public API.

Important
---------
These metrics measure numerical/representational alignment. They do not prove
that two tensors are semantically interchangeable. The merger should combine
similarity with tensor category, layer position, Fisher/task importance and
calibration evaluation.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "cosine_similarity_weights",
    "cka",
    "neuron_overlap",
    "sv_overlap",
    "similarity_bundle",
    "aggregate_similarity",
]


# =============================================================================
# Constants
# =============================================================================

DEFAULT_COSINE_ROWS = 4096
DEFAULT_CKA_ROWS = 512
DEFAULT_SV_ROWS = 2048

DEFAULT_CKA_EPS = 1e-8
DEFAULT_KERNEL_CHUNK = 128

_METRIC_KEYS = (
    "cosine",
    "cka_linear",
    "cka_rbf",
    "neuron_overlap",
    "sv_overlap",
)


# =============================================================================
# General helpers
# =============================================================================

def _flatten_2d(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Represent a tensor as [rows, features].

    For vectors:
        [D] -> [1, D]

    For matrices:
        [M, N] -> [M, N]

    For higher-rank tensors:
        [A, B, C, ...] -> [A, B*C*...]
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(tensor).__name__}."
        )

    tensor = tensor.detach()

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


def _stable_seed(
    a: torch.Tensor,
    b: torch.Tensor,
    seed: int = 42,
) -> int:
    """
    Build a deterministic seed from tensor metadata plus a user seed.

    This avoids hidden global RNG consumption while making subsampling stable
    across repeated similarity calls.
    """
    payload = (
        str(tuple(a.shape))
        + "|"
        + str(tuple(b.shape))
        + "|"
        + str(a.dtype)
        + "|"
        + str(b.dtype)
        + "|"
        + str(seed)
    ).encode("utf-8")

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        "little",
    ) % (2**31 - 1)


def _feature_align(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match feature dimensions without creating an oversized padded tensor.

    When feature dimensions differ, use the shared prefix. This is appropriate
    for a similarity diagnostic because the tensors are not necessarily directly
    mergeable anyway; the actual merger's projection/alignment layer handles
    shape adaptation separately.
    """
    columns = min(
        a.shape[1],
        b.shape[1],
    )

    if columns <= 0:
        return (
            a[:, :0],
            b[:, :0],
        )

    if (
        a.shape[1] != columns
        or b.shape[1] != columns
    ):
        a = a[:, :columns]
        b = b[:, :columns]

    return a, b


def _safe_subsample(
    a: torch.Tensor,
    b: torch.Tensor,
    max_rows: int = DEFAULT_CKA_ROWS,
    *,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert tensors to fp32, align feature dimensions and deterministically
    subsample rows.

    The previous implementation always took the first rows, which can bias
    similarity if the most informative structure is concentrated later in a
    tensor.
    """
    a_f = _flatten_2d(
        a.to(dtype=torch.float32)
    )

    b_f = _flatten_2d(
        b.to(dtype=torch.float32)
    )

    a_f, b_f = _feature_align(
        a_f,
        b_f,
    )

    if (
        a_f.shape[0] == 0
        or b_f.shape[0] == 0
        or a_f.shape[1] == 0
    ):
        return a_f[:0], b_f[:0]

    rows = min(
        int(max_rows),
        a_f.shape[0],
        b_f.shape[0],
    )

    if rows <= 0:
        return (
            a_f[:0],
            b_f[:0],
        )

    if (
        a_f.shape[0] == rows
        and b_f.shape[0] == rows
    ):
        return a_f, b_f

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        _stable_seed(
            a,
            b,
            seed,
        )
    )

    # Sampling indices on CPU avoids device-specific generator differences.
    a_indices = torch.randperm(
        a_f.shape[0],
        generator=generator,
        device="cpu",
    )[:rows]

    # Reset using a deterministic offset so A/B are not forced to select the
    # same row indices when their row counts differ.
    generator.manual_seed(
        _stable_seed(
            b,
            a,
            seed + 1,
        )
    )

    b_indices = torch.randperm(
        b_f.shape[0],
        generator=generator,
        device="cpu",
    )[:rows]

    a_indices = a_indices.to(
        device=a_f.device
    )
    b_indices = b_indices.to(
        device=b_f.device
    )

    return (
        a_f.index_select(
            0,
            a_indices,
        ),
        b_f.index_select(
            0,
            b_indices,
        ),
    )


def _sanitize_pair(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace non-finite inputs for metric computation.

    Similarity should not crash a merge because a single intermediate tensor
    contains NaN/Inf. The merger's safety module remains responsible for
    rejecting the underlying corruption.
    """
    a = torch.nan_to_num(
        a,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    b = torch.nan_to_num(
        b,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return a, b


def _clamp_score(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    if not math.isfinite(value):
        return 0.0

    return max(
        low,
        min(
            high,
            value,
        ),
    )


# =============================================================================
# Cosine similarity
# =============================================================================

def cosine_similarity_weights(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_rows: int = DEFAULT_COSINE_ROWS,
    seed: int = 42,
) -> float:
    """
    Compute cosine similarity between flattened/sampled tensor values.

    Returns a signed score in [-1, 1].
    """
    try:
        a_f, b_f = _safe_subsample(
            a,
            b,
            max_rows=max_rows,
            seed=seed,
        )

        if (
            a_f.numel() == 0
            or b_f.numel() == 0
        ):
            return 0.0

        a_f, b_f = _sanitize_pair(
            a_f,
            b_f,
        )

        a_flat = a_f.reshape(-1)
        b_flat = b_f.reshape(-1)

        # Match total flattened size after row/feature alignment.
        count = min(
            a_flat.numel(),
            b_flat.numel(),
        )

        if count <= 0:
            return 0.0

        a_flat = a_flat[:count]
        b_flat = b_flat[:count]

        norm_a = torch.linalg.vector_norm(
            a_flat
        )
        norm_b = torch.linalg.vector_norm(
            b_flat
        )

        if (
            norm_a <= DEFAULT_CKA_EPS
            or norm_b <= DEFAULT_CKA_EPS
        ):
            return 0.0

        similarity = (
            torch.dot(
                a_flat,
                b_flat,
            )
            / (
                norm_a
                * norm_b
                + DEFAULT_CKA_EPS
            )
        )

        if not torch.isfinite(
            similarity
        ):
            return 0.0

        return float(
            torch.clamp(
                similarity,
                -1.0,
                1.0,
            ).item()
        )

    except Exception:
        logger.debug(
            "Cosine similarity failed.",
            exc_info=True,
        )
        return 0.0


# =============================================================================
# Linear CKA
# =============================================================================

def _center_kernel(
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Center a Gram matrix without explicitly constructing a full centering
    matrix H, saving O(N^2) temporary memory.
    """
    row_mean = kernel.mean(
        dim=1,
        keepdim=True,
    )

    col_mean = kernel.mean(
        dim=0,
        keepdim=True,
    )

    grand_mean = kernel.mean()

    return (
        kernel
        - row_mean
        - col_mean
        + grand_mean
    )


def _linear_gram(
    x: torch.Tensor,
) -> torch.Tensor:
    return x @ x.transpose(
        0,
        1,
    )


def _rbf_gram_chunked(
    x: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = DEFAULT_KERNEL_CHUNK,
) -> torch.Tensor:
    """
    Build an RBF Gram matrix in chunks.

    Unlike torch.cdist over the whole matrix, this avoids a potentially huge
    intermediate [N, N, D] allocation.
    """
    n = x.shape[0]

    if n == 0:
        return x.new_empty(
            0,
            0,
        )

    # Squared norms.
    squared_norms = (
        x.pow(2)
        .sum(dim=1)
        .clamp_min(0.0)
    )

    if sigma is None:
        # Estimate median pairwise squared distance from a small deterministic
        # sample. Exact median over all pairs is unnecessarily expensive.
        sample_n = min(
            n,
            128,
        )

        sample = x[:sample_n]

        distances = (
            sample.pow(2).sum(dim=1, keepdim=True)
            + sample.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0
            * (sample @ sample.transpose(0, 1))
        ).clamp_min(0.0)

        valid = distances[
            torch.triu(
                torch.ones_like(
                    distances,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        ]

        if valid.numel() == 0:
            sigma_value = torch.tensor(
                1.0,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            sigma_value = torch.median(
                valid
            ).clamp_min(
                DEFAULT_CKA_EPS
            )

        sigma = sigma_value

    kernel = torch.empty(
        n,
        n,
        device=x.device,
        dtype=torch.float32,
    )

    sigma_value = float(
        sigma.item()
        if isinstance(
            sigma,
            torch.Tensor,
        )
        else sigma
    )

    sigma_value = max(
        sigma_value,
        DEFAULT_CKA_EPS,
    )

    denominator = (
        2.0
        * sigma_value
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):
        end = min(
            n,
            start + chunk_size,
        )

        block = x[
            start:end
        ]

        distances = (
            squared_norms[
                start:end
            ].unsqueeze(1)
            + squared_norms.unsqueeze(0)
            - 2.0
            * (
                block
                @ x.transpose(0, 1)
            )
        )

        distances = distances.clamp_min(
            0.0
        )

        kernel[
            start:end
        ] = torch.exp(
            -distances
            / denominator
        )

    return kernel


def cka(
    a: torch.Tensor,
    b: torch.Tensor,
    kernel: str = "linear",
    *,
    max_rows: int = DEFAULT_CKA_ROWS,
    seed: int = 42,
    chunk_size: int = DEFAULT_KERNEL_CHUNK,
) -> float:
    """
    Compute centered kernel alignment (CKA).

    Supported kernels:
        linear
        rbf

    Returns a value in [0, 1].
    """
    try:
        a_f, b_f = _safe_subsample(
            a,
            b,
            max_rows=max_rows,
            seed=seed,
        )

        if (
            a_f.shape[0] < 2
            or b_f.shape[0] < 2
        ):
            return 0.0

        # CKA needs corresponding examples/rows. Subsampling already picks
        # matched counts, but make this explicit for robustness.
        n = min(
            a_f.shape[0],
            b_f.shape[0],
        )

        a_f = a_f[:n]
        b_f = b_f[:n]

        a_f, b_f = _sanitize_pair(
            a_f,
            b_f,
        )

        normalized_kernel = str(
            kernel
        ).strip().lower()

        if normalized_kernel == "linear":
            gram_a = _linear_gram(
                a_f
            )
            gram_b = _linear_gram(
                b_f
            )

        elif normalized_kernel in {
            "rbf",
            "gaussian",
        }:
            gram_a = _rbf_gram_chunked(
                a_f,
                chunk_size=chunk_size,
            )
            gram_b = _rbf_gram_chunked(
                b_f,
                chunk_size=chunk_size,
            )

        else:
            raise ValueError(
                f"Unknown CKA kernel: {kernel!r}"
            )

        centered_a = _center_kernel(
            gram_a
        )
        centered_b = _center_kernel(
            gram_b
        )

        numerator = torch.sum(
            centered_a
            * centered_b
        )

        denominator = torch.sqrt(
            torch.sum(
                centered_a
                * centered_a
            )
            * torch.sum(
                centered_b
                * centered_b
            )
            + DEFAULT_CKA_EPS
        )

        if (
            not torch.isfinite(
                numerator
            )
            or not torch.isfinite(
                denominator
            )
            or denominator <= DEFAULT_CKA_EPS
        ):
            return 0.0

        score = (
            numerator
            / denominator
        )

        if not torch.isfinite(
            score
        ):
            return 0.0

        return _clamp_score(
            float(
                score.item()
            )
        )

    except Exception:
        logger.debug(
            "CKA computation failed.",
            exc_info=True,
        )
        return 0.0


# =============================================================================
# Neuron overlap
# =============================================================================

def neuron_overlap(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_rows: int = DEFAULT_CKA_ROWS,
    seed: int = 42,
) -> float:
    """
    Compare corresponding row/neuron directions.

    For incompatible feature widths, the shared feature prefix is used.
    """
    try:
        a_f, b_f = _safe_subsample(
            a,
            b,
            max_rows=max_rows,
            seed=seed,
        )

        if (
            a_f.shape[0] == 0
            or b_f.shape[0] == 0
            or a_f.shape[1] == 0
        ):
            return 0.0

        rows = min(
            a_f.shape[0],
            b_f.shape[0],
        )

        a_f = a_f[:rows]
        b_f = b_f[:rows]

        a_f, b_f = _sanitize_pair(
            a_f,
            b_f,
        )

        a_norm = F.normalize(
            a_f,
            p=2,
            dim=1,
            eps=DEFAULT_CKA_EPS,
        )

        b_norm = F.normalize(
            b_f,
            p=2,
            dim=1,
            eps=DEFAULT_CKA_EPS,
        )

        row_scores = (
            a_norm
            * b_norm
        ).sum(
            dim=1
        ).abs()

        score = row_scores.mean()

        if not torch.isfinite(
            score
        ):
            return 0.0

        return _clamp_score(
            float(
                score.item()
            )
        )

    except Exception:
        logger.debug(
            "Neuron overlap computation failed.",
            exc_info=True,
        )
        return 0.0


# =============================================================================
# Singular-value overlap
# =============================================================================

def sv_overlap(
    a: torch.Tensor,
    b: torch.Tensor,
    k: int = 64,
    *,
    max_rows: int = DEFAULT_SV_ROWS,
    seed: int = 42,
) -> float:
    """
    Compare the normalized singular-value spectra of two tensors.
    """
    try:
        a_f, b_f = _safe_subsample(
            a,
            b,
            max_rows=max_rows,
            seed=seed,
        )

        if (
            a_f.numel() == 0
            or b_f.numel() == 0
        ):
            return 0.0

        min_dim_a = min(
            a_f.shape
        )

        min_dim_b = min(
            b_f.shape
        )

        minimum = min(
            min_dim_a,
            min_dim_b,
        )

        if minimum <= 0:
            return 0.0

        q = min(
            max(
                1,
                int(k),
            ),
            minimum,
        )

        if minimum > 128:
            _, singular_a, _ = (
                torch.svd_lowrank(
                    a_f,
                    q=q,
                )
            )

            _, singular_b, _ = (
                torch.svd_lowrank(
                    b_f,
                    q=q,
                )
            )

        else:
            singular_a = torch.linalg.svdvals(
                a_f
            )
            singular_b = torch.linalg.svdvals(
                b_f
            )

        count = min(
            singular_a.numel(),
            singular_b.numel(),
        )

        if count <= 0:
            return 0.0

        singular_a = singular_a[
            :count
        ]
        singular_b = singular_b[
            :count
        ]

        # Normalize spectra so overall parameter scale does not dominate this
        # structural similarity metric.
        singular_a = singular_a / (
            torch.linalg.vector_norm(
                singular_a
            )
            + DEFAULT_CKA_EPS
        )

        singular_b = singular_b / (
            torch.linalg.vector_norm(
                singular_b
            )
            + DEFAULT_CKA_EPS
        )

        score = torch.dot(
            singular_a,
            singular_b,
        )

        if not torch.isfinite(
            score
        ):
            return 0.0

        return _clamp_score(
            float(
                score.item()
            )
        )

    except Exception:
        logger.debug(
            "Singular-value overlap computation failed.",
            exc_info=True,
        )
        return 0.0


# =============================================================================
# Similarity bundle
# =============================================================================

def similarity_bundle(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cosine_rows: int = DEFAULT_COSINE_ROWS,
    cka_rows: int = DEFAULT_CKA_ROWS,
    sv_rows: int = DEFAULT_SV_ROWS,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute the complete similarity feature bundle.

    Returns the same metric names used by FTRAIN's merger planner.
    """
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return {
            key: 0.0
            for key in _METRIC_KEYS
        }

    return {
        "cosine": cosine_similarity_weights(
            a,
            b,
            max_rows=cosine_rows,
            seed=seed,
        ),
        "cka_linear": cka(
            a,
            b,
            kernel="linear",
            max_rows=cka_rows,
            seed=seed,
        ),
        "cka_rbf": cka(
            a,
            b,
            kernel="rbf",
            max_rows=cka_rows,
            seed=seed,
        ),
        "neuron_overlap": neuron_overlap(
            a,
            b,
            max_rows=cka_rows,
            seed=seed,
        ),
        "sv_overlap": sv_overlap(
            a,
            b,
            max_rows=sv_rows,
            seed=seed,
        ),
    }


# =============================================================================
# Aggregate similarity
# =============================================================================

def aggregate_similarity(
    bundle: Dict[str, float],
    *,
    weights: Optional[Dict[str, float]] = None,
    signed_cosine: bool = False,
) -> float:
    """
    Aggregate a similarity bundle into a single [0, 1] score.

    Default weighting:
        cosine          0.30
        linear CKA      0.25
        RBF CKA         0.15
        neuron overlap 0.15
        singular overlap 0.15

    ``signed_cosine=False`` preserves the previous FTRAIN behavior of treating
    cosine agreement as magnitude of alignment. Set it to True when opposite
    directions should be strongly penalized.
    """
    if not isinstance(
        bundle,
        dict,
    ):
        return 0.0

    if weights is None:
        weights = {
            "cosine": 0.30,
            "cka_linear": 0.25,
            "cka_rbf": 0.15,
            "neuron_overlap": 0.15,
            "sv_overlap": 0.15,
        }

    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        try:
            weight = float(weight)
            value = float(
                bundle.get(
                    key,
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            not math.isfinite(weight)
            or weight <= 0
            or not math.isfinite(value)
        ):
            continue

        if key == "cosine" and not signed_cosine:
            value = abs(value)

        value = _clamp_score(
            value,
            0.0,
            1.0,
        )

        weighted_sum += (
            weight
            * value
        )

        total_weight += weight

    if total_weight <= 0:
        return 0.0

    return _clamp_score(
        weighted_sum
        / total_weight,
        0.0,
        1.0,
    )


# =============================================================================
# Developer smoke test
# =============================================================================

def _self_test() -> Dict[str, float]:
    """
    Lightweight test for development/CI.
    """
    generator = torch.Generator()
    generator.manual_seed(1234)

    a = torch.randn(
        256,
        128,
        generator=generator,
    )

    b = a * 0.9 + torch.randn(
        256,
        128,
        generator=generator,
    ) * 0.1

    bundle = similarity_bundle(
        a,
        b,
    )

    aggregate = aggregate_similarity(
        bundle
    )

    assert set(bundle) == set(
        _METRIC_KEYS
    )

    for key, value in bundle.items():
        assert math.isfinite(value), (
            f"{key} returned non-finite value."
        )
        assert 0.0 <= value <= 1.0 or key == "cosine"

    assert 0.0 <= aggregate <= 1.0

    # Shape mismatch should remain safe.
    c = torch.randn(
        128,
        64,
        generator=generator,
    )

    mismatch_bundle = similarity_bundle(
        a,
        c,
    )

    assert set(mismatch_bundle) == set(
        _METRIC_KEYS
    )

    return {
        "cosine": bundle["cosine"],
        "cka_linear": bundle["cka_linear"],
        "cka_rbf": bundle["cka_rbf"],
        "neuron_overlap": bundle["neuron_overlap"],
        "sv_overlap": bundle["sv_overlap"],
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    print(_self_test())
