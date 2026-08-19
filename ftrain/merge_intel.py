"""
FTRAIN Intelligent Merge Planner
=================================

Architecture-aware tensor analysis and merge planning for FTRAIN.

This module is responsible for answering:

    "Given tensor A and tensor B, what is the safest and most useful
     way to combine them?"

It does NOT perform the actual tensor merge.

The planner produces a TensorMergePlan that can later be consumed by
the merge engine / alignment / projection layer.

Major improvements
------------------
- Robust architecture-aware tensor classification.
- Better layer-depth extraction across common HF architectures.
- Explicit shape/dtype/device compatibility information.
- Safer similarity handling.
- Stable tensor importance estimation.
- Category-specific merge behavior.
- Conservative handling of routers, norms and embeddings.
- Better handling of MoE tensors.
- Cross-architecture planning support.
- Confidence score for every merge decision.
- Explicit projection/alignment requirements.
- Deterministic planning.
- Defensive handling of malformed statistics.
- No silent assumptions that differently shaped tensors are mergeable.
- Richer reasons/debug information.
- Backward-compatible core public classes/functions.

Important
---------
This planner does NOT magically make arbitrary architectures compatible.

For example:

    Llama hidden_size=4096
    Qwen hidden_size=3584

cannot safely be merged by simply doing:

    alpha * A + (1-alpha) * B

when their tensor shapes differ.

In such cases this module marks the tensor as requiring alignment/
projection. The actual projection/alignment implementation must be
performed by the merger.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .tensor_stats import compute_tensor_stats
from .similarity import similarity_bundle, aggregate_similarity


# =============================================================================
# Constants
# =============================================================================

EPS = 1e-8

MIN_ALPHA = 0.02
MAX_ALPHA = 0.98

# Category-specific baseline preference for model A.
#
# This is NOT the final alpha.
# It is a prior used by MergePlanner together with similarity,
# importance, entropy, depth and compatibility.
CAT_ALPHA: Dict[str, float] = {
    "embedding": 0.50,
    "lm_head": 0.50,
    "norm": 0.50,
    "router": 0.35,
    "shared_expert": 0.45,
    "moe_expert": 0.45,
    "attention": 0.50,
    "ffn": 0.50,
    "bias": 0.50,
    "other": 0.50,
}


# Categories that should normally NOT be interpolated blindly.
CRITICAL_CATEGORIES = {
    "norm",
    "router",
}


# Categories where alignment can be particularly dangerous.
SENSITIVE_CATEGORIES = {
    "embedding",
    "lm_head",
    "router",
    "norm",
}


# =============================================================================
# Name patterns
# =============================================================================

# Common embedding names.
_EMBEDDING_PATTERNS = (
    "embed_tokens",
    "tok_embeddings",
    "token_embedding",
    "token_embeddings",
    "word_embeddings",
    "word_embedding",
    "wte",
    "input_embeddings",
    "embedding",
)

# Common output-head names.
_LM_HEAD_PATTERNS = (
    "lm_head",
    "output",
    "output_head",
    "language_model_head",
    "final_logits",
)

# Normalization modules.
_NORM_PATTERNS = (
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "rms_norm",
    "ln_f",
    "ln_",
    "norm",
)

# Router / MoE routing.
_ROUTER_PATTERNS = (
    "router",
    "router_gate",
    "routing",
    "gate_logits",
    "gate_proj",
    "switch_gate",
    "moe_gate",
    "topk_gate",
)

# Shared expert structures.
_SHARED_EXPERT_PATTERNS = (
    "shared_expert",
    "shared_experts",
    "shared_mlp",
    "shared_ffn",
    "shared_ffn",
)

# MoE expert structures.
_EXPERT_PATTERNS = (
    ".experts.",
    ".expert.",
    "experts.",
    "expert.",
    "moe_experts",
    "moe_expert",
    "expert_weight",
)

# Attention projections.
_ATTENTION_PATTERNS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj",
    "kv_b_proj",
    "qkv",
    "qkv_proj",
    "wq",
    "wk",
    "wv",
    "wo",
    "query",
    "key",
    "value",
    "attention",
    "self_attn",
    "self_attention",
    "attn",
)

# Feed-forward projections.
_FFN_PATTERNS = (
    "gate_proj",
    "up_proj",
    "down_proj",
    "w1",
    "w2",
    "w3",
    "fc1",
    "fc2",
    "ffn",
    "mlp",
    "feed_forward",
    "feedforward",
)

# Bias tensors.
_BIAS_PATTERNS = (
    ".bias",
    "bias",
)


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert an arbitrary value to a finite float.

    Tensor statistics are produced by another FTRAIN module and may evolve
    over time, so this planner deliberately avoids assuming every field is
    always present or perfectly formatted.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    return value


def _clamp(
    value: float,
    low: float = MIN_ALPHA,
    high: float = MAX_ALPHA,
) -> float:
    return max(low, min(high, _safe_float(value, 0.5)))


def _normalize_similarity(value: Any) -> float:
    """
    Normalize similarity into [-1, 1].

    Most FTRAIN similarity implementations should already return something
    in this range, but defensive normalization prevents malformed metrics from
    corrupting the planner.
    """
    value = _safe_float(value, 0.0)

    if value > 1.0:
        return 1.0

    if value < -1.0:
        return -1.0

    return value


def _positive(value: Any, minimum: float = EPS) -> float:
    return max(_safe_float(value, 0.0), minimum)


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in tensor.shape)
    except Exception:
        return ()


def _dtype_name(tensor: torch.Tensor) -> str:
    try:
        return str(tensor.dtype)
    except Exception:
        return "unknown"


def _numel(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.numel())
    except Exception:
        return 0


def _is_matrix_like(tensor: torch.Tensor) -> bool:
    return tensor.ndim >= 2


def _shape_compatible(
    a: torch.Tensor,
    b: torch.Tensor,
) -> bool:
    return _tensor_shape(a) == _tensor_shape(b)


def _dtype_compatible(
    a: torch.Tensor,
    b: torch.Tensor,
) -> bool:
    return a.dtype == b.dtype


# =============================================================================
# Architecture / tensor classification
# =============================================================================


def _contains_any(
    name: str,
    patterns: Sequence[str],
) -> bool:
    return any(pattern in name for pattern in patterns)


def classify_tensor(name: str) -> str:
    """
    Classify a tensor into a merge category.

    Classification order is intentional.

    For example, a tensor named something like:

        model.layers.4.mlp.router.gate_proj.weight

    should be treated as a router tensor rather than an ordinary FFN gate.
    """
    n = str(name).lower().strip()

    # -------------------------------------------------------------------------
    # Bias
    # -------------------------------------------------------------------------

    if n.endswith(".bias") or n == "bias":
        return "bias"

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    if _contains_any(n, _EMBEDDING_PATTERNS):
        return "embedding"

    # -------------------------------------------------------------------------
    # LM head
    # -------------------------------------------------------------------------

    if _contains_any(n, _LM_HEAD_PATTERNS):
        # Avoid classifying generic "output" substrings inside unrelated
        # modules whenever possible.
        if (
            "lm_head" in n
            or n.endswith("output.weight")
            or ".output." in n
            or n.endswith("output")
        ):
            return "lm_head"

    # -------------------------------------------------------------------------
    # Router MUST be checked before generic gate/FFN classification.
    # -------------------------------------------------------------------------

    if _contains_any(n, _ROUTER_PATTERNS):
        return "router"

    # -------------------------------------------------------------------------
    # Shared expert
    # -------------------------------------------------------------------------

    if _contains_any(n, _SHARED_EXPERT_PATTERNS):
        return "shared_expert"

    # -------------------------------------------------------------------------
    # Individual MoE experts
    # -------------------------------------------------------------------------

    if _contains_any(n, _EXPERT_PATTERNS):
        return "moe_expert"

    # -------------------------------------------------------------------------
    # Norm
    # -------------------------------------------------------------------------

    if _contains_any(n, _NORM_PATTERNS):
        return "norm"

    # -------------------------------------------------------------------------
    # Attention
    # -------------------------------------------------------------------------

    if _contains_any(n, _ATTENTION_PATTERNS):
        return "attention"

    # -------------------------------------------------------------------------
    # FFN
    # -------------------------------------------------------------------------

    if _contains_any(n, _FFN_PATTERNS):
        return "ffn"

    return "other"


# =============================================================================
# Layer extraction
# =============================================================================


_LAYER_PATTERNS = (
    # HuggingFace / Llama / Qwen style:
    re.compile(r"(?:layers|h|blocks)\.(\d+)(?:\.|$)", re.IGNORECASE),

    # GPT-NeoX:
    re.compile(r"(?:layers|layer)\.(\d+)(?:\.|$)", re.IGNORECASE),

    # Some architectures use explicit block names.
    re.compile(r"(?:block|block_layer)\.(\d+)(?:\.|$)", re.IGNORECASE),

    # Numeric transformer block fallback.
    re.compile(r"\.(\d+)\.", re.IGNORECASE),
)


def extract_layer_index(
    name: str,
) -> Optional[int]:
    """
    Extract a layer index from a parameter name.

    Supports several common HuggingFace naming conventions.
    """
    n = str(name)

    for pattern in _LAYER_PATTERNS:
        match = pattern.search(n)

        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue

    return None


def extract_layer_depth(
    name: str,
    total_layers: int = 32,
) -> float:
    """
    Return normalized layer depth in [0, 1].

    0.0 = first layer
    1.0 = final layer

    Unknown layer locations are assigned the neutral midpoint 0.5.
    """
    total_layers = max(1, int(total_layers))

    idx = extract_layer_index(name)

    if idx is None:
        return 0.5

    denominator = max(1, total_layers - 1)

    return max(
        0.0,
        min(
            1.0,
            float(idx) / float(denominator),
        ),
    )


# =============================================================================
# Tensor statistics helpers
# =============================================================================


def _get_stat(
    stats: Any,
    field_name: str,
    default: float = 0.0,
) -> float:
    """
    Safely read a tensor-stat field.

    Supports dataclasses, objects and dictionaries.
    """
    if stats is None:
        return default

    if isinstance(stats, dict):
        return _safe_float(
            stats.get(field_name, default),
            default,
        )

    return _safe_float(
        getattr(stats, field_name, default),
        default,
    )


def _get_dead(
    stats: Any,
) -> bool:
    if stats is None:
        return False

    if isinstance(stats, dict):
        return bool(stats.get("dead", False))

    return bool(
        getattr(stats, "dead", False)
    )


def _get_numel(
    stats: Any,
    fallback: int,
) -> int:
    value = _get_stat(
        stats,
        "numel",
        float(fallback),
    )

    return max(0, int(value))


# =============================================================================
# Importance estimation
# =============================================================================


def _tensor_importance(
    stats: Any,
) -> float:
    """
    Estimate tensor importance using multiple statistics.

    The original planner used:

        l2_norm * (0.5 + effective_rank) * (1 + entropy)

    which can become excessively dominated by raw tensor magnitude.

    This implementation separates magnitude from structural information and
    compresses extreme values using log1p.

    The exact value is used comparatively between A and B, not as an absolute
    scientific importance score.
    """
    l2 = _positive(
        _get_stat(stats, "l2_norm", 0.0)
    )

    rank = max(
        0.0,
        _get_stat(stats, "effective_rank", 0.0),
    )

    entropy = max(
        0.0,
        _get_stat(stats, "entropy", 0.0),
    )

    numel = max(
        1,
        _get_numel(stats, 1),
    )

    # Normalize magnitude by tensor size.
    rms = l2 / math.sqrt(float(numel))

    magnitude_component = math.log1p(
        max(0.0, rms)
    )

    rank_component = 1.0 + min(
        rank,
        10.0,
    )

    entropy_component = 1.0 + min(
        entropy,
        10.0,
    )

    score = (
        magnitude_component
        * rank_component
        * entropy_component
    )

    return max(
        score,
        EPS,
    )


def _relative_importance(
    a_stats: Any,
    b_stats: Any,
) -> Tuple[float, float]:
    """
    Return normalized importance for A and B.
    """
    ia = _tensor_importance(a_stats)
    ib = _tensor_importance(b_stats)

    total = ia + ib

    if total <= EPS:
        return 0.5, 0.5

    return (
        ia / total,
        ib / total,
    )


# =============================================================================
# Merge plan
# =============================================================================


@dataclass
class TensorMergePlan:
    """
    Decision produced by MergePlanner.

    alpha
        Weight assigned to model A.

        final ≈ alpha * A + (1-alpha) * B

    strategy
        Actual merge strategy requested from the merger.

    projection
        Alignment mechanism required before merging.

    compatible
        Whether A and B can be directly operated on together.

    confidence
        Planner confidence in this decision.

    requires_alignment
        True when an architecture/name/shape alignment stage is required.

    metadata
        Additional machine-readable information for future FTRAIN modules.
    """

    name: str
    category: str
    alpha: float
    strategy: str

    projection: str = "identity"

    similarity: float = 0.0
    layer_depth: float = 0.5

    importance_a: float = 0.5
    importance_b: float = 0.5

    reason: str = ""

    compatible: bool = True
    requires_alignment: bool = False

    confidence: float = 0.0

    shape_a: Tuple[int, ...] = field(
        default_factory=tuple
    )

    shape_b: Tuple[int, ...] = field(
        default_factory=tuple
    )

    dtype_a: str = "unknown"
    dtype_b: str = "unknown"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# =============================================================================
# Analyzer
# =============================================================================


class MergeAnalyzer:
    """
    Analyze two tensors before planning their merge.

    The analyzer is intentionally side-effect free.
    """

    def analyze_pair(
        self,
        name: str,
        a: torch.Tensor,
        b: torch.Tensor,
        total_layers: int = 32,
    ) -> Dict[str, Any]:
        """
        Analyze one tensor pair.

        If shapes differ, similarity is NOT calculated directly.
        The merger must first perform an architecture alignment/projection.
        """
        if not isinstance(a, torch.Tensor):
            raise TypeError(
                f"Tensor A for '{name}' must be torch.Tensor, "
                f"got {type(a).__name__}."
            )

        if not isinstance(b, torch.Tensor):
            raise TypeError(
                f"Tensor B for '{name}' must be torch.Tensor, "
                f"got {type(b).__name__}."
            )

        sa = compute_tensor_stats(
            name,
            a,
        )

        sb = compute_tensor_stats(
            name,
            b,
        )

        compatible = _shape_compatible(
            a,
            b
        )

        sim = 0.0

        if compatible and a.numel() > 0:
            try:
                sim = aggregate_similarity(
                    similarity_bundle(
                        a,
                        b,
                    )
                )
            except Exception as exc:
                logger.debug(
                    "Similarity calculation failed for '%s': %s",
                    name,
                    exc,
                )
                sim = 0.0

        sim = _normalize_similarity(
            sim
        )

        category = classify_tensor(
            name
        )

        layer_depth = extract_layer_depth(
            name,
            total_layers,
        )

        importance_a, importance_b = (
            _relative_importance(
                sa,
                sb,
            )
        )

        return {
            "a": sa,
            "b": sb,

            "similarity": sim,

            "category": category,

            "layer_depth": layer_depth,

            "compatible": compatible,

            "shape_a": _tensor_shape(a),
            "shape_b": _tensor_shape(b),

            "dtype_a": _dtype_name(a),
            "dtype_b": _dtype_name(b),

            "numel_a": _numel(a),
            "numel_b": _numel(b),

            "importance_a": importance_a,
            "importance_b": importance_b,
        }


# =============================================================================
# Planner
# =============================================================================


class MergePlanner:
    """
    Architecture-aware merge planner.

    This class decides how each tensor should be merged, but does not execute
    the merge.
    """

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------

    @staticmethod
    def _confidence(
        similarity: float,
        compatible: bool,
        category: str,
        strategy: str,
    ) -> float:
        """
        Estimate confidence in a plan.

        This is a planning confidence metric, NOT a model-quality metric.
        """
        sim = abs(
            _normalize_similarity(similarity)
        )

        confidence = 0.45

        if compatible:
            confidence += 0.25

        if sim >= 0.85:
            confidence += 0.20
        elif sim >= 0.60:
            confidence += 0.10
        elif sim < 0.20:
            confidence -= 0.10

        if category in CRITICAL_CATEGORIES:
            confidence -= 0.05

        if strategy == "projection":
            confidence -= 0.10

        return _clamp(
            confidence,
            0.05,
            0.99,
        )

    # -------------------------------------------------------------------------
    # Depth adjustment
    # -------------------------------------------------------------------------

    @staticmethod
    def _depth_modifier(
        depth: float,
        category: str,
    ) -> float:
        """
        Compute a small positional prior.

        Earlier layers are generally more structural.
        Later layers tend to contain more task-specific behavior.

        The effect is intentionally small so depth never dominates actual
        tensor evidence.
        """
        depth = max(
            0.0,
            min(1.0, depth),
        )

        if category in (
            "embedding",
            "norm",
            "router",
        ):
            return 0.0

        # Slightly favor model A in early layers and B in later layers.
        # This is only a weak prior.
        return 0.04 * (
            0.5 - depth
        )

    # -------------------------------------------------------------------------
    # Similarity weighting
    # -------------------------------------------------------------------------

    @staticmethod
    def _similarity_weight(
        similarity: float,
        category: str,
    ) -> float:
        """
        Convert similarity into a stable blending prior.
        """
        sim = _normalize_similarity(
            similarity
        )

        # Negative cosine-like similarity indicates conflicting directions.
        if sim < 0.0:
            return 0.15

        if sim >= 0.95:
            weight = 1.00
        elif sim >= 0.85:
            weight = 0.90
        elif sim >= 0.70:
            weight = 0.75
        elif sim >= 0.50:
            weight = 0.60
        elif sim >= 0.30:
            weight = 0.40
        else:
            weight = 0.20

        # Critical tensors should receive less aggressive interpolation.
        if category in CRITICAL_CATEGORIES:
            weight *= 0.60

        return weight

    # -------------------------------------------------------------------------
    # Main planner
    # -------------------------------------------------------------------------

    def plan_for_pair(
        self,
        name: str,
        an: Dict[str, Any],
    ) -> TensorMergePlan:
        """
        Produce a merge plan for a single tensor pair.

        Parameters
        ----------
        name:
            Parameter/tensor name.

        an:
            Output from MergeAnalyzer.analyze_pair().
        """
        category = str(
            an.get(
                "category",
                classify_tensor(name),
            )
        )

        similarity = _normalize_similarity(
            an.get("similarity", 0.0)
        )

        depth = _safe_float(
            an.get("layer_depth", 0.5),
            0.5,
        )

        depth = max(
            0.0,
            min(1.0, depth),
        )

        sa = an.get("a")
        sb = an.get("b")

        compatible = bool(
            an.get("compatible", True)
        )

        shape_a = tuple(
            an.get("shape_a", ())
        )

        shape_b = tuple(
            an.get("shape_b", ())
        )

        dtype_a = str(
            an.get("dtype_a", "unknown")
        )

        dtype_b = str(
            an.get("dtype_b", "unknown")
        )

        # ---------------------------------------------------------------------
        # Dead tensor detection
        # ---------------------------------------------------------------------

        dead_a = _get_dead(sa)
        dead_b = _get_dead(sb)

        imp_a, imp_b = _relative_importance(
            sa,
            sb,
        )

        # ---------------------------------------------------------------------
        # A completely dead
        # ---------------------------------------------------------------------

        if dead_a and not dead_b:
            return TensorMergePlan(
                name=name,
                category=category,
                alpha=0.0,
                strategy="keep_b",
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=0.0,
                importance_b=1.0,
                reason="Model A tensor is structurally dead; preserving model B.",
                compatible=compatible,
                requires_alignment=not compatible,
                confidence=self._confidence(
                    similarity,
                    compatible,
                    category,
                    "keep_b",
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "dead_a": True,
                    "dead_b": False,
                },
            )

        # ---------------------------------------------------------------------
        # B completely dead
        # ---------------------------------------------------------------------

        if dead_b and not dead_a:
            return TensorMergePlan(
                name=name,
                category=category,
                alpha=1.0,
                strategy="keep_a",
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=1.0,
                importance_b=0.0,
                reason="Model B tensor is structurally dead; preserving model A.",
                compatible=compatible,
                requires_alignment=not compatible,
                confidence=self._confidence(
                    similarity,
                    compatible,
                    category,
                    "keep_a",
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "dead_a": False,
                    "dead_b": True,
                },
            )

        # ---------------------------------------------------------------------
        # Both dead
        # ---------------------------------------------------------------------

        if dead_a and dead_b:
            return TensorMergePlan(
                name=name,
                category=category,
                alpha=0.5,
                strategy="keep_a",
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=0.5,
                importance_b=0.5,
                reason="Both tensors are marked dead; preserving model A as deterministic fallback.",
                compatible=compatible,
                requires_alignment=not compatible,
                confidence=0.70,
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "dead_a": True,
                    "dead_b": True,
                },
            )

        # ---------------------------------------------------------------------
        # DIFFERENT SHAPES
        #
        # This is the most important improvement for cross-architecture
        # merging.
        # ---------------------------------------------------------------------

        if not compatible:
            return self._plan_incompatible(
                name=name,
                category=category,
                similarity=similarity,
                depth=depth,
                imp_a=imp_a,
                imp_b=imp_b,
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
            )

        # ---------------------------------------------------------------------
        # Critical architecture tensors
        #
        # Blind interpolation of LayerNorm/RMSNorm/router tensors can destroy
        # architecture behavior.
        # ---------------------------------------------------------------------

        if category == "norm":
            # If both are compatible, use a balanced but conservative blend.
            alpha = 0.50

            return TensorMergePlan(
                name=name,
                category=category,
                alpha=alpha,
                strategy="weighted",
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=imp_a,
                importance_b=imp_b,
                reason=(
                    "Normalization tensor detected; using conservative "
                    "shape-compatible interpolation."
                ),
                compatible=True,
                requires_alignment=False,
                confidence=self._confidence(
                    similarity,
                    True,
                    category,
                    "weighted",
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "critical_tensor": True,
                },
            )

        if category == "router":
            # Router weights are extremely sensitive.
            #
            # If alignment is strong, a mild blend is acceptable.
            # Otherwise preserve A instead of creating a potentially broken
            # routing function.
            if similarity >= 0.90:
                alpha = 0.65
                strategy = "weighted"
                reason = (
                    "Highly aligned MoE router; conservative weighted merge."
                )
            else:
                alpha = 1.0
                strategy = "keep_a"
                reason = (
                    "MoE router is architecture-critical and similarity is "
                    "insufficient for safe interpolation; preserving model A."
                )

            return TensorMergePlan(
                name=name,
                category=category,
                alpha=alpha,
                strategy=strategy,
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=imp_a,
                importance_b=imp_b,
                reason=reason,
                compatible=True,
                requires_alignment=False,
                confidence=self._confidence(
                    similarity,
                    True,
                    category,
                    strategy,
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "critical_tensor": True,
                    "router_safe_threshold": 0.90,
                },
            )

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------

        if category == "embedding":
            if similarity >= 0.90:
                alpha = 0.50
                strategy = "weighted"
                reason = (
                    "Embedding tensors are strongly aligned; balanced blend."
                )
            elif similarity >= 0.70:
                alpha = 0.65
                strategy = "weighted"
                reason = (
                    "Embedding alignment is moderate; conservative A-biased blend."
                )
            else:
                alpha = 1.0
                strategy = "keep_a"
                reason = (
                    "Embedding similarity is too low for safe direct blending."
                )

            return TensorMergePlan(
                name=name,
                category=category,
                alpha=alpha,
                strategy=strategy,
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=imp_a,
                importance_b=imp_b,
                reason=reason,
                compatible=True,
                requires_alignment=False,
                confidence=self._confidence(
                    similarity,
                    True,
                    category,
                    strategy,
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "sensitive_tensor": True,
                },
            )

        # ---------------------------------------------------------------------
        # LM head
        # ---------------------------------------------------------------------

        if category == "lm_head":
            if similarity >= 0.90:
                alpha = 0.50
                strategy = "weighted"
            elif similarity >= 0.70:
                alpha = 0.60
                strategy = "weighted"
            else:
                alpha = 1.0
                strategy = "keep_a"

            return TensorMergePlan(
                name=name,
                category=category,
                alpha=alpha,
                strategy=strategy,
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=imp_a,
                importance_b=imp_b,
                reason=(
                    "Language-model output head requires conservative "
                    "handling because logits are highly sensitive to weight changes."
                ),
                compatible=True,
                requires_alignment=False,
                confidence=self._confidence(
                    similarity,
                    True,
                    category,
                    strategy,
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "sensitive_tensor": True,
                },
            )

        # ---------------------------------------------------------------------
        # Very low similarity
        # ---------------------------------------------------------------------

        if similarity < 0.20:
            return TensorMergePlan(
                name=name,
                category=category,
                alpha=1.0,
                strategy="keep_a",
                projection="identity",
                similarity=similarity,
                layer_depth=depth,
                importance_a=imp_a,
                importance_b=imp_b,
                reason=(
                    "Tensor directions are weakly aligned; direct interpolation "
                    "could destroy useful features. Preserving model A."
                ),
                compatible=True,
                requires_alignment=False,
                confidence=self._confidence(
                    similarity,
                    True,
                    category,
                    "keep_a",
                ),
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                metadata={
                    "low_similarity": True,
                },
            )

        # ---------------------------------------------------------------------
        # Calculate adaptive alpha
        # ---------------------------------------------------------------------

        base_alpha = CAT_ALPHA.get(
            category,
            CAT_ALPHA["other"],
        )

        sim_weight = self._similarity_weight(
            similarity,
            category,
        )

        # Importance evidence.
        importance_component = imp_a

        # Entropy evidence.
        entropy_a = max(
            0.0,
            _get_stat(sa, "entropy", 0.0),
        )

        entropy_b = max(
            0.0,
            _get_stat(sb, "entropy", 0.0),
        )

        entropy_total = (
            entropy_a
            + entropy_b
            + EPS
        )

        entropy_component = (
            entropy_a
            / entropy_total
        )

        # Small layer prior.
        depth_component = self._depth_modifier(
            depth,
            category,
        )

        # ---------------------------------------------------------------------
        # Adaptive blend
        # ---------------------------------------------------------------------

        alpha = (
            0.25 * base_alpha
            + 0.25 * importance_component
            + 0.25 * sim_weight
            + 0.15 * entropy_component
            + 0.10 * 0.5
            + depth_component
        )

        # Strongly aligned tensors deserve more blending.
        if similarity >= 0.90:
            alpha = (
                0.80 * alpha
                + 0.20 * 0.5
            )

        # Moderate alignment remains conservative.
        elif similarity < 0.55:
            alpha = (
                0.70 * alpha
                + 0.30 * 0.65
            )

        # ---------------------------------------------------------------------
        # Variance protection
        # ---------------------------------------------------------------------

        std_a = _positive(
            _get_stat(sa, "std", 0.0)
        )

        std_b = _positive(
            _get_stat(sb, "std", 0.0)
        )

        if std_a > EPS and std_b > EPS:
            std_ratio = max(
                std_a / std_b,
                std_b / std_a,
            )

            # Large distribution mismatch means aggressive interpolation is
            # risky.
            if std_ratio >= 3.0:
                alpha = (
                    0.75 * alpha
                    + 0.25 * imp_a
                )

        # ---------------------------------------------------------------------
        # Clamp
        # ---------------------------------------------------------------------

        alpha = _clamp(
            alpha
        )

        # ---------------------------------------------------------------------
        # Select strategy
        # ---------------------------------------------------------------------

        if similarity >= 0.88:
            strategy = "weighted"
            projection = "identity"
            reason = (
                "High tensor alignment; adaptive weighted interpolation."
            )

        elif similarity >= 0.65:
            strategy = "slerp"
            projection = "identity"
            reason = (
                "Moderate-to-high alignment; spherical interpolation "
                "reduces destructive magnitude effects."
            )

        elif similarity >= 0.35:
            if _numel(sa) >= 16 and _numel(sb) >= 16:
                strategy = "ties"
                projection = "identity"
                reason = (
                    "Moderate alignment; TIES sign conflict resolution "
                    "is safer than naive interpolation."
                )
            else:
                strategy = "weighted"
                projection = "identity"
                reason = (
                    "Moderate alignment on a small tensor; weighted merge."
                )

        else:
            strategy = "keep_a"
            projection = "identity"
            alpha = 1.0
            reason = (
                "Weak alignment; preserving model A instead of combining "
                "conflicting feature directions."
            )

        confidence = self._confidence(
            similarity,
            True,
            category,
            strategy,
        )

        return TensorMergePlan(
            name=name,
            category=category,
            alpha=alpha,
            strategy=strategy,
            projection=projection,
            similarity=similarity,
            layer_depth=depth,
            importance_a=imp_a,
            importance_b=imp_b,
            reason=reason,
            compatible=True,
            requires_alignment=False,
            confidence=confidence,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            metadata={
                "base_alpha": base_alpha,
                "similarity_weight": sim_weight,
                "entropy_a": entropy_a,
                "entropy_b": entropy_b,
                "std_a": std_a,
                "std_b": std_b,
            },
        )

    # =========================================================================
    # Incompatible tensor planner
    # =========================================================================

    def _plan_incompatible(
        self,
        name: str,
        category: str,
        similarity: float,
        depth: float,
        imp_a: float,
        imp_b: float,
        shape_a: Tuple[int, ...],
        shape_b: Tuple[int, ...],
        dtype_a: str,
        dtype_b: str,
    ) -> TensorMergePlan:
        """
        Plan tensors whose shapes differ.

        This is critical for cross-architecture FTRAIN merging.

        We never pretend that:

            [4096, 4096]
        and
            [3584, 3584]

        are directly mergeable.

        Instead we request a projection/alignment stage.
        """
        if category == "router":
            # Router dimensionality mismatch is especially dangerous.
            strategy = "keep_a"
            projection = "architecture_router_alignment"
            alpha = 1.0

            reason = (
                "Router shapes differ across architectures. Direct merging "
                "is unsafe; preserving model A until a dedicated router "
                "alignment strategy is available."
            )

            confidence = 0.90

        elif category in (
            "embedding",
            "lm_head",
        ):
            strategy = "projection"
            projection = "vocab_or_hidden_projection"
            alpha = 0.50

            reason = (
                "Embedding/output dimensions differ. A vocabulary/hidden "
                "alignment projection is required before merging."
            )

            confidence = 0.60

        elif category == "norm":
            strategy = "projection"
            projection = "norm_dimension_alignment"
            alpha = 0.50

            reason = (
                "Normalization dimensions differ. Direct merge is forbidden; "
                "a dimension-aware alignment is required."
            )

            confidence = 0.70

        elif category == "attention":
            strategy = "projection"
            projection = "attention_head_alignment"
            alpha = 0.50

            reason = (
                "Attention tensor shapes differ. Head-count/hidden-dimension "
                "alignment is required before interpolation."
            )

            confidence = 0.50

        elif category == "moe_expert":
            strategy = "projection"
            projection = "expert_alignment"
            alpha = 0.50

            reason = (
                "MoE expert dimensions differ. Expert mapping/alignment is "
                "required before merging."
            )

            confidence = 0.45

        elif category == "ffn":
            strategy = "projection"
            projection = "ffn_dimension_alignment"
            alpha = 0.50

            reason = (
                "FFN tensor dimensions differ. Hidden/intermediate dimensions "
                "must be aligned before merging."
            )

            confidence = 0.50

        else:
            strategy = "projection"
            projection = "generic_shape_alignment"
            alpha = 0.50

            reason = (
                "Tensor shapes differ; generic learned/statistical projection "
                "is required before merging."
            )

            confidence = 0.35

        return TensorMergePlan(
            name=name,
            category=category,
            alpha=alpha,
            strategy=strategy,
            projection=projection,
            similarity=similarity,
            layer_depth=depth,
            importance_a=imp_a,
            importance_b=imp_b,
            reason=reason,
            compatible=False,
            requires_alignment=True,
            confidence=confidence,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            metadata={
                "cross_architecture": True,
                "direct_merge_forbidden": True,
                "shape_mismatch": True,
            },
        )


# =============================================================================
# Convenience API
# =============================================================================


def analyze_and_plan(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    total_layers: int = 32,
    planner: Optional[MergePlanner] = None,
) -> TensorMergePlan:
    """
    Convenience function for callers that want analysis + planning in one call.
    """
    analyzer = MergeAnalyzer()

    analysis = analyzer.analyze_pair(
        name=name,
        a=a,
        b=b,
        total_layers=total_layers,
    )

    if planner is None:
        planner = MergePlanner()

    return planner.plan_for_pair(
        name=name,
        an=analysis,
    )


__all__ = [
    "CAT_ALPHA",
    "TensorMergePlan",
    "MergeAnalyzer",
    "MergePlanner",
    "classify_tensor",
    "extract_layer_index",
    "extract_layer_depth",
    "analyze_and_plan",
]
