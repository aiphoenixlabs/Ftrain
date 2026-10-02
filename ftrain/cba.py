"""FTRAIN CBA — Captain Brain Alignment.

Deterministic, architecture-aware, conflict-aware routing for model merges.
Backward-compatible with the public CBA API used by FTRAIN's merger layer.

CBA analyzes weight-space evidence. It does not claim behavioral superiority
without activation/task/calibration evidence; optional Fisher and activation
statistics can be supplied to strengthen routing decisions.

Changelog (v4 -> v5)
---------------------
Every public name, dataclass field and function signature from v4 still
works; new fields default so old callers and old saved JSON reports are
unaffected. ``CBAReport.to_dict()["cba_version"]`` is now 5.

Bug fixes
  * ``TensorRecord.l2_norm`` (and everything derived from it: ``rel_norm``,
    ``importance``) over-estimated tensors that were both large enough to be
    sub-sampled (> ``_FLATTEN_CAP`` elements) and contained non-finite values:
    the non-finite entries were dropped from the sample *after* it was drawn,
    but the extrapolation still scaled up to the tensor's full element count
    instead of its estimated finite element count. Large corrupted tensors
    (embeddings, MLP projections) could look artificially "important".
  * ``TensorConflict.finite_ratio`` was hard-coded to ``1.0`` regardless of
    the actual data, so a pair with heavy NaN/Inf contamination reported the
    same confidence as a perfectly clean pair. It is now measured from the
    sampled pair and folded into ``confidence``.
  * ``plan_routing``'s per-*layer* protection for norm/embedding/head tensors
    keyed off a majority-vote category per layer. Embeddings and heads have
    no layer index at all (``_layer_of`` never matches their names), so that
    branch could never fire for them; norm tensors are always a numeric
    minority within a layer (2 of ~9 tensors in a standard block), so it
    essentially never fired for them either. In effect, layer-level
    protection was dead code, and ``CBARouting.layer_alphas`` /
    ``protected_layers`` under-reported how conservatively sensitive tensors
    were actually routed (the *per-tensor* directives were already correct).
    Layer-level protection is now a numel-weighted blend over each layer's
    real tensor composition, using the same protection constants as the
    per-tensor pass, so the two levels agree.
  * The category routing pass (embedding/attention/mlp/.../norm) called
    ``_evidence_score`` without ``activation_layer_a/b``, silently ignoring
    supplied activation evidence for every category-level decision, and had
    no category-specific conflict signal (conflicts are bucketed by
    ``layer_N`` for any tensor with a detected layer, so a bare category key
    such as ``"attention"`` almost never exists in ``per_layer`` and quietly
    fell back to the *global* conflict score). Category routing now
    aggregates real per-category conflict from ``conflicts.per_tensor`` and
    receives the same activation evidence as layer routing. ``CBARouting``
    gained ``category_actions``, ``category_conflicts`` and
    ``category_confidence`` to match the layer-level fields.
  * ``build_correspondence`` matched targets in alphabetical-name order and
    let the first target to reach a shared source claim it, so a later,
    possibly better-evidenced target could be starved of its correct match
    purely by name ordering. Matching is now two-pass: candidate scores are
    computed for every target first (without consuming sources), then
    targets are assigned in order of their score margin (least ambiguous
    first), so contested sources go to whichever target has the strongest,
    least-ambiguous case for them. The role-based fallback candidate pool
    (used when a name carries no reliable layer/canonical signal) was also
    capped at 256 alphabetically-sorted names; for architectures with more
    than ~256 tensors sharing one role this could silently drop the correct
    match. It is now capped by proximity to the expected translated layer
    instead of by name.

New
  * ``CompatibilityReport.unmatched_targets``: the A-side tensors that found
    no correspondence at all, for direct inspection instead of only the
    aggregate ratio.
  * ``CBAReport.summary()``: a short human-readable digest (decision,
    evidence, most decisive layers, critique) for logs/notebooks.
  * ``CBAReport.critique()``: convenience wrapper for
    ``critique_decision_record(self.decision.to_dict())``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "TensorRecord", "LayerRecord", "BrainInspection", "ParameterMatch",
    "CompatibilityReport", "TensorConflict", "LayerConflict", "ConflictReport",
    "CBADirective", "CBARouting", "CBADecision", "CBAReport",
    "inspect_state_dict", "build_correspondence", "check_compatibility",
    "analyze_conflicts", "plan_routing", "run_cba",
    "answer_three_questions", "critique_decision_record",
]

_EPS = 1e-12
_FLATTEN_CAP = 262_144
_PAIR_SAMPLE_CAP = 131_072
_ROLE_FALLBACK_CAP = 256
_CONFLICT_MEDIUM_CUTOFF = 0.35
_CONFLICT_HIGH_CUTOFF_DEFAULT = 0.65
_CONFLICT_CRITICAL_CUTOFF = 0.85
CONFLICT_LOW = "low"
CONFLICT_MEDIUM = "medium"
CONFLICT_HIGH = "high"
CONFLICT_CRITICAL = "critical"
_ROUTING_GAIN = 0.60
_ALPHA_MIN = 0.05
_ALPHA_MAX = 0.95
_SIMILAR_DIRECTION = 0.995
_MIN_CORRESPONDENCE_CONFIDENCE = 0.20
_MIN_CANONICAL_CONFIDENCE = 0.35
_EMBEDDING_PROTECTION = 0.85
_HEAD_PROTECTION = 0.80
_NORM_PROTECTION = 0.95
_BIAS_PROTECTION = 0.60
_LAYER_PROTECTION_FLAG_THRESHOLD = 0.20  # numel-weighted layer protection above which a layer is listed as "protected"

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:decoder\.)?layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:transformer\.)h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:transformer\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:model\.)?h\.(\d+)(?:\.|$)"),
)


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _clamp01(x: float) -> float:
    return _clamp(x, 0.0, 1.0)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def _safe_mean(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    try:
        return _safe_float(t.float().mean())
    except Exception:
        return 0.0


def _safe_std(t: torch.Tensor) -> float:
    if t.numel() <= 1:
        return 0.0
    try:
        return _safe_float(t.float().std(unbiased=False))
    except Exception:
        return 0.0


def _safe_norm(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    try:
        return _safe_float(torch.linalg.vector_norm(t.float()))
    except Exception:
        return 0.0


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else default


def _weighted_mean(values: Iterable[Tuple[float, float]], default: float = 0.0) -> float:
    pairs = [(float(v), max(0.0, float(w))) for v, w in values]
    total = sum(w for _, w in pairs)
    if total <= _EPS:
        return default
    return sum(v * w for v, w in pairs) / total


def _safe_log_ratio(a: float, b: float) -> float:
    return math.log(max(abs(a), _EPS) / max(abs(b), _EPS))


def _deterministic_indices(size: int, limit: int, device: torch.device) -> Optional[torch.Tensor]:
    if size <= 0 or size <= limit:
        return None
    return torch.linspace(0, size - 1, steps=limit, device=device, dtype=torch.float64).round().long()


def _deterministic_sample(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    flat = tensor.detach().float().reshape(-1)
    idx = _deterministic_indices(flat.numel(), limit, flat.device)
    return flat if idx is None else flat.index_select(0, idx)


def _estimated_l2(sample: torch.Tensor, total_numel: int) -> float:
    if sample.numel() == 0 or total_numel <= 0:
        return 0.0
    energy = _safe_float((sample * sample).sum())
    return math.sqrt(max(0.0, energy * (float(total_numel) / float(sample.numel()))))


def _finite_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Deterministically-sampled, finite-masked pair, plus the pre-mask sample
    size so callers can measure how much of the pair was actually usable."""
    x = _deterministic_sample(a, _PAIR_SAMPLE_CAP)
    y = _deterministic_sample(b, _PAIR_SAMPLE_CAP)
    if x.numel() != y.numel():
        return torch.empty(0), torch.empty(0), 0
    sampled = int(x.numel())
    mask = torch.isfinite(x) & torch.isfinite(y)
    return x[mask], y[mask], sampled


def _layer_of(name: str) -> Optional[int]:
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None


def _classify(name: str) -> str:
    s = name.lower()
    if any(x in s for x in ("embed_tokens", "tok_embeddings", "word_embeddings", ".embedding")):
        return "embedding"
    if any(x in s for x in ("lm_head", "output_projection", "language_model_head", "output.weight")):
        return "head"
    if any(x in s for x in ("self_attn", ".attention.", ".attn.", "attention.wq", "attention.wk", "attention.wv", "attention.wo")):
        return "attention"
    if any(x in s for x in (".mlp.", ".feed_forward.", ".ffn.", ".experts.", "gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if any(x in s for x in ("layernorm", "layer_norm", ".norm.", "ln_", ".ln")):
        return "norm"
    if s.endswith(".bias"):
        return "bias"
    return "other"


def _parameter_role(name: str) -> str:
    s = name.lower()
    if "q_proj" in s or s.endswith(".wq"):
        return "q_proj"
    if "k_proj" in s or s.endswith(".wk"):
        return "k_proj"
    if "v_proj" in s or s.endswith(".wv"):
        return "v_proj"
    if "o_proj" in s or s.endswith(".wo"):
        return "o_proj"
    if "gate_proj" in s or ".w1" in s:
        return "gate_proj"
    if "up_proj" in s or ".w3" in s:
        return "up_proj"
    if "down_proj" in s or ".w2" in s:
        return "down_proj"
    if any(x in s for x in ("embed_tokens", "tok_embeddings", "word_embeddings")):
        return "embedding"
    if "lm_head" in s or "output_projection" in s:
        return "lm_head"
    if "norm" in s:
        return "norm"
    if s.endswith(".bias"):
        return "bias"
    return "weight"


def _canonical_key(name: str) -> str:
    v = str(name)
    reps = (
        ("base_model.model.", ""), ("base_model.", ""), ("module.", ""),
        ("decoder.layers.", "layers."), ("transformer.layers.", "layers."),
        ("transformer.h.", "layers."), ("model.layers.", "layers."),
        ("transformer.wte", "embed_tokens"), ("model.embed_tokens", "embed_tokens"),
        ("tok_embeddings", "embed_tokens"), ("word_embeddings", "embed_tokens"),
        ("attention.wq", "self_attn.q_proj"), ("attention.wk", "self_attn.k_proj"),
        ("attention.wv", "self_attn.v_proj"), ("attention.wo", "self_attn.o_proj"),
        ("attn.wq", "self_attn.q_proj"), ("attn.wk", "self_attn.k_proj"),
        ("attn.wv", "self_attn.v_proj"), ("attn.wo", "self_attn.o_proj"),
        ("feed_forward.w1", "mlp.gate_proj"), ("feed_forward.w2", "mlp.down_proj"),
        ("feed_forward.w3", "mlp.up_proj"), ("ffn.w1", "mlp.gate_proj"),
        ("ffn.w2", "mlp.down_proj"), ("ffn.w3", "mlp.up_proj"),
        ("attention_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm"),
    )
    for a, b in reps:
        v = v.replace(a, b)
    v = re.sub(r"(?:layers|h)\.(\d+)\.", "layers.{layer}.", v)
    v = v.replace("self_attn.self_attn.", "self_attn.")
    return re.sub(r"\.{2,}", ".", v).strip(".")


def _sensitive(cat: str) -> bool:
    return cat in {"embedding", "head", "norm"}


def _protection_for_category(category: str) -> float:
    """Shared protection strength for a parameter category.

    Used both for per-tensor directives and for the numel-weighted per-layer
    blend, so the two levels can never silently disagree about how strongly a
    category should be protected.
    """
    if category == "norm":
        return _NORM_PROTECTION
    if category == "embedding":
        return _EMBEDDING_PROTECTION
    if category == "head":
        return _HEAD_PROTECTION
    if category == "bias":
        return _BIAS_PROTECTION
    return 0.0


@dataclass
class TensorRecord:
    name: str
    category: str
    layer: Optional[int]
    numel: int
    mean: float
    std: float
    l2_norm: float
    rel_norm: float
    mean_abs: float = 0.0
    rms: float = 0.0
    min_abs: float = 0.0
    max_abs: float = 0.0
    zero_ratio: float = 0.0
    near_zero_ratio: float = 0.0
    finite_ratio: float = 1.0
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: str = ""
    role: str = "weight"
    importance: float = 0.0
    energy_estimate: float = 0.0


@dataclass
class LayerRecord:
    key: str
    kind: str
    index: Optional[int]
    category: str
    tensors: int
    numel: int
    l2_norm: float
    rel_norm: float
    mean_std: float
    mean_abs: float = 0.0
    sparsity: float = 0.0
    importance: float = 0.0


@dataclass
class BrainInspection:
    total_tensors: int
    total_numel: int
    total_norm: float
    layer_count: int
    hidden_size: Optional[int]
    vocab_size: Optional[int]
    dtypes: Dict[str, int]
    tensors: Dict[str, TensorRecord]
    layers: Dict[str, LayerRecord]
    categories: Dict[str, LayerRecord]
    floating_tensors: int = 0
    nonfinite_tensors: int = 0
    zero_tensors: int = 0
    sensitive_tensors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tensors": self.total_tensors,
            "total_numel": self.total_numel,
            "total_norm": self.total_norm,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "dtypes": dict(self.dtypes),
            "floating_tensors": self.floating_tensors,
            "nonfinite_tensors": self.nonfinite_tensors,
            "zero_tensors": self.zero_tensors,
            "sensitive_tensors": self.sensitive_tensors,
            "tensors": {k: asdict(v) for k, v in self.tensors.items()},
            "layers": {k: asdict(v) for k, v in self.layers.items()},
            "categories": {k: asdict(v) for k, v in self.categories.items()},
        }


@dataclass
class ParameterMatch:
    target_name: str
    source_name: str
    method: str
    confidence: float
    shape_exact: bool
    role_exact: bool
    layer_a: Optional[int]
    layer_b: Optional[int]
    score: float
    shape_a: Tuple[int, ...] = field(default_factory=tuple)
    shape_b: Tuple[int, ...] = field(default_factory=tuple)
    canonical_exact: bool = False
    layer_distance: Optional[int] = None
    shape_similarity: float = 0.0


@dataclass
class TensorConflict:
    name: str
    layer: Optional[int]
    category: str
    direction_similarity: float
    sign_disagreement: float
    relative_delta: float
    score: float
    band: str
    cosine_similarity: float = 0.0
    norm_ratio: float = 1.0
    magnitude_disagreement: float = 0.0
    zero_disagreement: float = 0.0
    structural_disagreement: float = 0.0
    distribution_disagreement: float = 0.0
    row_structure_similarity: float = 0.0
    column_structure_similarity: float = 0.0
    finite_ratio: float = 1.0
    confidence: float = 0.0


@dataclass
class CompatibilityReport:
    compatible: bool
    mergeability: float
    name_overlap: float
    shape_match_ratio: float
    layer_count_a: int
    layer_count_b: int
    hidden_size_a: Optional[int]
    hidden_size_b: Optional[int]
    dtype_compatible: bool
    tokenizer_compatible: Optional[bool]
    notes: List[str] = field(default_factory=list)
    correspondence_ratio: float = 0.0
    role_match_ratio: float = 0.0
    exact_shape_correspondence_ratio: float = 0.0
    architecture_penalty: float = 0.0
    structural_confidence: float = 0.0
    unmatched_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerConflict:
    key: str
    kind: str
    score: float
    band: str
    tensors: int
    mean_cosine: float = 0.0
    mean_relative_delta: float = 0.0
    mean_sign_disagreement: float = 0.0
    mean_magnitude_disagreement: float = 0.0
    mean_confidence: float = 0.0
    weighted_importance: float = 0.0


@dataclass
class ConflictReport:
    global_score: float
    global_band: str
    high_conflict_layers: List[str]
    per_tensor: Dict[str, TensorConflict]
    per_layer: Dict[str, LayerConflict]
    critical_conflict_layers: List[str] = field(default_factory=list)
    conflict_tensor_ratio: float = 0.0
    mean_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_score": self.global_score,
            "global_band": self.global_band,
            "high_conflict_layers": list(self.high_conflict_layers),
            "critical_conflict_layers": list(self.critical_conflict_layers),
            "conflict_tensor_ratio": self.conflict_tensor_ratio,
            "mean_confidence": self.mean_confidence,
            "per_tensor": {k: asdict(v) for k, v in self.per_tensor.items()},
            "per_layer": {k: asdict(v) for k, v in self.per_layer.items()},
        }


@dataclass
class CBADirective:
    alpha_a: float
    action: str
    conflict: float
    reason: str
    confidence: float = 0.0
    importance: float = 0.0
    protection: float = 0.0
    source_key: Optional[str] = None
    layer_key: Optional[str] = None
    similarity: float = 0.0
    norm_ratio: float = 1.0
    fisher_a: float = 0.0
    fisher_b: float = 0.0
    activation_a: float = 0.0
    activation_b: float = 0.0
    evidence_score: float = 0.0
    risk: str = "unknown"
    alignment: str = "none"
    alignment_strength: float = 0.0


@dataclass
class CBARouting:
    layer_alphas: Dict[str, float]
    category_alphas: Dict[str, float]
    layer_actions: Dict[str, str]
    layer_conflicts: Dict[str, float]
    dominant_model: str
    directives: Dict[str, CBADirective]
    layer_confidence: Dict[str, float] = field(default_factory=dict)
    protected_layers: List[str] = field(default_factory=list)
    routing_evidence: Dict[str, float] = field(default_factory=dict)
    category_actions: Dict[str, str] = field(default_factory=dict)
    category_conflicts: Dict[str, float] = field(default_factory=dict)
    category_confidence: Dict[str, float] = field(default_factory=dict)

    def alpha_summary(self) -> str:
        def key_fn(k: str):
            m = re.fullmatch(r"layer_(\d+)", k)
            return (0, int(m.group(1))) if m else (1, k)
        out: List[str] = []
        for k in sorted(self.layer_alphas, key=key_fn):
            a = self.layer_alphas[k]
            out.append(
                f"{k}: A {a:.2f} B {1-a:.2f} "
                f"{self.layer_actions.get(k,'weighted')} "
                f"conf {self.layer_confidence.get(k,0.0):.2f} "
                f"conflict {self.layer_conflicts.get(k,0.0):.2f}"
            )
        for k, a in sorted(self.category_alphas.items()):
            out.append(
                f"{k}: A {a:.2f} B {1-a:.2f} "
                f"{self.category_actions.get(k,'weighted')} "
                f"conf {self.category_confidence.get(k,0.0):.2f} "
                f"conflict {self.category_conflicts.get(k,0.0):.2f}"
            )
        return "\n".join(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_alphas": dict(self.layer_alphas),
            "category_alphas": dict(self.category_alphas),
            "layer_actions": dict(self.layer_actions),
            "layer_conflicts": dict(self.layer_conflicts),
            "layer_confidence": dict(self.layer_confidence),
            "protected_layers": list(self.protected_layers),
            "routing_evidence": dict(self.routing_evidence),
            "category_actions": dict(self.category_actions),
            "category_conflicts": dict(self.category_conflicts),
            "category_confidence": dict(self.category_confidence),
            "dominant_model": self.dominant_model,
            "directives": {k: asdict(v) for k, v in self.directives.items()},
        }


@dataclass
class CBADecision:
    decision: str
    confidence: float
    evidence: List[str]
    risk: str
    alternative: str
    critique: Dict[str, Any]
    questions: Dict[str, str]
    dominant_model: str = "balanced"
    recommended_action: str = "weighted"
    policy_stability: float = 0.0
    measured_tensors: int = 0
    high_conflict_regions: int = 0
    critical_conflict_regions: int = 0
    structural_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CBAReport:
    inspection_a: BrainInspection
    inspection_b: BrainInspection
    compatibility: CompatibilityReport
    conflicts: ConflictReport
    routing: CBARouting
    decision: CBADecision
    correspondence: Dict[str, ParameterMatch] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cba_version": 5,
            "inspection_a": self.inspection_a.to_dict(),
            "inspection_b": self.inspection_b.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "conflicts": self.conflicts.to_dict(),
            "routing": self.routing.to_dict(),
            "decision": self.decision.to_dict(),
            "correspondence": {k: asdict(v) for k, v in self.correspondence.items()},
        }

    def save(self, path: str, *, overwrite: bool = False) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"CBA report already exists: {path}")
        data = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".ftrain_cba_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return path

    def critique(self) -> Dict[str, Any]:
        """``critique_decision_record`` applied to this report's own decision."""
        return critique_decision_record(self.decision.to_dict())

    def summary(self, max_layers: int = 12) -> str:
        """Short human-readable digest: decision, headline evidence, the most
        decisive routing regions, and the self-critique. Meant for logs and
        notebooks, not as a substitute for ``to_dict()``."""
        d, c = self.decision, self.compatibility
        lines = [
            f"CBA decision: {d.decision} (confidence {d.confidence:.2f}, dominant={d.dominant_model}, action={d.recommended_action})",
            f"Mergeability: {c.mergeability:.1f}/100 | compatible={c.compatible} | correspondence={c.correspondence_ratio:.0%}",
            f"Conflict: {self.conflicts.global_band} ({self.conflicts.global_score:.2f}) | risk: {d.risk}",
        ]
        if c.unmatched_targets:
            shown = ", ".join(c.unmatched_targets[:5])
            more = f" (+{len(c.unmatched_targets) - 5} more)" if len(c.unmatched_targets) > 5 else ""
            lines.append(f"Unmatched targets ({len(c.unmatched_targets)}): {shown}{more}")
        if self.routing.protected_layers:
            lines.append("Protected layers: " + ", ".join(self.routing.protected_layers[:max_layers]))
        lines.append("")
        lines.append("Most decisive regions:")
        decisive = sorted(self.routing.layer_alphas.items(), key=lambda x: abs(x[1] - 0.50), reverse=True)
        for key, alpha in decisive[:max_layers]:
            lines.append(
                f"  {key}: A={alpha:.2f} B={1-alpha:.2f} "
                f"[{self.routing.layer_actions.get(key,'weighted')}] "
                f"conflict={self.routing.layer_conflicts.get(key,0.0):.2f}"
            )
        crit = self.critique()
        lines.append("")
        lines.append(f"Self-critique: {crit['potential_weakness']}")
        lines.append(f"  claimed={crit['claimed_confidence']:.2f} justified_ceiling={crit['justified_confidence']:.2f}")
        lines.append(f"  {crit['recommendation']}")
        return "\n".join(lines)


def _row_signature(t: torch.Tensor, max_rows: int = 512) -> Optional[torch.Tensor]:
    if t.ndim != 2 or t.shape[0] == 0:
        return None
    x = t.detach().float()
    idx = _deterministic_indices(x.shape[0], max_rows, x.device)
    if idx is not None:
        x = x.index_select(0, idx)
    return torch.linalg.vector_norm(x, dim=1)


def _column_signature(t: torch.Tensor, max_cols: int = 512) -> Optional[torch.Tensor]:
    if t.ndim != 2 or t.shape[1] == 0:
        return None
    x = t.detach().float()
    idx = _deterministic_indices(x.shape[1], max_cols, x.device)
    if idx is not None:
        x = x.index_select(1, idx)
    return torch.linalg.vector_norm(x, dim=0)


def _signature_similarity(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return 0.0
    n = min(a.numel(), b.numel())
    if a.numel() != n:
        ia = torch.linspace(0, a.numel()-1, steps=n, device=a.device, dtype=torch.float64).round().long()
        a = a.index_select(0, ia)
    if b.numel() != n:
        ib = torch.linspace(0, b.numel()-1, steps=n, device=b.device, dtype=torch.float64).round().long()
        b = b.index_select(0, ib)
    na = _safe_norm(a) + _EPS
    nb = _safe_norm(b) + _EPS
    return _clamp(_safe_float(torch.dot(a, b) / (na * nb)), -1.0, 1.0)


def _scalar_evidence(mapping: Optional[Mapping[str, Any]], key: str) -> float:
    if mapping is None or key not in mapping:
        return 0.0
    v = mapping[key]
    if isinstance(v, torch.Tensor):
        try:
            v = v.float()
            v = v[torch.isfinite(v)]
            return max(0.0, _safe_mean(v)) if v.numel() else 0.0
        except Exception:
            return 0.0
    return max(0.0, _safe_float(v))


def inspect_state_dict(state_dict: Mapping[str, torch.Tensor]) -> BrainInspection:
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")

    tensors: Dict[str, TensorRecord] = {}
    dtypes: Dict[str, int] = {}
    total_numel = 0
    total_norm_sq = 0.0
    floating_tensors = nonfinite_tensors = zero_tensors = sensitive_tensors = 0
    hidden_size = vocab_size = None
    layer_indices: Set[int] = set()

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = tuple(int(v) for v in tensor.shape)
        numel = int(tensor.numel())
        total_numel += numel
        dtype_name = str(tensor.dtype).replace("torch.", "")
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1
        if not torch.is_floating_point(tensor):
            continue

        floating_tensors += 1
        layer = _layer_of(name)
        category = _classify(name)
        role = _parameter_role(name)
        if layer is not None:
            layer_indices.add(layer)
        if _sensitive(category):
            sensitive_tensors += 1

        if len(shape) == 2:
            s = name.lower()
            if any(x in s for x in ("embed_tokens", "tok_embeddings", "word_embeddings")):
                vocab_size, hidden_size = int(shape[0]), int(shape[1])
            elif "lm_head" in s and hidden_size is None:
                hidden_size = int(shape[1])

        sample = _deterministic_sample(tensor, _FLATTEN_CAP)
        finite = torch.isfinite(sample)
        finite_ratio = _safe_float(finite.float().mean()) if sample.numel() else 1.0
        if finite_ratio < 1.0:
            nonfinite_tensors += 1
        values = sample[finite]
        if values.numel() == 0:
            tensors[name] = TensorRecord(
                name=name, category=category, layer=layer, numel=numel,
                mean=0.0, std=0.0, l2_norm=0.0, rel_norm=0.0,
                shape=shape, dtype=dtype_name, role=role,
                finite_ratio=finite_ratio,
            )
            continue

        mean = _safe_mean(values)
        std = _safe_std(values)
        mean_abs = _safe_float(values.abs().mean())
        rms = math.sqrt(max(0.0, _safe_float((values * values).mean())))
        # Extrapolate the finite sample's energy to the *estimated finite
        # portion* of the full tensor (numel * finite_ratio), not the raw
        # element count. `values` already had non-finite entries stripped
        # out of the capped sample, so scaling by the full `numel` would
        # inflate the estimate for any large tensor (> _FLATTEN_CAP) that
        # also contains NaN/Inf -- exactly the tensors CBA most needs an
        # honest norm for, since `nonfinite_tensors` already flags them as
        # noteworthy.
        finite_numel_estimate = max(1, int(round(numel * finite_ratio)))
        min_abs = _safe_float(values.abs().min())
        max_abs = _safe_float(values.abs().max())
        zero_ratio = _safe_float((values == 0).float().mean())
        threshold = max(1e-8, 1e-3 * max(1.0, rms))
        near_zero_ratio = _safe_float((values.abs() <= threshold).float().mean())
        l2 = _estimated_l2(values, finite_numel_estimate)
        if l2 < 1e-12 or max_abs < 1e-12:
            zero_tensors += 1
        total_norm_sq += l2 * l2
        tensors[name] = TensorRecord(
            name=name, category=category, layer=layer, numel=numel,
            mean=mean, std=std, l2_norm=l2, rel_norm=0.0,
            mean_abs=mean_abs, rms=rms, min_abs=min_abs, max_abs=max_abs,
            zero_ratio=zero_ratio, near_zero_ratio=near_zero_ratio,
            finite_ratio=finite_ratio, shape=shape, dtype=dtype_name,
            role=role, energy_estimate=l2 * l2,
        )

    total_norm = math.sqrt(max(0.0, total_norm_sq))
    for rec in tensors.values():
        rec.rel_norm = rec.l2_norm / max(total_norm, _EPS)
        rec.importance = _clamp01(
            0.70 * min(1.0, rec.rel_norm * 100.0)
            + 0.30 * (1.0 - rec.near_zero_ratio)
        )

    layer_groups: Dict[int, List[str]] = {}
    category_groups: Dict[str, List[str]] = {}
    for name, rec in tensors.items():
        category_groups.setdefault(rec.category, []).append(name)
        if rec.layer is not None:
            layer_groups.setdefault(rec.layer, []).append(name)

    layers: Dict[str, LayerRecord] = {}
    categories: Dict[str, LayerRecord] = {}
    for layer, names in sorted(layer_groups.items()):
        counts: Dict[str, int] = {}
        for n in names:
            counts[tensors[n].category] = counts.get(tensors[n].category, 0) + 1
        category = max(counts, key=counts.get)
        norm = math.sqrt(sum(tensors[n].l2_norm ** 2 for n in names))
        numel = sum(tensors[n].numel for n in names)
        mean_std = _weighted_mean((tensors[n].std, tensors[n].numel) for n in names)
        mean_abs = _weighted_mean((tensors[n].mean_abs, tensors[n].numel) for n in names)
        sparsity = _weighted_mean((tensors[n].near_zero_ratio, tensors[n].numel) for n in names)
        importance = _weighted_mean((tensors[n].importance, tensors[n].numel) for n in names)
        key = f"layer_{layer}"
        layers[key] = LayerRecord(
            key=key, kind="layer", index=layer, category=category,
            tensors=len(names), numel=numel, l2_norm=norm,
            rel_norm=norm / max(total_norm, _EPS), mean_std=mean_std,
            mean_abs=mean_abs, sparsity=sparsity, importance=importance,
        )

    for category, names in sorted(category_groups.items()):
        norm = math.sqrt(sum(tensors[n].l2_norm ** 2 for n in names))
        numel = sum(tensors[n].numel for n in names)
        categories[category] = LayerRecord(
            key=category, kind="category", index=None, category=category,
            tensors=len(names), numel=numel, l2_norm=norm,
            rel_norm=norm / max(total_norm, _EPS),
            mean_std=_weighted_mean((tensors[n].std, tensors[n].numel) for n in names),
            mean_abs=_weighted_mean((tensors[n].mean_abs, tensors[n].numel) for n in names),
            sparsity=_weighted_mean((tensors[n].near_zero_ratio, tensors[n].numel) for n in names),
            importance=_weighted_mean((tensors[n].importance, tensors[n].numel) for n in names),
        )

    return BrainInspection(
        total_tensors=len(tensors), total_numel=total_numel, total_norm=total_norm,
        layer_count=len(layer_indices), hidden_size=hidden_size, vocab_size=vocab_size,
        dtypes=dtypes, tensors=tensors, layers=layers, categories=categories,
        floating_tensors=floating_tensors, nonfinite_tensors=nonfinite_tensors,
        zero_tensors=zero_tensors, sensitive_tensors=sensitive_tensors,
    )


def _layer_translate(layer: int, layers_a: int, layers_b: int) -> int:
    if layers_a <= 1 or layers_b <= 1:
        return 0
    return int(round((layer / float(layers_a - 1)) * (layers_b - 1)))


def _shape_similarity(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    score = 1.0
    for x, y in zip(a, b):
        score *= min(x, y) / max(x, y)
    return _clamp01(score)


def _score_candidates(
    target: str,
    rec: TensorRecord,
    tensor: torch.Tensor,
    state_dict_b: Mapping[str, torch.Tensor],
    exact: Set[str],
    canonical: Mapping[str, List[str]],
    role: Mapping[str, List[str]],
    layer_role: Mapping[Tuple[Optional[int], str], List[str]],
    layers_a: int,
    layers_b: int,
) -> Tuple[List[Tuple[float, str, bool, bool, Optional[int], float, Tuple[int, ...]]], Optional[int]]:
    """All scored candidate sources for one target, best first.

    Split out of :func:`build_correspondence` so scoring (cheap, order-
    independent) is fully separated from assignment (which needs to see every
    target's candidates before deciding who goes first).
    """
    tshape = tuple(tensor.shape)
    candidates: List[str] = []
    if target in exact:
        candidates.append(target)
    candidates.extend(canonical.get(_canonical_key(target), []))

    expected: Optional[int] = None
    if rec.layer is not None:
        expected = _layer_translate(rec.layer, layers_a, layers_b)
        r = rec.role
        candidates.extend(layer_role.get((expected, r), []))
        for d in (1, 2):
            for k in (expected - d, expected + d):
                if k >= 0:
                    candidates.extend(layer_role.get((k, r), []))

    # Role-only fallback (no reliable layer/canonical signal so far). When
    # there are more candidates than the cap, keep the ones nearest the
    # expected translated layer instead of an arbitrary alphabetical prefix,
    # so very deep models (hundreds of layers sharing one role) don't
    # silently lose the correct match to whichever name sorts first.
    role_pool = role.get(rec.role, [])
    if len(role_pool) > _ROLE_FALLBACK_CAP:
        if expected is not None:
            role_pool = sorted(
                role_pool,
                key=lambda s: (abs((_layer_of(s) if _layer_of(s) is not None else 10 ** 9) - expected), s),
            )[:_ROLE_FALLBACK_CAP]
        else:
            role_pool = sorted(role_pool)[:_ROLE_FALLBACK_CAP]
    candidates.extend(role_pool)

    scored: List[Tuple[float, str, bool, bool, Optional[int], float, Tuple[int, ...]]] = []
    for source in sorted(set(candidates)):
        st = state_dict_b.get(source)
        if not isinstance(st, torch.Tensor):
            continue
        sshape = tuple(st.shape)
        canon = _canonical_key(target) == _canonical_key(source)
        role_exact = rec.role == _parameter_role(source)
        shape_score = _shape_similarity(tshape, sshape)
        sl = _layer_of(source)
        layer_score = 0.0
        distance = None
        if rec.layer is not None and sl is not None:
            expected_sl = _layer_translate(rec.layer, layers_a, layers_b)
            distance = abs(sl - expected_sl)
            layer_score = max(0.0, 1.0 - distance / max(1, layers_b))
        score = _clamp01(
            0.65 * float(canon)
            + 0.25 * float(not canon and _canonical_key(target).split("layers.{layer}.")[-1] == _canonical_key(source).split("layers.{layer}.")[-1])
            + 0.15 * float(role_exact)
            + 0.15 * shape_score
            + 0.05 * layer_score
            + 0.10 * float(tshape == sshape)
        )
        scored.append((score, source, canon, role_exact, distance, shape_score, sshape))

    scored.sort(key=lambda c: (-c[0], c[1]))
    return scored, expected


def build_correspondence(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
) -> Dict[str, ParameterMatch]:
    layers_a = max(1, inspection_a.layer_count)
    layers_b = max(1, inspection_b.layer_count)
    exact = set(state_dict_b.keys())
    canonical: Dict[str, List[str]] = {}
    role: Dict[str, List[str]] = {}
    layer_role: Dict[Tuple[Optional[int], str], List[str]] = {}
    for n, t in state_dict_b.items():
        if not isinstance(t, torch.Tensor):
            continue
        canonical.setdefault(_canonical_key(n), []).append(n)
        r = _parameter_role(n)
        role.setdefault(r, []).append(n)
        layer_role.setdefault((_layer_of(n), r), []).append(n)
    for d in (canonical, role, layer_role):
        for v in d.values():
            v.sort()

    # Pass 1: score every target's candidates independently (no source is
    # consumed here), so assignment order in pass 2 can be based on evidence
    # rather than on however `inspection_a.tensors` happens to be ordered.
    scored_by_target: Dict[str, Tuple[List[Tuple[float, str, bool, bool, Optional[int], float, Tuple[int, ...]]], Optional[int]]] = {}
    for target, rec in inspection_a.tensors.items():
        tensor = state_dict_a.get(target)
        if not isinstance(tensor, torch.Tensor):
            continue
        scored_by_target[target] = _score_candidates(
            target, rec, tensor, state_dict_b, exact, canonical, role, layer_role, layers_a, layers_b,
        )

    def margin(target: str) -> float:
        scored, _ = scored_by_target[target]
        if not scored:
            return -1.0  # no candidates at all: resolve last, it won't get one anyway
        best = scored[0][0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        return best - second

    # Pass 2: assign in order of least ambiguity first (largest margin
    # between a target's best and second-best candidate), so a contested
    # source goes to whichever target has the strongest, most unambiguous
    # claim to it rather than to whichever target's name sorts first.
    priority = sorted(scored_by_target.keys(), key=lambda t: (-margin(t), t))

    used: Set[str] = set()
    out: Dict[str, ParameterMatch] = {}
    for target in priority:
        scored, expected = scored_by_target[target]
        best = next((c for c in scored if c[1] not in used), None)
        if best is None:
            continue
        score, source, canon, role_exact, distance, shape_score, sshape = best
        minimum = _MIN_CANONICAL_CONFIDENCE if canon else _MIN_CORRESPONDENCE_CONFIDENCE
        if score < minimum:
            continue
        rec = inspection_a.tensors[target]
        tshape = tuple(state_dict_a[target].shape)
        method = "exact" if source == target else "canonical" if canon else "translated" if expected is not None else "fuzzy"
        out[target] = ParameterMatch(
            target_name=target, source_name=source, method=method,
            confidence=score, shape_exact=(tshape == sshape), role_exact=role_exact,
            layer_a=rec.layer, layer_b=_layer_of(source), score=score,
            shape_a=tshape, shape_b=sshape, canonical_exact=canon,
            layer_distance=distance, shape_similarity=shape_score,
        )
        used.add(source)
    return out


def check_compatibility(
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    state_dict_a: Optional[Mapping[str, torch.Tensor]] = None,
    state_dict_b: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    correspondence: Optional[Mapping[str, ParameterMatch]] = None,
    vocab_size_b: Optional[int] = None,
) -> CompatibilityReport:
    names_a, names_b = set(inspection_a.tensors), set(inspection_b.tensors)
    inter, union = names_a & names_b, names_a | names_b
    name_overlap = len(inter) / max(1, len(union))
    shape_matches = shape_total = 0
    if state_dict_a is not None and state_dict_b is not None:
        for n in inter:
            a, b = state_dict_a.get(n), state_dict_b.get(n)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                shape_total += 1
                shape_matches += int(tuple(a.shape) == tuple(b.shape))
    shape_ratio = shape_matches / max(1, shape_total)

    corr_ratio = role_ratio = exact_shape_ratio = corr_conf = 0.0
    unmatched_targets: List[str] = []
    if correspondence:
        corr_ratio = len(correspondence) / max(1, inspection_a.total_tensors)
        role_ratio = sum(int(m.role_exact) for m in correspondence.values()) / max(1, len(correspondence))
        exact_shape_ratio = sum(int(m.shape_exact) for m in correspondence.values()) / max(1, len(correspondence))
        corr_conf = _mean(m.confidence for m in correspondence.values())
        unmatched_targets = sorted(set(inspection_a.tensors) - set(correspondence.keys()))

    hidden_match = inspection_a.hidden_size is not None and inspection_b.hidden_size is not None and inspection_a.hidden_size == inspection_b.hidden_size
    la, lb = max(1, inspection_a.layer_count), max(1, inspection_b.layer_count)
    layer_similarity = 1.0 - min(1.0, abs(la - lb) / max(la, lb))
    dtype_ok = bool(inspection_a.dtypes and inspection_b.dtypes)

    vb = vocab_size_b if vocab_size_b is not None else inspection_b.vocab_size
    tok_ok = None
    if inspection_a.vocab_size is not None and vb is not None:
        tol = max(256, int(0.01 * max(inspection_a.vocab_size, vb)))
        tok_ok = abs(inspection_a.vocab_size - vb) <= tol

    notes: List[str] = []
    if not hidden_match: notes.append("hidden sizes do not exactly match")
    if la != lb: notes.append("layer counts differ; positional translation is being used")
    if shape_ratio < 0.90: notes.append("exact-name shape compatibility is below 90%")
    if corr_ratio < 0.70: notes.append("correspondence coverage is incomplete")
    if corr_ratio >= 0.85: notes.append("most target parameters have a defensible source correspondence")
    if inspection_a.nonfinite_tensors or inspection_b.nonfinite_tensors: notes.append("non-finite tensors were detected")
    if tok_ok is False: notes.append("vocabulary sizes differ beyond tokenizer tolerance")
    if unmatched_targets: notes.append(f"{len(unmatched_targets)} target tensors have no correspondence match")

    penalty = 0.0
    if not hidden_match: penalty += 0.45
    penalty += 0.25 * (1.0 - layer_similarity)
    penalty += 0.30 * (1.0 - corr_ratio)
    penalty = _clamp01(penalty)
    merge_score = (
        0.28 * corr_ratio + 0.18 * corr_conf + 0.14 * exact_shape_ratio
        + 0.10 * role_ratio + 0.12 * layer_similarity + 0.10 * float(hidden_match)
        + 0.04 * float(dtype_ok) + 0.04 * float(tok_ok is not False)
    ) * (1.0 - 0.40 * penalty)
    mergeability = 100.0 * _clamp01(merge_score)
    compatible = corr_ratio >= 0.55 and mergeability >= 55.0 and not (inspection_a.nonfinite_tensors or inspection_b.nonfinite_tensors)
    structural_conf = _clamp01(
        0.45 * corr_ratio + 0.25 * corr_conf + 0.15 * layer_similarity + 0.15 * float(hidden_match)
    )
    return CompatibilityReport(
        compatible=compatible, mergeability=round(mergeability, 2), name_overlap=round(name_overlap, 4),
        shape_match_ratio=round(shape_ratio, 4), layer_count_a=inspection_a.layer_count, layer_count_b=inspection_b.layer_count,
        hidden_size_a=inspection_a.hidden_size, hidden_size_b=inspection_b.hidden_size,
        dtype_compatible=dtype_ok, tokenizer_compatible=tok_ok, notes=notes,
        correspondence_ratio=round(corr_ratio, 4), role_match_ratio=round(role_ratio, 4),
        exact_shape_correspondence_ratio=round(exact_shape_ratio, 4), architecture_penalty=round(penalty, 4),
        structural_confidence=round(structural_conf, 4), unmatched_targets=unmatched_targets,
    )


def _tensor_conflict(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    layer: Optional[int],
    category: str,
    high_cutoff: float,
) -> Optional[TensorConflict]:
    if tuple(a.shape) != tuple(b.shape):
        # Defensive: analyze_conflicts already guarantees this, but keeping
        # the guard here means the function is safe to call on its own.
        return None

    x, y, sampled = _finite_pair(a, b)
    if x.numel() == 0 or sampled == 0:
        return None
    finite_ratio = x.numel() / sampled

    na, nb = _safe_norm(x), _safe_norm(y)
    cosine = _safe_float(torch.dot(x, y) / max(na * nb, _EPS)) if na > _EPS and nb > _EPS else 0.0
    cosine = _clamp(cosine, -1.0, 1.0)
    direction_similarity = (cosine + 1.0) / 2.0
    relative_delta = _clamp01(_safe_norm(x - y) / max(na + nb, _EPS))

    ca, cb = x - x.mean(), y - y.mean()
    ta = max(1e-8, 0.05 * _safe_float(ca.abs().mean()))
    tb = max(1e-8, 0.05 * _safe_float(cb.abs().mean()))
    active = (ca.abs() > ta) | (cb.abs() > tb)
    sign_disagreement = _safe_float((((ca[active] * cb[active]) < 0).float().mean())) if bool(active.any()) else 0.0

    norm_ratio = nb / max(na, _EPS)
    magnitude_disagreement = _clamp01(abs(math.log(max(norm_ratio, _EPS))) / math.log(8.0))
    ma, mb, sa, sb = _safe_mean(x), _safe_mean(y), _safe_std(x), _safe_std(y)
    mean_disagreement = _clamp01(abs(ma - mb) / max(abs(ma), abs(mb), 1e-8))
    std_disagreement = _clamp01(abs(sa - sb) / max(sa, sb, 1e-8))
    distribution_disagreement = 0.5 * (mean_disagreement + std_disagreement)
    zero_disagreement = abs(_safe_float((x == 0).float().mean()) - _safe_float((y == 0).float().mean()))

    row_sim = col_sim = 0.0
    if a.ndim == b.ndim == 2 and tuple(a.shape) == tuple(b.shape):
        row_sim = (1.0 + _signature_similarity(_row_signature(a), _row_signature(b))) / 2.0
        col_sim = (1.0 + _signature_similarity(_column_signature(a), _column_signature(b))) / 2.0
        row_sim, col_sim = _clamp01(row_sim), _clamp01(col_sim)
    structural_disagreement = 0.5 * (1.0 - row_sim) + 0.5 * (1.0 - col_sim)

    score = _clamp01(
        0.30 * (1.0 - direction_similarity)
        + 0.22 * sign_disagreement
        + 0.16 * relative_delta
        + 0.10 * magnitude_disagreement
        + 0.08 * distribution_disagreement
        + 0.06 * zero_disagreement
        + 0.08 * structural_disagreement
    )
    # Confidence now reflects three independent things: enough sampled
    # elements to trust the statistics, matching rank (guaranteed here, kept
    # for defensive direct calls), and how much of the sampled pair was
    # actually finite -- a pair with heavy NaN/Inf contamination has real
    # data behind only a fraction of its score and should say so.
    confidence = _clamp01(
        0.60 * min(1.0, x.numel() / 8192.0)
        + 0.15 * float(a.ndim == b.ndim)
        + 0.25 * finite_ratio
    )
    band = CONFLICT_CRITICAL if score >= _CONFLICT_CRITICAL_CUTOFF else CONFLICT_HIGH if score >= high_cutoff else CONFLICT_MEDIUM if score >= _CONFLICT_MEDIUM_CUTOFF else CONFLICT_LOW
    return TensorConflict(
        name=name, layer=layer, category=category,
        direction_similarity=round(direction_similarity, 6),
        sign_disagreement=round(sign_disagreement, 6), relative_delta=round(relative_delta, 6),
        score=round(score, 6), band=band,
        cosine_similarity=round(cosine, 6), norm_ratio=round(norm_ratio, 6),
        magnitude_disagreement=round(magnitude_disagreement, 6), zero_disagreement=round(zero_disagreement, 6),
        structural_disagreement=round(structural_disagreement, 6), distribution_disagreement=round(distribution_disagreement, 6),
        row_structure_similarity=round(row_sim, 6), column_structure_similarity=round(col_sim, 6),
        finite_ratio=round(finite_ratio, 6), confidence=round(confidence, 6),
    )


def analyze_conflicts(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    *,
    correspondence: Optional[Mapping[str, ParameterMatch]] = None,
    high_cutoff: float = _CONFLICT_HIGH_CUTOFF_DEFAULT,
) -> ConflictReport:
    if not 0.0 < high_cutoff <= 1.0:
        raise ValueError("high_cutoff must be in (0,1]")
    correspondence = correspondence or build_correspondence(state_dict_a, state_dict_b, inspection_a, inspection_b)
    per_tensor: Dict[str, TensorConflict] = {}
    grouped: Dict[str, List[TensorConflict]] = {}

    for target, match in correspondence.items():
        a, b = state_dict_a.get(target), state_dict_b.get(match.source_name)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or tuple(a.shape) != tuple(b.shape):
            continue
        rec = inspection_a.tensors.get(target)
        if rec is None:
            continue
        c = _tensor_conflict(target, a, b, layer=rec.layer, category=rec.category, high_cutoff=high_cutoff)
        if c is None:
            continue
        c.confidence = round(_clamp01(c.confidence * match.confidence), 6)
        per_tensor[target] = c
        key = f"layer_{rec.layer}" if rec.layer is not None else rec.category
        grouped.setdefault(key, []).append(c)

    per_layer: Dict[str, LayerConflict] = {}
    for key, vals in sorted(grouped.items()):
        score = _mean(v.score for v in vals)
        importance = _mean(
            inspection_a.tensors[v.name].importance
            for v in vals
            if v.name in inspection_a.tensors
        )
        per_layer[key] = LayerConflict(
            key=key, kind="layer" if key.startswith("layer_") else "category",
            score=round(score, 6),
            band=CONFLICT_CRITICAL if score >= _CONFLICT_CRITICAL_CUTOFF else CONFLICT_HIGH if score >= high_cutoff else CONFLICT_MEDIUM if score >= _CONFLICT_MEDIUM_CUTOFF else CONFLICT_LOW,
            tensors=len(vals),
            mean_cosine=round(_mean(v.cosine_similarity for v in vals), 6),
            mean_relative_delta=round(_mean(v.relative_delta for v in vals), 6),
            mean_sign_disagreement=round(_mean(v.sign_disagreement for v in vals), 6),
            mean_magnitude_disagreement=round(_mean(v.magnitude_disagreement for v in vals), 6),
            mean_confidence=round(_mean(v.confidence for v in vals), 6),
            weighted_importance=round(importance, 6),
        )

    weighted: List[Tuple[float, float]] = []
    for name, c in per_tensor.items():
        imp = inspection_a.tensors.get(name).importance if name in inspection_a.tensors else 0.0
        weighted.append((c.score, 0.25 + 0.75 * _clamp01(imp)))
    global_score = _clamp01(_weighted_mean(weighted)) if weighted else 0.0
    global_band = CONFLICT_CRITICAL if global_score >= _CONFLICT_CRITICAL_CUTOFF else CONFLICT_HIGH if global_score >= high_cutoff else CONFLICT_MEDIUM if global_score >= _CONFLICT_MEDIUM_CUTOFF else CONFLICT_LOW
    high_layers = [k for k, v in sorted(per_layer.items()) if v.band in {CONFLICT_HIGH, CONFLICT_CRITICAL}]
    critical_layers = [k for k, v in sorted(per_layer.items()) if v.band == CONFLICT_CRITICAL]
    ratio = sum(v.score >= _CONFLICT_MEDIUM_CUTOFF for v in per_tensor.values()) / max(1, len(per_tensor))
    conf = _mean(v.confidence for v in per_tensor.values())
    return ConflictReport(
        global_score=round(global_score, 6), global_band=global_band,
        high_conflict_layers=high_layers, per_tensor=per_tensor, per_layer=per_layer,
        critical_conflict_layers=critical_layers, conflict_tensor_ratio=round(ratio, 6),
        mean_confidence=round(conf, 6),
    )


def _layer_rec(i: BrainInspection, key: str) -> Optional[LayerRecord]:
    return i.layers.get(key) or i.categories.get(key)


def _category_conflict_aggregate(conflicts: ConflictReport) -> Dict[str, Tuple[float, float]]:
    """True per-category (score, confidence), aggregated directly from
    ``per_tensor`` (which always carries a ``category`` field), rather than
    read from ``per_layer`` -- a tensor's conflict is bucketed there under
    ``layer_N`` whenever it has a detected layer index, so a bare category
    key such as ``"attention"`` or ``"mlp"`` is almost never present there
    even though plenty of attention/mlp tensors were actually measured.
    """
    grouped: Dict[str, List[TensorConflict]] = defaultdict(list)
    for c in conflicts.per_tensor.values():
        grouped[c.category].append(c)
    return {
        cat: (_mean(c.score for c in items), _mean(c.confidence for c in items))
        for cat, items in grouped.items()
    }


def _evidence_score(
    ia: BrainInspection,
    ib: BrainInspection,
    conflicts: ConflictReport,
    key: str,
    *,
    activation_layer_a: Optional[Mapping[str, Any]] = None,
    activation_layer_b: Optional[Mapping[str, Any]] = None,
    conflict_override: Optional[float] = None,
) -> float:
    a, b = _layer_rec(ia, key), _layer_rec(ib, key)
    if a is None or b is None:
        return 0.0
    importance = math.tanh(_safe_log_ratio(a.rel_norm + _EPS, b.rel_norm + _EPS))
    specialization = math.tanh(_safe_log_ratio(a.mean_std + _EPS, b.mean_std + _EPS))
    aa, ab = 1.0 - a.sparsity, 1.0 - b.sparsity
    activity = (aa - ab) / max(aa + ab, _EPS)
    evidence = 0.45 * importance + 0.30 * specialization + 0.15 * activity
    if activation_layer_a is not None and activation_layer_b is not None:
        va = _scalar_evidence(activation_layer_a, key)
        vb = _scalar_evidence(activation_layer_b, key)
        if va > 0 or vb > 0:
            evidence += 0.10 * _clamp((va - vb) / max(va + vb, _EPS), -1.0, 1.0)
    if conflict_override is not None:
        conflict = conflict_override
    else:
        conflict = conflicts.per_layer.get(key).score if key in conflicts.per_layer else conflicts.global_score
    return _clamp((evidence * (1.0 - 0.60 * _clamp01(conflict))), -1.0, 1.0)


def _smooth_alphas(alphas: Dict[str, float], conflicts: Dict[str, float]) -> Dict[str, float]:
    ordered = []
    for key, value in alphas.items():
        m = re.fullmatch(r"layer_(\d+)", key)
        if m:
            ordered.append((int(m.group(1)), key, value))
    ordered.sort()
    if len(ordered) < 3:
        return dict(alphas)
    out = dict(alphas)
    for i in range(1, len(ordered) - 1):
        _, key, current = ordered[i]
        previous, following = ordered[i - 1][2], ordered[i + 1][2]
        neighbor = (previous + following) / 2.0
        if abs(current - neighbor) < 0.22:
            s = 0.35 * (1.0 - conflicts.get(key, 0.0))
            out[key] = (1.0 - s) * current + s * neighbor
    return out


def _layer_protection_weights(inspection_a: BrainInspection, inspection_b: BrainInspection) -> Dict[str, float]:
    """Numel-weighted protection strength for every ``layer_N`` key, built
    from each layer's *actual* tensor composition in both models.

    Replaces a majority-vote-category check that could only ever protect a
    layer whose single most common tensor category was norm/embedding/head --
    embeddings and heads never carry a layer index at all, and norm tensors
    are always a minority within a mixed attention+mlp+norm block, so that
    check was dead for two of its three intended categories and essentially
    dead for the third.
    """
    grouped: Dict[str, List[TensorRecord]] = defaultdict(list)
    for rec in list(inspection_a.tensors.values()) + list(inspection_b.tensors.values()):
        if rec.layer is not None:
            grouped[f"layer_{rec.layer}"].append(rec)

    weights: Dict[str, float] = {}
    for key, recs in grouped.items():
        total = sum(r.numel for r in recs) or 1
        weighted = sum(_protection_for_category(r.category) * r.numel for r in recs)
        weights[key] = _clamp01(weighted / total)
    return weights


def plan_routing(
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    conflicts: ConflictReport,
    compatibility: CompatibilityReport,
    *,
    high_conflict_threshold: float = _CONFLICT_HIGH_CUTOFF_DEFAULT,
    fisher_a: Optional[Mapping[str, torch.Tensor]] = None,
    fisher_b: Optional[Mapping[str, torch.Tensor]] = None,
    activation_a: Optional[Mapping[str, Any]] = None,
    activation_b: Optional[Mapping[str, Any]] = None,
    activation_layer_a: Optional[Mapping[str, Any]] = None,
    activation_layer_b: Optional[Mapping[str, Any]] = None,
) -> CBARouting:
    if not 0.0 < high_conflict_threshold <= 1.0:
        raise ValueError("high_conflict_threshold must be in (0,1]")

    layer_alphas: Dict[str, float] = {}
    category_alphas: Dict[str, float] = {}
    layer_actions: Dict[str, str] = {}
    layer_conflicts: Dict[str, float] = {}
    layer_confidence: Dict[str, float] = {}
    routing_evidence: Dict[str, float] = {}
    protected: List[str] = []
    layer_protection = _layer_protection_weights(inspection_a, inspection_b)

    def layer_sort(k: str):
        m = re.fullmatch(r"layer_(\d+)", k)
        return (int(m.group(1)) if m else -1)

    all_layers = sorted(set(inspection_a.layers) | set(inspection_b.layers), key=layer_sort)
    for key in all_layers:
        ev = _evidence_score(inspection_a, inspection_b, conflicts, key, activation_layer_a=activation_layer_a, activation_layer_b=activation_layer_b)
        lc = conflicts.per_layer.get(key)
        cs = lc.score if lc is not None else conflicts.global_score
        conf = lc.mean_confidence if lc is not None else conflicts.mean_confidence
        alpha = 0.50 + _ROUTING_GAIN * ev
        if cs >= high_conflict_threshold:
            alpha = 0.50 + 0.45 * (alpha - 0.50)
        if cs >= _CONFLICT_CRITICAL_CUTOFF:
            alpha = 0.50 + 0.20 * (alpha - 0.50)
        protection_weight = layer_protection.get(key, 0.0)
        if protection_weight > 0.0:
            alpha = (1.0 - protection_weight) * alpha + protection_weight * 0.50
            if protection_weight >= _LAYER_PROTECTION_FLAG_THRESHOLD:
                protected.append(key)
        alpha = _clamp(alpha, _ALPHA_MIN, _ALPHA_MAX)
        action = "ties" if cs >= high_conflict_threshold else "slerp" if cs <= 0.15 and abs(alpha - 0.50) < 0.04 else "weighted"
        layer_alphas[key] = round(alpha, 5)
        layer_actions[key] = action
        layer_conflicts[key] = round(cs, 5)
        layer_confidence[key] = round(_clamp01(0.55 * conf + 0.45 * compatibility.structural_confidence), 5)
        routing_evidence[key] = round(ev, 5)

    layer_alphas = _smooth_alphas(layer_alphas, layer_conflicts)

    category_conflict_scores = _category_conflict_aggregate(conflicts)
    category_actions: Dict[str, str] = {}
    category_conflicts: Dict[str, float] = {}
    category_confidence: Dict[str, float] = {}
    for cat in sorted(set(inspection_a.categories) & set(inspection_b.categories)):
        cs, cconf = category_conflict_scores.get(cat, (conflicts.global_score, conflicts.mean_confidence))
        ev = _evidence_score(
            inspection_a, inspection_b, conflicts, cat,
            activation_layer_a=activation_layer_a, activation_layer_b=activation_layer_b,
            conflict_override=cs,
        )
        alpha = 0.50 + _ROUTING_GAIN * ev
        if cs >= high_conflict_threshold:
            alpha = 0.50 + 0.45 * (alpha - 0.50)
        if cs >= _CONFLICT_CRITICAL_CUTOFF:
            alpha = 0.50 + 0.20 * (alpha - 0.50)
        protection_weight = _protection_for_category(cat)
        if protection_weight:
            alpha = (1.0 - protection_weight) * alpha + protection_weight * 0.50
        alpha = _clamp(alpha, _ALPHA_MIN, _ALPHA_MAX)
        action = "ties" if cs >= high_conflict_threshold else "slerp" if cs <= 0.15 and abs(alpha - 0.50) < 0.04 else "weighted"
        category_alphas[cat] = round(alpha, 5)
        category_actions[cat] = action
        category_conflicts[cat] = round(cs, 5)
        category_confidence[cat] = round(_clamp01(0.55 * cconf + 0.45 * compatibility.structural_confidence), 5)

    directives: Dict[str, CBADirective] = {}
    for name, rec in inspection_a.tensors.items():
        conflict = conflicts.per_tensor.get(name)
        if rec.layer is not None:
            lk = f"layer_{rec.layer}"
            alpha = layer_alphas.get(lk, 0.50)
            action = layer_actions.get(lk, "weighted")
            layer_conflict = layer_conflicts.get(lk, 0.0)
            layer_conf = layer_confidence.get(lk, 0.0)
            evidence = routing_evidence.get(lk, 0.0)
        else:
            lk = None
            alpha = category_alphas.get(rec.category, 0.50)
            action = category_actions.get(rec.category, "weighted")
            layer_conflict = category_conflicts.get(rec.category, 0.0)
            layer_conf = category_confidence.get(rec.category, 0.0)
            evidence = 0.0

        tc = conflict.score if conflict else layer_conflict
        confidence = _clamp01(0.50 * layer_conf + 0.50 * (conflict.confidence if conflict else 0.0))
        similarity = conflict.cosine_similarity if conflict else 0.0
        norm_ratio = conflict.norm_ratio if conflict else 1.0

        if conflict and conflict.cosine_similarity >= _SIMILAR_DIRECTION and conflict.relative_delta < 0.10:
            alpha, action = 0.50, "slerp"
        if conflict and conflict.band in {CONFLICT_HIGH, CONFLICT_CRITICAL}:
            alpha, action = 0.50 + 0.35 * (alpha - 0.50), "ties"

        fa = _scalar_evidence(fisher_a, name)
        fb = _scalar_evidence(fisher_b, name)
        aa = _scalar_evidence(activation_a, name)
        ab = _scalar_evidence(activation_b, name)
        if fa > 0 and fb > 0:
            ratio = fa / max(fa + fb, _EPS)
            alpha = 0.65 * alpha + 0.35 * ratio
            if abs(fa - fb) / max(fa + fb, _EPS) > 0.50:
                action = "fisher"
        if aa > 0 and ab > 0:
            alpha = 0.80 * alpha + 0.20 * (aa / max(aa + ab, _EPS))

        protection = _protection_for_category(rec.category)
        if protection:
            alpha = (1.0 - protection) * alpha + protection * 0.50
        alpha = _clamp(alpha, _ALPHA_MIN, _ALPHA_MAX)
        importance = _clamp01(rec.importance)

        if action not in {"fisher", "ties", "slerp"} and alpha >= 0.93 and confidence >= 0.82 and tc <= 0.20 and protection <= 0.10:
            action = "keep_a"
        elif action not in {"fisher", "ties", "slerp"} and alpha <= 0.07 and confidence >= 0.82 and tc <= 0.20 and protection <= 0.10:
            action = "keep_b"

        if action == "keep_a":
            reason, risk = "high-confidence A dominance with low measured conflict", "low"
        elif action == "keep_b":
            reason, risk = "high-confidence B dominance with low measured conflict", "low"
        elif conflict and conflict.band == CONFLICT_CRITICAL:
            reason, risk = "critical tensor conflict; use conflict-aware sign consensus", "high"
        elif action == "ties":
            reason, risk = "high tensor conflict; conflict-aware merge requested", "medium"
        elif action == "slerp":
            reason, risk = "high structural similarity; spherical interpolation is appropriate", "low"
        elif action == "fisher":
            reason, risk = "Fisher importance differs strongly between parents", "medium"
        elif protection:
            reason, risk = f"sensitive {rec.category} parameter; conservative routing", "medium"
        else:
            reason, risk = "layer-aware importance + conflict routing", "low"

        alignment = "none"
        strength = 0.0
        if tc >= 0.65 and rec.category in {"attention", "mlp"}:
            alignment, strength = "orthogonal_review", _clamp01(tc)
        elif tc >= 0.35 and rec.role in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
            alignment, strength = "basis_review", _clamp01(tc)

        directives[name] = CBADirective(
            alpha_a=round(alpha, 5), action=action, conflict=round(tc, 5), reason=reason,
            confidence=round(confidence, 5), importance=round(importance, 5), protection=round(protection, 5),
            similarity=round(similarity, 5), norm_ratio=round(norm_ratio, 5), fisher_a=round(fa, 8), fisher_b=round(fb, 8),
            activation_a=round(aa, 8), activation_b=round(ab, 8), evidence_score=round(evidence, 5), risk=risk,
            alignment=alignment, alignment_strength=round(strength, 5), layer_key=lk,
        )

    mean_alpha = _mean(layer_alphas.values(), 0.50)
    dominant = "A" if mean_alpha >= 0.58 else "B" if mean_alpha <= 0.42 else "balanced"
    return CBARouting(
        layer_alphas=layer_alphas, category_alphas=category_alphas,
        layer_actions=layer_actions, layer_conflicts=layer_conflicts,
        dominant_model=dominant, directives=directives,
        layer_confidence=layer_confidence, protected_layers=sorted(set(protected)),
        routing_evidence=routing_evidence,
        category_actions=category_actions, category_conflicts=category_conflicts,
        category_confidence=category_confidence,
    )


def _policy_stability(routing: CBARouting) -> float:
    ordered = []
    for k, a in routing.layer_alphas.items():
        m = re.fullmatch(r"layer_(\d+)", k)
        if m:
            ordered.append((int(m.group(1)), a))
    ordered.sort()
    if len(ordered) < 2:
        return 0.5
    changes = [abs(b[1] - a[1]) for a, b in zip(ordered[:-1], ordered[1:])]
    return _clamp01(1.0 - 2.0 * _mean(changes))


def _build_decision(
    compatibility: CompatibilityReport,
    conflicts: ConflictReport,
    routing: CBARouting,
    ia: BrainInspection,
    ib: BrainInspection,
) -> CBADecision:
    stability = _policy_stability(routing)
    confidence = _clamp01(
        0.32 * compatibility.structural_confidence
        + 0.26 * conflicts.mean_confidence
        + 0.22 * stability
        + 0.20 * (1.0 - conflicts.global_score)
    )
    evidence = [
        f"mergeability={compatibility.mergeability:.1f}/100",
        f"correspondence coverage={compatibility.correspondence_ratio:.0%}",
        f"structural confidence={compatibility.structural_confidence:.2f}",
        f"global conflict={conflicts.global_score:.2f} ({conflicts.global_band})",
        f"conflict tensor ratio={conflicts.conflict_tensor_ratio:.0%}",
        f"routing stability={stability:.2f}",
    ]
    decisive = sorted(routing.layer_alphas.items(), key=lambda x: abs(x[1] - 0.50), reverse=True)[:5]
    evidence.extend(
        f"{k}: A={a:.2f} B={1-a:.2f} conflict={routing.layer_conflicts.get(k,0.0):.2f}"
        for k, a in decisive
    )
    if conflicts.critical_conflict_layers:
        risk = "critical conflict exists in " + ", ".join(conflicts.critical_conflict_layers[:4])
    elif not compatibility.compatible:
        risk = "structural compatibility is insufficient for a high-confidence static merge"
    elif conflicts.high_conflict_layers:
        risk = "high-conflict regions exist; conflict-aware directives are required"
    else:
        risk = "no critical weight-space conflict detected; task-level validation is still required"

    alternative = (
        "Preserve A as the stronger baseline and let B contribute selectively."
        if routing.dominant_model == "A"
        else "Use A as the protected baseline and allow B to contribute where evidence supports it."
        if routing.dominant_model == "B"
        else "Use balanced routing and adjust only after post-merge evaluation."
    )
    weaknesses = []
    if compatibility.correspondence_ratio < 0.80: weaknesses.append("incomplete parameter correspondence")
    if compatibility.hidden_size_a is not None and compatibility.hidden_size_b is not None and compatibility.hidden_size_a != compatibility.hidden_size_b: weaknesses.append("hidden-size mismatch")
    if conflicts.global_score >= 0.75: weaknesses.append("high global tensor conflict")
    if stability < 0.50: weaknesses.append("unstable layer routing")
    if ia.nonfinite_tensors or ib.nonfinite_tensors: weaknesses.append("non-finite tensors present")
    critique = {
        "potential_weakness": "; ".join(weaknesses) or "none identified",
        "missing_evidence": "activation-level behavior; task benchmarks; post-merge validation",
        "policy_stability": stability,
        "recommendation": "evaluate the final merged model against both parent models before deployment",
    }
    what = f"CBA inspected {ia.total_tensors} A tensors and {ib.total_tensors} B tensors; correspondence={compatibility.correspondence_ratio:.0%}; conflict={conflicts.global_band} ({conflicts.global_score:.2f})."
    why = (
        f"Strongest routing deviation is {decisive[0][0]} (A={decisive[0][1]:.2f}, B={1-decisive[0][1]:.2f})."
        if decisive else "Routing evidence is broadly balanced."
    )
    next_step = "apply per-tensor directives; inspect high-conflict regions; evaluate against both parents"
    recommended = "ties" if conflicts.critical_conflict_layers else "slerp" if abs(_mean(routing.layer_alphas.values(), 0.5) - 0.5) < 0.05 else "weighted"
    return CBADecision(
        decision="cba_routed_merge", confidence=round(confidence, 4), evidence=evidence, risk=risk,
        alternative=alternative, critique=critique, questions={"what": what, "why": why, "next": next_step},
        dominant_model=routing.dominant_model, recommended_action=recommended,
        policy_stability=round(stability, 4), measured_tensors=len(conflicts.per_tensor),
        high_conflict_regions=len(conflicts.high_conflict_layers),
        critical_conflict_regions=len(conflicts.critical_conflict_layers),
        structural_confidence=round(compatibility.structural_confidence, 4),
    )


def run_cba(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    *,
    high_conflict_threshold: float = _CONFLICT_HIGH_CUTOFF_DEFAULT,
    vocab_size_b: Optional[int] = None,
    fisher_a: Optional[Mapping[str, torch.Tensor]] = None,
    fisher_b: Optional[Mapping[str, torch.Tensor]] = None,
    activation_a: Optional[Mapping[str, Any]] = None,
    activation_b: Optional[Mapping[str, Any]] = None,
    activation_layer_a: Optional[Mapping[str, Any]] = None,
    activation_layer_b: Optional[Mapping[str, Any]] = None,
) -> CBAReport:
    ia = inspect_state_dict(state_dict_a)
    ib = inspect_state_dict(state_dict_b)
    correspondence = build_correspondence(state_dict_a, state_dict_b, ia, ib)
    compatibility = check_compatibility(
        ia, ib, state_dict_a, state_dict_b,
        correspondence=correspondence, vocab_size_b=vocab_size_b,
    )
    conflicts = analyze_conflicts(
        state_dict_a, state_dict_b, ia, ib,
        correspondence=correspondence, high_cutoff=high_conflict_threshold,
    )
    routing = plan_routing(
        ia, ib, conflicts, compatibility,
        high_conflict_threshold=high_conflict_threshold,
        fisher_a=fisher_a, fisher_b=fisher_b,
        activation_a=activation_a, activation_b=activation_b,
        activation_layer_a=activation_layer_a, activation_layer_b=activation_layer_b,
    )
    for target, match in correspondence.items():
        if target in routing.directives:
            routing.directives[target].source_key = match.source_name
    decision = _build_decision(compatibility, conflicts, routing, ia, ib)
    return CBAReport(ia, ib, compatibility, conflicts, routing, decision, correspondence)


def answer_three_questions(context: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")
    routing, conflicts = context.get("routing"), context.get("conflicts")
    compatibility = context.get("compatibility")
    if isinstance(routing, Mapping) and isinstance(conflicts, Mapping):
        alphas = routing.get("layer_alphas", {}) or {}
        dominant = routing.get("dominant_model", "balanced")
        score = _safe_float(conflicts.get("global_score", 0.0))
        band = conflicts.get("global_band", "low")
        mean_alpha = _mean(float(v) for v in alphas.values()) if alphas else 0.5
        what = f"CBA analyzed {len(alphas)} layer groups; routing={dominant}; mean A contribution={mean_alpha:.2f}; global conflict={band} ({score:.2f})."
        decisive = max(alphas.items(), key=lambda x: abs(float(x[1]) - 0.5), default=None)
        why = f"{decisive[0]} has the strongest routing split (A={float(decisive[1]):.2f})." if decisive else "Routing evidence is balanced."
        next_step = "apply directives and evaluate the merged model against both parents."
        if isinstance(compatibility, Mapping) and _safe_float(compatibility.get("mergeability", 0.0)) < 60.0:
            next_step = "validate architecture compatibility first; then apply directives and evaluate both parents."
        return {"what": what, "why": why, "next": next_step}
    return {
        "what": "CBA received incomplete routing context.",
        "why": "There is not enough evidence for a detailed explanation.",
        "next": "Run run_cba(...) to produce a complete CBA report.",
    }


def critique_decision_record(decision: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping")
    claimed = _clamp01(_safe_float(decision.get("confidence", 0.0)))
    evidence = decision.get("evidence")
    count = len(evidence) if isinstance(evidence, (list, tuple)) else 0
    ceiling = _clamp01(0.35 + 0.05 * count)
    weakness = []
    if claimed > ceiling:
        weakness.append(f"claimed confidence {claimed:.2f} exceeds evidence ceiling {ceiling:.2f}")
    if count == 0:
        weakness.append("no evidence items attached")
    return {
        "claimed_confidence": round(claimed, 4),
        "justified_confidence": round(ceiling, 4),
        "confidence": round(min(claimed, ceiling), 4),
        "potential_weakness": "; ".join(weakness) or "none identified",
        "missing_evidence": "activation statistics; task benchmarks; post-merge validation",
        "recommendation": "treat CBA as a weight-space controller until behavior-level evidence confirms the result",
    }
