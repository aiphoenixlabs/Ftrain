"""
FTRAIN CBA — Captain Brain Alignment (Phase 1)
==============================================

CBA is the central intelligence layer for model merging. It sits between
"load two state dicts" and "merge them" and replaces a blind global
``A = 0.5 / B = 0.5`` with a data-driven, per-layer routing decision.

Pipeline (mirrors the FTRAIN CBA diagram):

    MODEL A state dict ─┐
                        ├─ Brain Inspection
    MODEL B state dict ─┘
                        ↓
              Compatibility Analysis
                        ↓
                Conflict Detection
                        ↓
        Knowledge Importance Analysis
                        ↓
                CBA Alignment/Routing
                        ↓
           Captain Decision + Self-Critique
                        ↓
        Three Questions (WHAT / WHY / NEXT)
                        ↓
          CBADirectives consumed by Merger

Design rules honored by this module:

- Deterministic: identical inputs produce identical outputs. No RNG.
- Data-driven: every alpha is computed from measured weight statistics.
- Honest: every decision ships with confidence, evidence, risk and an
  alternative. When evidence is weak, confidence is low — never faked.
- Captain-optional: works fully without an LLM Captain. When a Captain
  exists, its judgement layers are added on top, never instead.
- No silent fallback: callers decide what happens when CBA fails.

This module is intentionally independent of ``captain`` to avoid circular
imports; ``captain`` imports CBA lazily.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "TensorRecord",
    "LayerRecord",
    "BrainInspection",
    "CompatibilityReport",
    "ConflictReport",
    "CBARouting",
    "CBADirective",
    "CBADecision",
    "CBAReport",
    "inspect_state_dict",
    "check_compatibility",
    "analyze_conflicts",
    "plan_routing",
    "run_cba",
    "answer_three_questions",
    "critique_decision_record",
]


# =============================================================================
# Tunables (fixed constants keep behavior deterministic across runs)
# =============================================================================

_FLATTEN_CAP = 262_144          # max elements inspected per tensor pair
_EPS = 1e-12

CONFLICT_LOW = "low"
CONFLICT_MEDIUM = "medium"
CONFLICT_HIGH = "high"

# Conflict score thresholds for the three bands.
_CONFLICT_MEDIUM_CUTOFF = 0.35
_CONFLICT_HIGH_CUTOFF_DEFAULT = 0.65

_ROUTING_GAIN = 0.55            # evidence -> alpha sensitivity
_ROUTING_MIN_ALPHA = 0.05
_ROUTING_MAX_ALPHA = 0.95
_HIGH_SIMILARITY = 0.995        # near-identical tensors
_DIRECTION_SIMILAR = 0.995


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TensorRecord:
    """Measured statistics for a single tensor of a single model."""

    name: str
    category: str
    layer: Optional[int]
    numel: int
    mean: float
    std: float
    l2_norm: float
    rel_norm: float          # share of the model's total L2 mass


@dataclass
class LayerRecord:
    """Aggregated per-layer (or per-category) view of one model."""

    key: str                 # e.g. "layer_12" or "category:attention"
    kind: str                # "layer" | "category"
    index: Optional[int]
    category: str            # dominant category for category-records
    tensors: int
    numel: int
    l2_norm: float
    rel_norm: float          # layer norm / model norm
    mean_std: float


@dataclass
class BrainInspection:
    """Result of inspecting one model's state dict."""

    total_tensors: int
    total_numel: int
    total_norm: float
    layer_count: int
    hidden_size: Optional[int]
    vocab_size: Optional[int]
    dtypes: Dict[str, int]                      # dtype name -> count
    tensors: Dict[str, TensorRecord]            # name -> record
    layers: Dict[str, LayerRecord]              # "layer_N" -> record
    categories: Dict[str, LayerRecord]          # "category_x" -> record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tensors": self.total_tensors,
            "total_numel": self.total_numel,
            "total_norm": self.total_norm,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "dtypes": dict(self.dtypes),
            "layers": {
                key: {
                    "tensors": rec.tensors,
                    "numel": rec.numel,
                    "l2_norm": rec.l2_norm,
                    "rel_norm": rec.rel_norm,
                }
                for key, rec in self.layers.items()
            },
            "categories": {
                key: {
                    "tensors": rec.tensors,
                    "rel_norm": rec.rel_norm,
                }
                for key, rec in self.categories.items()
            },
        }


@dataclass
class CompatibilityReport:
    """Architecture/tokenizer compatibility verdict for a model pair."""

    compatible: bool
    mergeability: float                     # 0..100
    name_overlap: float                     # Jaccard of parameter names
    shape_match_ratio: float                # matching shapes on common names
    layer_count_a: int
    layer_count_b: int
    hidden_size_a: Optional[int]
    hidden_size_b: Optional[int]
    dtype_compatible: bool
    tokenizer_compatible: Optional[bool]    # None when unknown
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "mergeability": self.mergeability,
            "name_overlap": self.name_overlap,
            "shape_match_ratio": self.shape_match_ratio,
            "layer_count_a": self.layer_count_a,
            "layer_count_b": self.layer_count_b,
            "hidden_size_a": self.hidden_size_a,
            "hidden_size_b": self.hidden_size_b,
            "dtype_compatible": self.dtype_compatible,
            "tokenizer_compatible": self.tokenizer_compatible,
            "notes": list(self.notes),
        }


@dataclass
class TensorConflict:
    """Conflict measurements for one common tensor."""

    name: str
    layer: Optional[int]
    category: str
    direction_similarity: float   # cosine(A, B); 1.0 = same direction
    sign_disagreement: float      # fraction of opposite-sign deviations
    relative_delta: float         # ||A-B|| / (||A||+||B||)
    score: float                  # 0..1 composite
    band: str                     # low | medium | high


@dataclass
class LayerConflict:
    key: str
    kind: str
    score: float
    band: str
    tensors: int


@dataclass
class ConflictReport:
    """Layer-level conflict profile for a model pair."""

    global_score: float
    global_band: str
    high_conflict_layers: List[str]
    per_tensor: Dict[str, TensorConflict]
    per_layer: Dict[str, LayerConflict]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_score": self.global_score,
            "global_band": self.global_band,
            "high_conflict_layers": list(self.high_conflict_layers),
            "per_layer": {
                key: {"score": rec.score, "band": rec.band, "tensors": rec.tensors}
                for key, rec in self.per_layer.items()
            },
        }


@dataclass
class CBADirective:
    """Per-tensor merge instruction produced by CBA."""

    alpha_a: float
    action: str                  # "weighted" | "ties" | "keep_a" | "keep_b"
    conflict: float
    reason: str


@dataclass
class CBARouting:
    """The alignment plan: per-layer and per-category A/B weights."""

    layer_alphas: Dict[str, float]           # "layer_12" -> alpha_a
    category_alphas: Dict[str, float]        # "attention" -> alpha_a
    layer_actions: Dict[str, str]
    layer_conflicts: Dict[str, float]
    dominant_model: str                      # "A" | "B" | "balanced"
    directives: Dict[str, CBADirective]      # per tensor name

    def alpha_summary(self) -> str:
        """Human-readable routing summary in the CBA showcase format."""
        lines: List[str] = []
        for key in sorted(
            self.layer_alphas,
            key=lambda item: int(item.split("_")[-1]) if item.startswith("layer_") else -1,
        ):
            alpha = self.layer_alphas[key]
            lines.append(f"{key}: A {alpha:.2f}  B {1.0 - alpha:.2f}")
        for key, alpha in sorted(self.category_alphas.items()):
            lines.append(f"{key}: A {alpha:.2f}  B {1.0 - alpha:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_alphas": dict(self.layer_alphas),
            "category_alphas": dict(self.category_alphas),
            "layer_actions": dict(self.layer_actions),
            "layer_conflicts": dict(self.layer_conflicts),
            "dominant_model": self.dominant_model,
        }


@dataclass
class CBADecision:
    """A Captain-grade decision record with honest confidence."""

    decision: str
    confidence: float                     # 0..1
    evidence: List[str]
    risk: str
    alternative: str
    critique: Dict[str, Any]              # self-criticism block
    questions: Dict[str, str]             # WHAT / WHY / NEXT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "risk": self.risk,
            "alternative": self.alternative,
            "critique": dict(self.critique),
            "questions": dict(self.questions),
        }


@dataclass
class CBAReport:
    """Full CBA analysis bundle (also the JSON export format)."""

    inspection_a: BrainInspection
    inspection_b: BrainInspection
    compatibility: CompatibilityReport
    conflicts: ConflictReport
    routing: CBARouting
    decision: CBADecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cba_version": 1,
            "inspection_a": self.inspection_a.to_dict(),
            "inspection_b": self.inspection_b.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "conflicts": self.conflicts.to_dict(),
            "routing": self.routing.to_dict(),
            "decision": self.decision.to_dict(),
        }

    def save(self, path: str) -> str:
        """Atomically write the report as JSON. Never overwrites silently."""
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=str)

        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".cba_", suffix=".json"
        )
        try:
            if os.path.exists(path):
                raise FileExistsError(
                    f"CBA report target already exists: {path}. "
                    "FTRAIN never overwrites existing reports silently."
                )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return path


# =============================================================================
# Classification helpers
# =============================================================================

_LAYER_RE = re.compile(r"layers[._](\d+)")


def _classify(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered or "tok_embeddings" in lowered:
        return "embedding"
    if "lm_head" in lowered or lowered.startswith(("output.", "head.")):
        return "head"
    if ".mlp." in lowered or ".feed_forward" in lowered or ".experts." in lowered:
        return "mlp"
    if ".self_attn." in lowered or ".attention." in lowered or ".attn." in lowered:
        return "attention"
    if "norm" in lowered:
        return "norm"
    if "layers" not in lowered:
        return "head" if ("head" in lowered or "output" in lowered) else "other"
    return "other"


def _layer_of(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _band(score: float, high_cutoff: float) -> str:
    if score >= high_cutoff:
        return CONFLICT_HIGH
    if score >= _CONFLICT_MEDIUM_CUTOFF:
        return CONFLICT_MEDIUM
    return CONFLICT_LOW


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _flat_sample(tensor: torch.Tensor) -> torch.Tensor:
    """Deterministic flattened sample of a tensor (stride-based, no RNG)."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= _FLATTEN_CAP:
        return flat.to(torch.float32)
    stride = int(math.ceil(flat.numel() / _FLATTEN_CAP))
    return flat[::stride].to(torch.float32)


# =============================================================================
# Stage 1 — Brain Inspection
# =============================================================================

def inspect_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> BrainInspection:
    """Measure the "brain" of a model from its state dict."""
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"state_dict must be a mapping, got {type(state_dict).__name__}."
        )

    tensors: Dict[str, TensorRecord] = {}
    layers: Dict[str, LayerRecord] = {}
    categories: Dict[str, LayerRecord] = {}
    dtypes: Dict[str, int] = {}

    total_norm_sq = 0.0
    total_numel = 0
    layer_indices: set = set()
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        dtype_name = str(tensor.dtype).replace("torch.", "")
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1

        flat = _flat_sample(tensor)
        numel = int(tensor.numel())
        norm = float(torch.linalg.norm(flat)) if numel else 0.0
        total_norm_sq += norm * norm
        total_numel += numel

        category = _classify(name)
        layer = _layer_of(name)
        if layer is not None:
            layer_indices.add(layer)

        if name.endswith("embed_tokens.weight") or "tok_embeddings" in name:
            if tensor.dim() == 2:
                vocab_size = int(tensor.shape[0])
                hidden_size = int(tensor.shape[1])
        elif name.endswith("lm_head.weight") and tensor.dim() == 2 and hidden_size is None:
            hidden_size = int(tensor.shape[1])

        tensors[name] = TensorRecord(
            name=name,
            category=category,
            layer=layer,
            numel=numel,
            mean=float(flat.mean()) if numel else 0.0,
            std=float(flat.std()) if numel > 1 else 0.0,
            l2_norm=norm,
            rel_norm=0.0,
        )

    total_norm = math.sqrt(max(total_norm_sq, 0.0))

    for record in tensors.values():
        record.rel_norm = record.l2_norm / max(total_norm, _EPS)

    def _aggregate(
        key: str,
        kind: str,
        index: Optional[int],
        category: str,
        names: List[str],
    ) -> LayerRecord:
        numel = sum(tensors[n].numel for n in names)
        norm = math.sqrt(sum(tensors[n].l2_norm ** 2 for n in names))
        return LayerRecord(
            key=key,
            kind=kind,
            index=index,
            category=category,
            tensors=len(names),
            numel=numel,
            l2_norm=norm,
            rel_norm=norm / max(total_norm, _EPS),
            mean_std=float(sum(tensors[n].std for n in names) / max(1, len(names))),
        )

    by_layer: Dict[int, List[str]] = {}
    by_category: Dict[str, List[str]] = {}
    for name, record in tensors.items():
        if record.layer is not None:
            by_layer.setdefault(record.layer, []).append(name)
        by_category.setdefault(record.category, []).append(name)

    for layer, names in sorted(by_layer.items()):
        dominant = max(
            { _classify(n) for n in names },
            key=lambda c: sum(1 for n in names if _classify(n) == c),
        )
        rec = _aggregate(f"layer_{layer}", "layer", layer, dominant, names)
        layers[rec.key] = rec

    for category, names in sorted(by_category.items()):
        rec = _aggregate(category, "category", None, category, names)
        categories[category] = rec

    return BrainInspection(
        total_tensors=len(tensors),
        total_numel=total_numel,
        total_norm=total_norm,
        layer_count=len(layer_indices),
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtypes=dtypes,
        tensors=tensors,
        layers=layers,
        categories=categories,
    )


# =============================================================================
# Stage 2 — Compatibility Analysis
# =============================================================================

def check_compatibility(
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    state_dict_a: Optional[Mapping[str, torch.Tensor]] = None,
    state_dict_b: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    vocab_size_b: Optional[int] = None,
) -> CompatibilityReport:
    """Decide whether two brains can be merged and how mergeable they are."""
    names_a = set(inspection_a.tensors)
    names_b = set(inspection_b.tensors)

    union = names_a | names_b
    intersection = names_a & names_b
    name_overlap = len(intersection) / max(1, len(union))

    shape_match = 0
    shape_total = 0
    for name in intersection:
        if state_dict_a is None or state_dict_b is None:
            break
        ta = state_dict_a.get(name)
        tb = state_dict_b.get(name)
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            continue
        shape_total += 1
        if tuple(ta.shape) == tuple(tb.shape):
            shape_match += 1

    shape_ratio = (shape_match / shape_total) if shape_total else (
        1.0 if name_overlap > 0.95 else 0.5
    )

    hidden_match = (
        inspection_a.hidden_size is not None
        and inspection_a.hidden_size == inspection_b.hidden_size
    )

    dtype_compatible = bool(inspection_a.dtypes and inspection_b.dtypes)

    layer_delta = abs(inspection_a.layer_count - inspection_b.layer_count)
    layer_closeness = 1.0 if inspection_a.layer_count == inspection_b.layer_count else (
        1.0 / (1.0 + layer_delta / max(1, max(inspection_a.layer_count, inspection_b.layer_count)))
    )

    tokenizer_compatible: Optional[bool] = None
    if vocab_size_b is not None and inspection_a.vocab_size is not None:
        diff = abs(inspection_a.vocab_size - vocab_size_b)
        tokenizer_compatible = diff <= max(256, int(0.01 * max(inspection_a.vocab_size, vocab_size_b)))

    notes: List[str] = []
    if not hidden_match:
        notes.append(
            f"hidden size mismatch: A={inspection_a.hidden_size} "
            f"B={inspection_b.hidden_size}"
        )
    if layer_delta:
        notes.append(f"layer count differs by {layer_delta}")
    if tokenizer_compatible is False:
        notes.append("tokenizer vocabularies differ beyond tolerance")
    if shape_total and shape_match < shape_total:
        notes.append(f"{shape_total - shape_match} shared tensors have different shapes")

    weights_total = 0.0
    score = 0.0
    contributions = (
        (name_overlap, 0.25),
        (shape_ratio, 0.35),
        (layer_closeness, 0.15),
        (1.0 if hidden_match else 0.0, 0.15),
        (1.0 if dtype_compatible else 0.0, 0.05),
        (1.0 if tokenizer_compatible is None else (1.0 if tokenizer_compatible else 0.0), 0.05),
    )
    for value, weight in contributions:
        score += value * weight
        weights_total += weight

    mergeability = 100.0 * score / weights_total
    compatible = (
        name_overlap > 0.5
        and shape_ratio > 0.9
        and hidden_match
    )

    return CompatibilityReport(
        compatible=compatible,
        mergeability=round(mergeability, 1),
        name_overlap=round(name_overlap, 4),
        shape_match_ratio=round(shape_ratio, 4),
        layer_count_a=inspection_a.layer_count,
        layer_count_b=inspection_b.layer_count,
        hidden_size_a=inspection_a.hidden_size,
        hidden_size_b=inspection_b.hidden_size,
        dtype_compatible=dtype_compatible,
        tokenizer_compatible=tokenizer_compatible,
        notes=notes,
    )


# =============================================================================
# Stage 3 — Conflict Detection
# =============================================================================

def _tensor_conflict(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    high_cutoff: float,
) -> Optional[TensorConflict]:
    ta = _flat_sample(a)
    tb = _flat_sample(b)

    if ta.numel() != tb.numel() or ta.numel() == 0:
        return None

    norm_a = float(torch.linalg.norm(ta))
    norm_b = float(torch.linalg.norm(tb))

    unit_a = ta / max(norm_a, _EPS)
    unit_b = tb / max(norm_b, _EPS)
    direction = float(torch.dot(unit_a, unit_b))
    direction_sim = _clamp01((direction + 1.0) / 2.0)  # -1..1 -> 0..1

    relative_delta = float(torch.linalg.norm(ta - tb)) / max(norm_a + norm_b, _EPS)

    centered_a = ta - ta.mean()
    centered_b = tb - tb.mean()
    denom = (centered_a.abs() + centered_b.abs()) > 1e-8
    if bool(denom.any()):
        sign_disagreement = float(
            ((torch.sign(centered_a) * torch.sign(centered_b)) < 0)[denom]
            .to(torch.float32)
            .mean()
        )
    else:
        sign_disagreement = 0.0

    score = _clamp01(
        0.5 * (1.0 - direction_sim)
        + 0.3 * sign_disagreement
        + 0.2 * min(1.0, relative_delta)
    )

    return TensorConflict(
        name=name,
        layer=_layer_of(name),
        category=_classify(name),
        direction_similarity=round(direction_sim, 6),
        sign_disagreement=round(sign_disagreement, 6),
        relative_delta=round(relative_delta, 6),
        score=round(score, 6),
        band=_band(score, high_cutoff),
    )


def analyze_conflicts(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    *,
    high_cutoff: float = _CONFLICT_HIGH_CUTOFF_DEFAULT,
) -> ConflictReport:
    """Detect LOW/MEDIUM/HIGH conflict per tensor and per layer."""
    if not 0.0 < high_cutoff <= 1.0:
        raise ValueError(
            f"high_cutoff must be in (0, 1], got {high_cutoff}."
        )

    per_tensor: Dict[str, TensorConflict] = {}
    by_key: Dict[str, List[TensorConflict]] = {}

    for name, tensor_a in state_dict_a.items():
        tensor_b = state_dict_b.get(name)
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            continue
        if tuple(tensor_a.shape) != tuple(tensor_b.shape):
            continue

        conflict = _tensor_conflict(name, tensor_a, tensor_b, high_cutoff)
        if conflict is None:
            continue

        per_tensor[name] = conflict
        if conflict.layer is not None:
            key = f"layer_{conflict.layer}"
            kind = "layer"
        else:
            key = conflict.category
            kind = "category"
        by_key.setdefault(key, []).append(conflict)

    per_layer: Dict[str, LayerConflict] = {}
    for key, conflicts in sorted(by_key.items()):
        score = sum(c.score for c in conflicts) / len(conflicts)
        per_layer[key] = LayerConflict(
            key=key,
            kind="layer" if key.startswith("layer_") else "category",
            score=round(score, 6),
            band=_band(score, high_cutoff),
            tensors=len(conflicts),
        )

    global_score = (
        sum(rec.score for rec in per_tensor.values()) / max(1, len(per_tensor))
        if per_tensor
        else 0.0
    )

    high_layers = [
        key
        for key, rec in sorted(per_layer.items())
        if rec.band == CONFLICT_HIGH
    ]

    return ConflictReport(
        global_score=round(global_score, 6),
        global_band=_band(global_score, high_cutoff),
        high_conflict_layers=high_layers,
        per_tensor=per_tensor,
        per_layer=per_layer,
    )


# =============================================================================
# Stage 4+5 — Importance + CBA Routing
# =============================================================================

def _importance_balance(inspection_a: BrainInspection, inspection_b: BrainInspection, key: str) -> float:
    """+1 favors A, -1 favors B, based on relative norm mass."""
    rec_a = inspection_a.layers.get(key) or inspection_a.categories.get(key)
    rec_b = inspection_b.layers.get(key) or inspection_b.categories.get(key)
    if rec_a is None or rec_b is None:
        return 0.0
    ratio = math.log((rec_a.rel_norm + _EPS) / (rec_b.rel_norm + _EPS))
    return math.tanh(ratio * 2.0)


def _evidence_for_key(
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    conflicts: ConflictReport,
    key: str,
) -> float:
    """Composite evidence in [-1, 1]; positive favors model A."""
    balance = _importance_balance(inspection_a, inspection_b, key)

    conflict_rec = conflicts.per_layer.get(key)
    conflict_score = conflict_rec.score if conflict_rec else conflicts.global_score

    layer_a = inspection_a.layers.get(key) or inspection_a.categories.get(key)
    layer_b = inspection_b.layers.get(key) or inspection_b.categories.get(key)
    specialization = 0.0
    if layer_a is not None and layer_b is not None and layer_a.mean_std > 0 and layer_b.mean_std > 0:
        # A layer whose weights vary more within itself carries more
        # specialized structure; treat relative dispersion as a weak signal.
        specialization = math.tanh(
            math.log((layer_a.mean_std + _EPS) / (layer_b.mean_std + _EPS))
        )

    evidence = 0.55 * balance + 0.25 * specialization
    # High conflict pulls routing toward neutrality — the safest response to
    # disagreement is consensus (handled via action="ties") not conviction.
    evidence *= (1.0 - 0.5 * _clamp01(conflict_score))
    return max(-1.0, min(1.0, evidence))


def plan_routing(
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
    conflicts: ConflictReport,
    compatibility: CompatibilityReport,
    *,
    high_conflict_threshold: float = _CONFLICT_HIGH_CUTOFF_DEFAULT,
) -> CBARouting:
    """Produce per-layer / per-category A/B weights and per-tensor directives."""
    layer_alphas: Dict[str, float] = {}
    category_alphas: Dict[str, float] = {}
    layer_actions: Dict[str, str] = {}
    layer_conflicts: Dict[str, float] = {}
    directives: Dict[str, CBADirective] = {}

    keys = sorted(
        set(inspection_a.layers) | set(inspection_b.layers)
    )

    for key in keys:
        evidence = _evidence_for_key(inspection_a, inspection_b, conflicts, key)
        alpha_a = 0.5 + _ROUTING_GAIN * evidence

        conflict_rec = conflicts.per_layer.get(key)
        conflict_score = conflict_rec.score if conflict_rec else 0.0

        if conflict_rec is not None and conflict_rec.band == CONFLICT_HIGH:
            action = "ties"
            alpha_a = 0.5 + (alpha_a - 0.5) * 0.5
        else:
            action = "weighted"

        depth_factor = 1.0
        if key.startswith("layer_"):
            index = int(key.split("_")[-1])
            max_layers = max(1, max(inspection_a.layer_count, inspection_b.layer_count))
            # Early layers carry more universal features: stay conservative.
            depth_factor = 0.85 + 0.15 * (index / max_layers)
            alpha_a = 0.5 + (alpha_a - 0.5) * depth_factor

        alpha_a = max(_ROUTING_MIN_ALPHA, min(_ROUTING_MAX_ALPHA, alpha_a))

        layer_alphas[key] = round(alpha_a, 4)
        layer_actions[key] = action
        layer_conflicts[key] = round(conflict_score, 4)

    for category, rec_a in sorted(inspection_a.categories.items()):
        rec_b = inspection_b.categories.get(category)
        if rec_b is None:
            continue
        evidence = _evidence_for_key(inspection_a, inspection_b, conflicts, category)
        alpha_a = max(_ROUTING_MIN_ALPHA, min(_ROUTING_MAX_ALPHA, 0.5 + _ROUTING_GAIN * evidence))
        category_alphas[category] = round(alpha_a, 4)

    # Per-tensor directives: layer routing first, category fallback.
    for name, record in inspection_a.tensors.items():
        if record.layer is not None:
            key = f"layer_{record.layer}"
            alpha_a = layer_alphas.get(key)
            action = layer_actions.get(key, "weighted")
            conflict = layer_conflicts.get(key, 0.0)
        else:
            alpha_a = category_alphas.get(record.category)
            action = "weighted"
            conflict = conflicts.per_tensor[name].score if name in conflicts.per_tensor else 0.0

        if alpha_a is None:
            continue

        conflict_rec = conflicts.per_tensor.get(name)
        if conflict_rec is not None:
            if conflict_rec.direction_similarity >= _DIRECTION_SIMILAR:
                action = "weighted"          # identical direction: any alpha works
                alpha_a = 0.5
            elif conflict_rec.band == CONFLICT_HIGH and action == "weighted":
                action = "ties"

        directives[name] = CBADirective(
            alpha_a=float(alpha_a),
            action=action,
            conflict=float(conflict),
            reason=(
                f"{record.category} routing; conflict={conflict:.3f}; "
                f"evidence_driven_alpha={alpha_a:.3f}"
            ),
        )

    mean_alpha = (
        sum(layer_alphas.values()) / len(layer_alphas)
        if layer_alphas
        else 0.5
    )
    if mean_alpha > 0.58:
        dominant = "A"
    elif mean_alpha < 0.42:
        dominant = "B"
    else:
        dominant = "balanced"

    return CBARouting(
        layer_alphas=layer_alphas,
        category_alphas=category_alphas,
        layer_actions=layer_actions,
        layer_conflicts=layer_conflicts,
        dominant_model=dominant,
        directives=directives,
    )


# =============================================================================
# Stage 6 — Captain Decision, Self-Critique, Three Questions
# =============================================================================

def _confidence(
    compatibility: CompatibilityReport,
    conflicts: ConflictReport,
    routing: CBARouting,
) -> float:
    """Honest confidence in [0, 1]. Low evidence => low confidence."""
    if not routing.layer_alphas:
        return 0.15

    count = len(routing.layer_alphas)
    volume = min(1.0, count / 12.0)

    margins = [abs(alpha - 0.5) for alpha in routing.layer_alphas.values()]
    mean_margin = sum(margins) / len(margins)
    decisiveness = min(1.0, mean_margin / 0.2)

    measured = min(1.0, len(conflicts.per_tensor) / max(1, count * 3))
    structural = compatibility.mergeability / 100.0

    confidence = 0.30 * volume + 0.25 * decisiveness + 0.25 * measured + 0.20 * structural
    return round(max(0.05, min(0.95, confidence)), 3)


def build_decision(
    compatibility: CompatibilityReport,
    conflicts: ConflictReport,
    routing: CBARouting,
    inspection_a: BrainInspection,
    inspection_b: BrainInspection,
) -> CBADecision:
    """Assemble decision + confidence + evidence + risk + alternative."""
    evidence: List[str] = []

    evidence.append(
        f"mergeability {compatibility.mergeability:.0f}/100 "
        f"(name overlap {compatibility.name_overlap:.0%}, "
        f"shape match {compatibility.shape_match_ratio:.0%})"
    )

    top_layers = sorted(
        routing.layer_alphas.items(),
        key=lambda item: abs(item[1] - 0.5),
        reverse=True,
    )[:3]
    for key, alpha in top_layers:
        evidence.append(
            f"{key} routes A={alpha:.2f}/B={1 - alpha:.2f} "
            f"(conflict {routing.layer_conflicts.get(key, 0.0):.2f})"
        )

    if routing.category_alphas:
        cat_text = ", ".join(
            f"{cat} A={alpha:.2f}" for cat, alpha in sorted(routing.category_alphas.items())
        )
        evidence.append(f"category routing: {cat_text}")

    evidence.append(
        f"global conflict {conflicts.global_score:.2f} ({conflicts.global_band}); "
        f"{len(conflicts.high_conflict_layers)} high-conflict groups"
    )

    decision = f"cba_routing_{routing.dominant_model}"

    if conflicts.high_conflict_layers:
        worst = conflicts.high_conflict_layers[0]
        risk = (
            f"catastrophic interference in {worst} "
            f"(conflict {conflicts.per_layer[worst].score:.2f}); "
            "naive averaging is unsafe there — sign-consensus (ties) applied"
        )
    elif not compatibility.compatible:
        risk = "structural incompatibility; merged model may degrade sharply"
    else:
        risk = "low: no high-conflict layers and structures align"

    alternative = (
        "increase weight of the non-dominant model inside "
        f"{min(routing.category_alphas, key=routing.category_alphas.get)}"
        if routing.category_alphas
        else "re-run CBA with calibration data for activation-level evidence"
    )

    confidence = _confidence(compatibility, conflicts, routing)

    # ----- Self-critique -------------------------------------------------
    weaknesses: List[str] = []
    missing: List[str] = []

    if not routing.layer_alphas:
        weaknesses.append("no layered structure detected; routing fell back to global statistics")
    if conflicts.per_tensor:
        max_conflict = max(rec.score for rec in conflicts.per_tensor.values())
        if max_conflict > 0.9:
            weaknesses.append(
                f"tensor conflict reaches {max_conflict:.2f}; decisions near this "
                "tensor are weakly grounded"
            )
    if len(inspection_a.tensors) < 8 or len(inspection_b.tensors) < 8:
        weaknesses.append("very small state dicts; statistical evidence is thin")

    missing.append("activation statistics (no calibration forward pass in Phase 1)")
    missing.append("benchmark performance deltas (evaluation engine lands in a later phase)")
    if compatibility.tokenizer_compatible is None:
        missing.append("tokenizer compatibility was not verifiable")

    recommendation = (
        "run merge-time calibration (Fisher) for the top-conflict layers "
        "before committing the final weights"
        if conflicts.high_conflict_layers
        else "proceed with CBA routing; re-run CBA after merge to verify drift"
    )

    critique = {
        "confidence": confidence,
        "potential_weakness": "; ".join(weaknesses) or "none identified",
        "missing_evidence": "; ".join(missing),
        "recommendation": recommendation,
    }

    # ----- Three questions ------------------------------------------------
    mean_alpha = (
        sum(routing.layer_alphas.values()) / len(routing.layer_alphas)
        if routing.layer_alphas
        else 0.5
    )

    what = (
        f"Model A carries {inspection_a.total_tensors} tensors "
        f"(norm mass {inspection_a.total_norm:.2f}); Model B "
        f"{inspection_b.total_tensors} ({inspection_b.total_norm:.2f}). "
        f"CBA routing favors {routing.dominant_model} "
        f"(mean alpha A={mean_alpha:.2f}). "
        f"Conflict: {conflicts.global_band} ({conflicts.global_score:.2f}); "
        f"mergeability {compatibility.mergeability:.0f}/100."
    )

    why_parts: List[str] = []
    if top_layers:
        key, alpha = top_layers[0]
        why_parts.append(
            f"{key} shows the strongest evidence split "
            f"(A={alpha:.2f}) from relative norm mass and dispersion"
        )
    if conflicts.high_conflict_layers:
        why_parts.append(
            f"high conflict in {', '.join(conflicts.high_conflict_layers[:3])} "
            "because weight direction and sign structure disagree"
        )
    if not why_parts:
        why_parts.append("models are statistically similar; evidence margins are small")

    why = "; ".join(why_parts) + "."

    next_parts: List[str] = ["apply per-layer CBA weights during the merge"]
    if conflicts.high_conflict_layers:
        next_parts.append(
            f"use sign-consensus (ties) for {', '.join(conflicts.high_conflict_layers[:3])}"
        )
    next_parts.append(
        "evaluate the merged model and re-run CBA to confirm the routing held"
    )
    next = "; ".join(next_parts) + "."

    questions = {"what": what, "why": why, "next": next}

    return CBADecision(
        decision=decision,
        confidence=confidence,
        evidence=evidence,
        risk=risk,
        alternative=alternative,
        critique=critique,
        questions=questions,
    )


# =============================================================================
# Public one-call pipeline
# =============================================================================

def run_cba(
    state_dict_a: Mapping[str, torch.Tensor],
    state_dict_b: Mapping[str, torch.Tensor],
    *,
    high_conflict_threshold: float = _CONFLICT_HIGH_CUTOFF_DEFAULT,
    vocab_size_b: Optional[int] = None,
) -> CBAReport:
    """Run the complete CBA pipeline over two state dicts."""
    inspection_a = inspect_state_dict(state_dict_a)
    inspection_b = inspect_state_dict(state_dict_b)

    compatibility = check_compatibility(
        inspection_a,
        inspection_b,
        state_dict_a,
        state_dict_b,
        vocab_size_b=vocab_size_b,
    )

    conflicts = analyze_conflicts(
        state_dict_a,
        state_dict_b,
        inspection_a,
        inspection_b,
        high_cutoff=high_conflict_threshold,
    )

    routing = plan_routing(
        inspection_a,
        inspection_b,
        conflicts,
        compatibility,
        high_conflict_threshold=high_conflict_threshold,
    )

    decision = build_decision(
        compatibility,
        conflicts,
        routing,
        inspection_a,
        inspection_b,
    )

    return CBAReport(
        inspection_a=inspection_a,
        inspection_b=inspection_b,
        compatibility=compatibility,
        conflicts=conflicts,
        routing=routing,
        decision=decision,
    )


# =============================================================================
# Standalone Captain reasoners (LLM-optional, deterministic)
# =============================================================================

def answer_three_questions(
    context: Mapping[str, Any],
) -> Dict[str, str]:
    """
    Answer WHAT / WHY / WHAT NEXT for a generic decision context.

    Accepts a ``CBAReport.to_dict()`` payload (uses its full statistics) or
    any mapping (falls back to an honest, evidence-aware generic answer).
    """
    if not isinstance(context, Mapping):
        raise TypeError(
            f"context must be a mapping, got {type(context).__name__}."
        )

    routing = context.get("routing")
    conflicts = context.get("conflicts")
    compatibility = context.get("compatibility")

    if (
        isinstance(routing, Mapping)
        and isinstance(conflicts, Mapping)
        and routing.get("layer_alphas")
    ):
        alphas = routing["layer_alphas"]
        mean_alpha = sum(alphas.values()) / len(alphas)
        dominant = routing.get("dominant_model", "balanced")
        global_score = float(conflicts.get("global_score", 0.0))
        global_band = str(conflicts.get("global_band", "low"))
        high_layers = list(conflicts.get("high_conflict_layers", []) or [])

        what = (
            f"CBA inspected {len(alphas)} layer groups; routing favors "
            f"{dominant} (mean alpha A={mean_alpha:.2f}); conflict is "
            f"{global_band} ({global_score:.2f})."
        )

        why_parts: List[str] = []
        most_decisive = max(
            alphas.items(), key=lambda item: abs(item[1] - 0.5), default=None
        )
        if most_decisive is not None:
            key, alpha = most_decisive
            why_parts.append(
                f"{key} has the strongest evidence split (A={alpha:.2f})"
            )
        if high_layers:
            why_parts.append(
                f"sign/direction disagreement drives high conflict in "
                f"{', '.join(str(x) for x in high_layers[:3])}"
            )
        if not why_parts:
            why_parts.append("evidence margins are small; models are similar")
        why = "; ".join(why_parts) + "."

        next_parts = ["apply the CBA layer weights during the merge"]
        if high_layers:
            next_parts.append(
                "keep sign-consensus (ties) on the high-conflict groups"
            )
        if isinstance(compatibility, Mapping) and compatibility.get("mergeability") is not None:
            mergeability = float(compatibility["mergeability"])
            if mergeability < 60.0:
                next_parts.append(
                    f"mergeability is only {mergeability:.0f}/100 — "
                    "validate on a small eval set before committing"
                )
        next_parts.append("re-run CBA after the merge to verify the routing held")
        next = "; ".join(next_parts) + "."

        return {"what": what, "why": why, "next": next}

    decision = str(context.get("decision", "unknown"))
    confidence = context.get("confidence")
    evidence = context.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, (list, tuple)) else 0

    what = (
        f"Decision context '{decision}' with "
        f"{evidence_count} evidence item(s)"
        + (f" and claimed confidence {confidence}" if confidence is not None else "")
        + "."
    )
    why = (
        "Evidence is deterministic and measured, but this context carries no "
        "CBA statistics; answers are intentionally generic."
        if evidence_count == 0
        else "The listed evidence items drive the decision; no deeper "
        "statistics are attached to this context."
    )
    next = (
        "attach a CBAReport (run_cba(...).to_dict()) or training metrics so "
        "the Captain can answer with measured evidence."
    )
    return {"what": what, "why": why, "next": next}


def critique_decision_record(
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Self-critique a decision record with honest, deterministic reasoning.

    Checks whether the claimed confidence is backed by evidence volume and
    flags what a Phase-1 analysis cannot know (activations, benchmarks).
    """
    if not isinstance(decision, Mapping):
        raise TypeError(
            f"decision must be a mapping, got {type(decision).__name__}."
        )

    try:
        claimed = float(decision.get("confidence", 0.0))
    except (TypeError, ValueError):
        claimed = 0.0

    claimed = _clamp01(claimed)
    evidence = decision.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, (list, tuple)) else 0

    # Evidence volume justifies at most ~0.6 + 0.05/item up to 0.95.
    justified = _clamp01(0.35 + 0.05 * evidence_count)

    weaknesses: List[str] = []
    if claimed > justified:
        weaknesses.append(
            f"claimed confidence {claimed:.2f} exceeds what "
            f"{evidence_count} evidence item(s) justify ({justified:.2f})"
        )

    existing_critique = decision.get("critique")
    if isinstance(existing_critique, Mapping):
        for key in ("potential_weakness", "missing_evidence", "recommendation"):
            value = existing_critique.get(key)
            if value and key == "potential_weakness" and value != "none identified":
                weaknesses.append(str(value))

    if evidence_count == 0:
        weaknesses.append("no evidence items attached to the decision")

    missing = "activation statistics; benchmark deltas; post-merge evaluation"
    recommendation = (
        "verify with a measured evaluation before acting; treat this "
        "decision as a proposal, not a verified result"
    )

    return {
        "confidence": round(min(claimed, justified), 3),
        "claimed_confidence": round(claimed, 3),
        "justified_confidence": round(justified, 3),
        "potential_weakness": "; ".join(weaknesses) or "none identified",
        "missing_evidence": missing,
        "recommendation": recommendation,
    }
