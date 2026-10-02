# ============================================================
# FTRAIN / PHOENIX INTELLIGENT BRAIN MERGER  (merger.py v3)
# ============================================================
"""
Two-model merger with a Captain-controlled policy, CBA routing, Fisher
awareness and a conservative cross-architecture adapter.

Pipeline
--------
    Model A (output architecture) + Model B (knowledge source)
        -> load (CPU, no device_map) + architecture analysis
        -> role-aware tensor correspondence (layer map, token map)
        -> optional Fisher information (source-key aware)
        -> CBA routing report
        -> Captain policy (LLM if available, otherwise rule-based; always clamped)
        -> per-tensor merge: weighted | SLERP | TIES-style | SVD-delta | Fisher
           (+ optional Procrustes), with shape adaptation where needed
        -> per-tensor safety validation + rollback to A
        -> calibration guard with alpha back-off and rollback to A
        -> targeted repair with rollback
        -> robust save (safetensors -> .bin fallbacks) + JSON report

Honesty notes (also written into the JSON report)
-------------------------------------------------
* Model A is ALWAYS the output architecture. Output tensors have A's shapes.
* Models with different hidden size, attention layout, depth, family or
  tokenizer are NOT equivalent. Their weight bases are unrelated, so the
  cross-architecture path (crop/pad/interpolate/head-select, layer
  resampling, token-row mapping) is an EXPERIMENTAL HEURISTIC. It is made
  conservative on purpose (alpha caps, partial-coverage updates, RMS
  matching, calibration guard) and flagged in the report. A successful run
  means "the arithmetic was safe", not "the models were compatible".
* Procrustes/SVD operate in weight space only and are accepted only when they
  measurably reduce weight-space distance. Function-level equivalence is not
  guaranteed.

Configuration
-------------
Every option is read with getattr(config, name, default) (or config[name] for
mappings), so a plain MergeConfig keeps working. Main keys: model_a, model_b,
output_dir, strategy, alpha, save_dtype, captain_model, calibration_data,
use_cba, cba_conflict_threshold, cba_projection, use_fisher,
merge_fisher_elementwise, allow_shape_adaptation, shape_strategy,
merge_adaptation, merge_adaptation_steps (alias: repair_steps),
merge_adaptation_lr, merge_global_loss_tolerance, merge_backoff,
merge_layer_interpolation, merge_experimental_alpha_cap,
merge_adapted_alpha_cap, merge_accelerator, merge_chunk_elements,
max_shard_size, maximum_power, trust_remote_code, merge_verbose,
merge_raise_on_error.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import logging
import math
import os
import platform
import re
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "Merger",
    "MergeError",
    "ArchInfo",
    "CaptainMergePolicy",
    "MergeTensorDecision",
    "MergeReport",
]

__version__ = "3.0.0"


# ============================================================
# OPTIONAL IMPORTS (all failures are tolerated)
# ============================================================

try:
    from transformers import AutoConfig as _TFConfig
    from transformers import AutoModelForCausalLM as _TFModel
    from transformers import AutoTokenizer as _TFTokenizer
except Exception:  # pragma: no cover - transformers is required at runtime
    _TFConfig = None
    _TFModel = None
    _TFTokenizer = None

# Kept for backward compatibility with older callers; Unsloth is imported
# lazily and only when explicitly requested (static merging never needs it).
FastLanguageModel = None
_UNSLOTH_OK = False

try:
    from .cpp_merge import (  # type: ignore
        fast_fisher_merge,
        fast_slerp,
        fast_ties,
        fast_weighted_avg,
    )

    _CPP_MERGE_OK = True
except Exception:
    _CPP_MERGE_OK = False
    fast_weighted_avg = None
    fast_slerp = None
    fast_ties = None
    fast_fisher_merge = None

try:
    from .tensor_state import compare_tensors  # type: ignore
except Exception:
    compare_tensors = None

try:
    from .data_utils import load_data  # type: ignore
except Exception:
    load_data = None

try:
    from .captain import PhoenixCaptain  # type: ignore
except Exception:
    PhoenixCaptain = None

try:
    from . import ui  # type: ignore
except Exception:
    ui = None


# ============================================================
# CONSTANTS
# ============================================================

_GB = 1024 ** 3
_EPS = 1e-12
_STATS_SAMPLE_CAP = 1_000_000
_VOCAB_ROW_SAMPLE = 20_000
_QUANTILE_CAP = 8_000_000  # torch.quantile refuses inputs above ~16.7M elements
_DEFAULT_CHUNK_ELEMENTS = 32 * 1024 * 1024
_FP16_SAFE_MAX = 60000.0

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")

_VALID_STRATEGIES = {"intelligent", "weighted", "slerp", "ties", "fisher", "svd"}
_SHAPE_STRATEGIES = {"crop_pad", "interpolate"}
_LAYER_PROFILES = {"flat", "protect_ends"}

_MATRIX_ROLES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
}
_NORM_ROLES = {"input_norm", "post_norm", "final_norm", "q_norm", "k_norm"}
_VOCAB_ROLES = {"embedding", "lm_head"}
_UNSUPPORTED_ROLES = {"qkv_fused", "moe_expert", "pos_embedding"}

# role -> name of the policy flag that protects it
_PROTECTION_FLAG = {
    "embedding": "protect_embeddings",
    "lm_head": "protect_lm_head",
    "input_norm": "protect_norms",
    "post_norm": "protect_norms",
    "final_norm": "protect_norms",
    "q_norm": "protect_norms",
    "k_norm": "protect_norms",
}
_ROLE_ALPHA_CAPS = {
    "embedding": 0.35,
    "lm_head": 0.40,
    "input_norm": 0.25,
    "post_norm": 0.25,
    "final_norm": 0.25,
    "q_norm": 0.25,
    "k_norm": 0.25,
}
_BIAS_ALPHA_CAP = 0.50

# axis semantics of 2-D weights: (out_axis_kind, in_axis_kind)
_WEIGHT_AXES = {
    "q_proj": ("heads", "hidden"),
    "k_proj": ("kv_heads", "hidden"),
    "v_proj": ("kv_heads", "hidden"),
    "o_proj": ("hidden", "heads"),
    "gate_proj": ("inter", "hidden"),
    "up_proj": ("inter", "hidden"),
    "down_proj": ("hidden", "inter"),
    "embedding": ("vocab", "hidden"),
    "lm_head": ("vocab", "hidden"),
}
_BIAS_AXES = {
    "q_proj": "heads",
    "k_proj": "kv_heads",
    "v_proj": "kv_heads",
    "o_proj": "hidden",
    "gate_proj": "inter",
    "up_proj": "inter",
    "down_proj": "hidden",
}


# ============================================================
# ERRORS
# ============================================================

class MergeError(RuntimeError):
    """Fatal merge failure carrying the pipeline stage it happened in."""

    def __init__(self, message: str, stage: str = "unknown"):
        super().__init__(message)
        self.stage = stage


# ============================================================
# SMALL UTILITIES
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    return _safe_float(value, default)


def _cfg(config: Any, names: Any, default: Any = None) -> Any:
    """First non-None value among attribute/key aliases of a config object."""
    if isinstance(names, str):
        names = (names,)
    for name in names:
        if isinstance(config, Mapping):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _finite(tensor: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(tensor).all().item())
    except Exception:
        return False


def _tensor_norm(tensor: torch.Tensor) -> float:
    try:
        return _safe_float(torch.linalg.vector_norm(tensor.float()))
    except Exception:
        return 0.0


def _sample_flat(tensor: torch.Tensor, cap: int = _STATS_SAMPLE_CAP) -> torch.Tensor:
    """Deterministic strided sample (flattened, float32). Same shape -> same indices."""
    flat = tensor.reshape(-1)
    n = flat.numel()
    if n > cap:
        step = (n + cap - 1) // cap
        flat = flat[::step]
    return flat.float()


def _robust_quantile(x: torch.Tensor, q: float) -> float:
    """torch.quantile with the 16M-element input limit handled by sub-sampling."""
    x = x.reshape(-1)
    if x.numel() == 0:
        return 0.0
    if x.numel() > _QUANTILE_CAP:
        step = (x.numel() + _QUANTILE_CAP - 1) // _QUANTILE_CAP
        x = x[::step]
    return _safe_float(torch.quantile(x.float(), q))


def _normalize_strategy(strategy: Any) -> str:
    value = str(strategy).strip().lower()
    aliases = {
        "weighted_avg": "weighted",
        "average": "weighted",
        "linear": "weighted",
        "fisher_merge": "fisher",
        "spherical": "slerp",
        "tie": "ties",
        "auto": "intelligent",
        "cba": "intelligent",
        "svd_delta": "svd",
    }
    return aliases.get(value, value)


def _json_safe(obj: Any, _depth: int = 0) -> Any:
    """Convert arbitrary report content into strict-JSON-compatible data."""
    if _depth > 14:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(asdict(obj), _depth + 1)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v, _depth + 1) for v in obj]
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if isinstance(obj, torch.device):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return _json_safe(obj.item(), _depth + 1)
        return {"tensor_shape": list(obj.shape), "dtype": str(obj.dtype).replace("torch.", "")}
    try:
        return _json_safe(float(obj), _depth + 1)
    except Exception:
        return str(obj)


def _round(x: Any, nd: int = 6) -> Any:
    return round(x, nd) if isinstance(x, float) and math.isfinite(x) else x


def _is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


# ============================================================
# TENSOR NAME PARSING / ROLES
# ============================================================

def _layer_index(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _parse_signature(name: str) -> Tuple[Optional[int], str, str]:
    """name -> (layer index, role, kind) where kind is weight|bias|other."""
    low = name.lower()
    comps = low.split(".")
    leaf = comps[-1]
    kind = leaf if leaf in ("weight", "bias") else "other"
    mods = set(comps[:-1]) if kind != "other" else set(comps)
    layer = _layer_index(name)

    def has(*cands: str) -> bool:
        return any(c in mods for c in cands)

    role = "other"
    if has("experts") or any(re.fullmatch(r"experts?_?\d+", c) for c in mods):
        role = "moe_expert"
    elif has("qkv_proj", "query_key_value", "c_attn", "wqkv"):
        role = "qkv_fused"
    elif has("q_proj", "wq"):
        role = "q_proj"
    elif has("k_proj", "wk"):
        role = "k_proj"
    elif has("v_proj", "wv"):
        role = "v_proj"
    elif has("o_proj", "wo", "out_proj"):
        role = "o_proj"
    elif has("gate_proj", "w1"):
        role = "gate_proj"
    elif has("up_proj", "w3"):
        role = "up_proj"
    elif has("down_proj", "w2"):
        role = "down_proj"
    elif has("q_norm"):
        role = "q_norm"
    elif has("k_norm"):
        role = "k_norm"
    elif has("input_layernorm", "attention_norm", "ln_1"):
        role = "input_norm"
    elif has("post_attention_layernorm", "ffn_norm", "ln_2"):
        role = "post_norm"
    elif layer is None:
        if has("embed_tokens", "wte", "word_embeddings", "tok_embeddings"):
            role = "embedding"
        elif has("lm_head") or comps == ["output", "weight"]:
            role = "lm_head"
        elif has("embed_positions", "wpe"):
            role = "pos_embedding"
        elif has("norm", "ln_f", "final_layernorm", "final_layer_norm"):
            role = "final_norm"
    return layer, role, kind


def _axis_kinds(role: str, kind: str, ndim: int) -> Optional[Tuple[str, ...]]:
    if ndim == 2 and kind == "weight" and role in _WEIGHT_AXES:
        return _WEIGHT_AXES[role]
    if ndim == 1 and kind == "bias" and role in _BIAS_AXES:
        return (_BIAS_AXES[role],)
    if ndim == 1 and role in ("input_norm", "post_norm", "final_norm"):
        return ("hidden",)
    if ndim == 1 and role in ("q_norm", "k_norm"):
        return ("head_dim",)
    return None


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ArchInfo:
    """Architecture facts used for cross-model decisions."""

    name: str = ""
    model_type: str = ""
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    tie_word_embeddings: bool = False
    num_parameters: int = 0
    layer_ids: List[int] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeTensorDecision:
    """Decision (and outcome) for one target tensor."""

    target_key: str
    source_key: Optional[str]

    role: str = "unknown"
    strategy: str = "weighted"
    alpha_b: float = 0.5

    # Similarity
    cosine: float = 0.0
    relative_delta: float = 0.0
    sign_conflict: float = 0.0
    overlap: float = 0.0

    # Scale
    norm_a: float = 0.0
    norm_b: float = 0.0
    norm_ratio: float = 1.0

    # Trust
    fisher_a: float = 0.0
    fisher_b: float = 0.0
    fisher_confidence: float = 0.0

    # Safety
    protected: bool = False
    shape_aligned: bool = False
    keep_a: bool = False
    keep_b: bool = False

    reason: str = ""

    # ---- v3 additions (all defaulted) ----
    match_method: str = "none"
    source_keys: List[str] = field(default_factory=list)
    source_weights: List[float] = field(default_factory=list)
    layer_a: Optional[int] = None
    layer_b: Optional[int] = None
    shape_a: Tuple[int, ...] = field(default_factory=tuple)
    shape_b: Tuple[int, ...] = field(default_factory=tuple)
    adaptation: str = "none"
    coverage: float = 1.0
    alpha_base: float = 0.5
    alpha_effective: float = 0.0
    cba_action: str = ""
    cba_alpha_b: Optional[float] = None
    cba_conflict: Optional[float] = None
    cba_confidence: Optional[float] = None
    status: str = "pending"  # merged | kept_a | reverted
    extra: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class CaptainMergePolicy:
    """
    Global policy produced by Phoenix Captain (or by the rule-based fallback).

    Captain is constrained to safe, known operations and every value is
    clamped by ``sanitized()``. ``adaptation_steps`` and ``calibration_steps``
    are two names for the same quantity and always stay in sync (setting
    either one updates the other).
    """

    strategy: str = "intelligent"
    alpha_b: float = 0.5
    conflict_threshold: float = 0.35
    trust_fisher: bool = True
    use_cba: bool = True
    use_projection: bool = False
    protect_embeddings: bool = True
    protect_lm_head: bool = True
    protect_norms: bool = True
    prefer_ties_for_conflicts: bool = True
    allow_shape_adaptation: bool = True
    shape_strategy: str = "crop_pad"
    calibration_steps: int = 50
    adaptation_steps: Optional[int] = None
    adaptation_lr: float = 1e-6
    confidence: float = 0.5
    explanation: str = ""
    layer_profile: str = "flat"
    adapted_alpha_cap: float = 0.15
    source: str = "config"  # config | rule | llm

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "adaptation_steps":
            if value is None:
                value = getattr(self, "calibration_steps", 50)
            object.__setattr__(self, "adaptation_steps", value)
            object.__setattr__(self, "calibration_steps", value)
        elif name == "calibration_steps":
            object.__setattr__(self, "calibration_steps", value)
            object.__setattr__(self, "adaptation_steps", value)
        else:
            object.__setattr__(self, name, value)

    def sanitized(self, alpha_ceiling: float = 1.0) -> "CaptainMergePolicy":
        """Clamp every field into its safe range (in place); returns self."""
        strategy = _normalize_strategy(self.strategy)
        self.strategy = strategy if strategy in _VALID_STRATEGIES else "intelligent"
        self.alpha_b = _clamp(_safe_float(self.alpha_b, 0.5), 0.0, min(1.0, alpha_ceiling))
        self.conflict_threshold = _clamp(_safe_float(self.conflict_threshold, 0.35), 0.05, 0.95)
        steps = _as_int(self.adaptation_steps, 50)
        self.adaptation_steps = int(_clamp(steps, 0, 500))
        self.adaptation_lr = _clamp(_safe_float(self.adaptation_lr, 1e-6), 1e-8, 1e-4)
        self.confidence = _clamp(_safe_float(self.confidence, 0.5), 0.0, 1.0)
        self.adapted_alpha_cap = _clamp(_safe_float(self.adapted_alpha_cap, 0.15), 0.0, 0.5)
        shape = str(self.shape_strategy).lower()
        self.shape_strategy = shape if shape in _SHAPE_STRATEGIES else "crop_pad"
        profile = str(self.layer_profile).lower()
        self.layer_profile = profile if profile in _LAYER_PROFILES else "flat"
        for name in (
            "trust_fisher", "use_cba", "use_projection", "prefer_ties_for_conflicts",
            "allow_shape_adaptation",
        ):
            setattr(self, name, bool(getattr(self, name)))
        # Guardrail: embeddings / lm_head / norms stay protected no matter who
        # (rules, config or an LLM Captain) proposes the policy.
        self.protect_embeddings = True
        self.protect_lm_head = True
        self.protect_norms = True
        return self


@dataclass
class MergeReport:
    """Complete merge diagnostics, written to ftrain_merge_report.json."""

    status: str = "running"  # running | success | rolled_back_to_model_a | failed
    error: Optional[str] = None
    failed_stage: Optional[str] = None
    experimental: bool = False
    cross_architecture: bool = False
    disclaimers: List[str] = field(default_factory=list)

    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_seconds: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)

    model_a: str = ""
    model_b: str = ""
    output_dir: str = ""
    strategy: str = ""

    captain_policy: Dict[str, Any] = field(default_factory=dict)
    captain: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)
    layer_map: Dict[str, Any] = field(default_factory=dict)
    vocab_map: Dict[str, Any] = field(default_factory=dict)
    matching: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    mean_cosine: float = 0.0
    mean_relative_delta: float = 0.0
    mean_conflict: float = 0.0

    fisher_used: bool = False
    fisher: Dict[str, Any] = field(default_factory=dict)
    cba_used: bool = False
    cba: Dict[str, Any] = field(default_factory=dict)

    global_guard: Dict[str, Any] = field(default_factory=dict)
    adaptation: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    save: Dict[str, Any] = field(default_factory=dict)

    unmatched: List[Dict[str, Any]] = field(default_factory=list)
    tensor_decisions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class TensorMatch:
    """Where a target (A) tensor gets its source (B) data from."""

    target_key: str
    role: str = "other"
    kind: str = "weight"
    layer_a: Optional[int] = None
    layer_b: Optional[int] = None
    sources: List[Tuple[str, float]] = field(default_factory=list)
    method: str = "none"
    reason: str = ""  # why it is unmatched


@dataclass
class _Adapted:
    tensor: torch.Tensor
    valid_shape: Tuple[int, ...]
    method: str = "none"
    adapted: bool = False
    coverage: float = 1.0


# ============================================================
# MODEL LOADING
# ============================================================

def _from_pretrained_compat(cls: Any, name: str, dtype: Any, kwargs: Dict[str, Any]):
    """from_pretrained across transformers 4.x/5.x (torch_dtype vs dtype,
    low_cpu_mem_usage requiring accelerate)."""
    attempts: List[Dict[str, Any]] = []
    for dtype_key in ("torch_dtype", "dtype"):
        for low_mem in (True, False):
            kw = dict(kwargs)
            if dtype is not None:
                kw[dtype_key] = dtype
            if low_mem:
                kw["low_cpu_mem_usage"] = True
            attempts.append(kw)
            if dtype is None:
                break
        if dtype is None:
            break
    last: Optional[BaseException] = None
    for kw in attempts:
        try:
            return cls.from_pretrained(name, **kw)
        except (TypeError, ValueError, ImportError) as exc:
            last = exc
    assert last is not None
    raise last


def _load_model_any(model_name: str, *, prefer_unsloth: bool = False, **kwargs):
    """
    Load (model, tokenizer) for merging.

    * device_map is never used (a static merge is a CPU tensor operation).
    * Unsloth is only used when explicitly requested and available.
    * Works across transformers versions.
    """
    load_kwargs = dict(kwargs)
    device_map = load_kwargs.pop("device_map", None)
    if device_map not in (None, "cpu"):
        logger.warning(
            "device_map=%r ignored: FTRAIN's static merger never uses device_map.",
            device_map,
        )

    if prefer_unsloth:
        try:  # lazy: importing unsloth patches transformers and is slow
            from unsloth import FastLanguageModel as _FLM  # type: ignore

            return _FLM.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:
            logger.warning(
                "Unsloth loading failed for %s: %s. Falling back to Transformers.",
                model_name, exc,
            )

    if _TFModel is None:
        raise RuntimeError("FTRAIN merger requires the `transformers` package.")

    if load_kwargs.pop("load_in_4bit", False):
        logger.warning("4-bit loading ignored: static merging needs full-precision weights.")
    dtype = load_kwargs.pop("dtype", None)
    load_kwargs.pop("max_seq_length", None)
    load_kwargs.pop("attn_implementation", None)
    trust = bool(load_kwargs.pop("trust_remote_code", False))
    if trust:
        load_kwargs["trust_remote_code"] = True

    model = _from_pretrained_compat(_TFModel, model_name, dtype, load_kwargs)

    tokenizer = None
    if _TFTokenizer is not None:
        try:
            tokenizer = _TFTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        except Exception as exc:
            logger.warning("Tokenizer for %s could not be loaded: %s", model_name, exc)
    return model, tokenizer


# ============================================================
# ARCHITECTURE ANALYSIS
# ============================================================

_EXTRA_CONFIG_KEYS = (
    "rope_theta", "rms_norm_eps", "layer_norm_epsilon", "hidden_act",
    "max_position_embeddings", "sliding_window", "attention_bias", "mlp_bias",
)


def _extract_arch(
    name: str,
    config: Any,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> ArchInfo:
    cfg = getattr(config, "text_config", None) or config

    def g(*names: str, default: Any = None) -> Any:
        for n in names:
            v = getattr(cfg, n, None)
            if v is not None:
                return v
        return default

    info = ArchInfo(name=name)
    info.model_type = str(g("model_type", default="") or "")
    info.hidden_size = _as_int(g("hidden_size", "n_embd", "d_model"), 0)
    info.num_layers = _as_int(g("num_hidden_layers", "n_layer", "num_layers"), 0)
    info.num_heads = _as_int(g("num_attention_heads", "n_head"), 0)
    info.num_kv_heads = _as_int(g("num_key_value_heads"), 0) or info.num_heads
    info.head_dim = _as_int(g("head_dim"), 0)
    if not info.head_dim and info.num_heads:
        info.head_dim = info.hidden_size // max(1, info.num_heads)
    info.intermediate_size = _as_int(g("intermediate_size", "ffn_dim", "n_inner"), 0)
    info.vocab_size = _as_int(g("vocab_size"), 0)
    info.tie_word_embeddings = bool(g("tie_word_embeddings", default=False))
    for key in _EXTRA_CONFIG_KEYS:
        value = g(key)
        if value is not None and isinstance(value, (int, float, str, bool)):
            info.extras[key] = value

    if state_dict is not None:
        ids: Set[int] = set()
        params = 0
        for key, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue
            params += int(tensor.numel())
            layer, role, kind = _parse_signature(key)
            if layer is not None:
                ids.add(layer)
            if tensor.ndim == 2 and kind == "weight":
                if role == "embedding":
                    info.vocab_size = info.vocab_size or int(tensor.shape[0])
                    info.hidden_size = info.hidden_size or int(tensor.shape[1])
                elif role in ("gate_proj", "up_proj") and not info.intermediate_size:
                    info.intermediate_size = int(tensor.shape[0])
        info.layer_ids = sorted(ids)
        info.num_layers = info.num_layers or len(ids)
        info.num_parameters = params
    return info


def _estimate_params(info: ArchInfo) -> int:
    h, layers = info.hidden_size, info.num_layers
    inter = info.intermediate_size or 4 * h
    per_layer = 4 * h * h + 3 * h * inter
    return int(layers * per_layer + info.vocab_size * h * (1 if info.tie_word_embeddings else 2))


def _compare_arch(a: ArchInfo, b: ArchInfo) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    for f in (
        "model_type", "hidden_size", "num_layers", "num_heads", "num_kv_heads",
        "head_dim", "intermediate_size", "vocab_size", "tie_word_embeddings",
    ):
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb:
            diffs[f] = {"a": va, "b": vb}
    extras = {
        k: {"a": a.extras[k], "b": b.extras[k]}
        for k in a.extras
        if k in b.extras and a.extras[k] != b.extras[k]
    }
    dimension = any(
        f in diffs for f in ("hidden_size", "num_heads", "num_kv_heads", "head_dim", "intermediate_size")
    )
    layers = "num_layers" in diffs
    family = "model_type" in diffs
    if not diffs and not extras:
        level = "identical"
    elif dimension or layers or family:
        level = "cross_architecture"
    else:
        level = "same_architecture_variant"
    return {
        "level": level,
        "differences": diffs,
        "semantic_config_differences": extras,
        "dimension_mismatch": dimension,
        "layer_mismatch": layers,
        "family_mismatch": family,
    }


# ============================================================
# TOKEN-AWARE VOCABULARY MAPPING
# ============================================================

def _bytes_to_unicode_inverse() -> Dict[str, int]:
    """Inverse of the GPT-2 byte<->unicode table used by byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _bytes_to_unicode_inverse()


def _looks_byte_level(vocab: Mapping[str, int]) -> bool:
    return sum(1 for t in vocab if "\u0120" in t) >= 5


def _normalize_token(token: str, byte_level: bool) -> Optional[str]:
    """Token string -> comparable surface text (None = not mappable by text)."""
    if not token:
        return None
    match = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", token)
    if match:  # sentencepiece byte fallback
        value = int(match.group(1), 16)
        return chr(value) if value < 128 else None
    if len(token) > 2 and token.startswith("<") and token.endswith(">"):
        return None  # special token such as <s> or <|im_start|>
    if byte_level:
        try:
            raw = bytes(_BYTE_DECODER[ch] for ch in token)
            return raw.decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return None
    return token.replace("\u2581", " ")


def _build_vocab_map(
    tok_a: Any, tok_b: Any, rows_a: int, rows_b: int
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    LongTensor [rows_a] giving, for each A token id, the matching B row (or -1).
    Tokens are matched by their surface text, so different tokenizer families
    (sentencepiece vs byte-level BPE) can still be related.
    """
    info: Dict[str, Any] = {"rows_a": rows_a, "rows_b": rows_b}
    if tok_a is None or tok_b is None:
        info["status"] = "tokenizer_unavailable"
        return None, info
    try:
        vocab_a = dict(tok_a.get_vocab())
        vocab_b = dict(tok_b.get_vocab())
    except Exception as exc:
        info["status"] = f"vocab_unavailable: {exc}"
        return None, info

    if vocab_a == vocab_b:
        n = min(rows_a, rows_b)
        mapping = torch.full((rows_a,), -1, dtype=torch.long)
        mapping[:n] = torch.arange(n)
        info.update(status="identical_vocab", mapped=n, coverage=n / max(1, rows_a))
        return mapping, info

    byte_a, byte_b = _looks_byte_level(vocab_a), _looks_byte_level(vocab_b)
    text_to_b: Dict[str, int] = {}
    for token, idx in sorted(vocab_b.items(), key=lambda kv: kv[1]):
        if idx >= rows_b:
            continue
        text = _normalize_token(token, byte_b)
        if text is not None:
            text_to_b.setdefault(text, idx)

    mapping = torch.full((rows_a,), -1, dtype=torch.long)
    mapped = 0
    for token, idx in vocab_a.items():
        if idx >= rows_a:
            continue
        text = _normalize_token(token, byte_a)
        if text is None:
            continue
        j = text_to_b.get(text)
        if j is not None:
            mapping[idx] = j
            mapped += 1

    special = 0
    for attr in ("bos_token_id", "eos_token_id", "unk_token_id"):
        ia, ib = getattr(tok_a, attr, None), getattr(tok_b, attr, None)
        if (
            isinstance(ia, int) and isinstance(ib, int)
            and 0 <= ia < rows_a and 0 <= ib < rows_b and mapping[ia] < 0
        ):
            mapping[ia] = ib
            special += 1

    total = mapped + special
    info.update(
        status="text_matched",
        tokens_in_a=len(vocab_a),
        mapped=total,
        mapped_by_text=mapped,
        mapped_special=special,
        coverage=total / max(1, rows_a),
        byte_level_a=byte_a,
        byte_level_b=byte_b,
    )
    return mapping, info


# ============================================================
# CALIBRATION DATA
# ============================================================

def _record_text(record: Any, tokenizer: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping):
        text = record.get("text")
        if isinstance(text, str) and text:
            return text
        messages = record.get("messages")
        if messages:
            try:
                if getattr(tokenizer, "chat_template", None):
                    return tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                pass
            return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        for key in ("content", "prompt", "question"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


class _LMDataset(Dataset):
    def __init__(self, records: Sequence[Any], tokenizer: Any, max_length: int):
        self.items: List[List[int]] = []
        for record in records:
            text = _record_text(record, tokenizer)
            if not text:
                continue
            try:
                ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
            except Exception:
                continue
            if len(ids) >= 2:
                self.items.append(list(ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> List[int]:
        return self.items[index]


def _make_collate(pad_id: int) -> Callable[[List[List[int]]], Dict[str, torch.Tensor]]:
    def collate(batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        length = max(len(x) for x in batch)
        ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.long)
        labels = torch.full((len(batch), length), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            t = torch.tensor(x, dtype=torch.long)
            ids[i, : len(x)] = t
            mask[i, : len(x)] = 1
            labels[i, : len(x)] = t
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


def _load_calibration_records(spec: Any) -> List[Any]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, str):
        if load_data is not None:
            try:
                return list(load_data(spec))
            except Exception as exc:
                logger.warning("load_data failed for calibration data (%s); trying built-in reader.", exc)
        if spec.startswith("hf://"):
            from datasets import load_dataset  # type: ignore

            return list(load_dataset(spec[5:], split="train"))
        if os.path.exists(spec):
            with open(spec, "r", encoding="utf-8") as handle:
                if spec.endswith(".jsonl"):
                    return [json.loads(line) for line in handle if line.strip()]
                data = json.load(handle)
                return list(data) if isinstance(data, list) else [data]
        raise FileNotFoundError(f"calibration data not found: {spec}")
    raise TypeError(f"unsupported calibration_data type: {type(spec).__name__}")


# ============================================================
# RESAMPLING / SHAPE ADAPTATION PRIMITIVES
# ============================================================

def _interp_axis(x: torch.Tensor, axis: int, n_target: int) -> torch.Tensor:
    moved = x.movedim(axis, -1).contiguous()
    lead = moved.shape[:-1]
    flat = moved.reshape(-1, 1, moved.shape[-1])
    if flat.shape[-1] == 1:
        y = flat.expand(-1, 1, n_target).clone()
    else:
        y = F.interpolate(flat, size=n_target, mode="linear", align_corners=True)
    return y.reshape(*lead, n_target).movedim(-1, axis).contiguous()


def _head_select(x: torch.Tensor, axis: int, heads_s: int, heads_t: int, head_dim: int) -> torch.Tensor:
    """Pick source heads proportionally (nearest) to fill the target head count."""
    if heads_t <= 1:
        idx_h = torch.zeros(1, dtype=torch.long)
    else:
        idx_h = torch.linspace(0, heads_s - 1, heads_t).round().long()
    idx = (idx_h[:, None] * head_dim + torch.arange(head_dim)[None, :]).reshape(-1)
    return x.index_select(axis, idx)


def _resample_axis(
    x: torch.Tensor,
    axis: int,
    n_target: int,
    kind: str,
    head_dim_src: int,
    head_dim_tgt: int,
    strategy: str,
) -> Tuple[torch.Tensor, int, str]:
    """Resize one axis. Returns (tensor, valid_extent, method)."""
    n_src = x.shape[axis]
    if n_src == n_target:
        return x, n_target, "none"
    if (
        kind in ("heads", "kv_heads")
        and head_dim_src
        and head_dim_tgt
        and head_dim_src == head_dim_tgt
        and n_src % head_dim_src == 0
        and n_target % head_dim_tgt == 0
    ):
        y = _head_select(x, axis, n_src // head_dim_src, n_target // head_dim_tgt, head_dim_src)
        return y, n_target, "head_select"
    if strategy == "interpolate":
        return _interp_axis(x, axis, n_target), n_target, "interpolate"
    n = min(n_src, n_target)
    y = x.narrow(axis, 0, n)
    if n_target > n:
        pad_shape = list(y.shape)
        pad_shape[axis] = n_target - n
        y = torch.cat([y, y.new_zeros(pad_shape)], dim=axis)
    return y.contiguous(), n, "crop_pad"


def _region(extent: Sequence[int]) -> Tuple[slice, ...]:
    return tuple(slice(0, int(n)) for n in extent)


def _row_chunks(extent: Sequence[int], max_elems: int) -> Iterator[Tuple[slice, ...]]:
    """Slices covering the region in row blocks of at most ~max_elems elements."""
    if len(extent) == 0:
        yield ()
        return
    n0 = int(extent[0])
    row_elems = 1
    for e in extent[1:]:
        row_elems *= int(e)
    step = max(1, max_elems // max(1, row_elems))
    tail = tuple(slice(0, int(e)) for e in extent[1:])
    for r0 in range(0, n0, step):
        yield (slice(r0, min(r0 + step, n0)),) + tail


# ============================================================
# MERGE OPERATORS (A and B are float32; results are new tensors)
# ============================================================

def _merge_weighted(a: torch.Tensor, b: torch.Tensor, extent: Sequence[int], alpha: float, chunk: int) -> torch.Tensor:
    out = a.clone()
    for sl in _row_chunks(extent, chunk):
        ac, bc = a[sl], b[sl]
        out[sl] = ac + alpha * (bc - ac)
    return out


def _merge_slerp(a: torch.Tensor, b: torch.Tensor, extent: Sequence[int], alpha: float, chunk: int) -> torch.Tensor:
    dot = na2 = nb2 = 0.0
    for sl in _row_chunks(extent, max(1, chunk // 2)):
        ac, bc = a[sl].double(), b[sl].double()
        dot += float((ac * bc).sum())
        na2 += float((ac * ac).sum())
        nb2 += float((bc * bc).sum())
    na, nb = math.sqrt(na2), math.sqrt(nb2)
    if na < 1e-12 or nb < 1e-12:
        return _merge_weighted(a, b, extent, alpha, chunk)
    cos = _clamp(dot / (na * nb), -0.9995, 0.9995)
    theta = math.acos(cos)
    sin_t = math.sin(theta)
    if abs(sin_t) < 1e-6:
        return _merge_weighted(a, b, extent, alpha, chunk)
    wa = math.sin((1.0 - alpha) * theta) / sin_t
    wb = math.sin(alpha * theta) / sin_t
    norm = (1.0 - alpha) * na + alpha * nb
    out = a.clone()
    for sl in _row_chunks(extent, chunk):
        out[sl] = (wa * (a[sl] / na) + wb * (b[sl] / nb)) * norm
    return out


def _merge_ties(a: torch.Tensor, b: torch.Tensor, extent: Sequence[int], alpha: float, conflict: float, chunk: int) -> torch.Tensor:
    """
    Two-model conflict-aware merge (TIES-style).

    With only two models this is not full multi-task TIES; instead:
    low-signal deltas (B - A) are trimmed, sign-conflicting entries are
    suppressed in proportion to the measured conflict, strong updates stay.
    """
    reg = _region(extent)
    sa, sb = _sample_flat(a[reg]), _sample_flat(b[reg])
    mag = (sb - sa).abs()
    mean_delta = _safe_float(mag.mean()) if mag.numel() else 0.0
    if mean_delta <= 1e-12:
        return a.clone()
    thr = max(mean_delta * 0.25, _robust_quantile(mag, 0.25))
    suppress = 0.75 * _clamp(conflict, 0.0, 1.0)
    out = a.clone()
    for sl in _row_chunks(extent, chunk):
        ac, bc = a[sl], b[sl]
        delta = bc - ac
        keep = (delta.abs() >= thr).float()
        disagree = ((ac * bc) < 0).float()
        out[sl] = ac + alpha * delta * keep * (1.0 - suppress * disagree)
    return out


def _merge_svd_delta(a: torch.Tensor, b: torch.Tensor, extent: Sequence[int], alpha: float, rank: int) -> Tuple[torch.Tensor, float]:
    """Merge only the dominant low-rank part of the delta B - A."""
    reg = _region(extent)
    ar, br = a[reg], b[reg]
    delta = br - ar
    dn = _tensor_norm(delta)
    if delta.ndim != 2 or dn <= 1e-12:
        return a.clone(), 0.0
    q = max(1, min(int(rank), min(delta.shape)))
    u, s, v = torch.svd_lowrank(delta, q=q, niter=2)
    low = (u * s) @ v.T
    energy = _tensor_norm(low) / dn
    out = a.clone()
    out[reg] = ar + alpha * low
    return out, energy


def _merge_fisher(
    a: torch.Tensor, b: torch.Tensor, fa: torch.Tensor, fb: torch.Tensor,
    extent: Sequence[int], alpha: float, med_a: float, med_b: float, chunk: int,
) -> torch.Tensor:
    """
    Elementwise Fisher-tempered blending: the prior odds from alpha are
    multiplied by the (tempered) Fisher ratio, so alpha stays a real control
    and Fisher only redistributes trust inside the tensor.
    """
    alpha = _clamp(alpha, 1e-3, 0.999)
    prior_odds = alpha / (1.0 - alpha)
    out = a.clone()
    for sl in _row_chunks(extent, chunk):
        fac = fa[sl].float() / max(med_a, 1e-30) + 1e-8
        fbc = fb[sl].float() / max(med_b, 1e-30) + 1e-8
        odds = prior_odds * (fbc / fac).pow(0.5)
        w = odds / (1.0 + odds)
        ac, bc = a[sl], b[sl]
        out[sl] = ac + w * (bc - ac)
    return out


def _procrustes_rotate(a: torch.Tensor, b: torch.Tensor, max_dim: int, min_gain: float) -> Tuple[Optional[torch.Tensor], float]:
    """
    Orthogonal Procrustes on the input basis: R = argmin ||B R - A||_F.
    Accepted only if it reduces ||A - B|| by at least ``min_gain`` (relative).
    """
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] > max_dim:
        return None, 0.0
    cross = b.T @ a
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    aligned = b @ (u @ vh)
    d0 = _tensor_norm(a - b)
    d1 = _tensor_norm(a - aligned)
    gain = (d0 - d1) / max(d0, _EPS)
    if gain >= min_gain and _finite(aligned):
        return aligned, gain
    return None, gain


def _relation(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Cheap, sample-based relationship between two same-shaped tensors."""
    a_s, b_s = _sample_flat(a), _sample_flat(b)
    total = a.numel()
    keep = torch.isfinite(a_s) & torch.isfinite(b_s)
    a_s, b_s = a_s[keep], b_s[keep]
    if a_s.numel() == 0:
        return {
            "cosine": 0.0, "relative_delta": 1.0, "sign_conflict": 1.0, "overlap": 0.0,
            "norm_a": 0.0, "norm_b": 0.0, "rms_a": 0.0, "rms_b": 0.0,
        }
    scale = total / max(1, a_s.numel())
    na2 = _safe_float((a_s * a_s).sum())
    nb2 = _safe_float((b_s * b_s).sum())
    norm_a, norm_b = math.sqrt(na2 * scale), math.sqrt(nb2 * scale)
    sna, snb = math.sqrt(na2), math.sqrt(nb2)
    cosine = _safe_float(torch.dot(a_s, b_s) / (sna * snb)) if sna > 1e-12 and snb > 1e-12 else 0.0
    rel_delta = _tensor_norm(b_s - a_s) / max(sna, 1e-12)
    ta = 0.05 * max(_safe_float(a_s.abs().mean()), 1e-12)
    tb = 0.05 * max(_safe_float(b_s.abs().mean()), 1e-12)
    active = (a_s.abs() > ta) & (b_s.abs() > tb)
    if bool(active.any()):
        sign_conflict = _safe_float(((a_s[active] * b_s[active]) < 0).float().mean())
    else:
        sign_conflict = 0.0
    n = max(1, a_s.numel())
    return {
        "cosine": _clamp(cosine, -1.0, 1.0),
        "relative_delta": rel_delta,
        "sign_conflict": sign_conflict,
        "overlap": _safe_float(active.float().mean()),
        "norm_a": norm_a,
        "norm_b": norm_b,
        "rms_a": math.sqrt(na2 / n),
        "rms_b": math.sqrt(nb2 / n),
    }


def _balanced_json_objects(text: str) -> List[str]:
    """All top-level balanced {...} substrings (string-literal aware)."""
    found: List[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : i + 1])
                start = -1
    return found


_POLICY_KEYS = {
    "strategy", "alpha_b", "conflict_threshold", "trust_fisher", "use_cba",
    "use_projection", "prefer_ties_for_conflicts", "allow_shape_adaptation",
    "shape_strategy", "calibration_steps", "adaptation_steps", "adaptation_lr",
    "confidence", "explanation", "layer_profile", "adapted_alpha_cap",
}


# ============================================================
# MERGER
# ============================================================

class _SourceIndex:
    """Index of B's parameters by (layer, role, kind)."""

    def __init__(self, keys: Iterable[str], skip: Set[str]):
        self.sig: Dict[str, Tuple[Optional[int], str, str]] = {}
        self.by_sig: Dict[Tuple[Optional[int], str, str], List[str]] = defaultdict(list)
        layer_ids: Set[int] = set()
        for key in keys:
            if key in skip:
                continue
            layer, role, kind = _parse_signature(key)
            self.sig[key] = (layer, role, kind)
            self.by_sig[(layer, role, kind)].append(key)
            if layer is not None:
                layer_ids.add(layer)
        for names in self.by_sig.values():
            names.sort()
        self.layer_ids: List[int] = sorted(layer_ids)


class Merger:
    """
    Merge Model B into Model A (A stays the output architecture).

    Usage::

        ok = Merger(config).merge()      # True on success
        report = merger.report           # MergeReport (also saved as JSON)
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, config: Any):
        self.config = config
        self.model_a = _cfg(config, ("model_a", "First", "first"))
        self.model_b = _cfg(config, ("model_b", "Second", "second"))
        if not self.model_a or not self.model_b:
            raise ValueError("Merger requires both model_a and model_b.")
        self.model_a, self.model_b = str(self.model_a), str(self.model_b)
        self.output_dir = str(_cfg(config, "output_dir", "./merged_model"))

        self.maximum_power = _as_bool(_cfg(config, ("maximum_power", "MaximumPower")), False)
        self.verbose = _as_bool(_cfg(config, ("merge_verbose", "verbose")), True)
        self.raise_on_error = _as_bool(_cfg(config, "merge_raise_on_error"), True)
        self.trust_remote_code = _as_bool(_cfg(config, "trust_remote_code"), False)

        raw_strategy = str(_cfg(config, "strategy", "intelligent")).strip().lower()
        self.strategy = _normalize_strategy(raw_strategy)
        if self.strategy not in _VALID_STRATEGIES:
            logger.warning("Unknown strategy %r; using 'intelligent'.", raw_strategy)
            self.strategy = "intelligent"
        self.alpha = _clamp(_as_float(_cfg(config, "alpha", 0.5), 0.5), 0.0, 1.0)

        # CBA
        self.use_cba = _as_bool(_cfg(config, "use_cba"), True) or raw_strategy == "cba"
        self.cba_conflict_threshold = _clamp(
            _as_float(_cfg(config, "cba_conflict_threshold", 0.35), 0.35), 0.05, 0.95
        )
        self.cba_projection = _as_bool(_cfg(config, ("cba_projection", "merge_projection")), False)
        self.cba_fallback = str(_cfg(config, "cba_fallback", "intelligent")).lower()

        # shape adaptation / structure
        self.allow_shape_adaptation = _as_bool(_cfg(config, "allow_shape_adaptation"), True)
        shape = str(_cfg(config, "shape_strategy", "crop_pad")).lower()
        self.shape_strategy = shape if shape in _SHAPE_STRATEGIES else "crop_pad"
        self.layer_interpolation = _as_bool(
            _cfg(config, "merge_layer_interpolation"), self.maximum_power
        )
        self.vocab_mapping = _as_bool(_cfg(config, "merge_vocab_mapping"), True)
        self.norm_matching = _as_bool(_cfg(config, "merge_norm_matching"), True)
        self.experimental_alpha_cap = _clamp(
            _as_float(_cfg(config, "merge_experimental_alpha_cap", 0.30), 0.30), 0.0, 1.0
        )
        self.allow_high_alpha_cross_arch = _as_bool(
            _cfg(config, "merge_allow_high_alpha_cross_arch"), False
        )
        self.adapted_alpha_cap = _clamp(
            _as_float(_cfg(config, "merge_adapted_alpha_cap", 0.15), 0.15), 0.0, 0.5
        )

        # Fisher
        self.use_fisher = _as_bool(_cfg(config, "use_fisher"), False)
        self.fisher_trust = _as_bool(_cfg(config, "merge_fisher_protection"), True)
        self.fisher_elementwise = _as_bool(_cfg(config, "merge_fisher_elementwise"), False)
        self.fisher_batches = max(1, _as_int(_cfg(config, "merge_fisher_batches", 16), 16))
        self.fisher_elementwise_max_params = _as_int(
            _cfg(config, "merge_fisher_elementwise_max_params", 1_500_000_000), 1_500_000_000
        )

        # adaptation / repair
        repair_steps = _as_int(_cfg(config, "repair_steps", 0), 0)
        default_steps = repair_steps if repair_steps > 0 else 50
        self.adaptation_enabled = _as_bool(_cfg(config, "merge_adaptation"), True)
        self.adaptation_steps = _as_int(_cfg(config, "merge_adaptation_steps", default_steps), default_steps)
        self.adaptation_lr = _as_float(_cfg(config, "merge_adaptation_lr", 1e-6), 1e-6)
        self.adaptation_weight_decay = _as_float(_cfg(config, "merge_adaptation_weight_decay", 0.01), 0.01)
        self.gradient_clip = _as_float(_cfg(config, "merge_gradient_clip", 1.0), 1.0)
        self.adaptation_max_tensors = _as_int(_cfg(config, "merge_adaptation_max_tensors", 32), 32)
        self.rollback_tolerance = _as_float(_cfg(config, "merge_rollback_tolerance", 0.02), 0.02)
        self.repair_max_params = _as_int(_cfg(config, "merge_repair_max_params", 2_000_000_000), 2_000_000_000)

        # calibration + global guard
        self.max_calibration_samples = max(1, _as_int(_cfg(config, "merge_calibration_samples", 128), 128))
        self.calibration_length = max(8, _as_int(_cfg(config, "merge_calibration_length", 512), 512))
        self.calibration_batch = max(1, _as_int(_cfg(config, "merge_calibration_batch", 1), 1))
        tolerance = _cfg(config, "merge_global_loss_tolerance", 0.5)
        self.global_loss_tolerance = None if tolerance is False else _as_float(tolerance, 0.5)
        self.backoff_enabled = _as_bool(_cfg(config, "merge_backoff"), True)
        self.rollback_save = _as_bool(_cfg(config, "merge_rollback_save"), True)
        self.explosion_ratio = _as_float(_cfg(config, "merge_explosion_ratio", 5.0), 5.0)
        self.collapse_ratio = _as_float(_cfg(config, "merge_collapse_ratio", 0.2), 0.2)

        # numerics / hardware
        self.chunk_elements = max(1024, _as_int(_cfg(config, "merge_chunk_elements", _DEFAULT_CHUNK_ELEMENTS), _DEFAULT_CHUNK_ELEMENTS))
        self.merge_accelerator = _as_bool(_cfg(config, "merge_accelerator"), False)
        self.accelerator_max_elements = _as_int(_cfg(config, "merge_accelerator_max_elements", 64 * 1024 * 1024), 64 * 1024 * 1024)
        self.procrustes_max_dim = _as_int(_cfg(config, "merge_procrustes_max_dim", 2048), 2048)
        self.procrustes_min_gain = _as_float(_cfg(config, "merge_procrustes_min_gain", 0.05), 0.05)
        self.svd_rank = _as_int(_cfg(config, "merge_svd_rank", 64), 64)
        self.svd_max_elements = _as_int(_cfg(config, "merge_svd_max_elements", 16 * 1024 * 1024), 16 * 1024 * 1024)
        self.use_cpp = _as_bool(_cfg(config, "merge_use_cpp"), False) and _CPP_MERGE_OK
        self.report_max_tensors = _as_int(_cfg(config, "merge_report_max_tensors", 5000), 5000)
        self.max_shard_size = _cfg(config, "max_shard_size")

        # save dtype
        save_dtype = str(_cfg(config, "save_dtype", "bf16")).lower()
        self.dtype = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }.get(save_dtype, torch.bfloat16)
        self._save_dtype = self.dtype
        self.load_dtype_pref = str(_cfg(config, "merge_load_dtype", "auto")).lower()

        # captain
        self.captain_model = _cfg(config, "captain_model")
        self.captain_max_new_tokens = _as_int(_cfg(config, "captain_max_new_tokens", 384), 384)
        self.captain_max_seconds = _as_float(_cfg(config, "captain_max_seconds", 90.0), 90.0)
        self.release_captain = _as_bool(_cfg(config, "captain_release_after_policy"), True)
        self.captain: Any = None

        # runtime state
        self.policy = self._base_policy()
        self.captain_policy = self.policy  # backward-compatible alias
        self._alpha_ceiling = 1.0
        self._experimental = False
        self._cross_arch = False
        self._arch_a = ArchInfo()
        self._arch_b = ArchInfo()
        self._arch_cmp: Dict[str, Any] = {}
        self._model_a_obj: Any = None
        self._tok_a: Any = None
        self._tok_b: Any = None
        self._sd_a: Dict[str, torch.Tensor] = {}
        self._sd_b: Dict[str, torch.Tensor] = {}
        self._param_names_a: Set[str] = set()
        self._param_names_b: Set[str] = set()
        self._index_b: Optional[_SourceIndex] = None
        self._sig_a: Dict[str, Tuple[Optional[int], str, str]] = {}
        self._layer_map: Dict[int, List[Tuple[int, float]]] = {}
        self._layer_pos_a: Dict[int, float] = {}
        self._matches: Dict[str, TensorMatch] = {}
        self._tie_alias: Dict[str, str] = {}
        self._vocab_map: Optional[torch.Tensor] = None
        self._vocab_info: Dict[str, Any] = {}
        self._decision_cache: Dict[str, MergeTensorDecision] = {}
        self._decisions: List[MergeTensorDecision] = []
        self._merge_records: List[MergeTensorDecision] = []
        self._cba_report: Any = None
        self._fisher_scalar_a: Optional[Dict[str, float]] = None
        self._fisher_scalar_b: Optional[Dict[str, float]] = None
        self._fisher_elem_a: Optional[Dict[str, torch.Tensor]] = None
        self._fisher_elem_b: Optional[Dict[str, torch.Tensor]] = None
        self._fisher_med = {"a": 1.0, "b": 1.0}
        self._loader_train: Optional[DataLoader] = None
        self._loader_val: Optional[DataLoader] = None
        self._compute_dtype = torch.float32
        self._timers: Dict[str, float] = {}

        self._report = MergeReport(
            model_a=self.model_a, model_b=self.model_b,
            output_dir=self.output_dir, strategy=self.strategy,
        )
        self._report.environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "merger_version": __version__,
        }

    @property
    def report(self) -> MergeReport:
        return self._report

    # ------------------------------------------------------------------
    # logging / memory helpers
    # ------------------------------------------------------------------

    def _say(self, message: str, level: str = "info") -> None:
        getattr(logger, level, logger.info)(message)
        if self.verbose:
            try:
                print(message, flush=True)
            except Exception:
                pass

    def _warn(self, message: str) -> None:
        self._report.warnings.append(message)
        self._say(f"WARNING: {message}", "warning")

    def _purge_memory(self) -> None:
        with contextlib.suppress(Exception):
            gc.collect()
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        with contextlib.suppress(Exception):
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available() and hasattr(xpu, "empty_cache"):
                xpu.empty_cache()
        with contextlib.suppress(Exception):
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()

    @contextlib.contextmanager
    def _stage(self, name: str):
        started = time.time()
        self._current_stage = name
        self._say(f"[merge] {name} ...")
        try:
            yield
        finally:
            self._report.stage_timings[name] = round(time.time() - started, 3)

    # ------------------------------------------------------------------
    # backward-compatible static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_index(name: str) -> Optional[int]:
        return _layer_index(name)

    @staticmethod
    def _parameter_role(key: str) -> str:
        return _parse_signature(key)[1]

    @classmethod
    def _canonical_key(cls, key: str) -> str:
        layer, role, kind = _parse_signature(key)
        prefix = "layers.{layer}" if layer is not None else "global"
        return f"{prefix}.{role}.{kind}"

    @staticmethod
    def _translate_layer_index(layer: int, layers_from: int, layers_to: int) -> int:
        """Map a layer index of a ``layers_from``-layer model onto a
        ``layers_to``-layer model (proportional position)."""
        if layers_from <= 1 or layers_to <= 1:
            return 0
        return int(round(layer / float(layers_from - 1) * (layers_to - 1)))

    @staticmethod
    def _extract_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            for key in ("response", "text", "answer", "content", "output"):
                if key in result:
                    return str(result[key])
            return json.dumps(dict(result), ensure_ascii=False, default=str)
        return str(result)

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse a policy JSON object out of free-form (even R1-style) text."""
        if not text:
            return None
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in cleaned:  # unterminated reasoning block
            cleaned = cleaned.split("<think>")[0]
        candidates = _balanced_json_objects(cleaned)
        for candidate in reversed(candidates):
            try:
                value = json.loads(candidate)
            except Exception:
                continue
            if isinstance(value, dict) and (_POLICY_KEYS & set(value)):
                return value
        try:
            value = json.loads(cleaned.strip())
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------

    def _base_policy(self) -> CaptainMergePolicy:
        return CaptainMergePolicy(
            strategy=self.strategy,
            alpha_b=self.alpha,
            conflict_threshold=self.cba_conflict_threshold,
            trust_fisher=self.fisher_trust,
            use_cba=self.use_cba,
            use_projection=self.cba_projection,
            allow_shape_adaptation=self.allow_shape_adaptation,
            shape_strategy=self.shape_strategy,
            adaptation_steps=self.adaptation_steps,
            adaptation_lr=self.adaptation_lr,
            adapted_alpha_cap=self.adapted_alpha_cap,
            source="config",
        )

    def _heuristic_policy(self) -> CaptainMergePolicy:
        """Deterministic rule-based policy used when no LLM Captain answers."""
        policy = self._base_policy()
        policy.source = "rule"
        notes: List[str] = []
        if self._experimental:
            if not self.allow_high_alpha_cross_arch:
                policy.alpha_b = min(policy.alpha_b, self.experimental_alpha_cap)
            policy.layer_profile = "protect_ends"
            notes.append(
                "experimental cross-architecture merge: conservative alpha, "
                "end layers protected, adapted tensors capped"
            )
        else:
            notes.append("compatible architectures: standard intelligent merge")
        policy.explanation = "; ".join(notes)
        return policy.sanitized(self._alpha_ceiling)

    def _generate_with_llm(self, model: Any, tok: Any, prompt: str) -> str:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        text = prompt
        if getattr(tok, "chat_template", None) and hasattr(tok, "apply_chat_template"):
            with contextlib.suppress(Exception):
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=3072).to(device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.captain_max_new_tokens,
            "do_sample": False,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        }
        if self.captain_max_seconds > 0:
            gen_kwargs["max_time"] = self.captain_max_seconds
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def _ask_captain(self, prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Ask the Captain for a policy JSON. Returns (parsed dict or None, raw text)."""
        captain = self.captain
        if captain is None:
            return None, ""
        raw = ""
        model = getattr(captain, "model", None)
        tok = getattr(captain, "tokenizer", None)
        if model is not None and tok is not None:
            try:
                raw = self._generate_with_llm(model, tok, prompt)
                parsed = self._parse_json(raw)
                if parsed:
                    return parsed, raw
            except Exception as exc:
                logger.debug("Captain LLM generation failed: %s", exc)
        for name in ("ask", "query", "generate", "run", "decide"):
            method = getattr(captain, name, None)
            if not callable(method):
                continue
            try:
                params = inspect.signature(method).parameters
                if "prompt" in params:
                    result = method(prompt=prompt)
                elif "question" in params:
                    result = method(question=prompt)
                else:
                    result = method(prompt)
                raw = self._extract_text(result)
                parsed = self._parse_json(raw)
                if parsed:
                    return parsed, raw
            except Exception as exc:
                logger.debug("Captain method %s failed: %s", name, exc)
        return None, raw

    def _release_captain(self) -> None:
        captain = self.captain
        if captain is None:
            return
        for attr in ("model", "tokenizer"):
            with contextlib.suppress(Exception):
                if getattr(captain, attr, None) is not None:
                    setattr(captain, attr, None)
        self._purge_memory()

    def _captain_global_policy(self) -> CaptainMergePolicy:
        policy = self._heuristic_policy()
        info: Dict[str, Any] = {"source": "rule", "llm_attempted": False}
        if self.captain is not None:
            info["llm_attempted"] = True
            cba_summary = self._cba_summary_text()
            a, b = self._arch_a, self._arch_b
            prompt = f"""You are Phoenix Captain, the strategic controller of FTRAIN's two-model merger.
Choose a SAFE global merge policy from the supported operations. Do not invent transformations.

MODEL A (output architecture): layers={a.num_layers} hidden={a.hidden_size} heads={a.num_heads} kv_heads={a.num_kv_heads} params={a.num_parameters:,} type={a.model_type}
MODEL B (knowledge source):   layers={b.num_layers} hidden={b.hidden_size} heads={b.num_heads} kv_heads={b.num_kv_heads} params={b.num_parameters:,} type={b.model_type}
Architecture comparison: {self._arch_cmp.get('level')}; experimental={self._experimental}
Tokenizer map: {self._vocab_info.get('status')} coverage={self._vocab_info.get('coverage')}
{cba_summary}

Operations: weighted, slerp, ties (conflict-aware), svd (low-rank delta), fisher, intelligent (per-tensor choice).
Embeddings, lm_head and norms are always protected. Alpha is the weight of Model B.
Return ONLY one JSON object:
{{"strategy":"intelligent","alpha_b":0.3,"conflict_threshold":0.35,"trust_fisher":true,"use_cba":true,"use_projection":false,"prefer_ties_for_conflicts":true,"allow_shape_adaptation":true,"shape_strategy":"crop_pad","adaptation_steps":50,"adaptation_lr":0.000001,"layer_profile":"flat","adapted_alpha_cap":0.15,"confidence":0.5,"explanation":"short reason"}}"""
            result, raw = self._ask_captain(prompt)
            info["raw_response_head"] = (raw or "")[:600]
            if result:
                before = asdict(policy)
                for key in _POLICY_KEYS - {"explanation"}:
                    if key in result:
                        try:
                            setattr(policy, key, result[key])
                        except Exception:
                            pass
                if "calibration_steps" in result and "adaptation_steps" not in result:
                    policy.adaptation_steps = result["calibration_steps"]
                policy.explanation = str(result.get("explanation", "") or policy.explanation)[:500]
                policy.source = "llm"
                policy.sanitized(self._alpha_ceiling)
                info["source"] = "llm"
                info["changed_fields"] = sorted(
                    k for k, v in asdict(policy).items() if before.get(k) != v
                )
            else:
                self._warn("Captain did not return a usable policy; using the rule-based policy.")
            if self.release_captain:
                self._release_captain()
        policy.sanitized(self._alpha_ceiling)
        self._report.captain = info
        return policy

    # ------------------------------------------------------------------
    # CBA
    # ------------------------------------------------------------------

    def _cba_summary_text(self) -> str:
        report = self._cba_report
        if report is None:
            return "CBA: not available"
        try:
            comp, conf = report.compatibility, report.conflicts
            return (
                f"CBA: mergeability={comp.mergeability:.1f}/100 compatible={comp.compatible} "
                f"conflict={conf.global_band} ({conf.global_score:.2f}) "
                f"high-conflict layers={len(conf.high_conflict_layers)}"
            )
        except Exception:
            return "CBA: available"

    def _fisher_for_cba(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Median-normalized scalar Fisher keyed by A's names (B translated via matches)."""
        if not self._fisher_scalar_a or not self._fisher_scalar_b:
            return None, None
        fa: Dict[str, float] = {}
        fb: Dict[str, float] = {}
        for key, match in self._matches.items():
            if not match.sources:
                continue
            va = self._fisher_value("a", key, match.role)
            vb = self._fisher_value("b", match.sources[0][0], match.role)
            if va is not None and vb is not None and va > 0 and vb > 0:
                fa[key] = va / self._fisher_med["a"]
                fb[key] = vb / self._fisher_med["b"]
        return (fa or None), (fb or None)

    def _run_cba(self) -> Any:
        self._cba_report = None
        if not self.use_cba:
            return None
        try:
            from .cba import run_cba  # type: ignore
        except Exception as exc:
            logger.info("CBA module unavailable: %s", exc)
            self._report.cba = {"used": False, "reason": f"module unavailable: {exc}"}
            if self.cba_fallback == "abort":
                raise MergeError("CBA requested but unavailable.", "cba") from exc
            return None
        try:
            params = inspect.signature(run_cba).parameters
            kwargs: Dict[str, Any] = {}
            if "high_conflict_threshold" in params:
                kwargs["high_conflict_threshold"] = self.cba_conflict_threshold
            fisher_a, fisher_b = self._fisher_for_cba()
            if fisher_a and "fisher_a" in params:
                kwargs["fisher_a"], kwargs["fisher_b"] = fisher_a, fisher_b
            if "vocab_size_b" in params and self._arch_b.vocab_size:
                kwargs["vocab_size_b"] = self._arch_b.vocab_size
            report = run_cba(self._sd_a, self._sd_b, **kwargs)
            self._cba_report = report
            summary: Dict[str, Any] = {"used": True}
            with contextlib.suppress(Exception):
                summary["compatibility"] = report.compatibility.to_dict()
                summary["decision"] = report.decision.to_dict()
                summary["global_conflict"] = report.conflicts.global_score
                summary["global_band"] = report.conflicts.global_band
                summary["high_conflict_layers"] = list(report.conflicts.high_conflict_layers)
            self._report.cba = summary
            return report
        except Exception as exc:
            if self.cba_fallback == "abort":
                raise MergeError("CBA execution failed.", "cba") from exc
            self._warn(f"CBA failed; continuing without CBA: {type(exc).__name__}: {exc}")
            self._report.cba = {"used": False, "reason": f"{type(exc).__name__}: {exc}"}
            return None

    def _cba_directive(self, key: str) -> Any:
        report = self._cba_report
        if report is None:
            return None
        try:
            return report.routing.directives.get(key)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # calibration data
    # ------------------------------------------------------------------

    def _build_calibration_loaders(self, tokenizer: Any) -> None:
        self._loader_train = self._loader_val = None
        spec = _cfg(self.config, "calibration_data")
        if not spec or tokenizer is None:
            return
        try:
            records = _load_calibration_records(spec)[: self.max_calibration_samples]
            dataset = _LMDataset(records, tokenizer, self.calibration_length)
        except Exception as exc:
            self._warn(f"Calibration data could not be prepared: {type(exc).__name__}: {exc}")
            return
        n = len(dataset)
        if n == 0:
            self._warn("Calibration data produced no usable samples.")
            return
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        collate = _make_collate(int(pad_id))
        if n >= 10:
            cut = max(1, int(n * 0.8))
            train_ds = torch.utils.data.Subset(dataset, list(range(0, cut)))
            val_ds = torch.utils.data.Subset(dataset, list(range(cut, n)))
        else:
            train_ds = val_ds = dataset
        self._loader_train = DataLoader(train_ds, batch_size=self.calibration_batch, shuffle=False, collate_fn=collate)
        self._loader_val = DataLoader(val_ds, batch_size=self.calibration_batch, shuffle=False, collate_fn=collate)
        self._say(f"[merge] calibration samples: {n} (train/val split: {n >= 10})")

    @staticmethod
    def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _runtime_device(self) -> torch.device:
        try:
            from .hardware import best_device  # type: ignore

            return torch.device(best_device())
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_device_safely(self, model: nn.Module, device: torch.device) -> torch.device:
        """Move a model, falling back to CPU when the accelerator is too small."""
        if device.type == "cpu":
            return device
        try:
            model.to(device)
            return device
        except Exception as exc:
            if _is_oom(exc) or "CUDA" in str(exc):
                self._warn(f"Model does not fit on {device}; falling back to CPU for this step.")
                with contextlib.suppress(Exception):
                    model.to("cpu")
                self._purge_memory()
                return torch.device("cpu")
            raise

    def _evaluate_model(
        self, model: nn.Module, loader: Optional[DataLoader],
        device: Optional[torch.device] = None, max_batches: int = 16,
    ) -> float:
        if loader is None:
            return float("inf")
        device = device or self._model_device(model)
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break
                try:
                    out = model(**self._move_batch(batch, device))
                    loss = getattr(out, "loss", None)
                    if loss is None or not torch.isfinite(loss):
                        continue
                    total += float(loss.item())
                    count += 1
                except Exception as exc:
                    if _is_oom(exc):
                        self._purge_memory()
                        break
                    logger.debug("Evaluation batch failed: %s", exc)
        return total / count if count else float("inf")

    # ------------------------------------------------------------------
    # Fisher information (source-key aware)
    # ------------------------------------------------------------------

    def _fisher_value(self, side: str, key: str, role: str) -> Optional[float]:
        store = self._fisher_scalar_a if side == "a" else self._fisher_scalar_b
        if not store:
            return None
        if key in store:
            return store[key]
        if role in _VOCAB_ROLES:  # tied embedding/lm_head appear under one name only
            other = "lm_head" if role == "embedding" else "embedding"
            for name, value in store.items():
                if _parse_signature(name)[1] == other:
                    return value
        return None

    def _fisher_tensor(self, side: str, key: str, role: str) -> Optional[torch.Tensor]:
        store = self._fisher_elem_a if side == "a" else self._fisher_elem_b
        if not store:
            return None
        if key in store:
            return store[key]
        if role in _VOCAB_ROLES:
            other = "lm_head" if role == "embedding" else "embedding"
            for name, value in store.items():
                if _parse_signature(name)[1] == other:
                    return value
        return None

    def _compute_fisher(self, model: nn.Module, side: str) -> None:
        """Empirical Fisher (mean squared gradient) with fp32 accumulation."""
        loader = self._loader_train
        if loader is None:
            return
        try:
            n_params = sum(p.numel() for p in model.parameters())
            elementwise = self.fisher_elementwise or self.strategy == "fisher"
            if elementwise and n_params > self.fisher_elementwise_max_params:
                self._warn(
                    f"Fisher elementwise disabled for model {side.upper()} ({n_params:,} params > "
                    f"{self.fisher_elementwise_max_params:,}); using per-tensor Fisher only."
                )
                elementwise = False
            device = self._to_device_safely(model, self._runtime_device())
            model.eval()
            scalar: Dict[str, float] = defaultdict(float)
            elem: Dict[str, torch.Tensor] = {}
            steps = 0
            for batch in loader:
                if steps >= self.fisher_batches:
                    break
                model.zero_grad(set_to_none=True)
                try:
                    loss = model(**self._move_batch(batch, device)).loss
                    if loss is None or not torch.isfinite(loss):
                        continue
                    loss.backward()
                except Exception as exc:
                    if _is_oom(exc):
                        self._warn("Fisher: out of memory; stopping early.")
                        self._purge_memory()
                        break
                    raise
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g2 = p.grad.detach().float().pow(2)
                    scalar[name] += _safe_float(g2.mean())
                    if elementwise:
                        cpu = g2.cpu()
                        if name in elem:
                            elem[name] += cpu
                        else:
                            elem[name] = cpu
                steps += 1
            model.zero_grad(set_to_none=True)
            if steps == 0:
                self._warn(f"Fisher for model {side.upper()} produced no valid steps.")
                return
            scalar_out = {k: v / steps for k, v in scalar.items()}
            if elementwise:
                for k in elem:
                    elem[k] /= steps
            positives = sorted(v for v in scalar_out.values() if v > 0)
            median = positives[len(positives) // 2] if positives else 1.0
            if side == "a":
                self._fisher_scalar_a, self._fisher_elem_a = scalar_out, (elem if elementwise else None)
            else:
                self._fisher_scalar_b, self._fisher_elem_b = scalar_out, (elem if elementwise else None)
            self._fisher_med[side] = max(median, 1e-30)
            self._report.fisher.setdefault(f"model_{side}", {}).update(
                steps=steps, tensors=len(scalar_out), median=median, elementwise=elementwise
            )
        except Exception as exc:
            self._warn(f"Fisher for model {side.upper()} failed: {type(exc).__name__}: {exc}")
        finally:
            with contextlib.suppress(Exception):
                model.zero_grad(set_to_none=True)
                model.to("cpu")
            self._purge_memory()

    def _fisher_alpha(self, key_a: str, key_b: Optional[str], role: str, default_alpha: float) -> Tuple[float, float, float, float]:
        """(alpha, fisher_a, fisher_b, confidence). Uses source-key lookup for B."""
        fa = self._fisher_value("a", key_a, role)
        fb = self._fisher_value("b", key_b, role) if key_b else None
        if not fa or not fb or fa <= 0 or fb <= 0:
            return default_alpha, fa or 0.0, fb or 0.0, 0.0
        la = math.log1p(fa / self._fisher_med["a"])
        lb = math.log1p(fb / self._fisher_med["b"])
        total = la + lb + 1e-12
        confidence = min(1.0, abs(la - lb) / total)
        ratio = lb / total
        trust = 0.5 if self._experimental else 1.0
        weight = 0.65 * confidence * trust
        alpha = (1.0 - weight) * default_alpha + weight * ratio
        return _clamp(alpha, 0.05, 0.95), fa, fb, confidence

    # ------------------------------------------------------------------
    # structure: layer map, matching, ties
    # ------------------------------------------------------------------

    def _build_layer_map(self, ids_a: List[int], ids_b: List[int]) -> Dict[int, List[Tuple[int, float]]]:
        """A layer -> [(B layer, weight)], proportional by position."""
        mapping: Dict[int, List[Tuple[int, float]]] = {}
        na, nb = len(ids_a), len(ids_b)
        for rank, layer in enumerate(ids_a):
            if nb == 0:
                mapping[layer] = []
                continue
            pos = 0.0 if na <= 1 else rank / (na - 1) * (nb - 1)
            j0 = min(max(int(math.floor(pos + 1e-9)), 0), nb - 1)
            frac = pos - j0
            if self.layer_interpolation and frac > 1e-6 and j0 + 1 < nb:
                mapping[layer] = [(ids_b[j0], 1.0 - frac), (ids_b[j0 + 1], frac)]
            else:
                mapping[layer] = [(ids_b[min(nb - 1, int(round(pos))) ], 1.0)]
        return mapping

    def _detect_ties(self) -> None:
        """Group A tensors that share storage (tied embeddings); merge one, copy to the rest."""
        groups: Dict[int, List[str]] = defaultdict(list)
        for key, tensor in self._sd_a.items():
            if torch.is_tensor(tensor) and tensor.numel() > 0 and torch.is_floating_point(tensor):
                groups[tensor.data_ptr()].append(key)
        self._tie_alias = {}
        for keys in groups.values():
            if len(keys) < 2:
                continue
            keys = sorted(keys)
            rep = next((k for k in keys if self._sig_a.get(k, (None, "", ""))[1] == "embedding"), keys[0])
            for k in keys:
                if k != rep:
                    self._tie_alias[k] = rep

    def _match_tensor(self, key_a: str) -> TensorMatch:
        layer_a, role, kind = self._sig_a[key_a]
        match = TensorMatch(target_key=key_a, role=role, kind=kind, layer_a=layer_a)
        idx = self._index_b
        assert idx is not None
        same_layers = self._arch_a.layer_ids == idx.layer_ids

        if key_a in self._param_names_a:
            pass
        else:
            match.reason = "buffer / non-parameter tensor (kept from A)"
            return match

        exact = key_a in idx.sig and idx.sig[key_a] == (layer_a, role, kind)
        if role in _UNSUPPORTED_ROLES or role == "other":
            if exact and (layer_a is None or same_layers):
                match.sources, match.method = [(key_a, 1.0)], "exact"
                match.layer_b = layer_a
            else:
                match.reason = f"role '{role}' has no safe cross-model mapping"
            return match

        if layer_a is None:
            if exact:
                match.sources, match.method = [(key_a, 1.0)], "exact"
            else:
                keys = idx.by_sig.get((None, role, kind))
                if keys:
                    match.sources, match.method = [(keys[0], 1.0)], "role"
                else:
                    match.reason = f"no '{role}' ({kind}) tensor in model B"
            return match

        if exact and same_layers:
            match.sources, match.method, match.layer_b = [(key_a, 1.0)], "exact", layer_a
            return match

        targets = self._layer_map.get(layer_a, [])
        sources: List[Tuple[str, float]] = []
        for layer_b, weight in targets:
            keys = idx.by_sig.get((layer_b, role, kind))
            if keys:
                sources.append((keys[0], weight))
        if sources:
            total = sum(w for _, w in sources)
            match.sources = [(k, w / total) for k, w in sources]
            match.method = "positional_interp" if len(sources) > 1 else "positional"
            match.layer_b = targets[0][0] if targets else None
        else:
            match.reason = f"layer counterpart in model B has no '{role}' ({kind}) tensor"
        return match

    # ------------------------------------------------------------------
    # shape adaptation
    # ------------------------------------------------------------------

    def _adapt_to_target(self, src: torch.Tensor, target: torch.Tensor, role: str, kind: str) -> Optional[_Adapted]:
        sshape, tshape = tuple(src.shape), tuple(target.shape)
        x = src.detach().float()
        if sshape == tshape:
            return _Adapted(x, tshape, "none", False, 1.0)
        if not self.policy.allow_shape_adaptation or src.ndim != target.ndim or src.ndim == 0:
            return None
        axes = _axis_kinds(role, kind, src.ndim)
        valid = list(tshape)
        methods: List[str] = []
        for axis in range(src.ndim):
            if x.shape[axis] == tshape[axis]:
                continue
            axis_kind = axes[axis] if axes else "generic"
            if axis_kind in ("heads",):
                hd_s, hd_t = self._arch_b.head_dim, self._arch_a.head_dim
            elif axis_kind == "kv_heads":
                hd_s, hd_t = self._arch_b.head_dim, self._arch_a.head_dim
            else:
                hd_s = hd_t = 0
            x, extent, method = _resample_axis(
                x, axis, tshape[axis], axis_kind, hd_s, hd_t, self.policy.shape_strategy
            )
            valid[axis] = extent
            methods.append(f"{axis_kind}:{method}")
        covered = 1.0
        for v, t in zip(valid, tshape):
            covered *= v / max(1, t)
        return _Adapted(x, tuple(valid), "+".join(methods), True, covered)

    def _gather_source(self, match: TensorMatch, target: torch.Tensor) -> Tuple[Optional[_Adapted], List[str]]:
        """Adapt (and blend, for layer interpolation) the matched B tensor(s) to A's shape."""
        notes: List[str] = []
        parts: List[Tuple[_Adapted, float, str]] = []
        for src_key, weight in match.sources:
            src = self._sd_b.get(src_key)
            if not torch.is_tensor(src) or not torch.is_floating_point(src):
                notes.append(f"source {src_key} is not a float tensor")
                continue
            if not _finite(src):
                notes.append(f"source {src_key} contains non-finite values")
                continue
            adapted = self._adapt_to_target(src, target, match.role, match.kind)
            if adapted is None:
                notes.append(f"source {src_key}: shape {tuple(src.shape)} cannot be adapted to {tuple(target.shape)}")
                continue
            parts.append((adapted, weight, src_key))
        if not parts:
            return None, notes
        if len(parts) == 1:
            return parts[0][0], notes
        total = sum(w for _, w, _ in parts)
        blended = sum(p.tensor * (w / total) for p, w, _ in parts)
        extent = tuple(min(vs) for vs in zip(*[p.valid_shape for p, _, _ in parts]))
        methods = "+".join(sorted({p.method for p, _, _ in parts}))
        coverage = min(p.coverage for p, _, _ in parts)
        return _Adapted(blended, extent, methods, any(p.adapted for p, _, _ in parts), coverage), notes

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------

    def _is_protected(self, role: str) -> bool:
        flag = _PROTECTION_FLAG.get(role)
        return bool(flag and getattr(self.policy, flag, True))

    def _layer_factor(self, layer: Optional[int]) -> float:
        if layer is None or self.policy.layer_profile != "protect_ends":
            return 1.0
        pos = self._layer_pos_a.get(layer, 0.5)
        return 0.6 + 0.4 * math.sin(math.pi * pos)

    def _make_decision(
        self, key_a: str, match: TensorMatch, rel: Dict[str, float],
        a: torch.Tensor, adapted: _Adapted, src_keys: List[str],
    ) -> MergeTensorDecision:
        policy = self.policy
        role, kind = match.role, match.kind
        notes: List[str] = []
        is_adapted = adapted.adapted
        protected = self._is_protected(role)
        forced = policy.strategy

        alpha = _clamp(policy.alpha_b * self._layer_factor(match.layer_a), 0.0, 1.0)

        fisher_src = src_keys[0] if src_keys else None
        alpha_f, fa, fb, fconf = self._fisher_alpha(key_a, fisher_src, role, alpha)
        if policy.trust_fisher and fconf > 0.0:
            alpha = alpha_f
            notes.append(f"Fisher shifts alpha (confidence {fconf:.2f})")

        norm_ratio = rel["norm_b"] / max(rel["norm_a"], 1e-12)
        if not is_adapted:
            if norm_ratio > 3.0:
                alpha *= 0.55
                notes.append("B scale >3x A")
            elif norm_ratio > 2.0:
                alpha *= 0.75
            elif norm_ratio < 0.33:
                alpha *= 0.65
                notes.append("B scale <1/3 A")

        # ---- strategy -------------------------------------------------
        numel = a.numel()
        elem_ok = (
            not is_adapted
            and self._fisher_tensor("a", key_a, role) is not None
            and fisher_src is not None
            and self._fisher_tensor("b", fisher_src, role) is not None
        )
        svd_ok = a.ndim == 2 and min(a.shape) >= 8 and numel <= self.svd_max_elements
        mergeable_matrix = role in _MATRIX_ROLES and kind == "weight" and a.ndim == 2
        strategy = "weighted"
        if protected or kind == "bias" or is_adapted or not mergeable_matrix:
            strategy = "weighted"
            if protected:
                notes.append("protected tensor: weighted only")
            if is_adapted:
                notes.append("shape-adapted: weighted only")
        elif forced != "intelligent":
            strategy = forced
            if forced == "fisher" and not elem_ok:
                strategy = "weighted"
                notes.append("elementwise Fisher unavailable; weighted")
            if forced == "svd" and not svd_ok:
                strategy = "weighted"
                notes.append("tensor too large/small for SVD; weighted")
        else:
            if rel["sign_conflict"] >= policy.conflict_threshold and policy.prefer_ties_for_conflicts:
                strategy = "ties"
                notes.append("high sign conflict")
            elif rel["cosine"] >= 0.92 and rel["relative_delta"] < 0.40:
                strategy = "slerp"
                notes.append("high structural similarity")
            elif rel["cosine"] >= 0.20 and rel["relative_delta"] >= 0.60 and svd_ok:
                strategy = "svd"
                notes.append("large correlated delta: low-rank merge")
            elif elem_ok and fconf >= 0.35:
                strategy = "fisher"
                notes.append("Fisher importance differs between parents")
            elif rel["cosine"] < 0.15 and rel["relative_delta"] > 1.0:
                alpha *= 0.55
                notes.append("weak structural agreement")

        # ---- CBA directive ---------------------------------------------
        decision_extra: Dict[str, Any] = {}
        cba_action = ""
        cba_alpha_b = cba_conflict = cba_conf = None
        keep_a = keep_b = False
        directive = self._cba_directive(key_a) if policy.use_cba else None
        if directive is not None:
            try:
                cba_alpha_b = 1.0 - float(getattr(directive, "alpha_a", 0.5))
                cba_conf = _clamp(_safe_float(getattr(directive, "confidence", 0.0)), 0.0, 1.0)
                cba_conflict = _safe_float(getattr(directive, "conflict", 0.0))
                cba_action = str(getattr(directive, "action", "")).lower()
                weight = _clamp(0.25 + 0.5 * cba_conf, 0.0, 0.75)
                alpha = (1.0 - weight) * alpha + weight * cba_alpha_b
                notes.append(f"CBA {cba_action} (conf {cba_conf:.2f})")
                if forced == "intelligent" and mergeable_matrix and not protected and not is_adapted:
                    if cba_action == "ties" and policy.prefer_ties_for_conflicts and strategy in ("weighted", "slerp", "svd"):
                        strategy = "ties"
                    elif cba_action == "slerp" and strategy == "weighted" and rel["cosine"] >= 0.8:
                        strategy = "slerp"
                    elif cba_action == "fisher" and strategy == "weighted" and elem_ok:
                        strategy = "fisher"
                if forced == "intelligent" and cba_conf >= 0.8 and not protected:
                    if cba_action == "keep_a":
                        keep_a = True
                    elif cba_action == "keep_b" and not self._experimental and not is_adapted:
                        keep_b = True
            except Exception as exc:
                logger.debug("CBA directive unusable for %s: %s", key_a, exc)

        # ---- caps ------------------------------------------------------
        cap = _ROLE_ALPHA_CAPS.get(role)
        if kind == "bias":
            cap = min(cap if cap is not None else 1.0, _BIAS_ALPHA_CAP)
        if cap is not None:
            alpha = min(alpha, cap)
        if is_adapted:
            alpha = min(alpha, policy.adapted_alpha_cap)
        alpha = _clamp(min(alpha, self._alpha_ceiling), 0.0, 1.0)
        if keep_b:
            alpha = 1.0
        if alpha < 0.005:
            keep_a = True

        return MergeTensorDecision(
            target_key=key_a,
            source_key=fisher_src,
            role=role,
            strategy=strategy,
            alpha_b=alpha,
            cosine=rel["cosine"],
            relative_delta=rel["relative_delta"],
            sign_conflict=rel["sign_conflict"],
            overlap=rel["overlap"],
            norm_a=rel["norm_a"],
            norm_b=rel["norm_b"],
            norm_ratio=norm_ratio,
            fisher_a=fa,
            fisher_b=fb,
            fisher_confidence=fconf,
            protected=protected,
            shape_aligned=is_adapted,
            keep_a=keep_a,
            keep_b=keep_b,
            reason="; ".join(notes) or "standard intelligent merge",
            match_method=match.method,
            source_keys=list(src_keys),
            source_weights=[w for _, w in match.sources],
            layer_a=match.layer_a,
            layer_b=match.layer_b,
            shape_a=tuple(a.shape),
            shape_b=tuple(adapted.tensor.shape),
            adaptation=adapted.method,
            coverage=adapted.coverage,
            alpha_base=alpha,
            alpha_effective=alpha,
            cba_action=cba_action,
            cba_alpha_b=_round(cba_alpha_b) if cba_alpha_b is not None else None,
            cba_conflict=_round(cba_conflict) if cba_conflict is not None else None,
            cba_confidence=_round(cba_conf) if cba_conf is not None else None,
            extra=decision_extra,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # executing one tensor
    # ------------------------------------------------------------------

    def _compute_device(self, numel: int) -> torch.device:
        if self.merge_accelerator and torch.cuda.is_available() and numel <= self.accelerator_max_elements:
            return torch.device("cuda")
        return torch.device("cpu")

    def _execute(
        self, decision: MergeTensorDecision, key_a: str, a: torch.Tensor, b: torch.Tensor,
        extent: Tuple[int, ...], rel: Dict[str, float], role: str, kind: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        alpha = decision.alpha_effective
        info: Dict[str, Any] = {}
        if decision.keep_a or alpha <= 0.0:
            return a, {"op": "keep_a"}
        chunk = self.chunk_elements
        reg = _region(extent) if extent else ()

        # scale (RMS) matching of B onto A's scale
        if self.norm_matching and role in _MATRIX_ROLES | _VOCAB_ROLES and kind == "weight":
            ra, rb = rel["rms_a"], rel["rms_b"]
            if rb > 1e-12 and ra > 1e-12:
                scale = ra / rb
                if decision.shape_aligned:
                    scale = _clamp(scale, 0.05, 20.0)
                    b = b * scale
                    info["rms_scale"] = round(scale, 4)
                elif 0.5 <= scale <= 2.0:
                    b = b * scale
                    info["rms_scale"] = round(scale, 4)

        if decision.keep_b and not decision.shape_aligned and tuple(a.shape) == tuple(b.shape):
            return b.clone(), {**info, "op": "keep_b"}

        strategy = decision.strategy

        if (
            self.policy.use_projection and strategy in ("weighted", "slerp", "ties", "svd")
            and role in _MATRIX_ROLES and not decision.shape_aligned and a.ndim == 2
        ):
            aligned, gain = _procrustes_rotate(a, b, self.procrustes_max_dim, self.procrustes_min_gain)
            if aligned is not None:
                b = aligned
                info["procrustes_gain"] = round(gain, 4)
                decision.notes.append(f"Procrustes aligned B (gain {gain:.1%})")

        if strategy == "fisher":
            fa = self._fisher_tensor("a", key_a, role)
            fb = self._fisher_tensor("b", decision.source_key or "", role)
            if fa is not None and fb is not None and tuple(fa.shape) == tuple(a.shape) and tuple(fb.shape) == tuple(a.shape):
                return _merge_fisher(a, b, fa, fb, extent, alpha, self._fisher_med["a"], self._fisher_med["b"], chunk), {**info, "op": "fisher"}
            strategy = "weighted"
        if strategy == "svd":
            if a.ndim == 2 and a.numel() <= self.svd_max_elements:
                out, energy = _merge_svd_delta(a, b, extent, alpha, self.svd_rank)
                if energy >= 0.20:
                    return out, {**info, "op": "svd", "energy": round(energy, 4)}
                info["svd_energy"] = round(energy, 4)
            strategy = "weighted"
        if strategy == "ties":
            return _merge_ties(a, b, extent, alpha, decision.sign_conflict, chunk), {**info, "op": "ties"}
        if strategy == "slerp":
            return _merge_slerp(a, b, extent, alpha, chunk), {**info, "op": "slerp"}
        return _merge_weighted(a, b, extent, 
