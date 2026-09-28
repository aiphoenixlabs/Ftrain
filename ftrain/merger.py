# ============================================================
# FTRAIN / PHOENIX INTELLIGENT BRAIN MERGER
# ============================================================
#
# Deep architecture:
#
#   Model A + Model B
#         |
#         v
#   Architecture Mapper
#         |
#         v
#   Tensor Intelligence
#       /    |    \
#  Fisher   CBA   Tensor State
#       \    |    /
#         Captain
#            |
#            v
#      Local Merge Policy
#            |
#     +------+------+------+
#     |      |      |      |
# weighted SLERP  TIES Fisher
#     +------+------+------+
#            |
#            v
#      Protection Layer
#            |
#            v
#      Safety Validation
#            |
#            v
#   Targeted Brain Repair
#            |
#            v
#      Rollback Guard
#            |
#            v
#       Final Model
#
# IMPORTANT:
# - Model A is always the output architecture.
# - Different tensor names are translated.
# - Different layer counts are mapped positionally.
# - Shape conversion is conservative.
# - device_map="auto" is NEVER used for normal training.
# - CBA is treated as a decision system, not arbitrary weight movement.
# ============================================================

from __future__ import annotations

import gc
import inspect
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    from unsloth import FastLanguageModel

    _UNSLOTH_OK = True

except Exception:

    FastLanguageModel = None
    _UNSLOTH_OK = False


try:

    from transformers import (
        AutoModelForCausalLM as _TFModel,
        AutoTokenizer as _TFTokenizer,
    )

except Exception:

    _TFModel = None
    _TFTokenizer = None


try:

    from .cpp_merge import (
        fast_weighted_avg,
        fast_slerp,
        fast_ties,
        fast_fisher_merge,
    )

    _CPP_MERGE_OK = True

except Exception:

    _CPP_MERGE_OK = False

    fast_weighted_avg = None
    fast_slerp = None
    fast_ties = None
    fast_fisher_merge = None


try:

    from .tensor_state import compare_tensors

except Exception:

    compare_tensors = None


try:

    from .merge_intel import (
        MergeAnalyzer,
        MergePlanner,
    )

except Exception:

    MergeAnalyzer = None
    MergePlanner = None


try:

    from .merge_advanced import (
        compute_fisher,
    )

except Exception:

    compute_fisher = None


try:

    from .safety import (
        check_state_dict,
        sanitize,
    )

except Exception:

    check_state_dict = None
    sanitize = None


try:

    from .data_utils import load_data

except Exception:

    load_data = None


try:

    from .dataset import (
        FtrainDataset,
        collate,
    )

except Exception:

    FtrainDataset = None
    collate = None


try:

    from .captain import PhoenixCaptain

except Exception:

    PhoenixCaptain = None


try:

    from . import ui

except Exception:

    ui = None


# ============================================================
# CONSTANTS
# ============================================================

_GB = 1024 ** 3

_LAYER_PATTERNS = (
    r"(?:model\.)?layers\.(\d+)\.",
    r"decoder\.layers\.(\d+)\.",
    r"transformer\.h\.(\d+)\.",
    r"transformer\.layers\.(\d+)\.",
    r"(?:model\.)?h\.(\d+)\.",
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class MergeTensorDecision:
    """
    Decision made for one target tensor.
    """

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

    # Reason
    reason: str = ""


@dataclass
class CaptainMergePolicy:
    """
    Global policy produced by Phoenix Captain.

    Captain is constrained to safe, known operations.
    It does not get permission to invent arbitrary tensor
    transformations.
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

    adaptation_lr: float = 1e-6

    confidence: float = 0.5

    explanation: str = ""


@dataclass
class MergeReport:
    """
    Complete merge diagnostics.
    """

    started_at: float = field(
        default_factory=time.time
    )

    finished_at: Optional[float] = None

    duration_seconds: float = 0.0

    model_a: str = ""

    model_b: str = ""

    output_dir: str = ""

    strategy: str = ""

    captain_policy: Dict[str, Any] = field(
        default_factory=dict
    )

    architecture: Dict[str, Any] = field(
        default_factory=dict
    )

    counts: Dict[str, int] = field(
        default_factory=dict
    )

    mean_cosine: float = 0.0

    mean_relative_delta: float = 0.0

    mean_conflict: float = 0.0

    fisher_used: bool = False

    cba_used: bool = False

    adaptation: Dict[str, Any] = field(
        default_factory=dict
    )

    safety: Dict[str, Any] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )


# ============================================================
# SAFE UTILITIES
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        value = float(value)

        if math.isfinite(value):

            return value

    except Exception:

        pass

    return default


def _finite(
    tensor: torch.Tensor,
) -> bool:

    try:

        return bool(
            torch.isfinite(
                tensor
            ).all().item()
        )

    except Exception:

        return False


def _tensor_numel(
    tensor: torch.Tensor,
) -> int:

    try:

        return int(
            tensor.numel()
        )

    except Exception:

        return 0


def _tensor_norm(
    tensor: torch.Tensor,
) -> float:

    try:

        return _safe_float(
            torch.linalg.vector_norm(
                tensor.float()
            )
        )

    except Exception:

        return 0.0


def _normalize_strategy(
    strategy: str,
) -> str:

    strategy = str(
        strategy
    ).strip().lower()

    aliases = {
        "weighted_avg": "weighted",
        "average": "weighted",
        "linear": "weighted",
        "fisher_merge": "fisher",
        "spherical": "slerp",
        "tie": "ties",
        "auto": "intelligent",
        "cba": "intelligent",
    }

    return aliases.get(
        strategy,
        strategy,
    )


# ============================================================
# MODEL LOADING
# ============================================================

def _load_model_any(
    model_name: str,
    *,
    prefer_unsloth: bool = False,
    **kwargs,
):
    """
    Load a model safely.

    Unsloth is used when explicitly requested and available.
    Raw tensor merging itself does not require Unsloth.
    """

    if (
        prefer_unsloth
        and _UNSLOTH_OK
        and FastLanguageModel is not None
    ):

        try:

            return (
                FastLanguageModel.from_pretrained(
                    model_name,
                    **kwargs,
                )
            )

        except Exception as exc:

            logger.warning(
                "Unsloth loading failed for %s: %s. "
                "Falling back to Transformers.",
                model_name,
                exc,
            )

    if _TFModel is None:

        raise RuntimeError(
            "FTRAIN merger requires Transformers "
            "or Unsloth."
        )

    load_kwargs = dict(
        kwargs
    )

    load_in_4bit = bool(
        load_kwargs.pop(
            "load_in_4bit",
            False,
        )
    )

    dtype = load_kwargs.pop(
        "dtype",
        None,
    )

    # Unsloth-specific.
    load_kwargs.pop(
        "max_seq_length",
        None,
    )

    load_kwargs.pop(
        "attn_implementation",
        None,
    )

    if dtype is not None:

        load_kwargs[
            "torch_dtype"
        ] = dtype

    if load_in_4bit:

        try:

            import bitsandbytes  # noqa: F401

            from transformers import (
                BitsAndBytesConfig,
            )

            load_kwargs[
                "quantization_config"
            ] = BitsAndBytesConfig(
                load_in_4bit=True
            )

        except Exception:

            logger.warning(
                "4-bit requested but bitsandbytes "
                "is unavailable. Loading normal weights."
            )

    model = _TFModel.from_pretrained(
        model_name,
        **load_kwargs,
    )

    tokenizer = (
        _TFTokenizer.from_pretrained(
            model_name
        )
    )

    return model, tokenizer


# ============================================================
# MERGER
# ============================================================

class Merger:

    def __init__(
        self,
        config,
    ):

        self.config = config

        self.model_a = config.model_a
        self.model_b = config.model_b

        self.output_dir = config.output_dir

        self.strategy = _normalize_strategy(
            getattr(
                config,
                "strategy",
                "intelligent",
            )
        )

        self.alpha = float(
            getattr(
                config,
                "alpha",
                0.5,
            )
        )

        # ----------------------------------------------------
        # CBA
        # ----------------------------------------------------

        self.use_cba = bool(
            getattr(
                config,
                "use_cba",
                True,
            )
        )

        if self.strategy == "cba":

            self.strategy = "intelligent"
            self.use_cba = True

        self.cba_conflict_threshold = float(
            getattr(
                config,
                "cba_conflict_threshold",
                0.35,
            )
        )

        self.cba_projection = bool(
            getattr(
                config,
                "cba_projection",
                False,
            )
        )

        self.cba_fallback = str(
            getattr(
                config,
                "cba_fallback",
                "intelligent",
            )
        )

        # ----------------------------------------------------
        # Tensor behavior
        # ----------------------------------------------------

        self.allow_shape_adaptation = bool(
            getattr(
                config,
                "allow_shape_adaptation",
                True,
            )
        )

        self.shape_strategy = str(
            getattr(
                config,
                "shape_strategy",
                "crop_pad",
            )
        )

        # ----------------------------------------------------
        # Fisher
        # ----------------------------------------------------

        self.use_fisher = bool(
            getattr(
                config,
                "use_fisher",
                False,
            )
        )

        self.use_fisher_protection = bool(
            getattr(
                config,
                "merge_fisher_protection",
                True,
            )
        )

        # ----------------------------------------------------
        # Adaptation
        # ----------------------------------------------------

        self.adaptation_enabled = bool(
            getattr(
                config,
                "merge_adaptation",
                True,
            )
        )

        self.adaptation_steps = int(
            getattr(
                config,
                "merge_adaptation_steps",
                50,
            )
        )

        self.adaptation_lr = float(
            getattr(
                config,
                "merge_adaptation_lr",
                1e-6,
            )
        )

        self.adaptation_weight_decay = float(
            getattr(
                config,
                "merge_adaptation_weight_decay",
                0.01,
            )
        )

        self.gradient_clip = float(
            getattr(
                config,
                "merge_gradient_clip",
                1.0,
            )
        )

        self.adaptation_max_tensors = int(
            getattr(
                config,
                "merge_adaptation_max_tensors",
                32,
            )
        )

        self.rollback_tolerance = float(
            getattr(
                config,
                "merge_rollback_tolerance",
                0.02,
            )
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        self.max_calibration_samples = int(
            getattr(
                config,
                "merge_calibration_samples",
                128,
            )
        )

        self.calibration_length = int(
            getattr(
                config,
                "merge_calibration_length",
                512,
            )
        )

        # ----------------------------------------------------
        # Candidate selection
        # ----------------------------------------------------

        self.candidate_search = bool(
            getattr(
                config,
                "merge_candidate_search",
                False,
            )
        )

        self.candidate_search_max_params = int(
            getattr(
                config,
                "merge_candidate_search_max_params",
                3_000_000_000,
            )
        )

        # ----------------------------------------------------
        # Merge device
        # ----------------------------------------------------

        self.merge_accelerator = bool(
            getattr(
                config,
                "merge_accelerator",
                False,
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_dtype = str(
            getattr(
                config,
                "save_dtype",
                "bf16",
            )
        ).lower()

        self.dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(
            save_dtype,
            torch.bfloat16,
        )

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.captain = None

        self.captain_policy = (
            CaptainMergePolicy(
                strategy=self.strategy,
                alpha_b=self.alpha,
            )
        )

        self.fisher_a = None
        self.fisher_b = None

        self._decisions: List[
            MergeTensorDecision
        ] = []

        self._merge_records: List[
            MergeTensorDecision
        ] = []

        self._cba_report = None

        self._report = MergeReport(
            model_a=self.model_a,
            model_b=self.model_b,
            output_dir=self.output_dir,
            strategy=self.strategy,
        )

        # ----------------------------------------------------
        # Captain
        # ----------------------------------------------------

        captain_model = getattr(
            config,
            "captain_model",
            None,
        )

        if (
            captain_model
            and PhoenixCaptain is not None
        ):

            try:

                from .config import TrainConfig

                cap_cfg = TrainConfig(
                    model_name=captain_model,
                    captain_model=captain_model,
                    captain_mode="llm",
                    answer_mode="auto_yes",
                )

                self.captain = PhoenixCaptain(
                    cap_cfg
                )

            except Exception as exc:

                logger.warning(
                    "Captain initialization failed: %s",
                    exc,
                )

    # ========================================================
    # MEMORY
    # ========================================================

    def _purge_memory(
        self,
    ):

        try:

            gc.collect()

        except Exception:

            pass

        try:

            if torch.cuda.is_available():

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        except Exception:

            pass

        try:

            xpu = getattr(
                torch,
                "xpu",
                None,
            )

            if (
                xpu is not None
                and xpu.is_available()
                and hasattr(
                    xpu,
                    "empty_cache",
                )
            ):

                xpu.empty_cache()

        except Exception:

            pass

        try:

            mps = getattr(
                torch,
                "mps",
                None,
            )

            if (
                mps is not None
                and hasattr(
                    mps,
                    "empty_cache",
                )
            ):

                mps.empty_cache()

        except Exception:

            pass

    # ========================================================
    # LAYER DETECTION
    # ========================================================

    @staticmethod
    def _layer_index(
        name: str,
    ) -> Optional[int]:

        for pattern in _LAYER_PATTERNS:

            match = re.search(
                pattern,
                name,
            )

            if match:

                return int(
                    match.group(1)
                )

        return None

    def _get_num_layers(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> int:

        layers = []

        for key in state_dict.keys():

            index = self._layer_index(
                key
            )

            if index is not None:

                layers.append(
                    index
                )

        if not layers:

            return 1

        return (
            max(layers)
            + 1
        )

    # ========================================================
    # LAYER TRANSLATION
    # ========================================================

    @staticmethod
    def _translate_layer_index(
        layer: int,
        source_layers: int,
        target_layers: int,
    ) -> int:

        if (
            source_layers <= 1
            or target_layers <= 1
        ):

            return 0

        position = (
            layer
            / float(
                target_layers - 1
            )
        )

        return int(
            round(
                position
                * (
                    source_layers - 1
                )
            )
        )

    # ========================================================
    # PARAMETER ROLE
    # ========================================================

    @staticmethod
    def _parameter_role(
        key: str,
    ) -> str:

        key_l = key.lower()

        if (
            "embed_tokens" in key_l
            or "wte" in key_l
            or "word_embeddings" in key_l
            or "embedding" in key_l
        ):

            return "embedding"

        if (
            "lm_head" in key_l
            or key_l.endswith(
                "output.weight"
            )
        ):

            return "lm_head"

        if (
            "layernorm" in key_l
            or "layer_norm" in key_l
            or "ln_" in key_l
            or key_l.endswith(
                "norm.weight"
            )
        ):

            return "norm"

        if (
            "q_proj" in key_l
            or key_l.endswith(
                ".wq"
            )
        ):

            return "q_proj"

        if (
            "k_proj" in key_l
            or key_l.endswith(
                ".wk"
            )
        ):

            return "k_proj"

        if (
            "v_proj" in key_l
            or key_l.endswith(
                ".wv"
            )
        ):

            return "v_proj"

        if (
            "o_proj" in key_l
            or key_l.endswith(
                ".wo"
            )
        ):

            return "o_proj"

        if (
            "gate_proj" in key_l
            or key_l.endswith(
                ".w1"
            )
        ):

            return "gate_proj"

        if (
            "up_proj" in key_l
            or key_l.endswith(
                ".w3"
            )
        ):

            return "up_proj"

        if (
            "down_proj" in key_l
            or key_l.endswith(
                ".w2"
            )
        ):

            return "down_proj"

        if (
            key_l.endswith(
                ".bias"
            )
        ):

            return "bias"

        return "weight"

    # ========================================================
    # CANONICAL PARAMETER NAME
    # ========================================================

    @classmethod
    def _canonical_key(
        cls,
        key: str,
    ) -> str:

        value = key

        replacements = [
            (
                "decoder.layers",
                "layers",
            ),
            (
                "transformer.h",
                "layers",
            ),
            (
                "transformer.layers",
                "layers",
            ),
            (
                "model.layers",
                "layers",
            ),
            (
                "model.embed_tokens",
                "embed_tokens",
            ),
            (
                "transformer.wte",
                "embed_tokens",
            ),

            # Attention aliases.
            (
                "attention.wq",
                "self_attn.q_proj",
            ),
            (
                "attention.wk",
                "self_attn.k_proj",
            ),
            (
                "attention.wv",
                "self_attn.v_proj",
            ),
            (
                "attention.wo",
                "self_attn.o_proj",
            ),

            (
                "attn.wq",
                "self_attn.q_proj",
            ),
            (
                "attn.wk",
                "self_attn.k_proj",
            ),
            (
                "attn.wv",
                "self_attn.v_proj",
            ),
            (
                "attn.wo",
                "self_attn.o_proj",
            ),

            (
                "q_proj",
                "self_attn.q_proj",
            ),
            (
                "k_proj",
                "self_attn.k_proj",
            ),
            (
                "v_proj",
                "self_attn.v_proj",
            ),
            (
                "o_proj",
                "self_attn.o_proj",
            ),

            # FFN aliases.
            (
                "feed_forward.w1",
                "mlp.gate_proj",
            ),
            (
                "feed_forward.w2",
                "mlp.down_proj",
            ),
            (
                "feed_forward.w3",
                "mlp.up_proj",
            ),

            (
                "w1",
                "gate_proj",
            ),
            (
                "w2",
                "down_proj",
            ),
            (
                "w3",
                "up_proj",
            ),

            # Norm aliases.
            (
                "attention_norm",
                "input_layernorm",
            ),
            (
                "ffn_norm",
                "post_attention_layernorm",
            ),
        ]

        for source, target in replacements:

            value = value.replace(
                source,
                target,
            )

        # Normalize layer index placeholder.
        value = re.sub(
            r"(?:layers|h)\.\d+\.",
            "layers.{layer}.",
            value,
        )

        # Remove wrapper differences.
        value = re.sub(
            r"^(?:base_model\.)+",
            "",
            value,
        )

        value = re.sub(
            r"^(?:model\.)+",
            "",
            value,
        )

        return value

    # ========================================================
    # MATCH SCORE
    # ========================================================

    def _match_score(
        self,
        key_a: str,
        key_b: str,
        shape_a: Tuple[int, ...],
        shape_b: Tuple[int, ...],
        layers_a: int,
        layers_b: int,
    ) -> float:

        score = 0.0

        canonical_a = (
            self._canonical_key(
                key_a
            )
        )

        canonical_b = (
            self._canonical_key(
                key_b
            )
        )

        # Same canonical structure.
        if canonical_a == canonical_b:

            score += 100.0

        # Same role.
        if (
            self._parameter_role(
                key_a
            )
            ==
            self._parameter_role(
                key_b
            )
        ):

            score += 15.0

        # Exact shape.
        if shape_a == shape_b:

            score += 20.0

        # Compatible rank.
        elif (
            len(shape_a)
            == len(shape_b)
        ):

            score += 5.0

        # Layer position.
        la = self._layer_index(
            key_a
        )

        lb = self._layer_index(
            key_b
        )

        if (
            la is not None
            and lb is not None
        ):

            expected_b = (
                self._translate_layer_index(
                    la,
                    layers_a,
                    layers_b,
                )
            )

            distance = abs(
                expected_b
                - lb
            )

            score += max(
                0.0,
                15.0
                - 3.0 * distance,
            )

        # Same suffix.
        suffix_a = canonical_a.split(
            "layers.{layer}."
        )[-1]

        suffix_b = canonical_b.split(
            "layers.{layer}."
        )[-1]

        if suffix_a == suffix_b:

            score += 10.0

        return score

    # ========================================================
    # MATCH KEY
    # ========================================================

    def _find_matching_key(
        self,
        key_a: str,
        sd_a: Mapping[str, torch.Tensor],
        sd_b: Mapping[str, torch.Tensor],
        layers_a: int,
        layers_b: int,
    ) -> Optional[str]:

        # Exact.
        if key_a in sd_b:

            return key_a

        target_tensor = sd_a.get(
            key_a
        )

        if target_tensor is None:

            return None

        shape_a = tuple(
            target_tensor.shape
        )

        # ----------------------------------------------------
        # First pass: canonical matching
        # ----------------------------------------------------

        canonical_a = (
            self._canonical_key(
                key_a
            )
        )

        candidates = []

        for key_b, tensor_b in sd_b.items():

            canonical_b = (
                self._canonical_key(
                    key_b
                )
            )

            if (
                canonical_a
                == canonical_b
            ):

                candidates.append(
                    (
                        1000.0
                        + (
                            50.0
                            if tuple(
                                tensor_b.shape
                            )
                            == shape_a
                            else 0.0
                        ),
                        key_b,
                    )
                )

        if candidates:

            candidates.sort(
                reverse=True
            )

            return candidates[0][1]

        # ----------------------------------------------------
        # Scored fallback
        # ----------------------------------------------------

        best_key = None
        best_score = -float(
            "inf"
        )

        role_a = self._parameter_role(
            key_a
        )

        layer_a = self._layer_index(
            key_a
        )

        expected_layer_b = None

        if layer_a is not None:

            expected_layer_b = (
                self._translate_layer_index(
                    layer_a,
                    layers_a,
                    layers_b,
                )
            )

        for key_b, tensor_b in sd_b.items():

            role_b = (
                self._parameter_role(
                    key_b
                )
            )

            if role_a != role_b:

                continue

            layer_b = self._layer_index(
                key_b
            )

            if (
                expected_layer_b is not None
                and layer_b is not None
                and abs(
                    layer_b
                    - expected_layer_b
                )
                > 2
            ):

                continue

            score = self._match_score(
                key_a,
                key_b,
                shape_a,
                tuple(
                    tensor_b.shape
                ),
                layers_a,
                layers_b,
            )

            if score > best_score:

                best_score = score
                best_key = key_b

        # Avoid absurd fuzzy matches.
        if (
            best_key is None
            or best_score < 20.0
        ):

            return None

        return best_key

    # ========================================================
    # SHAPE ADAPTATION
    # ========================================================

    @staticmethod
    def _crop_pad_1d(
        source: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:

        target_n = target_shape[0]

        out = torch.zeros(
            target_n,
            dtype=torch.float32,
            device=source.device,
        )

        n = min(
            target_n,
            source.shape[0],
        )

        out[:n] = source[:n]

        return out

    @staticmethod
    def _crop_pad_2d(
        source: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:

        rows, cols = target_shape

        out = torch.zeros(
            rows,
            cols,
            dtype=torch.float32,
            device=source.device,
        )

        r = min(
            rows,
            source.shape[0],
        )

        c = min(
            cols,
            source.shape[1],
        )

        out[:r, :c] = (
            source[:r, :c]
        )

        return out

    @staticmethod
    def _interpolate_2d(
        source: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:

        import torch.nn.functional as F

        x = source.float()

        x = x.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        x = F.interpolate(
            x,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )

        return x[0, 0]

    def _align_tensor(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        role: str = "weight",
    ) -> Tuple[
        Optional[torch.Tensor],
        bool,
    ]:

        if (
            source.shape
            == target.shape
        ):

            return (
                source,
                False,
            )

        if not self.allow_shape_adaptation:

            return (
                None,
                False,
            )

        if source.ndim != target.ndim:

            return (
                None,
                False,
            )

        if (
            role in (
                "embedding",
                "lm_head",
            )
        ):

            # Vocabulary dimensions should NOT be blurred
            # with interpolation. Preserve matching token rows.
            if source.ndim == 2:

                return (
                    self._crop_pad_2d(
                        source,
                        tuple(
                            target.shape
                        ),
                    ),
                    True,
                )

        if source.ndim == 1:

            return (
                self._crop_pad_1d(
                    source,
                    tuple(
                        target.shape
                    ),
                ),
                True,
            )

        if source.ndim == 2:

            if self.shape_strategy == "interpolate":

                try:

                    return (
                        self._interpolate_2d(
                            source,
                            tuple(
                                target.shape
                            ),
                        ),
                        True,
                    )

                except Exception:

                    pass

            return (
                self._crop_pad_2d(
                    source,
                    tuple(
                        target.shape
                    ),
                ),
                True,
            )

        # Conservative generic tensor adaptation.
        out = torch.zeros(
            target.shape,
            dtype=torch.float32,
            device=source.device,
        )

        slices = tuple(
            slice(
                0,
                min(
                    sa,
                    sb,
                ),
            )
            for sa, sb in zip(
                source.shape,
                target.shape,
            )
        )

        try:

            out[slices] = source[
                slices
            ]

            return (
                out,
                True,
            )

        except Exception:

            return (
                None,
                False,
            )

    # ========================================================
    # PROCRUSTES
    # ========================================================

    def _procrustes_align(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:

        if (
            target.ndim != 2
            or source.ndim != 2
            or target.shape
            != source.shape
        ):

            return source

        try:

            a = target.float()
            b = source.float()

            a_mean = a.mean(
                dim=0,
                keepdim=True,
            )

            b_mean = b.mean(
                dim=0,
                keepdim=True,
            )

            a_centered = (
                a
                - a_mean
            )

            b_centered = (
                b
                - b_mean
            )

            cross = (
                b_centered.T
                @ a_centered
            )

            u, _, vh = torch.linalg.svd(
                cross,
                full_matrices=False,
            )

            rotation = (
                u
                @ vh
            )

            aligned = (
                b_centered
                @ rotation
                + a_mean
            )

            return aligned

        except Exception as exc:

            logger.debug(
                "Procrustes failed: %s",
                exc,
            )

            return source

    # ========================================================
    # TIES-LIKE CONFLICT MERGE
    # ========================================================

    def _conflict_aware_merge(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha_b: float,
        conflict: float,
    ) -> torch.Tensor:
        """
        Two-model conflict-aware merge.

        This isn't pretending that two models provide the full
        TIES information used by multi-task task-vector merging.

        Instead:

            delta = B - A

        Small low-signal deltas are trimmed.
        High conflict regions are suppressed.
        Strong updates receive more trust.
        """

        a32 = a.float()
        b32 = b.float()

        delta = (
            b32
            - a32
        )

        magnitude = (
            delta.abs()
        )

        mean_delta = float(
            magnitude.mean().item()
        )

        if mean_delta <= 1e-12:

            return a32

        # ----------------------------------------------------
        # Adaptive trim threshold
        # ----------------------------------------------------

        q = torch.quantile(
            magnitude.flatten(),
            0.25,
        )

        threshold = max(
            mean_delta * 0.25,
            float(q.item()),
        )

        keep = (
            magnitude
            >= threshold
        )

        # ----------------------------------------------------
        # Conflict suppression
        # ----------------------------------------------------

        signs_disagree = (
            (a32 * b32)
            < 0
        )

        suppression = (
            1.0
            - (
                float(
                    min(
                        1.0,
                        max(
                            0.0,
                            conflict,
                        ),
                    )
                )
                * 0.75
                * signs_disagree.float()
            )
        )

        # Low signal -> almost no update.
        trimmed_delta = (
            delta
            * keep.float()
            * suppression
        )

        return (
            a32
            + float(alpha_b)
            * trimmed_delta
        )

    # ========================================================
    # SLERP
    # ========================================================

    def _slerp(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:

        a32 = a.float().reshape(
            -1
        )

        b32 = b.float().reshape(
            -1
        )

        na = (
            torch.linalg.vector_norm(
                a32
            )
            + 1e-12
        )

        nb = (
            torch.linalg.vector_norm(
                b32
            )
            + 1e-12
        )

        ua = a32 / na
        ub = b32 / nb

        dot = torch.clamp(
            torch.dot(
                ua,
                ub,
            ),
            -0.9995,
            0.9995,
        )

        theta = torch.acos(
            dot
        )

        sin_theta = torch.sin(
            theta
        )

        if (
            not torch.isfinite(
                sin_theta
            )
            or abs(
                float(
                    sin_theta.item()
                )
            )
            < 1e-6
        ):

            return (
                (1.0 - alpha)
                * a32
                + alpha
                * b32
            ).reshape(
                a.shape
            )

        w_a = (
            torch.sin(
                (1.0 - alpha)
                * theta
            )
            / sin_theta
        )

        w_b = (
            torch.sin(
                alpha
                * theta
            )
            / sin_theta
        )

        # Preserve interpolation of norms.
        norm = (
            (1.0 - alpha) * na
            + alpha * nb
        )

        result = (
            w_a * ua
            + w_b * ub
        ) * norm

        return result.reshape(
            a.shape
        )

    # ========================================================
    # FISHER TRUST
    # ========================================================

    @staticmethod
    def _mean_fisher(
        fisher: Optional[
            Mapping[str, torch.Tensor]
        ],
        name: str,
    ) -> float:

        if (
            fisher is None
            or name not in fisher
        ):

            return 0.0

        try:

            value = fisher[
                name
            ].float()

            value = value[
                torch.isfinite(value)
            ]

            if value.numel() == 0:

                return 0.0

            return _safe_float(
                value.mean()
            )

        except Exception:

            return 0.0

    def _fisher_alpha(
        self,
        name: str,
        default_alpha: float,
    ) -> Tuple[
        float,
        float,
        float,
    ]:

        fa = self._mean_fisher(
            self.fisher_a,
            name,
        )

        fb = self._mean_fisher(
            self.fisher_b,
            name,
        )

        if (
            fa <= 0.0
            or fb <= 0.0
        ):

            return (
                default_alpha,
                fa,
                fb,
            )

        # Log compression prevents one tensor from dominating.
        fa_log = math.log1p(
            fa
        )

        fb_log = math.log1p(
            fb
        )

        total = (
            fa_log
            + fb_log
            + 1e-12
        )

        confidence = min(
            1.0,
            abs(
                fa_log
                - fb_log
            )
            / total,
        )

        alpha = (
            fb_log
            / total
        )

        # Blend with user/Captain prior.
        alpha = (
            0.65 * alpha
            + 0.35 * default_alpha
        )

        return (
            float(
                max(
                    0.05,
                    min(
                        0.95,
                        alpha,
                    ),
                )
            ),
            fa,
            fb,
        )

    # ========================================================
    # TENSOR RELATION
    # ========================================================

    def _tensor_relation(
        self,
        name: str,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Cheap tensor relationship.

        Uses tensor_state.compare_tensors when available.
        Falls back to direct vector metrics.
        """

        if compare_tensors is not None:

            try:

                rel = compare_tensors(
                    name,
                    a,
                    b,
                    max_elements=100_000,
                )

                return {
                    "cosine":
                        float(
                            rel.cosine_similarity
                        ),

                    "relative_delta":
                        float(
                            rel.relative_delta_l2
                        ),

                    "sign_conflict":
                        float(
                            rel.strong_conflict_ratio
                        ),

                    "overlap":
                        float(
                            rel.activation_overlap
                        ),

                    "norm_a":
                        float(
                            rel.norm_a
                        ),

                    "norm_b":
                        float(
                            rel.norm_b
                        ),
                }

            except Exception as exc:

                logger.debug(
                    "tensor_state relation failed: %s",
                    exc,
                )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        a32 = a.float().reshape(
            -1
        )

        b32 = b.float().reshape(
            -1
        )

        finite = (
            torch.isfinite(
                a32
            )
            &
            torch.isfinite(
                b32
            )
        )

        a32 = a32[
            finite
        ]

        b32 = b32[
            finite
        ]

        if a32.numel() == 0:

            return {
                "cosine": 0.0,
                "relative_delta": 1.0,
                "sign_conflict": 1.0,
                "overlap": 0.0,
                "norm_a": 0.0,
                "norm_b": 0.0,
            }

        norm_a = _tensor_norm(
            a32
        )

        norm_b = _tensor_norm(
            b32
        )

        cosine = 0.0

        if (
            norm_a > 1e-12
            and norm_b > 1e-12
        ):

            cosine = _safe_float(
                torch.dot(
                    a32,
                    b32,
                )
                / (
                    norm_a
                    * norm_b
                )
            )

        delta = (
            b32 - a32
        )

        relative_delta = (
            _tensor_norm(
                delta
            )
            / max(
                norm_a,
                1e-12,
            )
        )

        active = (
            (
                a32.abs()
                > 0.05
                * max(
                    float(
                        a32.abs().mean()
                        .item()
                    ),
                    1e-12,
                )
            )
            &
            (
                b32.abs()
                > 0.05
                * max(
                    float(
                        b32.abs().mean()
                        .item()
                    ),
                    1e-12,
                )
            )
        )

        if active.any():

            sign_conflict = _safe_float(
                (
                    (
                        a32[active]
                        * b32[active]
                    )
                    < 0
                )
                .float()
                .mean()
            )

        else:

            sign_conflict = 0.0

        overlap = _safe_float(
            active.float().mean()
        )

        return {
            "cosine": cosine,
            "relative_delta":
                relative_delta,
            "sign_conflict":
                sign_conflict,
            "overlap":
                overlap,
            "norm_a":
                norm_a,
            "norm_b":
                norm_b,
        }

    # ========================================================
    # LOCAL INTELLIGENCE
    # ========================================================

    def _build_tensor_decision(
        self,
        key_a: str,
        key_b: str,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> MergeTensorDecision:

        role = self._parameter_role(
            key_a
        )

        relation = self._tensor_relation(
            key_a,
            a,
            b,
        )

        cosine = relation[
            "cosine"
        ]

        relative_delta = relation[
            "relative_delta"
        ]

        conflict = relation[
            "sign_conflict"
        ]

        norm_a = relation[
            "norm_a"
        ]

        norm_b = relation[
            "norm_b"
        ]

        norm_ratio = (
            norm_b
            / max(
                norm_a,
                1e-12,
            )
        )

        # ----------------------------------------------------
        # Fisher
        # ----------------------------------------------------

        (
            alpha,
            fisher_a,
            fisher_b,
        ) = self._fisher_alpha(
            key_a,
            self.captain_policy.alpha_b,
        )

        fisher_total = (
            fisher_a
            + fisher_b
            + 1e-12
        )

        fisher_confidence = (
            abs(
                fisher_b
                - fisher_a
            )
            / fisher_total
        )

        # ----------------------------------------------------
        # Protection
        # ----------------------------------------------------

        protected = False

        if role == "embedding":
            protected = (
                self.captain_policy
                .protect_embeddings
            )

        elif role == "lm_head":
            protected = (
                self.captain_policy
                .protect_lm_head
            )

        elif role == "norm":
            protected = (
                self.captain_policy
                .protect_norms
            )

        # ----------------------------------------------------
        # Alpha caps for sensitive parameters.
        # ----------------------------------------------------

        if role == "embedding":

            alpha = min(
                alpha,
                0.35,
            )

        elif role == "lm_head":

            alpha = min(
                alpha,
                0.40,
            )

        elif role == "norm":

            alpha = min(
                alpha,
                0.25,
            )

        elif role == "bias":

            alpha = min(
                alpha,
                0.50,
            )

        # ----------------------------------------------------
        # Scale mismatch
        # ----------------------------------------------------

        if norm_ratio > 3.0:

            alpha *= 0.55

        elif norm_ratio > 2.0:

            alpha *= 0.75

        elif norm_ratio < 0.33:

            alpha *= 0.65

        # ----------------------------------------------------
        # Strategy
        # ----------------------------------------------------

        strategy = _normalize_strategy(
            self.captain_policy.strategy
        )

        reason = []

        # High conflict.
        if (
            conflict
            >= self.captain_policy
            .conflict_threshold
        ):

            if (
                self.captain_policy
                .prefer_ties_for_conflicts
            ):

                strategy = "ties"

                reason.append(
                    "high sign conflict"
                )

        # Very high similarity.
        elif (
            cosine >= 0.92
            and relative_delta < 0.40
        ):

            if strategy == "intelligent":

                strategy = "slerp"

                reason.append(
                    "high structural similarity"
                )

        # Very low similarity.
        elif (
            cosine < 0.15
            and relative_delta > 1.0
        ):

            strategy = "weighted"

            alpha *= 0.55

            reason.append(
                "weak structural agreement"
            )

        # Fisher preference.
        if (
            self.captain_policy.trust_fisher
            and fisher_a > 0.0
            and fisher_b > 0.0
        ):

            if (
                fisher_b > 1.75 * fisher_a
            ):

                alpha = min(
                    0.80,
                    alpha
                    + 0.10,
                )

                reason.append(
                    "Fisher favors B"
                )

            elif (
                fisher_a > 1.75 * fisher_b
            ):

                alpha = max(
                    0.10,
                    alpha
                    - 0.10,
                )

                reason.append(
                    "Fisher favors A"
                )

        # Protected tensor fallback.
        if protected:

            if role == "norm":

                strategy = "weighted"

                alpha = min(
                    alpha,
                    0.20,
                )

            elif role == "embedding":

                strategy = "weighted"

            elif role == "lm_head":

                strategy = "weighted"

        # Prevent pathological alpha.
        alpha = max(
            0.0,
            min(
                1.0,
                float(alpha),
            ),
        )

        return MergeTensorDecision(
            target_key=key_a,
            source_key=key_b,
            role=role,
            strategy=strategy,
            alpha_b=alpha,
            cosine=cosine,
            relative_delta=relative_delta,
            sign_conflict=conflict,
            overlap=relation[
                "overlap"
            ],
            norm_a=norm_a,
            norm_b=norm_b,
            norm_ratio=norm_ratio,
            fisher_a=fisher_a,
            fisher_b=fisher_b,
            fisher_confidence=fisher_confidence,
            protected=protected,
            reason="; ".join(
                reason
            )
            or "standard intelligent merge",
        )

    # ========================================================
    # MERGE TENSOR
    # ========================================================

    def _merge_tensor(
        self,
        decision: MergeTensorDecision,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:

        alpha = decision.alpha_b

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if not _finite(a):

            return b

        if not _finite(b):

            return a

        # ----------------------------------------------------
        # Keep A
        # ----------------------------------------------------

        if decision.keep_a:

            return a

        if decision.keep_b:

            return b

        # ----------------------------------------------------
        # Norm-aware scaling
        # ----------------------------------------------------

        a32 = a.float()
        b32 = b.float()

        if (
            decision.norm_a > 1e-8
            and decision.norm_b > 1e-8
        ):

            ratio = (
                decision.norm_a
                / decision.norm_b
            )

            # Only repair moderate scale mismatch.
            if (
                0.5
                <= ratio
                <= 2.0
            ):

                b32 = (
                    b32
                    * float(ratio)
                )

        strategy = decision.strategy

        # ----------------------------------------------------
        # Fisher
        # ----------------------------------------------------

        if strategy == "fisher":

            if (
                self.fisher_a is not None
                and self.fisher_b is not None
            ):

                fa = self.fisher_a.get(
                    decision.target_key
                )

                fb = self.fisher_b.get(
                    decision.target_key
                )

                if (
                    fa is not None
                    and fb is not None
                ):

                    try:

                        fa = fa.to(
                            device=a32.device,
                            dtype=torch.float32,
                        )

                        fb = fb.to(
                            device=a32.device,
                            dtype=torch.float32,
                        )

                        denom = (
                            fa
                            + fb
                            + 1e-8
                        )

                        merged = (
                            fa
                            * a32
                            + fb
                            * b32
                        ) / denom

                        if _finite(
                            merged
                        ):

                            return merged

                    except Exception as exc:

                        logger.debug(
                            "Fisher tensor merge failed "
                            "for %s: %s",
                            decision.target_key,
                            exc,
                        )

            strategy = "weighted"

        # ----------------------------------------------------
        # TIES / conflict
        # ----------------------------------------------------

        if strategy == "ties":

            return self._conflict_aware_merge(
                a32,
                b32,
                alpha,
                decision.sign_conflict,
            )

        # ----------------------------------------------------
        # SLERP
        # ----------------------------------------------------

        if strategy == "slerp":

            try:

                if (
                    _CPP_MERGE_OK
                    and fast_slerp is not None
                ):

                    result = fast_slerp(
                        a32,
                        b32,
                        alpha,
                    )

                else:

                    result = self._slerp(
                        a32,
                        b32,
                        alpha,
                    )

                if _finite(result):

                    return result

            except Exception as exc:

                logger.debug(
                    "SLERP failed for %s: %s",
                    decision.target_key,
                    exc,
                )

            strategy = "weighted"

        # ----------------------------------------------------
        # Weighted
        # ----------------------------------------------------

        if strategy == "weighted":

            try:

                if (
                    _CPP_MERGE_OK
                    and fast_weighted_avg is not None
                ):

                    result = fast_weighted_avg(
                        a32,
                        b32,
                        alpha,
                    )

                else:

                    result = (
                        (1.0 - alpha)
                        * a32
                        + alpha
                        * b32
                    )

                if _finite(result):

                    return result

            except Exception as exc:

                logger.debug(
                    "Weighted merge failed for %s: %s",
                    decision.target_key,
                    exc,
                )

        # Final safe fallback.
        return (
            (1.0 - alpha)
            * a32
            + alpha
            * b32
        )

    # ========================================================
    # CAPTAIN GLOBAL DECISION
    # ========================================================

    @staticmethod
    def _extract_text(
        result: Any,
    ) -> str:

        if result is None:

            return ""

        if isinstance(
            result,
            str,
        ):

            return result

        if isinstance(
            result,
            Mapping,
        ):

            for key in (
                "response",
                "text",
                "answer",
                "content",
                "output",
            ):

                if key in result:

                    return str(
                        result[key]
                    )

            return json.dumps(
                dict(result),
                ensure_ascii=False,
            )

        return str(
            result
        )

    @staticmethod
    def _parse_json(
        text: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if not text:

            return None

        # Direct JSON.
        try:

            value = json.loads(
                text
            )

            if isinstance(
                value,
                dict,
            ):

                return value

        except Exception:

            pass

        # Fenced JSON.
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL,
        )

        if match:

            try:

                value = json.loads(
                    match.group(1)
                )

                if isinstance(
                    value,
                    dict,
                ):

                    return value

            except Exception:

                pass

        # First JSON object.
        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):

            try:

                value = json.loads(
                    text[
                        start:end + 1
                    ]
                )

                if isinstance(
                    value,
                    dict,
                ):

                    return value

            except Exception:

                pass

        return None

    def _ask_captain(
        self,
        prompt: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if self.captain is None:

            return None

        methods = (
            "ask",
            "query",
            "generate",
            "run",
            "decide",
        )

        for method_name in methods:

            method = getattr(
                self.captain,
                method_name,
                None,
            )

            if method is None:

                continue

            try:

                signature = inspect.signature(
                    method
                )

                kwargs = {}

                if "prompt" in signature.parameters:

                    kwargs["prompt"] = prompt

                    result = method(
                        **kwargs
                    )

                elif "question" in signature.parameters:

                    kwargs["question"] = prompt

                    result = method(
                        **kwargs
                    )

                else:

                    result = method(
                        prompt
                    )

                text = self._extract_text(
                    result
                )

                return self._parse_json(
                    text
                )

            except Exception as exc:

                logger.debug(
                    "Captain method %s failed: %s",
                    method_name,
                    exc,
                )

        return None

    def _captain_global_policy(
        self,
        *,
        layers_a: int,
        layers_b: int,
        params_a: int,
        params_b: int,
    ) -> CaptainMergePolicy:

        policy = CaptainMergePolicy(
            strategy=self.strategy,
            alpha_b=self.alpha,
            conflict_threshold=(
                self.cba_conflict_threshold
            ),
            trust_fisher=(
                self.use_fisher_protection
            ),
            use_cba=self.use_cba,
            use_projection=self.cba_projection,
            allow_shape_adaptation=(
                self.allow_shape_adaptation
            ),
            shape_strategy=self.shape_strategy,
            adaptation_steps=(
                self.adaptation_steps
            ),
            adaptation_lr=(
                self.adaptation_lr
            ),
        )

        if self.captain is None:

            return policy

        prompt = f"""
You are Phoenix Captain, the strategic controller of FTRAIN's
two-brain model merger.

Your job is NOT to invent arbitrary neural transformations.
Choose a safe global merge policy from the supported operations.

MODEL A:
- output architecture
- layers: {layers_a}
- parameters: {params_a:,}

MODEL B:
- source knowledge
- layers: {layers_b}
- parameters: {params_b:,}

FTRAIN features available:
- weighted merge
- SLERP
- conflict-aware TIES-style merge
- Fisher-weighted merge
- CBA (Captain Brain Alignment)
- optional Procrustes alignment
- protected embeddings
- protected lm_head
- protected normalization parameters
- conservative shape adaptation
- targeted post-merge calibration

Return ONLY valid JSON:

{{
  "strategy": "intelligent|weighted|slerp|ties|fisher",
  "alpha_b": 0.0,
  "conflict_threshold": 0.0,
  "trust_fisher": true,
  "use_cba": true,
  "use_projection": false,
  "protect_embeddings": true,
  "protect_lm_head": true,
  "protect_norms": true,
  "prefer_ties_for_conflicts": true,
  "allow_shape_adaptation": true,
  "shape_strategy": "crop_pad|interpolate",
  "calibration_steps": 50,
  "adaptation_lr": 0.000001,
  "confidence": 0.0,
  "explanation": "brief reason"
}}

Do not choose unsupported strategies.
"""

        result = self._ask_captain(
            prompt
        )

        if not result:

            return policy

        # ----------------------------------------------------
        # Sanitized Captain controls.
        # ----------------------------------------------------

        try:

            policy.strategy = _normalize_strategy(
                str(
                    result.get(
                        "strategy",
                        policy.strategy,
                    )
                )
            )

        except Exception:

            pass

        if policy.strategy not in {
            "intelligent",
            "weighted",
            "slerp",
            "ties",
            "fisher",
        }:

            policy.strategy = "intelligent"

        for field_name in (
            "alpha_b",
            "conflict_threshold",
            "calibration_steps",
            "adaptation_lr",
            "confidence",
        ):

            if field_name in result:

                try:

                    value = float(
                        result[field_name]
                    )

                    if field_name in (
                        "alpha_b",
                        "conflict_threshold",
                        "confidence",
                    ):

                        value = max(
                            0.0,
                            min(
                                1.0,
                                value,
                            ),
                        )

                    elif field_name == "calibration_steps":

                        value = max(
                            0,
                            min(
                                500,
                                int(value),
                            ),
                        )

                    elif field_name == "adaptation_lr":

                        value = max(
                            1e-8,
                            min(
                                1e-3,
                                value,
                            ),
                        )

                    setattr(
                        policy,
                        field_name,
                        value,
                    )

                except Exception:

                    pass

        for field_name in (
            "trust_fisher",
            "use_cba",
            "use_projection",
            "protect_embeddings",
            "protect_lm_head",
            "protect_norms",
            "prefer_ties_for_conflicts",
            "allow_shape_adaptation",
        ):

            if field_name in result:

                setattr(
                    policy,
                    field_name,
                    bool(
                        result[field_name]
                    ),
                )

        shape_strategy = str(
            result.get(
                "shape_strategy",
                policy.shape_strategy,
            )
        ).lower()

        if shape_strategy in {
            "crop_pad",
            "interpolate",
        }:

            policy.shape_strategy = (
                shape_strategy
            )

        policy.explanation = str(
            result.get(
                "explanation",
                "",
            )
        )

        return policy

    # ========================================================
    # CBA
    # ========================================================

    def _run_cba(
        self,
        sd_a: Mapping[str, torch.Tensor],
        sd_b: Mapping[str, torch.Tensor],
    ):

        self._cba_report = None

        if not self.captain_policy.use_cba:

            return None

        try:

            from .cba import (
                run_cba,
            )

        except Exception as exc:

            logger.info(
                "CBA module unavailable: %s",
                exc,
            )

            if self.cba_fallback == "abort":

                raise RuntimeError(
                    "CBA requested but unavailable."
                )

            return None

        try:

            report = run_cba(
                sd_a,
                sd_b,
                high_conflict_threshold=(
                    self.captain_policy
                    .conflict_threshold
                ),
            )

            self._cba_report = report

            return report

        except Exception as exc:

            if self.cba_fallback == "abort":

                raise RuntimeError(
                    "CBA execution failed."
                ) from exc

            logger.warning(
                "CBA failed; "
                "continuing without CBA: %s",
                exc,
            )

            return None

    # ========================================================
    # CALIBRATION DATA
    # ========================================================

    def _build_calibration_loader(
        self,
        tokenizer,
    ):

        if (
            load_data is None
            or FtrainDataset is None
            or collate is None
        ):

            return None

        calibration_data = getattr(
            self.config,
            "calibration_data",
            None,
        )

        if not calibration_data:

            return None

        try:

            data = load_data(
                calibration_data
            )

            data = list(
                data
            )[
                : self.max_calibration_samples
            ]

            dataset = FtrainDataset(
                data,
                tokenizer,
                self.calibration_length,
            )

            return DataLoader(
                dataset,
                batch_size=int(
                    getattr(
                        self.config,
                        "merge_calibration_batch",
                        1,
                    )
                ),
                shuffle=False,
                collate_fn=partial(
                    collate,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        or 0
                    ),
                ),
            )

        except Exception as exc:

            logger.warning(
                "Calibration loader failed: %s",
                exc,
            )

            return None

    # ========================================================
    # BATCH DEVICE
    # ========================================================

    @staticmethod
    def _move_batch(
        batch: Mapping[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:

        output = {}

        for key, value in batch.items():

            if torch.is_tensor(
                value
            ):

                output[key] = value.to(
                    device,
                    non_blocking=(
                        device.type
                        == "cuda"
                    ),
                )

            else:

                output[key] = value

        return output

    # ========================================================
    # MODEL INPUT DEVICE
    # ========================================================

    @staticmethod
    def _model_device(
        model: nn.Module,
    ) -> torch.device:

        # Prefer embeddings.
        for name, param in model.named_parameters():

            lower = name.lower()

            if (
                "embed_tokens"
                in lower
                or "wte"
                in lower
            ):

                return param.device

        try:

            return next(
                model.parameters()
            ).device

        except StopIteration:

            return torch.device(
                "cpu"
            )

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    def _evaluate_model(
        self,
        model,
        loader,
        device: Optional[
            torch.device
        ] = None,
        max_batches: int = 16,
    ) -> float:

        if loader is None:

            return float(
                "inf"
            )

        if device is None:

            device = self._model_device(
                model
            )

        model.eval()

        total = 0.0
        count = 0

        with torch.inference_mode():

            for index, batch in enumerate(
                loader
            ):

                if index >= max_batches:

                    break

                batch = self._move_batch(
                    batch,
                    device,
                )

                try:

                    outputs = model(
                        **batch
                    )

                    loss = getattr(
                        outputs,
                        "loss",
                        None,
                    )

                    if (
                        loss is None
                        or not torch.isfinite(
                            loss
                        )
                    ):

                        continue

                    total += float(
                        loss.item()
                    )

                    count += 1

                except RuntimeError as exc:

                    if "out of memory" in str(
                        exc
                    ).lower():

                        self._purge_memory()

                        break

                    logger.debug(
                        "Evaluation batch failed: %s",
                        exc,
                    )

                except Exception as exc:

                    logger.debug(
                        "Evaluation failed: %s",
                        exc,
                    )

        return (
            total / max(
                1,
                count,
            )
        )

    # ========================================================
    # TARGETED BRAIN REPAIR
    # ========================================================

    def _select_repair_keys(
        self,
    ) -> List[str]:

        decisions = list(
            self._merge_records
        )

        # Highest conflict + largest relative change.
        decisions.sort(
            key=lambda d: (
                d.sign_conflict
                * 2.0
                + d.relative_delta
                + (
                    1.0
                    - max(
                        -1.0,
                        min(
                            1.0,
                            d.cosine,
                        )
                    )
                ),
            ),
            reverse=True,
        )

        selected = []

        for decision in decisions:

            if decision.protected:

                continue

            if (
                decision.strategy
                not in (
                    "ties",
                    "weighted",
                    "fisher",
                    "slerp",
                )
            ):

                continue

            selected.append(
                decision.target_key
            )

            if (
                len(selected)
                >= self.adaptation_max_tensors
            ):

                break

        return selected

    def _targeted_adaptation(
        self,
        model,
        tokenizer,
        loader,
        device,
    ) -> Dict[str, Any]:

        result = {
            "enabled": False,
            "steps": 0,
            "initial_loss": float(
                "inf"
            ),
            "final_loss": float(
                "inf"
            ),
            "best_loss": float(
                "inf"
            ),
            "rolled_back": False,
            "repair_tensors": [],
        }

        if (
            not self.adaptation_enabled
            or loader is None
        ):

            return result

        repair_keys = (
            self._select_repair_keys()
        )

        if not repair_keys:

            return result

        # ----------------------------------------------------
        # Put model on training device if possible.
        # ----------------------------------------------------

        try:

            model.to(
                device
            )

        except Exception as exc:

            logger.warning(
                "Targeted adaptation skipped; "
                "model cannot fit on %s: %s",
                device,
                exc,
            )

            return result

        # ----------------------------------------------------
        # Freeze everything except selected tensors.
        # ----------------------------------------------------

        repair_set = set(
            repair_keys
        )

        trainable = []

        for name, parameter in (
            model.named_parameters()
        ):

            enabled = (
                name in repair_set
            )

            parameter.requires_grad = (
                enabled
            )

            if enabled:

                trainable.append(
                    parameter
                )

        if not trainable:

            return result

        # ----------------------------------------------------
        # Snapshot selected parameters.
        # ----------------------------------------------------

        original = {}

        for name, parameter in (
            model.named_parameters()
        ):

            if (
                name in repair_set
            ):

                original[name] = (
                    parameter.detach()
                    .clone()
                )

        optimizer = torch.optim.AdamW(
            trainable,
            lr=(
                self.captain_policy
                .adaptation_lr
            ),
            weight_decay=(
                self.adaptation_weight_decay
            ),
        )

        initial_loss = (
            self._evaluate_model(
                model,
                loader,
                device,
                max_batches=8,
            )
        )

        result.update(
            {
                "enabled": True,
                "initial_loss": initial_loss,
                "best_loss": initial_loss,
                "repair_tensors": repair_keys,
            }
        )

        if not math.isfinite(
            initial_loss
        ):

            result["enabled"] = False

            return result

        best_state = {
            key: value.clone()
            for key, value in original.items()
        }

        best_loss = initial_loss

        iterator = iter(
            loader
        )

        for step in range(
            self.captain_policy
            .calibration_steps
        ):

            try:

                batch = next(
                    iterator
                )

            except StopIteration:

                iterator = iter(
                    loader
                )

                batch = next(
                    iterator
                )

            batch = self._move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            try:

                model.train()

                outputs = model(
                    **batch
                )

                loss = getattr(
                    outputs,
                    "loss",
                    None,
                )

                if (
                    loss is None
                    or not torch.isfinite(
                        loss
                    )
                ):

                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    self.gradient_clip,
                )

                optimizer.step()

                value = float(
                    loss.detach().item()
                )

                if value < best_loss:

                    best_loss = value

                    best_state = {
                        name:
                            parameter.detach()
                            .clone()

                        for name, parameter
                        in model.named_parameters()
                        if name in repair_set
                    }

                result[
                    "steps"
                ] = step + 1

            except RuntimeError as exc:

                if "out of memory" in str(
                    exc
                ).lower():

                    logger.warning(
                        "OOM during targeted brain repair."
                    )

                    self._purge_memory()

                    break

                logger.debug(
                    "Adaptation step failed: %s",
                    exc,
                )

        # ----------------------------------------------------
        # Evaluate the best state.
        # ----------------------------------------------------

        with torch.no_grad():

            for name, parameter in (
                model.named_parameters()
            ):

                if name in best_state:

                    parameter.copy_(
                        best_state[name].to(
                            parameter.device,
                            parameter.dtype,
                        )
                    )

        final_loss = (
            self._evaluate_model(
                model,
                loader,
                device,
                max_batches=8,
            )
        )

        result[
            "final_loss"
        ] = final_loss

        result[
            "best_loss"
        ] = best_loss

        # ----------------------------------------------------
        # Rollback if repair hurt model.
        # ----------------------------------------------------

        if (
            math.isfinite(
                final_loss
            )
            and
            final_loss
            >
            initial_loss
            * (
                1.0
                + self.rollback_tolerance
            )
        ):

            logger.warning(
                "Targeted repair degraded calibration loss. "
                "Rolling back."
            )

            with torch.no_grad():

                for name, parameter in (
                    model.named_parameters()
                ):

                    if name in original:

                        parameter.copy_(
                            original[name].to(
                                parameter.device,
                                parameter.dtype,
                            )
                        )

            result[
                "rolled_back"
            ] = True

            final_loss = (
                self._evaluate_model(
                    model,
                    loader,
                    device,
                    max_batches=8,
                )
            )

            result[
                "final_loss"
            ] = final_loss

        # ----------------------------------------------------
        # Restore trainability.
        # ----------------------------------------------------

        for parameter in (
            model.parameters()
        ):

            parameter.requires_grad = True

        result[
            "improvement"
        ] = (
            (
                initial_loss
                - result["final_loss"]
            )
            / max(
                abs(
                    initial_loss
                ),
                1e-8,
            )
        )

        return result

    # ========================================================
    # SAFETY VALIDATION
    # ========================================================

    def _run_safety(
        self,
        merged_state,
        baseline,
    ) -> Dict[str, Any]:

        result = {
            "checked": True,
            "ok": True,
            "sanitized": False,
            "nonfinite": 0,
        }

        # ----------------------------------------------------
        # Direct finite check
        # ----------------------------------------------------

        for name, tensor in (
            merged_state.items()
        ):

            if (
                torch.is_tensor(
                    tensor
                )
                and
                torch.is_floating_point(
                    tensor
                )
            ):

                if not _finite(
                    tensor
                ):

                    result[
                        "nonfinite"
                    ] += 1

        if result[
            "nonfinite"
        ]:

            result["ok"] = False

        # ----------------------------------------------------
        # FTRAIN safety module
        # ----------------------------------------------------

        if (
            check_state_dict is not None
            and baseline is not None
        ):

            try:

                report = check_state_dict(
                    merged_state,
                    baseline=baseline,
                    norm_collapse_factor=0.10,
                )

                result[
                    "summary"
                ] = report.summary()

                result[
                    "ok"
                ] = bool(
                    report.ok
                )

                if not report.ok:

                    if sanitize is not None:

                        repaired = sanitize(
                            merged_state,
                            baseline,
                            norm_collapse_factor=0.10,
                        )

                        if repaired is not None:

                            merged_state = repaired

                            result[
                                "sanitized"
                            ] = True

                            result[
                                "ok"
                            ] = True

            except Exception as exc:

                result[
                    "warning"
                ] = str(
                    exc
                )

        return (
            result,
            merged_state,
        )

    # ========================================================
    # STATIC MERGE
    # ========================================================

    def _static_merge(
        self,
        sd_a: Dict[str, torch.Tensor],
        sd_b: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        layers_a = self._get_num_layers(
            sd_a
        )

        layers_b = self._get_num_layers(
            sd_b
        )

        keys_b = list(
            sd_b.keys()
        )

        total = len(
            sd_a
        )

        matched = 0
        missing = 0
        adapted = 0
        failed = 0

        counts = {
            "weighted": 0,
            "slerp": 0,
            "ties": 0,
            "fisher": 0,
            "keep_a": 0,
            "keep_b": 0,
            "missing": 0,
            "shape_adapted": 0,
        }

        cosine_sum = 0.0
        delta_sum = 0.0
        conflict_sum = 0.0

        for index, key_a in enumerate(
            list(sd_a.keys())
        ):

            if (
                index % max(
                    1,
                    total // 20,
                )
                == 0
                or index == total - 1
            ):

                logger.info(
                    "FTRAIN merge: %d/%d",
                    index + 1,
                    total,
                )

            tensor_a = sd_a[
                key_a
            ]

            if not torch.is_tensor(
                tensor_a
            ):

                continue

            key_b = self._find_matching_key(
                key_a,
                sd_a,
                sd_b,
                layers_a,
                layers_b,
            )

            if key_b is None:

                missing += 1
                counts[
                    "missing"
                ] += 1

                continue

            tensor_b = sd_b[
                key_b
            ]

            # ------------------------------------------------
            # Non-floating buffers
            # ------------------------------------------------

            if (
                not torch.is_floating_point(
                    tensor_a
                )
                or
                not torch.is_floating_point(
                    tensor_b
                )
            ):

                # Keep target architecture's metadata/buffer.
                continue

            # ------------------------------------------------
            # Move only current tensors.
            # ------------------------------------------------

            if self.merge_accelerator:

                if torch.cuda.is_available():

                    merge_device = torch.device(
                        "cuda"
                    )

                else:

                    merge_device = (
                        torch.device(
                            "cpu"
                        )
                    )

            else:

                merge_device = torch.device(
                    "cpu"
                )

            a = tensor_a.detach().to(
                merge_device,
                dtype=torch.float32,
            )

            b = tensor_b.detach().to(
                merge_device,
                dtype=torch.float32,
            )

            role = self._parameter_role(
                key_a
            )

            # ------------------------------------------------
            # Shape alignment
            # ------------------------------------------------

            b_aligned, was_adapted = (
                self._align_tensor(
                    b,
                    a,
                    role=role,
                )
            )

            if b_aligned is None:

                failed += 1

                del a
                del b

                continue

            b = b_aligned

            if was_adapted:

                adapted += 1

                counts[
                    "shape_adapted"
                ] += 1

            # ------------------------------------------------
            # Relation
            # ------------------------------------------------

            try:

                decision = (
                    self._build_tensor_decision(
                        key_a,
                        key_b,
                        a,
                        b,
                    )
                )

            except Exception as exc:

                logger.debug(
                    "Tensor decision failed for %s: %s",
                    key_a,
                    exc,
                )

                decision = (
                    MergeTensorDecision(
                        target_key=key_a,
                        source_key=key_b,
                        role=role,
                        strategy="weighted",
                        alpha_b=self.captain_policy.alpha_b,
                        reason="decision fallback",
                    )
                )

            decision.shape_aligned = (
                was_adapted
            )

            # ------------------------------------------------
            # CBA-specific directive
            # ------------------------------------------------

            directive = None

            if self._cba_report is not None:

                try:

                    routing = getattr(
                        self._cba_report,
                        "routing",
                        None,
                    )

                    directives = getattr(
                        routing,
                        "directives",
                        {},
                    )

                    directive = directives.get(
                        key_a
                    )

                except Exception:

                    directive = None

            if directive is not None:

                action = str(
                    getattr(
                        directive,
                        "action",
                        "",
                    )
                ).lower()

                alpha_a = getattr(
                    directive,
                    "alpha_a",
                    None,
                )

                if (
                    action == "ties"
                    and self.captain_policy
                    .prefer_ties_for_conflicts
                ):

                    decision.strategy = (
                        "ties"
                    )

                if alpha_a is not None:

                    try:

                        alpha_a = float(
                            alpha_a
                        )

                        decision.alpha_b = max(
                            0.0,
                            min(
                                1.0,
                                1.0 - alpha_a,
                            ),
                        )

                    except Exception:

                        pass

            # ------------------------------------------------
            # Optional projection
            # ------------------------------------------------

            if (
                self.captain_policy
                .use_projection
                and
                self.cba_projection
                and
                a.ndim == 2
                and
                b.ndim == 2
            ):

                try:

                    b = (
                        self._procrustes_align(
                            a,
                            b,
                        )
                    )

                    decision.reason += (
                        "; projection aligned"
                    )

                except Exception:

                    pass

            # ------------------------------------------------
            # Merge
            # ------------------------------------------------

            try:

                merged = self._merge_tensor(
                    decision,
                    a,
                    b,
                )

            except Exception as exc:

                logger.warning(
                    "Merge failed for %s: %s. "
                    "Keeping Model A.",
                    key_a,
                    exc,
                )

                merged = a

            if not _finite(
                merged
            ):

                failed += 1

                merged = a

            else:

                matched += 1

            cosine_sum += (
                decision.cosine
            )

            delta_sum += (
                decision.relative_delta
            )

            conflict_sum += (
                decision.sign_conflict
            )

            strategy = (
                decision.strategy
            )

            counts[
                strategy
                if strategy in counts
                else "weighted"
            ] += 1

            self._decisions.append(
                decision
            )

            self._merge_records.append(
                decision
            )

            sd_a[key_a] = (
                merged
                .detach()
                .cpu()
                .to(
                    self.dtype
                )
            )

            del a
            del b
            del merged

        counts[
            "missing"
        ] = missing

        self._report.counts = {
            **counts,
            "matched": matched,
            "missing": missing,
            "shape_adapted": adapted,
            "failed": failed,
            "total": total,
        }

        self._report.mean_cosine = (
            cosine_sum
            / max(
                1,
                matched,
            )
        )

        self._report.mean_relative_delta = (
            delta_sum
            / max(
                1,
                matched,
            )
        )

        self._report.mean_conflict = (
            conflict_sum
            / max(
                1,
                matched,
            )
        )

        return sd_a

    # ========================================================
    # CANDIDATE SEARCH
    # ========================================================

    def _candidate_strategies(
        self,
    ) -> List[str]:

        strategies = [
            "intelligent",
            "weighted",
            "ties",
        ]

        if (
            self.fisher_a is not None
            and
            self.fisher_b is not None
        ):

            strategies.append(
                "fisher"
            )

        return strategies

    # ========================================================
    # BUILD CANDIDATE MODEL
    # ========================================================

    def _load_target_model(
        self,
    ):

        model, tokenizer = (
            _load_model_any(
                self.model_a,
                prefer_unsloth=False,
                load_in_4bit=False,
                dtype=self.dtype,
            )
        )

        return model, tokenizer

    # ========================================================
    # COMPUTE FISHER
    # ========================================================

    def _compute_fisher_if_needed(
        self,
        calibration_loader,
        device,
    ):

        if (
            not self.use_fisher
            or compute_fisher is None
            or calibration_loader is None
        ):

            return

        logger.info(
            "Computing Fisher information for Model A..."
        )

        try:

            model_a, _ = (
                _load_model_any(
                    self.model_a,
                    prefer_unsloth=False,
                    load_in_4bit=False,
                    dtype=torch.float16,
                )
            )

            try:

                model_a.to(
                    device
                )

                self.fisher_a = (
                    compute_fisher(
                        model_a,
                        calibration_loader,
                        device,
                    )
                )

            finally:

                del model_a
                self._purge_memory()

        except Exception as exc:

            logger.warning(
                "Model A Fisher failed: %s",
                exc,
            )

            self.fisher_a = None

        logger.info(
            "Computing Fisher information for Model B..."
        )

        try:

            model_b, _ = (
                _load_model_any(
                    self.model_b,
                    prefer_unsloth=False,
                    load_in_4bit=False,
                    dtype=torch.float16,
                )
            )

            try:

                model_b.to(
                    device
                )

                self.fisher_b = (
                    compute_fisher(
                        model_b,
                        calibration_loader,
                        device,
                    )
                )

            finally:

                del model_b
                self._purge_memory()

        except Exception as exc:

            logger.warning(
                "Model B Fisher failed: %s",
                exc,
            )

            self.fisher_b = None

        self._report.fisher_used = (
            self.fisher_a is not None
            and
            self.fisher_b is not None
        )

    # ========================================================
    # MAIN MERGE
    # ========================================================

    def merge(
        self,
    ) -> bool:

        started = time.time()

        logger.info(
            "================================================"
        )

        logger.info(
            "FTRAIN PHOENIX INTELLIGENT BRAIN MERGER"
        )

        logger.info(
            "Model A: %s",
            self.model_a,
        )

        logger.info(
            "Model B: %s",
            self.model_b,
        )

        # ====================================================
        # DEVICE
        # ====================================================

        try:

            from .hardware import (
                best_device,
            )

            runtime_device = (
                best_device()
            )

        except Exception:

            runtime_device = (
                torch.device(
                    "cuda"
                )
                if torch.cuda.is_available()
                else torch.device(
                    "cpu"
                )
            )

        logger.info(
            "Runtime device: %s",
            runtime_device,
        )

        # ====================================================
        # STEP 1
        # LOAD STATE DICTS
        # ====================================================

        logger.info(
            "Loading Model A..."
        )

        model_a, tokenizer = (
            _load_model_any(
                self.model_a,
                prefer_unsloth=False,
                load_in_4bit=False,
                dtype=torch.float16,
            )
        )

        sd_a = {
            key:
                value.detach()
                .cpu()
                for key, value
                in model_a.state_dict().items()
        }

        params_a = sum(
            _tensor_numel(
                tensor
            )
            for tensor in sd_a.values()
            if torch.is_tensor(
                tensor
            )
        )

        del model_a
        self._purge_memory()

        logger.info(
            "Loading Model B..."
        )

        model_b, _ = (
            _load_model_any(
                self.model_b,
                prefer_unsloth=False,
                load_in_4bit=False,
                dtype=torch.float16,
            )
        )

        sd_b = {
            key:
                value.detach()
                .cpu()
                for key, value
                in model_b.state_dict().items()
        }

        params_b = sum(
            _tensor_numel(
                tensor
            )
            for tensor in sd_b.values()
            if torch.is_tensor(
                tensor
            )
        )

        del model_b
        self._purge_memory()

        layers_a = self._get_num_layers(
            sd_a
        )

        layers_b = self._get_num_layers(
            sd_b
        )

        self._report.architecture = {
            "model_a_layers": layers_a,
            "model_b_layers": layers_b,
            "model_a_parameters": params_a,
            "model_b_parameters": params_b,
            "architecture": "Model A output architecture",
        }

        # ====================================================
        # STEP 2
        # CALIBRATION LOADER
        # ====================================================

        calibration_loader = (
            self._build_calibration_loader(
                tokenizer
            )
        )

        # ====================================================
        # STEP 3
        # FISHER
        # ====================================================

        if (
            self.use_fisher
            and calibration_loader is not None
        ):

            self._compute_fisher_if_needed(
                calibration_loader,
                runtime_device,
            )

        # ====================================================
        # STEP 4
        # CAPTAIN
        # ====================================================

        self.captain_policy = (
            self._captain_global_policy(
                layers_a=layers_a,
                layers_b=layers_b,
                params_a=params_a,
                params_b=params_b,
            )
        )

        self._report.captain_policy = (
            asdict(
                self.captain_policy
            )
        )

        logger.info(
            "Captain policy: %s",
            asdict(
                self.captain_policy
            ),
        )

        # ====================================================
        # STEP 5
        # CBA
        # ====================================================

        self.use_cba = (
            self.captain_policy.use_cba
        )

        self._run_cba(
            sd_a,
            sd_b,
        )

        self._report.cba_used = (
            self._cba_report is not None
        )

        # ====================================================
        # STEP 6
        # BASELINE
        # ====================================================

        # Only keep a baseline copy if safety/repair needs it.
        baseline = None

        if (
            self.adaptation_enabled
            or check_state_dict is not None
        ):

            baseline = {}

            for key, tensor in sd_a.items():

                if torch.is_tensor(
                    tensor
                ):

                    baseline[
                        key
                    ] = tensor.clone()

        # ====================================================
        # STEP 7
        # MERGE
        # ====================================================

        self.strategy = (
            self.captain_policy.strategy
        )

        logger.info(
            "Starting intelligent tensor merge..."
        )

        merged_state = self._static_merge(
            sd_a,
            sd_b,
        )

        del sd_b
        self._purge_memory()

        # ====================================================
        # STEP 8
        # SAFETY
        # ====================================================

        safety_result, merged_state = (
            self._run_safety(
                merged_state,
                baseline,
            )
        )

        self._report.safety = (
            safety_result
        )

        # ====================================================
        # STEP 9
        # LOAD MERGED MODEL
        # ====================================================

        logger.info(
            "Loading merged candidate model..."
        )

        merged_model, tokenizer = (
            self._load_target_model()
        )

        missing, unexpected = (
            merged_model.load_state_dict(
                merged_state,
                strict=False,
            )
        )

        if missing:

            logger.info(
                "Missing target keys after merge: %d",
                len(
                    missing
                ),
            )

        if unexpected:

            logger.info(
                "Unexpected target keys after merge: %d",
                len(
                    unexpected
                ),
            )

        # ====================================================
        # STEP 10
        # CALIBRATION EVAL
        # ====================================================

        pre_adaptation_loss = float(
            "inf"
        )

        if calibration_loader is not None:

            try:

                merged_model.to(
                    runtime_device
                )

                pre_adaptation_loss = (
                    self._evaluate_model(
                        merged_model,
                        calibration_loader,
                        runtime_device,
                        max_batches=16,
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Pre-adaptation evaluation failed: %s",
                    exc,
                )

        # ====================================================
        # STEP 11
        # TARGETED BRAIN REPAIR
        # ====================================================

        adaptation_result = {
            "enabled": False,
            "initial_loss":
                pre_adaptation_loss,
            "final_loss":
                pre_adaptation_loss,
        }

        if (
            self.adaptation_enabled
            and calibration_loader is not None
        ):

            adaptation_result = (
                self._targeted_adaptation(
                    merged_model,
                    tokenizer,
                    calibration_loader,
                    runtime_device,
                )
            )

        self._report.adaptation = (
            adaptation_result
        )

        # ====================================================
        # STEP 12
        # FINAL SAFETY
        # ====================================================

        logger.info(
            "Running final safety scan..."
        )

        try:

            final_state = (
                merged_model.state_dict()
            )

            bad_tensors = []

            for name, tensor in (
                final_state.items()
            ):

                if (
                    torch.is_floating_point(
                        tensor
                    )
                    and not _finite(
                        tensor
                    )
                ):

                    bad_tensors.append(
                        name
                    )

            if bad_tensors:

                logger.warning(
                    "Detected %d non-finite "
                    "tensors after adaptation.",
                    len(
                        bad_tensors
                    ),
                )

                if baseline is not None:

                    with torch.no_grad():

                        for name in bad_tensors:

                            if name in baseline:

                                final_state[
                                    name
                                ].copy_(
                                    baseline[
                                        name
                                    ].to(
                                        final_state[
                                            name
                                        ].device,
                                        final_state[
                                            name
                                        ].dtype,
                                    )
                                )

        except Exception as exc:

            logger.warning(
                "Final safety scan failed: %s",
                exc,
            )

        # ====================================================
        # STEP 13
        # CPU BEFORE SAVE
        # ====================================================

        try:

            merged_model.to(
                "cpu"
            )

        except Exception:

            pass

        self._purge_memory()

        # ====================================================
        # STEP 14
        # SAVE
        # ====================================================

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        logger.info(
            "Saving final merged brain..."
        )

        save_kwargs = {
            "safe_serialization": True,
        }

        max_shard_size = getattr(
            self.config,
            "max_shard_size",
            None,
        )

        if max_shard_size:

            save_kwargs[
                "max_shard_size"
            ] = max_shard_size

        try:

            merged_model.save_pretrained(
                self.output_dir,
                **save_kwargs,
            )

        except TypeError:

            # Older Transformers.
            merged_model.save_pretrained(
                self.output_dir
            )

        tokenizer.save_pretrained(
            self.output_dir
        )

        # ====================================================
        # STEP 15
        # REPORT
        # ====================================================

        self._report.finished_at = (
            time.time()
        )

        self._report.duration_seconds = (
            self._report.finished_at
            - started
        )

        # Most conflicted regions.
        conflict_regions = sorted(
            self._merge_records,
            key=lambda d: (
                d.sign_conflict
                * 2.0
                + d.relative_delta
            ),
            reverse=True,
        )[:20]

        self._report.architecture[
            "highest_conflict_tensors"
        ] = [
            {
                "name":
                    decision.target_key,
                "strategy":
                    decision.strategy,
                "cosine":
                    decision.cosine,
                "relative_delta":
                    decision.relative_delta,
                "conflict":
                    decision.sign_conflict,
                "alpha_b":
                    decision.alpha_b,
                "reason":
                    decision.reason,
            }
            for decision
            in conflict_regions
        ]

        metadata = {
            "ftrain_merger": "phoenix_deep",
            "timestamp": time.time(),
            "model_a": self.model_a,
            "model_b": self.model_b,
            "strategy": self.strategy,
            "captain_policy": asdict(
                self.captain_policy
            ),
            "report": asdict(
                self._report
            ),
        }

        metadata_path = os.path.join(
            self.output_dir,
            "ftrain_merge_report.json",
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                metadata,
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        logger.info(
            "================================================"
        )

        logger.info(
            "PHOENIX MERGE COMPLETE"
        )

        logger.info(
            "Matched tensors: %d",
            self._report.counts.get(
                "matched",
                0,
            ),
        )

        logger.info(
            "Missing tensors: %d",
            self._report.counts.get(
                "missing",
                0,
            ),
        )

        logger.info(
            "Shape-adapted tensors: %d",
            self._report.counts.get(
                "shape_adapted",
                0,
            ),
        )

        logger.info(
            "Mean cosine: %.4f",
            self._report.mean_cosine,
        )

        logger.info(
            "Mean relative delta: %.4f",
            self._report.mean_relative_delta,
        )

        logger.info(
            "Mean conflict: %.4f",
            self._report.mean_conflict,
        )

        if (
            math.isfinite(
                pre_adaptation_loss
            )
        ):

            logger.info(
                "Pre-repair calibration loss: %.5f",
                pre_adaptation_loss,
            )

        final_loss = (
            adaptation_result.get(
                "final_loss",
                float("inf"),
            )
        )

        if math.isfinite(
            final_loss
        ):

            logger.info(
                "Final calibration loss: %.5f",
                final_loss,
            )

        logger.info(
            "Output: %s",
            self.output_dir,
        )

        logger.info(
            "================================================"
        )

        # ====================================================
        # CLEANUP
        # ====================================================

        del merged_model
        del tokenizer
        del merged_state

        if baseline is not None:

            del baseline

        self._purge_memory()

        return True
