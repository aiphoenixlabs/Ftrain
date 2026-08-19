"""
FTRAIN Configuration System
===========================

Centralized, validated configuration objects for:

    • Training
    • PEFT / LoRA / DoRA
    • Learning-rate scheduling
    • Adaptive Captain control
    • Dataset processing
    • GRPO
    • Dashboards / reporting
    • Model merging
    • Alignment / repair

Design goals
------------
• Strong validation before expensive model loading begins.
• Backward compatibility with existing FTRAIN configuration names.
• Safe serialization to JSON.
• Explicit normalization of common configuration mistakes.
• No silent acceptance of dangerous numerical values.
• Clear errors for contradictory options.
• Stable configuration contracts between api.py, core.py, callbacks,
  PhoenixCaptain and Merger.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    get_type_hints,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TrainConfig",
    "MergeConfig",
]


# =============================================================================
# Shared helpers
# =============================================================================


def _require_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string, "
            f"got {type(value).__name__}."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return value


def _optional_string(
    value: Optional[str],
    field_name: str,
) -> Optional[str]:
    """Validate an optional string."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string or None, "
            f"got {type(value).__name__}."
        )

    value = value.strip()

    return value or None


def _finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
    """Validate a finite floating-point configuration value."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must be numeric, got {value!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite, got {value!r}."
        )

    if minimum is not None:
        if strict_minimum and value <= minimum:
            raise ValueError(
                f"{field_name} must be > {minimum}, got {value}."
            )
        if not strict_minimum and value < minimum:
            raise ValueError(
                f"{field_name} must be >= {minimum}, got {value}."
            )

    if maximum is not None:
        if strict_maximum and value >= maximum:
            raise ValueError(
                f"{field_name} must be < {maximum}, got {value}."
            )
        if not strict_maximum and value > maximum:
            raise ValueError(
                f"{field_name} must be <= {maximum}, got {value}."
            )

    return value


def _positive_int(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = False,
) -> int:
    """Validate an integer that must be positive or optionally non-negative."""
    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be an integer, not bool."
        )

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must be an integer, got {value!r}."
        ) from exc

    minimum = 0 if allow_zero else 1

    if integer < minimum:
        comparison = ">=" if allow_zero else ">"
        raise ValueError(
            f"{field_name} must be {comparison} 0, got {integer}."
        )

    return integer


def _bool_value(
    value: Any,
    field_name: str,
) -> bool:
    """Validate boolean options without accepting arbitrary truthy objects."""
    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be a boolean, "
            f"got {type(value).__name__}."
        )

    return value


def _enum_value(
    value: Any,
    field_name: str,
    allowed: Sequence[str],
) -> str:
    """Validate a string enum."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string, "
            f"got {type(value).__name__}."
        )

    normalized = value.strip().lower()

    if normalized not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValueError(
            f"Invalid {field_name}={value!r}. "
            f"Expected one of: {allowed_text}."
        )

    return normalized


def _clean_string_list(
    value: Optional[Sequence[str]],
    field_name: str,
) -> Optional[List[str]]:
    """Validate and normalize a list of strings."""
    if value is None:
        return None

    if isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a list of strings, not one string."
        )

    result: List[str] = []

    for item in value:
        if not isinstance(item, str):
            raise TypeError(
                f"{field_name} entries must be strings, "
                f"got {type(item).__name__}."
            )

        item = item.strip()

        if item:
            result.append(item)

    return result


def _json_safe(value: Any) -> Any:
    """
    Convert common Python/NumPy objects into JSON-safe representations.

    Callable objects are intentionally represented by name rather than
    serialized as executable code.
    """
    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if callable(value):
        return getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", repr(value)),
        )

    # NumPy scalars.
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    # Last-resort representation.
    return str(value)


def _write_json(
    data: Mapping[str, Any],
    path: str | os.PathLike[str],
) -> Path:
    """Atomically-ish write configuration JSON with UTF-8 encoding."""
    target = Path(path).expanduser()

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = target.with_suffix(
        target.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _json_safe(data),
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        file.write("\n")

    temporary.replace(target)

    return target


def _read_json(
    path: str | os.PathLike[str],
) -> Dict[str, Any]:
    """Read a JSON configuration file."""
    source = Path(path).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Configuration path is not a file: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {source}."
        )

    return data


# =============================================================================
# Training configuration
# =============================================================================


@dataclass
class TrainConfig:
    """
    Complete FTRAIN training configuration.

    The object is intentionally mutable so higher-level orchestration code can
    adjust options programmatically before training begins.
    """

    # -------------------------------------------------------------------------
    # Model / output
    # -------------------------------------------------------------------------

    model_name: str

    captain_model: Optional[str] = None
    output_dir: str = "./ftrain_output"
    logging_dir: str = "./ftrain_logs"
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3

    family: str = "auto"
    seed: int = 42

    # -------------------------------------------------------------------------
    # Quantization / PEFT
    # -------------------------------------------------------------------------

    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32

    use_custom_lora: bool = False
    use_dora: bool = False
    auto_lora_targets: bool = False

    lora_target_count: int = 4
    lora_target_modules: Optional[List[str]] = None

    use_unsloth_lora: bool = True

    lora_a_lr_mult: float = 2.0
    lora_b_lr_mult: float = 1.0

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1

    layerwise_lr_decay: float = 0.85
    swiglu_gate_boost: float = 1.2
    moe_router_lr_multiplier: float = 0.5

    max_grad_norm: float = 1.0

    # -------------------------------------------------------------------------
    # LR finder / scheduler
    # -------------------------------------------------------------------------

    use_lr_finder: bool = False
    lr_finder_start_lr: float = 1e-7
    lr_finder_end_lr: float = 10.0
    lr_finder_iter: int = 100

    use_cosine_restarts: bool = False
    restart_interval: int = 50

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    max_steps: int = 100

    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    max_seq_length: int = 2048

    eval_interval: int = 10
    checkpoint_interval: int = 20

    dataloader_num_workers: int = 2
    pin_memory: bool = True

    gradient_checkpointing_enable: bool = False

    use_adaptive_accumulation: bool = False
    target_batch_tokens: int = 8192

    use_hf_trainer: bool = True
    use_unsloth_trainer: bool = True

    # -------------------------------------------------------------------------
    # Captain
    # -------------------------------------------------------------------------

    captain_enabled: bool = True

    # Keep the original public values for compatibility.
    captain_mode: Literal[
        "rule",
        "adaptive",
        "strict",
        "llm",
    ] = "rule"

    captain_interval: int = 5
    captain_min_interval: float = 2.0

    captain_clamp: List[float] = field(
        default_factory=lambda: [0.25, 2.5]
    )

    captain_velocity_window: int = 3

    # New explicit Captain controls used by the upgraded callback.
    captain_strict: bool = False

    captain_memory_size: int = 20
    captain_reward_window: int = 10

    captain_mult_min: float = 0.25
    captain_mult_max: float = 2.5

    captain_gradient_collapse_threshold: float = 1e-6
    captain_brain_activity_threshold: float = 0.20
    captain_brain_low_threshold: float = 0.05
    captain_instability_grad_norm: float = 5.0
    captain_plateau_window: int = 5

    captain_max_seq_length: int = 2048
    captain_max_new_tokens: int = 160

    captain_load_in_4bit: bool = True
    captain_dtype: Optional[str] = None

    captain_do_sample: bool = False
    captain_temperature: float = 0.2

    captain_layer_boost: float = 2.0

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    data_sources: Optional[List[str]] = None

    data_balance_strategy: Literal[
        "tokens",
        "examples",
        "equal",
    ] = "tokens"

    use_packing: bool = False
    train_on_response_only: bool = False
    mask_thinking: bool = False
    group_by_length: bool = True

    data_perplexity_filter: bool = False
    data_perplexity_keep_pct: float = 0.8

    data_dedup: bool = False
    data_dedup_threshold: float = 0.9

    # -------------------------------------------------------------------------
    # Answers / logging / dashboard
    # -------------------------------------------------------------------------

    answer_mode: Literal[
        "auto_yes",
        "interactive",
        "strict",
    ] = "auto_yes"

    report_to: Literal[
        "none",
        "wandb",
        "tensorboard",
    ] = "none"

    auto_resume: bool = False

    use_dashboard: bool = False
    dashboard_port: int = 7860

    show_model_progress: bool = True

    # -------------------------------------------------------------------------
    # GRPO
    # -------------------------------------------------------------------------

    use_grpo: bool = False
    grpo_num_generations: int = 6

    grpo_reward_funcs: Optional[
        List[Callable[..., Any]]
    ] = None

    # =========================================================================
    # Validation
    # =========================================================================

    def __post_init__(self) -> None:
        """Normalize and validate the complete training configuration."""
        self._normalize()
        self.validate()

    def _normalize(self) -> None:
        """Normalize user-facing configuration values."""
        self.model_name = _require_non_empty_string(
            self.model_name,
            "model_name",
        )

        self.captain_model = _optional_string(
            self.captain_model,
            "captain_model",
        )

        self.output_dir = _require_non_empty_string(
            self.output_dir,
            "output_dir",
        )

        self.logging_dir = _require_non_empty_string(
            self.logging_dir,
            "logging_dir",
        )

        self.resume_from_checkpoint = _optional_string(
            self.resume_from_checkpoint,
            "resume_from_checkpoint",
        )

        self.family = _require_non_empty_string(
            self.family,
            "family",
        ).lower()

        self.lora_target_modules = _clean_string_list(
            self.lora_target_modules,
            "lora_target_modules",
        )

        self.data_sources = _clean_string_list(
            self.data_sources,
            "data_sources",
        )

        # Canonical Captain mode compatibility.
        #
        # "adaptive" means rule engine + adaptive behavior.
        # "strict" means LLM/rule operation with strict failure semantics.
        #
        # The downstream Captain itself uses the canonical operating modes
        # rule/llm/disabled. The derived captain_strict flag carries strictness.
        if self.captain_mode == "adaptive":
            self.captain_mode = "rule"

        elif self.captain_mode == "strict":
            self.captain_mode = (
                "llm"
                if self.captain_model
                else "rule"
            )
            self.captain_strict = True

        # If Captain is explicitly disabled, make its operating mode coherent.
        if not self.captain_enabled:
            self.captain_mode = "rule"

        self.captain_clamp = [
            float(value)
            for value in self.captain_clamp
        ]

        # Keep explicit bounds synchronized with legacy captain_clamp.
        if len(self.captain_clamp) >= 2:
            self.captain_mult_min = float(
                self.captain_clamp[0]
            )
            self.captain_mult_max = float(
                self.captain_clamp[1]
            )

        # Normalize LR finder ordering.
        if self.lr_finder_start_lr > self.lr_finder_end_lr:
            (
                self.lr_finder_start_lr,
                self.lr_finder_end_lr,
            ) = (
                self.lr_finder_end_lr,
                self.lr_finder_start_lr,
            )

    def validate(self) -> None:
        """
        Perform strict semantic validation.

        This runs after normalization so downstream components can assume the
        configuration obeys its invariants.
        """
        # ---------------------------------------------------------------------
        # Basic integers
        # ---------------------------------------------------------------------

        self.save_total_limit = _positive_int(
            self.save_total_limit,
            "save_total_limit",
        )

        self.seed = _positive_int(
            self.seed,
            "seed",
            allow_zero=True,
        )

        self.lora_r = _positive_int(
            self.lora_r,
            "lora_r",
        )

        self.lora_alpha = _positive_int(
            self.lora_alpha,
            "lora_alpha",
        )

        self.lora_target_count = _positive_int(
            self.lora_target_count,
            "lora_target_count",
        )

        self.warmup_steps = _positive_int(
            self.warmup_steps,
            "warmup_steps",
            allow_zero=True,
        )

        self.max_steps = _positive_int(
            self.max_steps,
            "max_steps",
        )

        self.per_device_batch_size = _positive_int(
            self.per_device_batch_size,
            "per_device_batch_size",
        )

        self.gradient_accumulation_steps = _positive_int(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )

        self.max_seq_length = _positive_int(
            self.max_seq_length,
            "max_seq_length",
        )

        self.eval_interval = _positive_int(
            self.eval_interval,
            "eval_interval",
            allow_zero=True,
        )

        self.checkpoint_interval = _positive_int(
            self.checkpoint_interval,
            "checkpoint_interval",
            allow_zero=True,
        )

        self.dataloader_num_workers = _positive_int(
            self.dataloader_num_workers,
            "dataloader_num_workers",
            allow_zero=True,
        )

        self.target_batch_tokens = _positive_int(
            self.target_batch_tokens,
            "target_batch_tokens",
        )

        # ---------------------------------------------------------------------
        # Boolean values
        # ---------------------------------------------------------------------

        boolean_fields = (
            "load_in_4bit",
            "use_custom_lora",
            "use_dora",
            "auto_lora_targets",
            "use_unsloth_lora",
            "use_lr_finder",
            "use_cosine_restarts",
            "pin_memory",
            "gradient_checkpointing_enable",
            "use_adaptive_accumulation",
            "use_hf_trainer",
            "use_unsloth_trainer",
            "captain_enabled",
            "captain_strict",
            "use_packing",
            "train_on_response_only",
            "mask_thinking",
            "group_by_length",
            "data_perplexity_filter",
            "data_dedup",
            "auto_resume",
            "use_dashboard",
            "show_model_progress",
            "use_grpo",
        )

        for field_name in boolean_fields:
            setattr(
                self,
                field_name,
                _bool_value(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        # ---------------------------------------------------------------------
        # Learning rates
        # ---------------------------------------------------------------------

        self.learning_rate = _finite_float(
            self.learning_rate,
            "learning_rate",
            minimum=0.0,
            strict_minimum=True,
        )

        self.lr_finder_start_lr = _finite_float(
            self.lr_finder_start_lr,
            "lr_finder_start_lr",
            minimum=0.0,
            strict_minimum=True,
        )

        self.lr_finder_end_lr = _finite_float(
            self.lr_finder_end_lr,
            "lr_finder_end_lr",
            minimum=0.0,
            strict_minimum=True,
        )

        if self.lr_finder_start_lr >= self.lr_finder_end_lr:
            raise ValueError(
                "lr_finder_start_lr must be smaller than "
                "lr_finder_end_lr."
            )

        self.lora_a_lr_mult = _finite_float(
            self.lora_a_lr_mult,
            "lora_a_lr_mult",
            minimum=0.0,
            strict_minimum=True,
        )

        self.lora_b_lr_mult = _finite_float(
            self.lora_b_lr_mult,
            "lora_b_lr_mult",
            minimum=0.0,
            strict_minimum=True,
        )

        # ---------------------------------------------------------------------
        # Scheduler
        # ---------------------------------------------------------------------

        self.warmup_ratio = _finite_float(
            self.warmup_ratio,
            "warmup_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        self.min_lr_ratio = _finite_float(
            self.min_lr_ratio,
            "min_lr_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        self.layerwise_lr_decay = _finite_float(
            self.layerwise_lr_decay,
            "layerwise_lr_decay",
            minimum=0.0,
            maximum=1.0,
        )

        if self.layerwise_lr_decay == 0.0:
            logger.warning(
                "layerwise_lr_decay=0 means deeper layers can receive "
                "an extremely small learning rate."
            )

        self.swiglu_gate_boost = _finite_float(
            self.swiglu_gate_boost,
            "swiglu_gate_boost",
            minimum=0.0,
            strict_minimum=True,
        )

        self.moe_router_lr_multiplier = _finite_float(
            self.moe_router_lr_multiplier,
            "moe_router_lr_multiplier",
            minimum=0.0,
            strict_minimum=True,
        )

        self.max_grad_norm = _finite_float(
            self.max_grad_norm,
            "max_grad_norm",
            minimum=0.0,
            strict_minimum=True,
        )

        self.restart_interval = _positive_int(
            self.restart_interval,
            "restart_interval",
        )

        self.lr_finder_iter = _positive_int(
            self.lr_finder_iter,
            "lr_finder_iter",
        )

        # ---------------------------------------------------------------------
        # Captain
        # ---------------------------------------------------------------------

        self.captain_interval = _positive_int(
            self.captain_interval,
            "captain_interval",
        )

        self.captain_min_interval = _finite_float(
            self.captain_min_interval,
            "captain_min_interval",
            minimum=0.0,
        )

        if len(self.captain_clamp) != 2:
            raise ValueError(
                "captain_clamp must contain exactly two values: "
                "[minimum, maximum]."
            )

        captain_low = _finite_float(
            self.captain_clamp[0],
            "captain_clamp[0]",
            minimum=0.0,
            strict_minimum=True,
        )

        captain_high = _finite_float(
            self.captain_clamp[1],
            "captain_clamp[1]",
            minimum=0.0,
            strict_minimum=True,
        )

        if captain_low > captain_high:
            raise ValueError(
                "captain_clamp minimum cannot exceed maximum."
            )

        self.captain_clamp = [
            captain_low,
            captain_high,
        ]

        self.captain_mult_min = captain_low
        self.captain_mult_max = captain_high

        self.captain_velocity_window = _positive_int(
            self.captain_velocity_window,
            "captain_velocity_window",
        )

        self.captain_memory_size = _positive_int(
            self.captain_memory_size,
            "captain_memory_size",
        )

        self.captain_reward_window = _positive_int(
            self.captain_reward_window,
            "captain_reward_window",
        )

        self.captain_gradient_collapse_threshold = _finite_float(
            self.captain_gradient_collapse_threshold,
            "captain_gradient_collapse_threshold",
            minimum=0.0,
        )

        self.captain_brain_activity_threshold = _finite_float(
            self.captain_brain_activity_threshold,
            "captain_brain_activity_threshold",
            minimum=0.0,
        )

        self.captain_brain_low_threshold = _finite_float(
            self.captain_brain_low_threshold,
            "captain_brain_low_threshold",
            minimum=0.0,
        )

        self.captain_instability_grad_norm = _finite_float(
            self.captain_instability_grad_norm,
            "captain_instability_grad_norm",
            minimum=0.0,
        )

        self.captain_plateau_window = _positive_int(
            self.captain_plateau_window,
            "captain_plateau_window",
            allow_zero=False,
        )

        self.captain_max_seq_length = _positive_int(
            self.captain_max_seq_length,
            "captain_max_seq_length",
        )

        self.captain_max_new_tokens = _positive_int(
            self.captain_max_new_tokens,
            "captain_max_new_tokens",
        )

        self.captain_temperature = _finite_float(
            self.captain_temperature,
            "captain_temperature",
            minimum=0.0,
            strict_minimum=True,
        )

        self.captain_layer_boost = _finite_float(
            self.captain_layer_boost,
            "captain_layer_boost",
            minimum=0.0,
            strict_minimum=True,
        )

        if self.captain_enabled and self.captain_mode == "llm":
            if not self.captain_model:
                raise ValueError(
                    "captain_model must be provided when "
                    "captain_mode='llm'."
                )

        # ---------------------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------------------

        self.data_perplexity_keep_pct = _finite_float(
            self.data_perplexity_keep_pct,
            "data_perplexity_keep_pct",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )

        self.data_dedup_threshold = _finite_float(
            self.data_dedup_threshold,
            "data_dedup_threshold",
            minimum=0.0,
            maximum=1.0,
        )

        self.data_balance_strategy = _enum_value(
            self.data_balance_strategy,
            "data_balance_strategy",
            ("tokens", "examples", "equal"),
        )

        # ---------------------------------------------------------------------
        # UI / reporting
        # ---------------------------------------------------------------------

        self.dashboard_port = _positive_int(
            self.dashboard_port,
            "dashboard_port",
        )

        if not 1 <= self.dashboard_port <= 65535:
            raise ValueError(
                "dashboard_port must be between 1 and 65535."
            )

        self.report_to = _enum_value(
            self.report_to,
            "report_to",
            ("none", "wandb", "tensorboard"),
        )

        self.answer_mode = _enum_value(
            self.answer_mode,
            "answer_mode",
            ("auto_yes", "interactive", "strict"),
        )

        # ---------------------------------------------------------------------
        # GRPO
        # ---------------------------------------------------------------------

        self.grpo_num_generations = _positive_int(
            self.grpo_num_generations,
            "grpo_num_generations",
        )

        if self.grpo_reward_funcs is not None:
            if isinstance(self.grpo_reward_funcs, str):
                raise TypeError(
                    "grpo_reward_funcs must be a list of callables."
                )

            for index, function in enumerate(
                self.grpo_reward_funcs
            ):
                if not callable(function):
                    raise TypeError(
                        f"grpo_reward_funcs[{index}] must be callable."
                    )

        # ---------------------------------------------------------------------
        # Cross-option validation
        # ---------------------------------------------------------------------

        if (
            self.use_custom_lora
            and self.lora_target_modules is None
            and not self.auto_lora_targets
        ):
            raise ValueError(
                "use_custom_lora=True requires either "
                "lora_target_modules or auto_lora_targets=True."
            )

        if (
            self.auto_lora_targets
            and self.lora_target_count <= 0
        ):
            raise ValueError(
                "lora_target_count must be positive when "
                "auto_lora_targets=True."
            )

        if (
            self.use_dora
            and not self.use_custom_lora
            and not self.use_unsloth_lora
        ):
            logger.warning(
                "use_dora=True while both custom LoRA and Unsloth LoRA "
                "are disabled. Dora may have no effect."
            )

        if (
            self.gradient_checkpointing_enable
            and self.max_seq_length < 128
        ):
            logger.warning(
                "Gradient checkpointing is enabled with a very small "
                "max_seq_length=%d.",
                self.max_seq_length,
            )

        if (
            self.use_lr_finder
            and self.lr_finder_iter < 10
        ):
            logger.warning(
                "LR finder is enabled with only %d iterations. "
                "This may produce an unreliable estimate.",
                self.lr_finder_iter,
            )

        if (
            self.use_cosine_restarts
            and self.restart_interval > self.max_steps
        ):
            logger.warning(
                "restart_interval=%d exceeds max_steps=%d.",
                self.restart_interval,
                self.max_steps,
            )

        if (
            self.report_to != "none"
            and not self.use_hf_trainer
        ):
            logger.warning(
                "report_to=%s is configured while use_hf_trainer=False. "
                "The selected backend may ignore this setting.",
                self.report_to,
            )

        if (
            self.use_dashboard
            and not 1 <= self.dashboard_port <= 65535
        ):
            raise ValueError(
                "use_dashboard requires a valid dashboard_port."
            )

        if (
            self.use_grpo
            and not self.grpo_reward_funcs
        ):
            logger.warning(
                "GRPO is enabled but no custom reward functions were "
                "provided. Downstream code must supply/default rewards."
            )

        if (
            self.use_unsloth_trainer
            and self.use_hf_trainer
        ):
            logger.info(
                "Both HF Trainer and Unsloth Trainer are enabled. "
                "The training backend selection must be resolved by core.py."
            )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(
        self,
        *,
        include_runtime_objects: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert configuration into a JSON-friendly dictionary.

        By default runtime-only reward functions are excluded.
        """
        data = asdict(self)

        if not include_runtime_objects:
            data.pop(
                "grpo_reward_funcs",
                None,
            )

        return _json_safe(data)

    def save(
        self,
        path: str | os.PathLike[str],
    ) -> Path:
        """Save configuration to JSON."""
        target = _write_json(
            self.to_dict(),
            path,
        )

        logger.info(
            "Saved TrainConfig to %s",
            target,
        )

        return target

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "TrainConfig":
        """Construct TrainConfig from a dictionary."""
        if not isinstance(data, Mapping):
            raise TypeError(
                "TrainConfig.from_dict() expects a mapping."
            )

        allowed_fields = {
            item.name
            for item in fields(cls)
        }

        cleaned = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        ignored = set(data) - allowed_fields

        if ignored:
            logger.warning(
                "Ignoring unknown TrainConfig fields: %s",
                ", ".join(sorted(ignored)),
            )

        return cls(**cleaned)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
    ) -> "TrainConfig":
        """Load TrainConfig from JSON."""
        return cls.from_dict(
            _read_json(path)
        )


# =============================================================================
# Merge configuration
# =============================================================================


@dataclass
class MergeConfig:
    """
    Complete FTRAIN model-merging configuration.
    """

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    model_a: str
    model_b: str

    captain_model: Optional[str] = None

    output_dir: str = "./merged_model"
    name: str = "auto"

    # -------------------------------------------------------------------------
    # Merge format / strategy
    # -------------------------------------------------------------------------

    save_dtype: Literal[
        "fp16",
        "bf16",
        "fp32",
    ] = "bf16"

    strategy: Literal[
        "intelligent",
        "linear",
        "slerp",
        "ties",
        "dare",
    ] = "intelligent"

    alpha: float = 0.5

    # -------------------------------------------------------------------------
    # Advanced merging
    # -------------------------------------------------------------------------

    use_fisher: bool = True
    use_dare: bool = False
    use_task_arithmetic: bool = False
    knowledge_preservation: bool = False
    merge_rollback: bool = False

    system_prompt_merger: str = (
        "You are an expert model merger."
    )

    calibration_data: Optional[Any] = None

    repair_steps: int = 0

    merge_knowledge_distill: bool = False

    force_cuda_merge: bool = False

    # -------------------------------------------------------------------------
    # Hugging Face / alignment
    # -------------------------------------------------------------------------

    hugging: bool = False
    hugging_token: Optional[str] = None

    align_grpo: bool = False
    align_grpo_steps: int = 50

    align_grpo_reward_funcs: Optional[
        List[Callable[..., Any]]
    ] = None

    # =========================================================================
    # Validation
    # =========================================================================

    def __post_init__(self) -> None:
        self._normalize()
        self.validate()

    def _normalize(self) -> None:
        """Normalize merge configuration values."""
        self.model_a = _require_non_empty_string(
            self.model_a,
            "model_a",
        )

        self.model_b = _require_non_empty_string(
            self.model_b,
            "model_b",
        )

        self.captain_model = _optional_string(
            self.captain_model,
            "captain_model",
        )

        self.output_dir = _require_non_empty_string(
            self.output_dir,
            "output_dir",
        )

        self.name = _require_non_empty_string(
            self.name,
            "name",
        )

        self.system_prompt_merger = _require_non_empty_string(
            self.system_prompt_merger,
            "system_prompt_merger",
        )

        self.hugging_token = _optional_string(
            self.hugging_token,
            "hugging_token",
        )

        # Do not expose secret Hugging Face tokens through ordinary config
        # serialization.
        if self.model_a == self.model_b:
            raise ValueError(
                "model_a and model_b cannot be the same."
            )

    def validate(self) -> None:
        """Validate all merge options and their interactions."""
        # ---------------------------------------------------------------------
        # Numeric values
        # ---------------------------------------------------------------------

        self.alpha = _finite_float(
            self.alpha,
            "alpha",
            minimum=0.0,
            maximum=1.0,
        )

        self.repair_steps = _positive_int(
            self.repair_steps,
            "repair_steps",
            allow_zero=True,
        )

        self.align_grpo_steps = _positive_int(
            self.align_grpo_steps,
            "align_grpo_steps",
        )

        # ---------------------------------------------------------------------
        # Enums
        # ---------------------------------------------------------------------

        self.save_dtype = _enum_value(
            self.save_dtype,
            "save_dtype",
            ("fp16", "bf16", "fp32"),
        )

        self.strategy = _enum_value(
            self.strategy,
            "strategy",
            (
                "intelligent",
                "linear",
                "slerp",
                "ties",
                "dare",
            ),
        )

        # ---------------------------------------------------------------------
        # Booleans
        # ---------------------------------------------------------------------

        boolean_fields = (
            "use_fisher",
            "use_dare",
            "use_task_arithmetic",
            "knowledge_preservation",
            "merge_rollback",
            "merge_knowledge_distill",
            "force_cuda_merge",
            "hugging",
            "align_grpo",
        )

        for field_name in boolean_fields:
            setattr(
                self,
                field_name,
                _bool_value(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        # ---------------------------------------------------------------------
        # Cross-option checks
        # ---------------------------------------------------------------------

        if (
            self.strategy == "ties"
            and self.use_dare
        ):
            logger.warning(
                "strategy='ties' and use_dare=True are both enabled. "
                "The strategy implementation must decide which behavior "
                "takes precedence."
            )

        if (
            self.strategy == "dare"
            and not self.use_dare
        ):
            logger.info(
                "strategy='dare' selected; enabling use_dare automatically."
            )
            self.use_dare = True

        if (
            self.merge_knowledge_distill
            and self.repair_steps <= 0
        ):
            logger.warning(
                "merge_knowledge_distill=True but repair_steps=0. "
                "Knowledge distillation may have no opportunity to run."
            )

        if self.hugging and not self.hugging_token:
            raise ValueError(
                "hugging=True requires hugging_token."
            )

        if (
            self.align_grpo
            and self.align_grpo_steps <= 0
        ):
            raise ValueError(
                "align_grpo_steps must be positive when "
                "align_grpo=True."
            )

        if self.align_grpo_reward_funcs is not None:
            if isinstance(
                self.align_grpo_reward_funcs,
                str,
            ):
                raise TypeError(
                    "align_grpo_reward_funcs must be a list "
                    "of callables."
                )

            for index, function in enumerate(
                self.align_grpo_reward_funcs
            ):
                if not callable(function):
                    raise TypeError(
                        f"align_grpo_reward_funcs[{index}] "
                        "must be callable."
                    )

        # Fisher information requires meaningful calibration/importance
        # estimates. The merge engine can decide exactly how to obtain them,
        # but making the configuration obviously contradictory should be
        # surfaced.
        if (
            self.use_fisher
            and self.calibration_data is None
        ):
            logger.warning(
                "use_fisher=True but calibration_data=None. "
                "The merger must obtain or approximate Fisher information "
                "internally."
            )

        if self.force_cuda_merge:
            logger.info(
                "force_cuda_merge=True. Merge execution may fail on machines "
                "without CUDA."
            )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(
        self,
        *,
        include_sensitive: bool = False,
        include_runtime_objects: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert configuration to a JSON-friendly dictionary.

        Sensitive Hugging Face credentials are excluded by default.
        Runtime-only calibration/reward objects are also excluded.
        """
        data = asdict(self)

        data.pop(
            "calibration_data",
            None,
        )

        if not include_runtime_objects:
            data.pop(
                "align_grpo_reward_funcs",
                None,
            )

        if not include_sensitive:
            data.pop(
                "hugging_token",
                None,
            )

        return _json_safe(data)

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        include_sensitive: bool = False,
    ) -> Path:
        """Save MergeConfig to JSON without exposing secrets by default."""
        target = _write_json(
            self.to_dict(
                include_sensitive=include_sensitive,
            ),
            path,
        )

        logger.info(
            "Saved MergeConfig to %s",
            target,
        )

        return target

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "MergeConfig":
        """Construct MergeConfig from a dictionary."""
        if not isinstance(data, Mapping):
            raise TypeError(
                "MergeConfig.from_dict() expects a mapping."
            )

        allowed_fields = {
            item.name
            for item in fields(cls)
        }

        cleaned = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        ignored = set(data) - allowed_fields

        if ignored:
            logger.warning(
                "Ignoring unknown MergeConfig fields: %s",
                ", ".join(sorted(ignored)),
            )

        return cls(**cleaned)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
    ) -> "MergeConfig":
        """Load MergeConfig from JSON."""
        return cls.from_dict(
            _read_json(path)
        )
