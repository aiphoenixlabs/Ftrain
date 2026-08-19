"""
FTRAIN Model Presets
====================

Model-family-aware defaults for FTRAIN training and LoRA configuration.

This module intentionally contains configuration only. It does not import
Transformers, PEFT, Unsloth, CUDA, or model implementations.

Responsibilities
----------------
• Identify common model families.
• Provide sensible LoRA target modules.
• Provide family-specific learning-rate defaults.
• Provide LoRA rank defaults.
• Provide attention implementation preferences.
• Keep preset data immutable.
• Normalize common model-family aliases.
• Return defensive copies to callers.
• Support custom preset registration.

Design goals
------------
FTRAIN should be able to use:

    get_preset("qwen")

as well as:

    get_preset("Qwen2.5")
    get_preset("Llama-3")
    get_preset("DeepSeek-V3")

without requiring every caller to know the exact internal family key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "ModelPreset",
    "FAMILY_PRESETS",
    "get_preset",
    "normalize_family",
    "register_preset",
]


# =============================================================================
# Default LoRA targets
# =============================================================================

_BASE_LORA_TARGETS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


_DEEPSEEK_LORA_TARGETS: Tuple[str, ...] = (
    *_BASE_LORA_TARGETS,
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
)


# =============================================================================
# Model preset
# =============================================================================


@dataclass(frozen=True)
class ModelPreset:
    """
    Immutable configuration preset for a model family.

    Parameters
    ----------
    lora_targets:
        LoRA target module names.

    learning_rate:
        Recommended starting learning rate.

    lora_r:
        LoRA rank.

    attn_implementation:
        Preferred Transformers attention implementation.

        ``None`` means FTRAIN should let the model/framework decide.
    """

    lora_targets: Tuple[str, ...]
    learning_rate: float
    lora_r: int
    attn_implementation: Optional[str]

    def __post_init__(self) -> None:
        # ---------------------------------------------------------------------
        # Normalize and validate LoRA targets
        # ---------------------------------------------------------------------

        if not self.lora_targets:
            raise ValueError(
                "ModelPreset.lora_targets cannot be empty."
            )

        normalized_targets = tuple(
            dict.fromkeys(
                str(target).strip()
                for target in self.lora_targets
                if str(target).strip()
            )
        )

        if not normalized_targets:
            raise ValueError(
                "ModelPreset.lora_targets must contain at least one "
                "non-empty target."
            )

        object.__setattr__(
            self,
            "lora_targets",
            normalized_targets,
        )

        # ---------------------------------------------------------------------
        # Validate learning rate
        # ---------------------------------------------------------------------

        try:
            lr = float(
                self.learning_rate
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "learning_rate must be numeric."
            ) from exc

        if lr <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        object.__setattr__(
            self,
            "learning_rate",
            lr,
        )

        # ---------------------------------------------------------------------
        # Validate LoRA rank
        # ---------------------------------------------------------------------

        try:
            rank = int(
                self.lora_r
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lora_r must be an integer."
            ) from exc

        if rank <= 0:
            raise ValueError(
                "lora_r must be greater than zero."
            )

        object.__setattr__(
            self,
            "lora_r",
            rank,
        )

        # ---------------------------------------------------------------------
        # Normalize attention implementation
        # ---------------------------------------------------------------------

        if self.attn_implementation is not None:
            attention = str(
                self.attn_implementation
            ).strip()

            if not attention:
                attention = None

            object.__setattr__(
                self,
                "attn_implementation",
                attention,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the preset into a normal mutable dictionary.

        A new list is returned for ``lora_targets`` so callers cannot mutate
        the internal preset.
        """
        return {
            "lora_targets": list(
                self.lora_targets
            ),
            "learning_rate": self.learning_rate,
            "lora_r": self.lora_r,
            "attn_implementation": self.attn_implementation,
        }


# =============================================================================
# Built-in presets
# =============================================================================


FAMILY_PRESETS: Dict[str, ModelPreset] = {
    "qwen": ModelPreset(
        lora_targets=_BASE_LORA_TARGETS,
        learning_rate=2e-4,
        lora_r=16,
        attn_implementation="flash_attention_2",
    ),

    "llama": ModelPreset(
        lora_targets=_BASE_LORA_TARGETS,
        learning_rate=2e-4,
        lora_r=16,
        attn_implementation="flash_attention_2",
    ),

    "deepseek": ModelPreset(
        lora_targets=_DEEPSEEK_LORA_TARGETS,
        learning_rate=1.5e-4,
        lora_r=16,
        attn_implementation="flash_attention_2",
    ),

    "generic": ModelPreset(
        lora_targets=_BASE_LORA_TARGETS,
        learning_rate=2e-4,
        lora_r=16,
        attn_implementation=None,
    ),
}


# =============================================================================
# Family aliases
# =============================================================================


_FAMILY_ALIASES: Dict[str, str] = {
    # Qwen
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2.5": "qwen",
    "qwen2.5-instruct": "qwen",
    "qwen3": "qwen",
    "qwen3-instruct": "qwen",

    # Llama
    "llama": "llama",
    "llama2": "llama",
    "llama-2": "llama",
    "llama3": "llama",
    "llama-3": "llama",
    "llama3.1": "llama",
    "llama-3.1": "llama",
    "llama3.2": "llama",
    "llama-3.2": "llama",
    "llama3.3": "llama",
    "llama-3.3": "llama",
    "llama4": "llama",
    "llama-4": "llama",

    # DeepSeek
    "deepseek": "deepseek",
    "deepseek-v2": "deepseek",
    "deepseek-v2.5": "deepseek",
    "deepseek-v3": "deepseek",
    "deepseek-v3.1": "deepseek",
    "deepseek-r1": "deepseek",
    "deepseek-r1-distill": "deepseek",

    # Generic
    "generic": "generic",
    "auto": "generic",
    "unknown": "generic",
}


# =============================================================================
# Normalization
# =============================================================================


def _clean_family_name(
    family: Any,
) -> str:
    """
    Convert arbitrary family input into a normalized comparison string.
    """
    if family is None:
        return "generic"

    value = str(
        family
    ).strip().lower()

    if not value:
        return "generic"

    # Normalize separators so:
    #
    #   "DeepSeek_V3"
    #   "deepseek-v3"
    #   "DeepSeek V3"
    #
    # can all be handled consistently.
    value = value.replace(
        "_",
        "-",
    )

    value = " ".join(
        value.split()
    )

    value = value.replace(
        " ",
        "-",
    )

    return value


def normalize_family(
    family: Any,
) -> str:
    """
    Normalize a family identifier to an internal FTRAIN family key.

    Examples
    --------
    ::

        normalize_family("Qwen2.5")
        # "qwen"

        normalize_family("Llama-3.1")
        # "llama"

        normalize_family("DeepSeek-V3")
        # "deepseek"

        normalize_family(None)
        # "generic"
    """
    cleaned = _clean_family_name(
        family
    )

    # Direct alias lookup.
    if cleaned in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[
            cleaned
        ]

    # -------------------------------------------------------------------------
    # Prefix detection
    # -------------------------------------------------------------------------

    if cleaned.startswith(
        "qwen"
    ):
        return "qwen"

    if cleaned.startswith(
        "llama"
    ):
        return "llama"

    if cleaned.startswith(
        "deepseek"
    ):
        return "deepseek"

    return "generic"


# =============================================================================
# Preset registration
# =============================================================================


def register_preset(
    family: str,
    preset: ModelPreset,
    *,
    aliases: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> None:
    """
    Register a custom model-family preset.

    Example
    -------

    ::

        register_preset(
            "my_model",
            ModelPreset(
                lora_targets=(
                    "q_proj",
                    "v_proj",
                ),
                learning_rate=1e-4,
                lora_r=32,
                attn_implementation=None,
            ),
        )
    """
    if not isinstance(
        preset,
        ModelPreset,
    ):
        raise TypeError(
            "preset must be an instance of ModelPreset."
        )

    key = _clean_family_name(
        family
    )

    if key in FAMILY_PRESETS and not overwrite:
        raise ValueError(
            f"A preset for '{key}' already exists. "
            "Use overwrite=True to replace it."
        )

    FAMILY_PRESETS[
        key
    ] = preset

    if aliases:
        for alias in aliases:
            alias_key = _clean_family_name(
                alias
            )

            if (
                alias_key in _FAMILY_ALIASES
                and not overwrite
            ):
                raise ValueError(
                    f"The family alias '{alias}' already exists."
                )

            _FAMILY_ALIASES[
                alias_key
            ] = key


# =============================================================================
# Public preset API
# =============================================================================


def get_preset(
    family: Any,
) -> Dict[str, Any]:
    """
    Return a defensive dictionary copy of the appropriate model preset.

    Unknown families automatically receive the ``generic`` preset.

    Examples
    --------
    ::

        preset = get_preset("qwen")

    ::

        preset = get_preset("Qwen2.5")

    ::

        preset = get_preset("DeepSeek-V3")

    The returned dictionary can safely be modified by the caller without
    changing FTRAIN's global preset registry.
    """
    normalized = normalize_family(
        family
    )

    preset = FAMILY_PRESETS.get(
        normalized
    )

    if preset is None:
        preset = FAMILY_PRESETS[
            "generic"
        ]

    result = preset.to_dict()

    # Add metadata that is useful to higher-level FTRAIN code while keeping
    # backwards compatibility with the original four fields.
    result["family"] = normalized

    return result
