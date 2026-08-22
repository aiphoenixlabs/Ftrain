"""
FTRAIN Model Utilities v1.1
===========================

Centralized model/runtime utilities used by the FTRAIN engine.

Provides
--------
seed_everything
    Deterministic seeding across Python, NumPy, PyTorch, CUDA and optionally
    common distributed/runtime libraries.

get_family
    Robust model-family detection from model names, configs, paths, and model
    objects.

get_num_layers
    Safely detect transformer depth from model/configuration objects.

is_moe
    Detect Mixture-of-Experts architectures from common config conventions and
    model structures.

count_params
    Count total/trainable/frozen parameters with useful diagnostics.

Additional helpers
------------------
get_model_dtype
get_model_device
has_quantization
is_model_dispatched
get_vocab_size
get_hidden_size
get_attention_heads
get_intermediate_size
get_model_summary

Design goals
------------
• Defensive behavior.
• No CUDA assumptions.
• Deterministic experiments.
• Compatibility with Unsloth and Hugging Face Transformers.
• Compatibility with wrapped/PEFT models.
• Useful diagnostics for the training and merging engines.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Constants
# =============================================================================

FamilyValue = str

KNOWN_FAMILIES = (
    "qwen",
    "deepseek",
    "llama",
    "gemma",
    "phi",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen2_moe",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v4",
    "gemma2",
    "gemma3",
    "phi3",
    "phi4",
    "mistral",
    "mixtral",
)

FAMILY_ALIASES = {
    "qwen2": "qwen",
    "qwen3": "qwen",
    "qwen2_moe": "qwen",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseek_v4": "deepseek",
    "mixtral": "mistral",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "phi3": "phi",
    "phi4": "phi",
}

_LAYER_ATTRIBUTES = (
    "num_hidden_layers",
    "num_layers",
    "n_layer",
    "num_layers_decoder",
    "decoder_layers",
)

_HEAD_ATTRIBUTES = (
    "num_attention_heads",
    "num_heads",
    "n_head",
)

_KV_HEAD_ATTRIBUTES = (
    "num_key_value_heads",
    "num_kv_heads",
    "n_kv_head",
)

_HIDDEN_ATTRIBUTES = (
    "hidden_size",
    "d_model",
    "dim",
    "model_dim",
)

_INTERMEDIATE_ATTRIBUTES = (
    "intermediate_size",
    "ffn_dim",
    "intermediate_dim",
    "d_ff",
)

_VOCAB_ATTRIBUTES = (
    "vocab_size",
    "n_vocab",
)

_EXPERT_ATTRIBUTES = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "num_experts_per_tok",
)


# =============================================================================
# Generic helpers
# =============================================================================

def _config_from_model(
    model_or_config: Any,
) -> Any:
    """
    Return the most useful configuration object.

    Handles:
        model.config
        PEFT wrappers
        Unsloth wrappers
        raw config objects
    """
    if model_or_config is None:
        return None

    config = getattr(
        model_or_config,
        "config",
        None,
    )

    if config is not None:
        return config

    # PEFT models sometimes expose the wrapped base model.
    base_model = getattr(
        model_or_config,
        "base_model",
        None,
    )

    if base_model is not None:
        base_config = getattr(
            base_model,
            "config",
            None,
        )

        if base_config is not None:
            return base_config

    return model_or_config


def _get_first_attr(
    obj: Any,
    names: Tuple[str, ...],
    default: Any = None,
) -> Any:
    if obj is None:
        return default

    for name in names:
        try:
            value = getattr(
                obj,
                name,
                None,
            )
        except Exception:
            continue

        if value is None:
            continue

        return value

    return default


def _safe_positive_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default

    return result if result > 0 else default


def _normalize_name(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = False,
    benchmark: Optional[bool] = None,
) -> int:
    """
    Seed all common random-number generators used by FTRAIN.

    Parameters
    ----------
    seed:
        Base random seed.

    deterministic:
        When True, request deterministic PyTorch algorithms where supported.

    benchmark:
        Optional cuDNN benchmark setting. When omitted, it is chosen from
        ``deterministic``:
            deterministic=True  -> benchmark=False
            deterministic=False -> leave backend default unchanged

    Returns
    -------
    int
        The normalized seed.

    Notes
    -----
    Completely deterministic GPU training is not guaranteed for every model
    kernel, but this configures the major RNG sources consistently.
    """
    try:
        normalized_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"seed must be an integer, got {seed!r}"
        ) from exc

    # Make negative seeds deterministic too.
    normalized_seed %= 2**32

    os.environ["PYTHONHASHSEED"] = str(
        normalized_seed
    )

    random.seed(
        normalized_seed
    )

    np.random.seed(
        normalized_seed
    )

    try:
        import torch

        torch.manual_seed(
            normalized_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed(
                normalized_seed
            )
            torch.cuda.manual_seed_all(
                normalized_seed
            )

            try:
                torch.backends.cuda.matmul.allow_tf32 = not deterministic
                torch.backends.cudnn.allow_tf32 = not deterministic
            except Exception:
                pass

            try:
                torch.backends.cudnn.deterministic = bool(
                    deterministic
                )

                if benchmark is None:
                    torch.backends.cudnn.benchmark = not deterministic
                else:
                    torch.backends.cudnn.benchmark = bool(
                        benchmark
                    )

            except Exception:
                pass

            try:
                if deterministic:
                    torch.use_deterministic_algorithms(
                        True
                    )
            except Exception:
                # Some installed torch versions/backends cannot enable all
                # deterministic kernels.
                pass

    except ImportError:
        # model_utils can still be imported by lightweight tooling without
        # PyTorch installed.
        pass

    return normalized_seed


# =============================================================================
# Model family detection
# =============================================================================

def get_family(
    name: Any,
) -> str:
    """
    Detect a normalized model family.

    Accepts:
        model identifier
        local model path
        model object
        config object

    Returns one of:
        qwen, deepseek, llama, gemma, phi, mistral, generic
    """
    if name is None:
        return "generic"

    # Model/config object.
    config = _config_from_model(
        name
    )

    model_type = _normalize_name(
        getattr(
            config,
            "model_type",
            None,
        )
    )

    architectures = getattr(
        config,
        "architectures",
        None,
    )

    architecture_text = ""

    if isinstance(
        architectures,
        (list, tuple),
    ):
        architecture_text = " ".join(
            _normalize_name(value)
            for value in architectures
        )

    # Give explicit model_type priority.
    search_text = " ".join(
        part
        for part in (
            model_type,
            architecture_text,
            _normalize_name(
                getattr(
                    name,
                    "name_or_path",
                    None,
                )
            ),
            _normalize_name(name),
        )
        if part
    )

    # Order matters: more specific families first.
    ordered = (
        ("deepseek", "deepseek"),
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("phi", "phi"),
    )

    for marker, family in ordered:
        if marker in search_text:
            return family

    return "generic"


# =============================================================================
# Transformer architecture information
# =============================================================================

def get_num_layers(
    model: Any,
) -> int:
    """
    Safely detect transformer depth.

    Returns 0 when it cannot be determined.
    """
    config = _config_from_model(
        model
    )

    value = _get_first_attr(
        config,
        _LAYER_ATTRIBUTES,
        0,
    )

    return _safe_positive_int(
        value,
        0,
    )


def get_hidden_size(
    model: Any,
) -> int:
    """Return hidden/model width, or 0 if unavailable."""
    config = _config_from_model(
        model
    )

    return _safe_positive_int(
        _get_first_attr(
            config,
            _HIDDEN_ATTRIBUTES,
            0,
        ),
        0,
    )


def get_intermediate_size(
    model: Any,
) -> int:
    """Return FFN intermediate width, or 0 if unavailable."""
    config = _config_from_model(
        model
    )

    return _safe_positive_int(
        _get_first_attr(
            config,
            _INTERMEDIATE_ATTRIBUTES,
            0,
        ),
        0,
    )


def get_attention_heads(
    model: Any,
) -> int:
    """Return the number of attention heads."""
    config = _config_from_model(
        model
    )

    return _safe_positive_int(
        _get_first_attr(
            config,
            _HEAD_ATTRIBUTES,
            0,
        ),
        0,
    )


def get_kv_heads(
    model: Any,
) -> int:
    """
    Return the number of key/value heads.

    Falls back to full attention-head count when the model does not expose a
    separate GQA/MQA field.
    """
    config = _config_from_model(
        model
    )

    value = _safe_positive_int(
        _get_first_attr(
            config,
            _KV_HEAD_ATTRIBUTES,
            0,
        ),
        0,
    )

    if value > 0:
        return value

    return get_attention_heads(
        model
    )


def get_vocab_size(
    model: Any,
) -> int:
    """Return vocabulary size."""
    config = _config_from_model(
        model
    )

    return _safe_positive_int(
        _get_first_attr(
            config,
            _VOCAB_ATTRIBUTES,
            0,
        ),
        0,
    )


def get_head_dimension(
    model: Any,
) -> int:
    """
    Calculate attention head dimension when possible.
    """
    hidden = get_hidden_size(
        model
    )

    heads = get_attention_heads(
        model
    )

    if hidden <= 0 or heads <= 0:
        return 0

    return hidden // heads


# =============================================================================
# MoE detection
# =============================================================================

def is_moe(
    model: Any,
) -> bool:
    """
    Detect Mixture-of-Experts architectures.

    Checks configuration first and then common module/parameter naming when
    possible.
    """
    if model is None:
        return False

    config = _config_from_model(
        model
    )

    # ------------------------------------------------------------
    # Explicit configuration signals
    # ------------------------------------------------------------

    for attribute in (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
    ):
        try:
            value = getattr(
                config,
                attribute,
                None,
            )
        except Exception:
            value = None

        if value is not None:
            try:
                if int(value) > 1:
                    return True
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------
    # Nested MoE configuration
    # ------------------------------------------------------------

    for attribute in (
        "moe_config",
        "sparse_config",
        "expert_config",
    ):
        nested = getattr(
            config,
            attribute,
            None,
        )

        if nested is None:
            continue

        for expert_attribute in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
        ):
            value = getattr(
                nested,
                expert_attribute,
                None,
            )

            if value is not None:
                try:
                    if int(value) > 1:
                        return True
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------
    # Module-name fallback
    # ------------------------------------------------------------

    try:
        for name, _module in model.named_modules():
            lname = name.lower()

            if (
                "experts" in lname
                or "expert" in lname
                or "router" in lname
                or "moe" in lname
            ):
                return True

    except Exception:
        pass

    # ------------------------------------------------------------
    # Parameter-name fallback
    # ------------------------------------------------------------

    try:
        for name, _parameter in model.named_parameters():
            lname = name.lower()

            if (
                ".experts." in lname
                or ".expert." in lname
                or "router" in lname
                or "moe" in lname
            ):
                return True

    except Exception:
        pass

    return False


def get_num_experts(
    model: Any,
) -> int:
    """Return configured MoE expert count, or 0."""
    config = _config_from_model(
        model
    )

    value = _get_first_attr(
        config,
        (
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
        ),
        0,
    )

    return _safe_positive_int(
        value,
        0,
    )


def get_experts_per_token(
    model: Any,
) -> int:
    """Return top-k experts per token when exposed."""
    config = _config_from_model(
        model
    )

    value = _get_first_attr(
        config,
        (
            "num_experts_per_tok",
            "num_selected_experts",
            "top_k",
            "num_experts_per_token",
        ),
        0,
    )

    return _safe_positive_int(
        value,
        0,
    )


# =============================================================================
# Model dtype / device / quantization
# =============================================================================

def get_model_dtype(
    model: Any,
) -> Optional[Any]:
    """
    Return a representative model parameter dtype.

    Uses the first parameter with a valid dtype.
    """
    if model is None:
        return None

    try:
        for parameter in model.parameters():
            return parameter.dtype
    except Exception:
        return None

    return None


def get_model_device(
    model: Any,
) -> Optional[Any]:
    """
    Return a representative model device.

    For dispatched models, ``hf_device_map`` is consulted first.
    """
    if model is None:
        return None

    try:
        device_map = getattr(
            model,
            "hf_device_map",
            None,
        )

        if isinstance(
            device_map,
            dict,
        ) and device_map:
            values = list(
                device_map.values()
            )

            for value in values:
                if value not in {
                    "disk",
                    "cpu",
                    None,
                }:
                    try:
                        import torch
                        return torch.device(value)
                    except Exception:
                        pass

            try:
                import torch
                return torch.device(
                    str(values[0])
                )
            except Exception:
                pass

    except Exception:
        pass

    try:
        return next(
            model.parameters()
        ).device

    except Exception:
        return None


def has_quantization(
    model: Any,
) -> bool:
    """
    Detect common Hugging Face/Unsloth quantization configurations.
    """
    if model is None:
        return False

    config = _config_from_model(
        model
    )

    # Common HF bitsandbytes fields.
    for attribute in (
        "quantization_config",
        "_load_in_4bit",
        "_load_in_8bit",
        "load_in_4bit",
        "load_in_8bit",
    ):
        try:
            value = getattr(
                model,
                attribute,
                None,
            )

            if value is None:
                value = getattr(
                    config,
                    attribute,
                    None,
                )

            if value is not None:
                if isinstance(
                    value,
                    bool,
                ):
                    if value:
                        return True

                else:
                    name = _normalize_name(
                        type(value).__name__
                    )

                    if "quant" in name:
                        return True

                    if "bitsandbytes" in name:
                        return True

        except Exception:
            continue

    # Parameter dtypes/classes can also expose quantized modules.
    try:
        for module in model.modules():
            module_name = _normalize_name(
                type(module).__name__
            )

            if any(
                token in module_name
                for token in (
                    "4bit",
                    "8bit",
                    "bnb",
                    "bitsandbytes",
                    "quantized",
                )
            ):
                return True

    except Exception:
        pass

    return False


def is_model_dispatched(
    model: Any,
) -> bool:
    """Return whether Transformers/Accelerate has assigned a device map."""
    if model is None:
        return False

    try:
        device_map = getattr(
            model,
            "hf_device_map",
            None,
        )

        return bool(
            isinstance(
                device_map,
                dict,
            )
            and device_map
        )

    except Exception:
        return False


# =============================================================================
# Parameter statistics
# =============================================================================

def count_params(
    model: Any,
) -> Dict[str, Union[int, float, str]]:
    """
    Count total/trainable/frozen parameters.

    Returns
    -------
    dict
        total
        trainable
        frozen
        pct_trainable
        pct_frozen
        tensors
        trainable_tensors
    """
    if model is None:
        return {
            "total": 0,
            "trainable": 0,
            "frozen": 0,
            "pct_trainable": 0.0,
            "pct_frozen": 0.0,
            "tensors": 0,
            "trainable_tensors": 0,
        }

    total = 0
    trainable = 0
    tensors = 0
    trainable_tensors = 0

    try:
        for parameter in model.parameters():
            count = int(
                parameter.numel()
            )

            total += count
            tensors += 1

            if parameter.requires_grad:
                trainable += count
                trainable_tensors += 1

    except Exception as exc:
        raise RuntimeError(
            "Unable to count model parameters."
        ) from exc

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "pct_trainable": (
            100.0 * trainable / max(1, total)
        ),
        "pct_frozen": (
            100.0 * frozen / max(1, total)
        ),
        "tensors": tensors,
        "trainable_tensors": trainable_tensors,
    }


# =============================================================================
# Architecture summary
# =============================================================================

def get_model_summary(
    model: Any,
) -> Dict[str, Any]:
    """
    Return a compact architecture summary used by FTRAIN diagnostics and the
    Captain.
    """
    parameters = count_params(
        model
    )

    summary = {
        "family": get_family(model),
        "layers": get_num_layers(model),
        "hidden_size": get_hidden_size(model),
        "intermediate_size": get_intermediate_size(model),
        "attention_heads": get_attention_heads(model),
        "kv_heads": get_kv_heads(model),
        "head_dim": get_head_dimension(model),
        "vocab_size": get_vocab_size(model),
        "is_moe": is_moe(model),
        "num_experts": get_num_experts(model),
        "experts_per_token": get_experts_per_token(model),
        "dtype": str(
            get_model_dtype(model)
        ),
        "device": str(
            get_model_device(model)
        ),
        "quantized": has_quantization(model),
        "dispatched": is_model_dispatched(model),
        **parameters,
    }

    return summary


# =============================================================================
# Compatibility aliases
# =============================================================================

def model_family(
    name: Any,
) -> str:
    """Backward-compatible alias for ``get_family``."""
    return get_family(name)


def parameter_count(
    model: Any,
) -> Dict[str, Union[int, float, str]]:
    """Backward-compatible alias for ``count_params``."""
    return count_params(model)


__all__ = [
    "seed_everything",
    "get_family",
    "model_family",
    "get_num_layers",
    "get_hidden_size",
    "get_intermediate_size",
    "get_attention_heads",
    "get_kv_heads",
    "get_head_dimension",
    "get_vocab_size",
    "is_moe",
    "get_num_experts",
    "get_experts_per_token",
    "count_params",
    "parameter_count",
    "get_model_dtype",
    "get_model_device",
    "has_quantization",
    "is_model_dispatched",
    "get_model_summary",
]

path = Path("/mnt/data/ftrain_model_utils_FULL_enhanced_v1_1.py")
path.write_text(code, encoding="utf-8")
compile(code, str(path), "exec")
print(f"Created: {path}")
print(f"Lines: {len(code.splitlines())}")
# Basic import-level exercise without requiring torch model objects.
ns={}
exec(compile(code, str(path), "exec"), ns)
print(ns["get_family"]("unsloth/DeepSeek-R1-Distill-Qwen-1.5B"))
print(ns["get_family"]("meta-llama/Llama-3.2-1B-Instruct"))
print(ns["get_family"]("google/gemma-3-1b"))
print(ns["count_params"](None))
print(ns["get_model_summary"](None))
print("Self-check: PASS")
♀♀♀
null
  File "/home/oai/share/ftrain_model_utils_FULL_enhanced_v1_1.py", line 677
    def parameter_count(
TabError: inconsistent use of tabs and spaces in indentation
