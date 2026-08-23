
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


_LAYER_ATTRS: Tuple[str, ...] = (
    "num_hidden_layers", "num_layers", "n_layer",
    "num_layers_decoder", "decoder_layers",
)
_HIDDEN_ATTRS: Tuple[str, ...] = (
    "hidden_size", "d_model", "dim", "model_dim",
)
_INTERMEDIATE_ATTRS: Tuple[str, ...] = (
    "intermediate_size", "ffn_dim", "intermediate_dim", "d_ff",
)
_HEAD_ATTRS: Tuple[str, ...] = (
    "num_attention_heads", "num_heads", "n_head",
)
_KV_HEAD_ATTRS: Tuple[str, ...] = (
    "num_key_value_heads", "num_kv_heads", "n_kv_head",
)
_VOCAB_ATTRS: Tuple[str, ...] = (
    "vocab_size", "n_vocab",
)
_EXPERT_ATTRS: Tuple[str, ...] = (
    "num_local_experts", "num_experts", "n_routed_experts",
)


def _config(model_or_config: Any) -> Any:
    if model_or_config is None:
        return None
    cfg = getattr(model_or_config, "config", None)
    if cfg is not None:
        return cfg
    base = getattr(model_or_config, "base_model", None)
    return getattr(base, "config", model_or_config) if base is not None else model_or_config


def _first(obj: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        try:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        except Exception:
            pass
    return default


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        value = int(value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _name(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = False,
    benchmark: Optional[bool] = None,
) -> int:
    """Seed Python, NumPy and PyTorch RNGs."""
    try:
        seed = int(seed) % (2 ** 32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer, got {seed!r}") from exc

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            try:
                torch.backends.cudnn.deterministic = deterministic
                torch.backends.cudnn.benchmark = (
                    not deterministic if benchmark is None else bool(benchmark)
                )
                torch.backends.cuda.matmul.allow_tf32 = not deterministic
                torch.backends.cudnn.allow_tf32 = not deterministic
            except Exception:
                pass
            if deterministic:
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
    except ImportError:
        pass

    return seed


def get_family(name: Any) -> str:
    """Detect a normalized FTRAIN model family."""
    cfg = _config(name)
    text = " ".join(
        part for part in (
            _name(getattr(cfg, "model_type", None)),
            " ".join(_name(x) for x in (getattr(cfg, "architectures", None) or [])),
            _name(getattr(name, "name_or_path", None)),
            _name(name),
        ) if part
    )

    if "deepseek" in text:
        return "deepseek"
    if "qwen" in text:
        return "qwen"
    if "llama" in text:
        return "llama"
    if "gemma" in text:
        return "gemma"
    if "mistral" in text or "mixtral" in text:
        return "mistral"
    if "phi" in text:
        return "phi"
    return "generic"


def get_num_layers(model: Any) -> int:
    return _positive_int(_first(_config(model), _LAYER_ATTRS, 0))


def get_hidden_size(model: Any) -> int:
    return _positive_int(_first(_config(model), _HIDDEN_ATTRS, 0))


def get_intermediate_size(model: Any) -> int:
    return _positive_int(_first(_config(model), _INTERMEDIATE_ATTRS, 0))


def get_attention_heads(model: Any) -> int:
    return _positive_int(_first(_config(model), _HEAD_ATTRS, 0))


def get_kv_heads(model: Any) -> int:
    value = _positive_int(_first(_config(model), _KV_HEAD_ATTRS, 0))
    return value if value > 0 else get_attention_heads(model)


def get_head_dimension(model: Any) -> int:
    hidden = get_hidden_size(model)
    heads = get_attention_heads(model)
    return hidden // heads if hidden > 0 and heads > 0 else 0


def get_vocab_size(model: Any) -> int:
    return _positive_int(_first(_config(model), _VOCAB_ATTRS, 0))


def is_moe(model: Any) -> bool:
    """Detect common MoE configurations and module structures."""
    if model is None:
        return False

    cfg = _config(model)

    for attr in _EXPERT_ATTRS:
        try:
            value = getattr(cfg, attr, None)
            if value is not None and int(value) > 1:
                return True
        except (TypeError, ValueError):
            pass

    for attr in ("moe_config", "sparse_config", "expert_config"):
        nested = getattr(cfg, attr, None)
        if nested is None:
            continue
        for expert_attr in ("num_experts", "num_local_experts", "n_routed_experts"):
            try:
                value = getattr(nested, expert_attr, None)
                if value is not None and int(value) > 1:
                    return True
            except (TypeError, ValueError):
                pass

    try:
        for module_name, _ in model.named_modules():
            name = module_name.lower()
            if "router" in name or "moe" in name or ".experts" in name or "experts." in name:
                return True
    except Exception:
        pass

    return False


def get_num_experts(model: Any) -> int:
    return _positive_int(
        _first(
            _config(model),
            ("num_local_experts", "num_experts", "n_routed_experts"),
            0,
        )
    )


def get_experts_per_token(model: Any) -> int:
    return _positive_int(
        _first(
            _config(model),
            ("num_experts_per_tok", "num_selected_experts",
             "num_experts_per_token", "top_k"),
            0,
        )
    )


def get_model_dtype(model: Any) -> Optional[Any]:
    if model is None:
        return None
    try:
        return next(model.parameters()).dtype
    except Exception:
        return None


def get_model_device(model: Any) -> Optional[Any]:
    if model is None:
        return None

    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            import torch
            for value in device_map.values():
                if value in (None, "disk"):
                    continue
                try:
                    return torch.device(value)
                except Exception:
                    continue
    except Exception:
        pass

    try:
        return next(model.parameters()).device
    except Exception:
        return None


def has_quantization(model: Any) -> bool:
    if model is None:
        return False

    cfg = _config(model)

    for obj in (model, cfg):
        for attr in (
            "quantization_config",
            "_load_in_4bit", "_load_in_8bit",
            "load_in_4bit", "load_in_8bit",
        ):
            try:
                value = getattr(obj, attr, None)
            except Exception:
                continue

            if isinstance(value, bool) and value:
                return True

            if value is not None:
                typename = type(value).__name__.lower()
                if "quant" in typename or "bitsandbytes" in typename or "bnb" in typename:
                    return True

    try:
        for module in model.modules():
            typename = type(module).__name__.lower()
            if any(x in typename for x in ("4bit", "8bit", "bnb", "bitsandbytes", "quantized")):
                return True
    except Exception:
        pass

    return False


def is_model_dispatched(model: Any) -> bool:
    if model is None:
        return False
    try:
        device_map = getattr(model, "hf_device_map", None)
        return isinstance(device_map, dict) and bool(device_map)
    except Exception:
        return False


def count_params(model: Any) -> Dict[str, Union[int, float]]:
    """Return total, trainable and frozen parameter statistics."""
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

    total = trainable = tensors = trainable_tensors = 0

    for param in model.parameters():
        count = int(param.numel())
        total += count
        tensors += 1
        if param.requires_grad:
            trainable += count
            trainable_tensors += 1

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "pct_trainable": 100.0 * trainable / max(1, total),
        "pct_frozen": 100.0 * frozen / max(1, total),
        "tensors": tensors,
        "trainable_tensors": trainable_tensors,
    }


def get_model_summary(model: Any) -> Dict[str, Any]:
    return {
        "family": get_family(model),
        "layers": get_num_layers(model),
        "hidden_size": get_hidden_size(model),
        "intermediate_size": get_intermediate_size(model),
        "attention_heads": get_attention_heads(model),
        "kv_heads": get_kv_heads(model),
        "head_dimension": get_head_dimension(model),
        "vocab_size": get_vocab_size(model),
        "is_moe": is_moe(model),
        "num_experts": get_num_experts(model),
        "experts_per_token": get_experts_per_token(model),
        "dtype": str(get_model_dtype(model)),
        "device": str(get_model_device(model)),
        "quantized": has_quantization(model),
        "dispatched": is_model_dispatched(model),
        **count_params(model),
    }


def model_family(name: Any) -> str:
    return get_family(name)


def parameter_count(model: Any) -> Dict[str, Union[int, float]]:
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
