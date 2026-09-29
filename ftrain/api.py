"""
FTRAIN High-Level API
=====================

Deep-enhanced public orchestration layer for FTRAIN.

Public API
----------
    train.fire(...)
    merge.fire(...)
    test()

The API layer is deliberately thin: model/training/merge logic stays in
``core``, ``merger``, ``data_utils`` and the lower-level utilities.

Enhancements over the original API
-----------------------------------
- Strict public-argument validation.
- Alias handling for legacy argument names.
- Deterministic seed support when the caller provides ``seed``.
- Safer dataset splitting, including Hugging Face Dataset-like objects.
- Configuration validation before expensive model work.
- Optional ``dry_run`` mode for both training and merging.
- Optional ``return_metadata`` wrapper for orchestration diagnostics.
- Explicit ``MaximumPower=True`` plumbing without keyword collisions.
- Better logging and stage timing.
- Safer output-directory normalization.
- Package health diagnostics that do not load models.
- Backward-compatible public exports.
"""

from __future__ import annotations

import inspect
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from . import rewards
from .config import MergeConfig, TrainConfig
from .core import Ftrain
from .data_utils import load_data
from .merger import Merger

__all__ = [
    "train",
    "merge",
    "test",
    "xml_format_reward",
    "math_exact_reward",
    "python_exec_reward",
]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

_DEFAULT_TRAIN_OUTPUT_DIR = "./ftrain_output"
_DEFAULT_MERGE_OUTPUT_DIR = "./merged_model"
_DEFAULT_VALIDATION_RATIO = 0.10
_MIN_VALIDATION_RATIO = 0.0
_MAX_VALIDATION_RATIO = 1.0

_RESERVED_TRAIN_KEYS = {
    "model_name",
    "captain_model",
    "max_steps",
    "answer_mode",
    "captain_mode",
    "output_dir",
    "maximum_power",
}

_RESERVED_MERGE_KEYS = {
    "model_a",
    "model_b",
    "captain_model",
    "output_dir",
    "maximum_power",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_model_name(model: Optional[str], argument_name: str) -> str:
    """Validate a model identifier or local path without forcing existence."""
    if model is None:
        raise ValueError(f"'{argument_name}' must be provided.")
    if not isinstance(model, str):
        raise TypeError(
            f"'{argument_name}' must be a string, got {type(model).__name__}."
        )
    model = model.strip()
    if not model:
        raise ValueError(f"'{argument_name}' cannot be empty.")
    return model


def _validate_steps(steps: int) -> int:
    """Validate the requested number of training steps."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(
            f"'Steps' must be an integer, got {type(steps).__name__}."
        )
    if steps <= 0:
        raise ValueError(f"'Steps' must be greater than zero, got {steps}.")
    return steps


def _validate_validation_ratio(ratio: float) -> float:
    """Validate and normalize a validation split ratio."""
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TypeError(
            "'validation_ratio' must be a number between 0 and 1, "
            f"got {type(ratio).__name__}."
        )

    ratio = float(ratio)
    if not _MIN_VALIDATION_RATIO <= ratio <= _MAX_VALIDATION_RATIO:
        raise ValueError(
            "'validation_ratio' must be between 0.0 and 1.0, "
            f"got {ratio}."
        )
    return ratio


def _validate_bool_flag(value: Any, argument_name: str, *, allow_none: bool = False) -> bool:
    """Strict boolean validation for public API flags."""
    if value is None and allow_none:
        return False
    if isinstance(value, bool):
        return value
    raise TypeError(
        f"'{argument_name}' must be a boolean (True/False), "
        f"got {type(value).__name__}."
    )


def _validate_positive_int(value: Any, argument_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"'{argument_name}' must be an integer, got {type(value).__name__}."
        )
    if value <= 0:
        raise ValueError(f"'{argument_name}' must be > 0, got {value}.")
    return value


def _normalize_output_dir(output_dir: Any, default: str) -> str:
    if output_dir is None:
        output_dir = default
    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError(
            "'output_dir' must be a string or path-like object, "
            f"got {type(output_dir).__name__}."
        )

    value = os.fspath(output_dir).strip()
    if not value:
        raise ValueError("'output_dir' cannot be empty.")
    return value


def _maybe_seed(seed: Optional[int]) -> Optional[int]:
    """Seed common Python RNGs when a seed is explicitly provided."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"'seed' must be an integer or None, got {type(seed).__name__}."
        )
    if seed < 0:
        raise ValueError(f"'seed' must be >= 0, got {seed}.")

    random.seed(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    return seed


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _dataset_len(data: Any) -> int:
    try:
        size = len(data)
    except TypeError as exc:
        raise TypeError(
            "The object returned by 'load_data()' must be sized "
            "(it must implement __len__)."
        ) from exc

    if size < 0:
        raise ValueError(f"Loaded dataset reported an invalid length: {size}.")
    return size


def _slice_dataset(data: Any, start: int, stop: Optional[int] = None) -> Any:
    """Best-effort slicing for normal Python/Arrow/HF dataset-like objects."""
    end = start if stop is None else stop

    # Standard slicing.
    try:
        return data[start:end]
    except (TypeError, IndexError, KeyError):
        pass

    # Hugging Face Dataset-style ``select``.
    if hasattr(data, "select"):
        try:
            indices = list(range(start, end))
            return data.select(indices)
        except Exception:
            pass

    raise TypeError(
        "The loaded dataset does not support slicing or Dataset.select(). "
        "FTRAIN requires a sliceable dataset for automatic validation splits."
    )


def _split_data(
    data: Any,
    validation_ratio: float = _DEFAULT_VALIDATION_RATIO,
    *,
    seed: Optional[int] = None,
    shuffle: bool = False,
) -> Tuple[Any, Any]:
    """
    Split data into training and validation portions.

    By default, the original ordering is preserved. When ``shuffle=True``,
    common Dataset-like ``shuffle`` APIs are used where available; otherwise
    the function falls back to deterministic index reordering when possible.
    """
    validation_ratio = _validate_validation_ratio(validation_ratio)
    dataset_size = _dataset_len(data)

    if dataset_size == 0:
        raise ValueError(
            "The loaded dataset is empty. FTRAIN cannot start training "
            "without at least one training sample."
        )

    if dataset_size == 1 or validation_ratio <= 0.0:
        return data, None

    working = data
    if shuffle:
        if hasattr(data, "shuffle"):
            try:
                # HF Dataset uses a seed argument.
                working = data.shuffle(seed=seed)
            except TypeError:
                working = data.shuffle()
        else:
            try:
                indices = list(range(dataset_size))
                rng = random.Random(seed)
                rng.shuffle(indices)
                working = [data[i] for i in indices]
            except Exception as exc:
                raise TypeError(
                    "shuffle=True was requested, but the dataset does not "
                    "expose a usable shuffle() method or index access."
                ) from exc

    validation_size = max(1, int(round(dataset_size * validation_ratio)))
    validation_size = min(validation_size, dataset_size - 1)
    split_index = dataset_size - validation_size

    train_data = _slice_dataset(working, 0, split_index)
    validation_data = _slice_dataset(working, split_index, dataset_size)

    if _dataset_len(train_data) == 0:
        raise RuntimeError("Internal dataset splitting error: training split is empty.")

    if _dataset_len(validation_data) == 0:
        LOGGER.warning("Validation split is empty; continuing without validation.")
        validation_data = None

    return train_data, validation_data


# ---------------------------------------------------------------------------
# Config construction helpers
# ---------------------------------------------------------------------------


def _accepted_init_keys(cls: Any) -> Optional[set[str]]:
    """Return explicit constructor keys when introspection is available."""
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return None

    accepted: set[str] = set()
    has_varkw = False
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_varkw = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(name)

    return None if has_varkw else accepted


def _filter_supported_kwargs(cls: Any, kwargs: Mapping[str, Any], *, strict: bool) -> Dict[str, Any]:
    """Validate/filter config kwargs based on the installed config signature."""
    payload = dict(kwargs)
    accepted = _accepted_init_keys(cls)
    if accepted is None:
        return payload

    unsupported = sorted(set(payload) - accepted)
    if unsupported:
        message = (
            f"Unsupported {cls.__name__} option(s): "
            + ", ".join(unsupported)
        )
        if strict:
            raise TypeError(message)
        LOGGER.warning(message + "; ignoring them.")
        for key in unsupported:
            payload.pop(key, None)

    return payload


def _build_train_config(
    *,
    model: str,
    captain: Optional[str],
    steps: int,
    answer: str,
    output_dir: str,
    extra_kwargs: Mapping[str, Any],
    maximum_power: bool = False,
    strict_config: bool = True,
) -> TrainConfig:
    config_kwargs: Dict[str, Any] = dict(extra_kwargs)
    for key in _RESERVED_TRAIN_KEYS:
        config_kwargs.pop(key, None)

    config_kwargs.update(
        {
            "model_name": model,
            "captain_model": captain,
            "max_steps": steps,
            "answer_mode": answer,
            "captain_mode": "llm" if captain else "rule",
            "output_dir": output_dir,
            "maximum_power": bool(maximum_power),
        }
    )

    config_kwargs = _filter_supported_kwargs(
        TrainConfig,
        config_kwargs,
        strict=strict_config,
    )

    try:
        config = TrainConfig(**config_kwargs)
    except TypeError as exc:
        raise TypeError(
            "Failed to construct TrainConfig. Check the supplied training "
            "options against your installed TrainConfig."
        ) from exc

    if hasattr(config, "validate"):
        try:
            config.validate()
        except Exception as exc:
            raise ValueError(f"TrainConfig validation failed: {exc}") from exc

    return config


def _build_merge_config(
    *,
    model_a: str,
    model_b: str,
    captain: Optional[str],
    output_dir: str,
    extra_kwargs: Mapping[str, Any],
    maximum_power: bool = False,
    strict_config: bool = True,
) -> MergeConfig:
    config_kwargs: Dict[str, Any] = dict(extra_kwargs)
    for key in _RESERVED_MERGE_KEYS:
        config_kwargs.pop(key, None)

    config_kwargs.update(
        {
            "model_a": model_a,
            "model_b": model_b,
            "captain_model": captain,
            "output_dir": output_dir,
            "maximum_power": bool(maximum_power),
        }
    )

    config_kwargs = _filter_supported_kwargs(
        MergeConfig,
        config_kwargs,
        strict=strict_config,
    )

    try:
        config = MergeConfig(**config_kwargs)
    except TypeError as exc:
        raise TypeError(
            "Failed to construct MergeConfig. Check the supplied merge "
            "options against your installed MergeConfig."
        ) from exc

    if hasattr(config, "validate"):
        try:
            config.validate()
        except Exception as exc:
            raise ValueError(f"MergeConfig validation failed: {exc}") from exc

    return config


# ---------------------------------------------------------------------------
# Metadata / diagnostics helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    *,
    operation: str,
    started_at: float,
    success: bool,
    output_dir: Optional[str],
    maximum_power: bool,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
    captain: Optional[str] = None,
    steps: Optional[int] = None,
    dataset_size: Optional[int] = None,
    validation_size: Optional[int] = None,
    result: Any = None,
) -> Dict[str, Any]:
    return {
        "operation": operation,
        "success": bool(success),
        "elapsed_seconds": round(max(0.0, time.perf_counter() - started_at), 4),
        "output_dir": output_dir,
        "maximum_power": bool(maximum_power),
        "model_a": model_a,
        "model_b": model_b,
        "captain": captain,
        "steps": steps,
        "dataset_size": dataset_size,
        "validation_size": validation_size,
        "result_type": type(result).__name__ if result is not None else None,
    }


def _wrap_result(result: Any, metadata: Dict[str, Any], *, return_metadata: bool) -> Any:
    if not return_metadata:
        return result
    return {
        "result": result,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Training API
# ---------------------------------------------------------------------------


class train:
    """High-level FTRAIN training interface."""

    @staticmethod
    def fire(
        Model: str,
        Data: Any,
        Steps: int = 100,
        Captain: Optional[str] = None,
        Answer: str = "auto_yes",
        MaximumPower: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run the complete FTRAIN training pipeline.

        Extra orchestration options
        ----------------------------
        output_dir:
            Output directory. Defaults to ``./ftrain_output``.
        validation_ratio / val_split:
            Fraction reserved for validation. Defaults to ``0.10``.
        shuffle_validation_split:
            Shuffle before splitting. Defaults to False.
        seed:
            Optional deterministic seed.
        dry_run:
            Validate and build the config without starting training.
        return_metadata:
            Return ``{"result": ..., "metadata": ...}``.
        strict_config:
            Reject unknown TrainConfig keywords instead of warning/ignoring.
        """
        started = time.perf_counter()

        model = _validate_model_name(Model, "Model")
        steps = _validate_steps(Steps)

        if not isinstance(Answer, str):
            raise TypeError(
                f"'Answer' must be a string, got {type(Answer).__name__}."
            )
        answer = Answer.strip()
        if not answer:
            raise ValueError("'Answer' cannot be empty.")

        captain = (
            _validate_model_name(Captain, "Captain")
            if Captain is not None
            else None
        )

        runtime_kwargs: Dict[str, Any] = dict(kwargs)
        requested_power = MaximumPower
        if requested_power is None:
            requested_power = runtime_kwargs.pop("maximum_power", False)
        else:
            runtime_kwargs.pop("maximum_power", None)
        maximum_power = _validate_bool_flag(requested_power, "MaximumPower")

        output_dir = _normalize_output_dir(
            runtime_kwargs.pop("output_dir", _DEFAULT_TRAIN_OUTPUT_DIR),
            _DEFAULT_TRAIN_OUTPUT_DIR,
        )

        validation_ratio = runtime_kwargs.pop(
            "validation_ratio",
            runtime_kwargs.pop("val_split", _DEFAULT_VALIDATION_RATIO),
        )
        shuffle_split = runtime_kwargs.pop("shuffle_validation_split", False)
        shuffle_split = _validate_bool_flag(
            shuffle_split,
            "shuffle_validation_split",
        )

        seed = runtime_kwargs.pop("seed", None)
        _maybe_seed(seed)

        dry_run = _validate_bool_flag(
            runtime_kwargs.pop("dry_run", False),
            "dry_run",
        )
        return_metadata = _validate_bool_flag(
            runtime_kwargs.pop("return_metadata", False),
            "return_metadata",
        )
        strict_config = _validate_bool_flag(
            runtime_kwargs.pop("strict_config", True),
            "strict_config",
        )

        LOGGER.info(
            "FTRAIN train.fire: model=%s steps=%d captain=%s maximum_power=%s output=%s",
            model,
            steps,
            captain or "disabled",
            maximum_power,
            output_dir,
        )

        data = load_data(Data)
        train_data, val_data = _split_data(
            data,
            validation_ratio=validation_ratio,
            seed=seed,
            shuffle=shuffle_split,
        )

        config = _build_train_config(
            model=model,
            captain=captain,
            steps=steps,
            answer=answer,
            output_dir=output_dir,
            extra_kwargs=runtime_kwargs,
            maximum_power=maximum_power,
            strict_config=strict_config,
        )

        if dry_run:
            metadata = _make_metadata(
                operation="train",
                started_at=started,
                success=True,
                output_dir=output_dir,
                maximum_power=maximum_power,
                captain=captain,
                steps=steps,
                dataset_size=_dataset_len(data),
                validation_size=(
                    _dataset_len(val_data) if val_data is not None else 0
                ),
            )
            return _wrap_result(config, metadata, return_metadata=return_metadata)

        engine = Ftrain(config, train_data, val_data)
        result = engine.train()

        metadata = _make_metadata(
            operation="train",
            started_at=started,
            success=True,
            output_dir=output_dir,
            maximum_power=maximum_power,
            captain=captain,
            steps=steps,
            dataset_size=_dataset_len(data),
            validation_size=(
                _dataset_len(val_data) if val_data is not None else 0
            ),
            result=result,
        )
        LOGGER.info("FTRAIN train.fire completed in %.2fs", metadata["elapsed_seconds"])
        return _wrap_result(result, metadata, return_metadata=return_metadata)


# ---------------------------------------------------------------------------
# Merge API
# ---------------------------------------------------------------------------


class merge:
    """High-level FTRAIN model-merging interface."""

    @staticmethod
    def fire(
        First: Optional[str] = None,
        Second: Optional[str] = None,
        Captain: Optional[str] = None,
        MaximumPower: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run the complete FTRAIN merge pipeline.

        Legacy aliases:
            Model_a -> First
            Model_b -> Second

        Extra orchestration options:
            output_dir, seed, dry_run, return_metadata, strict_config,
            allow_self_merge.
        """
        started = time.perf_counter()
        runtime_kwargs: Dict[str, Any] = dict(kwargs)

        model_a = First
        model_b = Second

        if model_a is None:
            model_a = runtime_kwargs.pop("Model_a", None)
        else:
            runtime_kwargs.pop("Model_a", None)

        if model_b is None:
            model_b = runtime_kwargs.pop("Model_b", None)
        else:
            runtime_kwargs.pop("Model_b", None)

        model_a = _validate_model_name(model_a, "First")
        model_b = _validate_model_name(model_b, "Second")

        captain = (
            _validate_model_name(Captain, "Captain")
            if Captain is not None
            else None
        )

        requested_power = MaximumPower
        if requested_power is None:
            requested_power = runtime_kwargs.pop("maximum_power", False)
        else:
            runtime_kwargs.pop("maximum_power", None)
        maximum_power = _validate_bool_flag(requested_power, "MaximumPower")

        allow_self_merge = _validate_bool_flag(
            runtime_kwargs.pop("allow_self_merge", False),
            "allow_self_merge",
        )
        if model_a == model_b and not allow_self_merge:
            raise ValueError(
                "The two merge inputs resolve to the same model. "
                "Pass allow_self_merge=True only for an intentional self-merge."
            )

        output_dir = _normalize_output_dir(
            runtime_kwargs.pop("output_dir", _DEFAULT_MERGE_OUTPUT_DIR),
            _DEFAULT_MERGE_OUTPUT_DIR,
        )

        seed = runtime_kwargs.pop("seed", None)
        _maybe_seed(seed)

        dry_run = _validate_bool_flag(
            runtime_kwargs.pop("dry_run", False),
            "dry_run",
        )
        return_metadata = _validate_bool_flag(
            runtime_kwargs.pop("return_metadata", False),
            "return_metadata",
        )
        strict_config = _validate_bool_flag(
            runtime_kwargs.pop("strict_config", True),
            "strict_config",
        )

        LOGGER.info(
            "FTRAIN merge.fire: A=%s B=%s captain=%s maximum_power=%s output=%s",
            model_a,
            model_b,
            captain or "disabled",
            maximum_power,
            output_dir,
        )

        config = _build_merge_config(
            model_a=model_a,
            model_b=model_b,
            captain=captain,
            output_dir=output_dir,
            extra_kwargs=runtime_kwargs,
            maximum_power=maximum_power,
            strict_config=strict_config,
        )

        if dry_run:
            metadata = _make_metadata(
                operation="merge",
                started_at=started,
                success=True,
                output_dir=output_dir,
                maximum_power=maximum_power,
                model_a=model_a,
                model_b=model_b,
                captain=captain,
            )
            return _wrap_result(config, metadata, return_metadata=return_metadata)

        merger = Merger(config)
        result = merger.merge()

        metadata = _make_metadata(
            operation="merge",
            started_at=started,
            success=True,
            output_dir=output_dir,
            maximum_power=maximum_power,
            model_a=model_a,
            model_b=model_b,
            captain=captain,
            result=result,
        )
        LOGGER.info("FTRAIN merge.fire completed in %.2fs", metadata["elapsed_seconds"])
        return _wrap_result(result, metadata, return_metadata=return_metadata)


# ---------------------------------------------------------------------------
# Diagnostics / smoke test
# ---------------------------------------------------------------------------


def test(*, verbose: bool = True, include_optional: bool = True) -> bool:
    """
    Run a lightweight package-level health check.

    No model is loaded and no GPU memory is allocated.
    """
    required_objects = {
        "train.fire": getattr(train, "fire", None),
        "merge.fire": getattr(merge, "fire", None),
        "xml_format_reward": xml_format_reward,
        "math_exact_reward": math_exact_reward,
        "python_exec_reward": python_exec_reward,
    }

    missing = [
        name
        for name, obj in required_objects.items()
        if obj is None or not callable(obj)
    ]
    if missing:
        raise RuntimeError(
            "FTRAIN API health check failed. Missing or invalid exports: "
            + ", ".join(missing)
        )

    # Validate that the config classes remain constructible at the API layer.
    if include_optional:
        try:
            inspect.signature(TrainConfig)
            inspect.signature(MergeConfig)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "FTRAIN API config introspection failed."
            ) from exc

    if verbose:
        print("✅ FTRAIN API health check passed")
        print("✅ train.fire available")
        print("✅ merge.fire available")
        print("✅ reward functions available")
        print("✅ config validation/introspection available")

    return True


# ---------------------------------------------------------------------------
# Public reward exports
# ---------------------------------------------------------------------------


xml_format_reward = rewards.xml_format_reward
math_exact_reward = rewards.math_exact_reward
python_exec_reward = rewards.python_exec_reward
