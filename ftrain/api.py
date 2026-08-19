"""
FTRAIN High-Level API
=====================

Stable public entry points for the FTRAIN training, merging, and diagnostic
pipelines.

Public API:
    train.fire(...)
    merge.fire(...)
    test()

The implementation in this module intentionally acts as an orchestration
layer. Heavy model/training/merging logic belongs in the underlying modules
(`core`, `merger`, `data_utils`, etc.).
"""

from __future__ import annotations

import logging
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
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_TRAIN_OUTPUT_DIR = "./ftrain_output"
_DEFAULT_MERGE_OUTPUT_DIR = "./merged_model"

_MIN_VALIDATION_RATIO = 0.0
_MAX_VALIDATION_RATIO = 1.0


def _validate_model_name(model: Optional[str], argument_name: str) -> str:
    """
    Validate a model identifier/path before passing it deeper into FTRAIN.

    Model names may be Hugging Face IDs, local paths, or other identifiers,
    so this helper intentionally does not try to validate filesystem or
    repository existence here.
    """
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
    """Validate and normalize a dataset validation split ratio."""
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


def _split_data(
    data: Any,
    validation_ratio: float = 0.10,
) -> Tuple[Any, Any]:
    """
    Split loaded data into training and validation portions.

    The function intentionally operates on the object returned by
    ``load_data`` rather than assuming it is a Python list. This keeps it
    compatible with sequence-like datasets and common dataset abstractions.

    For a non-empty dataset:
      - training data always receives at least one sample
      - validation data is ``None`` when a validation split is impossible
      - validation ratio ``0`` disables validation

    Returns:
        (train_data, validation_data)
    """
    validation_ratio = _validate_validation_ratio(validation_ratio)

    try:
        dataset_size = len(data)
    except TypeError as exc:
        raise TypeError(
            "The object returned by 'load_data()' must be sized "
            "(it must implement __len__)."
        ) from exc

    if dataset_size < 0:
        raise ValueError(
            f"Loaded dataset reported an invalid length: {dataset_size}."
        )

    if dataset_size == 0:
        raise ValueError(
            "The loaded dataset is empty. FTRAIN cannot start training "
            "without at least one training sample."
        )

    if dataset_size == 1 or validation_ratio <= 0.0:
        return data, None

    # Calculate validation count while guaranteeing at least one training
    # sample. This fixes the original behavior where small datasets could
    # accidentally produce an empty training split.
    validation_size = int(round(dataset_size * validation_ratio))
    validation_size = max(1, validation_size)
    validation_size = min(validation_size, dataset_size - 1)

    split_index = dataset_size - validation_size

    try:
        train_data = data[:split_index]
        validation_data = data[split_index:]
    except (TypeError, IndexError) as exc:
        raise TypeError(
            "The loaded dataset does not support slicing. "
            "FTRAIN currently requires a sliceable dataset for its "
            "automatic train/validation split."
        ) from exc

    if len(train_data) == 0:
        raise RuntimeError(
            "Internal dataset splitting error: training split is empty."
        )

    if len(validation_data) == 0:
        LOGGER.warning(
            "Validation splitting produced an empty validation set; "
            "continuing without validation."
        )
        validation_data = None

    return train_data, validation_data


def _build_train_config(
    *,
    model: str,
    captain: Optional[str],
    steps: int,
    answer: str,
    output_dir: str,
    extra_kwargs: Mapping[str, Any],
) -> TrainConfig:
    """
    Construct TrainConfig while preventing duplicate keyword collisions.

    Explicit public API arguments always take precedence over values supplied
    through **kwargs.
    """
    config_kwargs: Dict[str, Any] = dict(extra_kwargs)

    # These are explicitly controlled by the public API.
    reserved = {
        "model_name",
        "captain_model",
        "max_steps",
        "answer_mode",
        "captain_mode",
        "output_dir",
    }

    for key in reserved:
        config_kwargs.pop(key, None)

    config_kwargs.update(
        {
            "model_name": model,
            "captain_model": captain,
            "max_steps": steps,
            "answer_mode": answer,
            "captain_mode": "llm" if captain else "rule",
            "output_dir": output_dir,
        }
    )

    try:
        return TrainConfig(**config_kwargs)
    except TypeError as exc:
        raise TypeError(
            "Failed to construct TrainConfig. "
            "Check the supplied training options and make sure every "
            "keyword is supported by your installed TrainConfig."
        ) from exc


def _build_merge_config(
    *,
    model_a: str,
    model_b: str,
    captain: Optional[str],
    output_dir: str,
    extra_kwargs: Mapping[str, Any],
) -> MergeConfig:
    """
    Construct MergeConfig while preventing duplicate keyword collisions.
    """
    config_kwargs: Dict[str, Any] = dict(extra_kwargs)

    reserved = {
        "model_a",
        "model_b",
        "captain_model",
        "output_dir",
    }

    for key in reserved:
        config_kwargs.pop(key, None)

    config_kwargs.update(
        {
            "model_a": model_a,
            "model_b": model_b,
            "captain_model": captain,
            "output_dir": output_dir,
        }
    )

    try:
        return MergeConfig(**config_kwargs)
    except TypeError as exc:
        raise TypeError(
            "Failed to construct MergeConfig. "
            "Check the supplied merge options and make sure every keyword "
            "is supported by your installed MergeConfig."
        ) from exc


# ---------------------------------------------------------------------------
# Training API
# ---------------------------------------------------------------------------


class train:
    """
    High-level FTRAIN training interface.

    Example:
        train.fire(
            Model="Qwen/Qwen2.5-0.5B-Instruct",
            Data="dataset.json",
            Steps=500,
        )
    """

    @staticmethod
    def fire(
        Model: str,
        Data: Any,
        Steps: int = 100,
        Captain: Optional[str] = None,
        Answer: str = "auto_yes",
        **kwargs: Any,
    ) -> Any:
        """
        Run the complete FTRAIN training pipeline.

        Parameters:
            Model:
                Base model identifier or local model path.

            Data:
                Dataset accepted by ``load_data``.

            Steps:
                Maximum number of training steps.

            Captain:
                Optional captain/reviewer LLM model.

            Answer:
                Answer/reward mode.

            output_dir:
                Output directory. Defaults to ``./ftrain_output``.

            validation_ratio:
                Fraction of the dataset reserved for validation.
                Defaults to ``0.10``.

        Returns:
            Whatever ``Ftrain(...).train()`` returns.

        Raises:
            ValueError:
                Invalid required arguments or empty dataset.

            TypeError:
                Invalid argument types or unsupported configuration options.
        """
        model = _validate_model_name(Model, "Model")
        steps = _validate_steps(Steps)

        if not isinstance(Answer, str):
            raise TypeError(
                f"'Answer' must be a string, got {type(Answer).__name__}."
            )

        answer = Answer.strip()

        if not answer:
            raise ValueError("'Answer' cannot be empty.")

        if Captain is not None:
            captain = _validate_model_name(Captain, "Captain")
        else:
            captain = None

        runtime_kwargs: Dict[str, Any] = dict(kwargs)

        output_dir = runtime_kwargs.pop(
            "output_dir",
            _DEFAULT_TRAIN_OUTPUT_DIR,
        )

        if output_dir is None:
            output_dir = _DEFAULT_TRAIN_OUTPUT_DIR

        if not isinstance(output_dir, str):
            raise TypeError(
                "'output_dir' must be a string, "
                f"got {type(output_dir).__name__}."
            )

        output_dir = output_dir.strip()

        if not output_dir:
            raise ValueError("'output_dir' cannot be empty.")

        # Support an explicit validation ratio without leaking this
        # orchestration-level setting into TrainConfig unless that config
        # explicitly expects it.
        validation_ratio = runtime_kwargs.pop(
            "validation_ratio",
            runtime_kwargs.pop("val_split", 0.10),
        )

        LOGGER.info(
            "Starting FTRAIN training: model=%s, steps=%d, captain=%s, "
            "output_dir=%s",
            model,
            steps,
            captain or "disabled",
            output_dir,
        )

        # Load data once. Any expensive parsing/tokenization handled by
        # load_data therefore remains centralized.
        data = load_data(Data)

        train_data, val_data = _split_data(
            data,
            validation_ratio=validation_ratio,
        )

        LOGGER.info(
            "Dataset prepared: total=%d, train=%d, validation=%s",
            len(data),
            len(train_data),
            len(val_data) if val_data is not None else "disabled",
        )

        config = _build_train_config(
            model=model,
            captain=captain,
            steps=steps,
            answer=answer,
            output_dir=output_dir,
            extra_kwargs=runtime_kwargs,
        )

        engine = Ftrain(
            config,
            train_data,
            val_data,
        )

        result = engine.train()

        LOGGER.info(
            "FTRAIN training pipeline completed successfully for model=%s",
            model,
        )

        return result


# ---------------------------------------------------------------------------
# Model merge API
# ---------------------------------------------------------------------------


class merge:
    """
    High-level FTRAIN model merging interface.

    Example:
        merge.fire(
            First="model_a",
            Second="model_b",
            output_dir="./merged",
        )
    """

    @staticmethod
    def fire(
        First: Optional[str] = None,
        Second: Optional[str] = None,
        Captain: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the FTRAIN intelligent model-merging pipeline.

        Both ``First`` and ``Second`` are required.

        Backward-compatible aliases are also accepted:
            Model_a
            Model_b

        Explicit ``First``/``Second`` values take precedence over aliases.
        """
        runtime_kwargs: Dict[str, Any] = dict(kwargs)

        # Preserve compatibility with the previous API.
        model_a = First
        model_b = Second

        if model_a is None:
            model_a = runtime_kwargs.pop("Model_a", None)
        else:
            # Do not allow a stale alias to create confusing behavior.
            runtime_kwargs.pop("Model_a", None)

        if model_b is None:
            model_b = runtime_kwargs.pop("Model_b", None)
        else:
            runtime_kwargs.pop("Model_b", None)

        model_a = _validate_model_name(model_a, "First")
        model_b = _validate_model_name(model_b, "Second")

        if Captain is not None:
            captain = _validate_model_name(Captain, "Captain")
        else:
            captain = None

        # Avoid accidentally merging a model with itself unless the caller
        # explicitly disables the check via allow_self_merge=True.
        allow_self_merge = bool(
            runtime_kwargs.pop("allow_self_merge", False)
        )

        if model_a == model_b and not allow_self_merge:
            raise ValueError(
                "The two merge inputs resolve to the same model. "
                "Pass allow_self_merge=True only when an intentional "
                "self-merge is required."
            )

        output_dir = runtime_kwargs.pop(
            "output_dir",
            _DEFAULT_MERGE_OUTPUT_DIR,
        )

        if output_dir is None:
            output_dir = _DEFAULT_MERGE_OUTPUT_DIR

        if not isinstance(output_dir, str):
            raise TypeError(
                "'output_dir' must be a string, "
                f"got {type(output_dir).__name__}."
            )

        output_dir = output_dir.strip()

        if not output_dir:
            raise ValueError("'output_dir' cannot be empty.")

        LOGGER.info(
            "Starting FTRAIN merge: model_a=%s, model_b=%s, captain=%s, "
            "output_dir=%s",
            model_a,
            model_b,
            captain or "disabled",
            output_dir,
        )

        config = _build_merge_config(
            model_a=model_a,
            model_b=model_b,
            captain=captain,
            output_dir=output_dir,
            extra_kwargs=runtime_kwargs,
        )

        merger = Merger(config)
        result = merger.merge()

        LOGGER.info(
            "FTRAIN model merge completed successfully: output_dir=%s",
            output_dir,
        )

        return result


# ---------------------------------------------------------------------------
# Diagnostics / smoke test
# ---------------------------------------------------------------------------


def test() -> bool:
    """
    Run a lightweight package-level health check.

    This intentionally does not load a model or allocate GPU memory.
    It verifies that the public API and reward exports are available.

    Returns:
        ``True`` when the package API is available.

    Raises:
        RuntimeError:
            If a required public component is unexpectedly unavailable.
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

    print("✅ FTRAIN API health check passed")
    print("✅ train.fire available")
    print("✅ merge.fire available")
    print("✅ reward functions available")

    return True

xml_format_reward = rewards.xml_format_reward
math_exact_reward = rewards.math_exact_reward
python_exec_reward = rewards.python_exec_reward
