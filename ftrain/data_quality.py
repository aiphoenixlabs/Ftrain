"""
FTRAIN Data Quality
=====================

High-performance and defensive utilities for:

    • Perplexity calculation
    • Perplexity-based dataset filtering
    • Exact dataset deduplication
    • Multi-source dataset balancing

Design goals
------------

• Never mutate the caller's dataset.
• Preserve original examples exactly.
• Work with Hugging Face datasets and normal Python sequences where possible.
• Handle malformed examples defensively.
• Avoid unnecessary model/device state changes.
• Prevent numerical overflow in perplexity calculations.
• Support deterministic deduplication.
• Provide multiple balancing strategies.
• Avoid pathological dataset explosion.
• Keep compatibility with FTRAIN's existing callers.

Expected example formats
-------------------------

Plain text:

    {"text": "Hello world"}

Chat/conversation:

    {"messages": [...]}

Generic examples are also supported and are converted to a deterministic
string representation when necessary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "compute_perplexity",
    "filter_by_perplexity",
    "deduplicate",
    "balance_datasets",
]


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_MAX_LENGTH = 512
_DEFAULT_KEEP_PCT = 0.80

# exp(88) is already extremely large for a practical perplexity value.
# Keeping the exponent below ~88 avoids floating-point overflow.
_MAX_EXPONENT = 88.0

# Prevent accidental pathological memory growth from an invalid balancing
# configuration.
_MAX_BALANCE_MULTIPLIER = 100_000

# Stable fallback used when an example contains neither text nor messages.
_EMPTY_TEXT = ""


# =============================================================================
# General helpers
# =============================================================================


def _validate_max_length(
    max_length: Any,
) -> int:
    """Validate and normalize tokenizer maximum sequence length."""
    if isinstance(max_length, bool):
        raise TypeError(
            "max_length must be an integer, not bool."
        )

    try:
        value = int(max_length)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"max_length must be an integer, got {max_length!r}."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"max_length must be greater than zero, got {value}."
        )

    return value


def _validate_keep_pct(
    keep_pct: Any,
) -> float:
    """
    Validate perplexity filtering percentage.

    ``keep_pct`` represents the quantile to use as the threshold.

    Examples
    --------
    0.8 -> keep approximately the lowest 80% by perplexity.
    1.0 -> keep everything up to the maximum.
    0.0 -> keep only the minimum-scoring examples.
    """
    try:
        value = float(keep_pct)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"keep_pct must be a number, got {keep_pct!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"keep_pct must be finite, got {keep_pct!r}."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"keep_pct must be between 0.0 and 1.0, got {value}."
        )

    return value


def _to_serializable(
    value: Any,
) -> Any:
    """
    Convert common Python / NumPy / tensor values into deterministic
    JSON-compatible representations.

    This is primarily used for deduplication.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            if math.isnan(value):
                return "__nan__"

            if math.isinf(value):
                return "__inf__" if value > 0 else "__-inf__"

        return value

    if isinstance(value, bytes):
        return {
            "__bytes__": value.hex(),
        }

    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    if isinstance(value, np.generic):
        return _to_serializable(
            value.item()
        )

    if torch.is_tensor(value):
        # Avoid moving arbitrary tensors to CPU unless absolutely necessary.
        # Dataset examples normally won't contain tensors, but this makes the
        # deduplicator deterministic if they do.
        tensor = value.detach().cpu()

        return {
            "__tensor__": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_serializable(item)
            for item in value
        ]

    if isinstance(value, set):
        return [
            _to_serializable(item)
            for item in sorted(
                value,
                key=lambda item: repr(item),
            )
        ]

    # Last-resort deterministic representation.
    return repr(value)


def _example_to_text(
    example: Any,
) -> str:
    """
    Extract a meaningful textual representation from a dataset example.

    Priority:
        1. ``text``
        2. ``messages``
        3. deterministic JSON representation
        4. string representation
    """
    if example is None:
        return _EMPTY_TEXT

    if isinstance(example, Mapping):
        text = example.get("text")

        if isinstance(text, str):
            return text

        if text is not None:
            return str(text)

        messages = example.get("messages")

        if messages is not None:
            if isinstance(messages, str):
                return messages

            try:
                return json.dumps(
                    _to_serializable(messages),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                return str(messages)

        # Generic dictionary fallback.
        try:
            return json.dumps(
                _to_serializable(example),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return str(example)

    if isinstance(example, str):
        return example

    return str(example)


def _iter_examples(
    data: Any,
) -> Iterable[Any]:
    """
    Return an iterable over dataset examples.

    Supports:
        • Python lists
        • tuples
        • Hugging Face Dataset
        • generators
        • other iterable datasets
    """
    if data is None:
        return ()

    return data


def _safe_len(
    data: Any,
) -> int:
    """Return len(data) where possible, otherwise zero."""
    try:
        return len(data)
    except (TypeError, AttributeError):
        return 0


# =============================================================================
# Perplexity
# =============================================================================


def compute_perplexity(
    text: str,
    model: Any,
    tok: Any,
    dev: Any,
    ml: int = _DEFAULT_MAX_LENGTH,
) -> float:
    """
    Compute language-model perplexity for one text sample.

    Parameters
    ----------
    text:
        Input text.

    model:
        Hugging Face-compatible causal language model.

    tok:
        Hugging Face-compatible tokenizer.

    dev:
        Torch device, for example ``"cuda"`` or ``torch.device("cuda")``.

    ml:
        Maximum tokenized sequence length.

    Returns
    -------
    float
        Perplexity. ``inf`` is returned when the sample cannot be evaluated.

    Notes
    -----
    The function:

    • does not calculate gradients;
    • handles empty input;
    • avoids exp() overflow;
    • supports tokenizers without a pad token;
    • restores the model's original training/evaluation mode.
    """
    max_length = _validate_max_length(
        ml
    )

    if text is None:
        return float("inf")

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return float("inf")

    if model is None:
        raise ValueError(
            "model cannot be None."
        )

    if tok is None:
        raise ValueError(
            "tok cannot be None."
        )

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    try:
        encoded = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    except Exception:
        logger.debug(
            "Tokenizer failed while computing perplexity.",
            exc_info=True,
        )
        return float("inf")

    if encoded is None:
        return float("inf")

    if "input_ids" not in encoded:
        logger.debug(
            "Tokenizer output has no input_ids."
        )
        return float("inf")

    input_ids = encoded["input_ids"]

    if input_ids is None:
        return float("inf")

    if not torch.is_tensor(input_ids):
        try:
            input_ids = torch.as_tensor(
                input_ids
            )
        except Exception:
            return float("inf")

    if input_ids.numel() == 0:
        return float("inf")

    # Causal LM perplexity requires at least enough tokens for the model to
    # produce a next-token prediction. A single token cannot produce a normal
    # shifted loss.
    if input_ids.shape[-1] < 2:
        return float("inf")

    # -------------------------------------------------------------------------
    # Move tensors to target device
    # -------------------------------------------------------------------------

    try:
        if hasattr(encoded, "to"):
            encoded = encoded.to(
                dev
            )
        else:
            encoded = {
                key: (
                    value.to(dev)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in encoded.items()
            }

    except Exception:
        logger.debug(
            "Failed to move tokenized input to device %r.",
            dev,
            exc_info=True,
        )
        return float("inf")

    # -------------------------------------------------------------------------
    # Model evaluation
    # -------------------------------------------------------------------------

    was_training = bool(
        getattr(
            model,
            "training",
            False,
        )
    )

    try:
        # ``eval()`` disables dropout and makes perplexity deterministic.
        model.eval()

        with torch.inference_mode():
            outputs = model(
                **encoded,
                labels=encoded["input_ids"],
            )

            loss = getattr(
                outputs,
                "loss",
                None,
            )

            if loss is None:
                return float("inf")

            loss_value = float(
                loss.detach().float().item()
            )

    except (RuntimeError, ValueError, TypeError):
        logger.debug(
            "Model failed while computing perplexity.",
            exc_info=True,
        )
        return float("inf")

    finally:
        # Restore whatever state the caller had before this function.
        if was_training:
            model.train()
        else:
            model.eval()

    # -------------------------------------------------------------------------
    # Numerical safety
    # -------------------------------------------------------------------------

    if not math.isfinite(loss_value):
        return float("inf")

    if loss_value < 0.0:
        # Cross-entropy should not be negative for a standard language model.
        # Treat impossible output as invalid rather than returning a misleading
        # perplexity.
        logger.debug(
            "Model returned negative perplexity loss: %f.",
            loss_value,
        )
        return float("inf")

    if loss_value >= _MAX_EXPONENT:
        return float("inf")

    try:
        perplexity = math.exp(
            loss_value
        )
    except OverflowError:
        return float("inf")

    if not math.isfinite(perplexity):
        return float("inf")

    return perplexity


# =============================================================================
# Perplexity filtering
# =============================================================================


def filter_by_perplexity(
    data: Any,
    model: Any,
    tok: Any,
    dev: Any,
    keep_pct: float = _DEFAULT_KEEP_PCT,
    ml: int = _DEFAULT_MAX_LENGTH,
) -> List[Any]:
    """
    Filter dataset examples using language-model perplexity.

    Lower perplexity means the model finds the example more predictable.

    Parameters
    ----------
    data:
        Iterable of dataset examples.

    model:
        Language model used to calculate perplexity.

    tok:
        Corresponding tokenizer.

    dev:
        Torch device.

    keep_pct:
        Quantile threshold between 0 and 1.

        Example:
            0.80 keeps approximately the lowest 80% of finite-scoring samples.

    ml:
        Maximum token sequence length.

    Returns
    -------
    list
        Original examples whose perplexity is at or below the threshold.

    Important
    ---------
    The returned examples are the ORIGINAL objects. This function does not
    rewrite or mutate dataset records.
    """
    if data is None:
        return []

    keep_pct = _validate_keep_pct(
        keep_pct
    )

    max_length = _validate_max_length(
        ml
    )

    # Materialize once because ``data`` can be a generator.
    examples = list(
        _iter_examples(data)
    )

    if not examples:
        return []

    logger.info(
        "FTRAIN perplexity filtering: evaluating %d examples.",
        len(examples),
    )

    scores: List[float] = []

    for index, example in enumerate(
        examples
    ):
        text = _example_to_text(
            example
        )

        score = compute_perplexity(
            text=text,
            model=model,
            tok=tok,
            dev=dev,
            ml=max_length,
        )

        scores.append(
            score
        )

        if (
            (index + 1) % 100 == 0
            or index + 1 == len(examples)
        ):
            logger.debug(
                "Perplexity filtering progress: %d/%d.",
                index + 1,
                len(examples),
            )

    # -------------------------------------------------------------------------
    # Ignore infinite scores when calculating the threshold.
    #
    # If every example is invalid, return an empty dataset instead of letting
    # NumPy produce a confusing threshold.
    # -------------------------------------------------------------------------

    finite_scores = [
        score
        for score in scores
        if math.isfinite(score)
    ]

    if not finite_scores:
        logger.warning(
            "Perplexity filtering produced no finite scores."
        )
        return []

    threshold = float(
        np.quantile(
            np.asarray(
                finite_scores,
                dtype=np.float64,
            ),
            keep_pct,
        )
    )

    # -------------------------------------------------------------------------
    # Keep examples at or below threshold.
    #
    # ``keep_pct == 0`` should still preserve the minimum-scoring examples.
    # -------------------------------------------------------------------------

    filtered = [
        example
        for example, score in zip(
            examples,
            scores,
        )
        if math.isfinite(score)
        and score <= threshold
    ]

    logger.info(
        "FTRAIN perplexity filtering complete: "
        "%d/%d examples retained (%.2f%%), threshold=%.6f.",
        len(filtered),
        len(examples),
        (
            100.0 * len(filtered) / len(examples)
            if examples
            else 0.0
        ),
        threshold,
    )

    return filtered


# =============================================================================
# Deduplication
# =============================================================================


def _stable_example_hash(
    example: Any,
) -> str:
    """
    Create a deterministic SHA-256 fingerprint for a dataset example.

    SHA-256 is used instead of MD5 because this is a correctness-oriented
    content fingerprint, not merely a tiny checksum.
    """
    normalized = _to_serializable(
        example
    )

    try:
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(
            "utf-8",
        )
    except (TypeError, ValueError):
        payload = repr(
            normalized
        ).encode(
            "utf-8",
            errors="replace",
        )

    return hashlib.sha256(
        payload
    ).hexdigest()


def deduplicate(
    data: Any,
) -> List[Any]:
    """
    Remove exact duplicate examples while preserving original order.

    Parameters
    ----------
    data:
        Iterable of examples.

    Returns
    -------
    list
        Deduplicated examples.

    Notes
    -----
    Deduplication is content-based rather than object-identity-based.

    Example:

        {"text": "hello"}
        {"text": "hello"}

    are considered duplicates even though they are separate Python objects.
    """
    if data is None:
        return []

    unique: List[Any] = []
    seen = set()

    for example in _iter_examples(data):
        fingerprint = _stable_example_hash(
            example
        )

        if fingerprint in seen:
            continue

        seen.add(
            fingerprint
        )

        unique.append(
            example
        )

    logger.info(
        "FTRAIN deduplication: %d -> %d examples.",
        _safe_len(data),
        len(unique),
    )

    return unique


# =============================================================================
# Dataset balancing
# =============================================================================


def _example_size(
    example: Any,
) -> int:
    """
    Estimate textual size of one example.

    This intentionally uses character count rather than tokenization because
    tokenizing every example just to balance datasets would be expensive.

    The value is an approximate weighting signal, not a true token count.
    """
    return len(
        _example_to_text(
            example
        )
    )


def _dataset_text_size(
    dataset: Any,
) -> int:
    """Estimate total textual size of a dataset."""
    total = 0

    for example in _iter_examples(dataset):
        total += _example_size(
            example
        )

    return total


def _repeat_dataset(
    dataset: Sequence[Any],
    multiplier: int,
) -> List[Any]:
    """
    Repeat a dataset safely.

    Returns a new list and never mutates the original sequence.
    """
    if not dataset or multiplier <= 0:
        return []

    multiplier = min(
        multiplier,
        _MAX_BALANCE_MULTIPLIER,
    )

    return list(
        dataset
    ) * multiplier


def _balance_by_examples(
    sources: List[Sequence[Any]],
) -> List[Any]:
    """
    Balance sources according to example count.

    Each source is repeated enough times to approach the largest source.
    """
    if not sources:
        return []

    max_size = max(
        (
            len(source)
            for source in sources
        ),
        default=0,
    )

    if max_size <= 0:
        return []

    balanced: List[Any] = []

    for source in sources:
        if not source:
            continue

        multiplier = max(
            1,
            math.ceil(
                max_size / len(source)
            ),
        )

        repeated = _repeat_dataset(
            source,
            multiplier,
        )

        # Avoid overshooting the target by a huge amount.
        target = min(
            len(repeated),
            max_size,
        )

        balanced.extend(
            repeated[:target]
        )

    return balanced


def _balance_by_tokens(
    sources: List[Sequence[Any]],
) -> List[Any]:
    """
    Approximate token balancing using character counts.

    Character counts are deliberately used here instead of invoking a
    tokenizer because ``balance_datasets`` does not receive a tokenizer.
    """
    if not sources:
        return []

    sizes = [
        _dataset_text_size(
            source
        )
        for source in sources
    ]

    max_size = max(
        sizes,
        default=0,
    )

    if max_size <= 0:
        return []

    balanced: List[Any] = []

    for source, size in zip(
        sources,
        sizes,
    ):
        if not source or size <= 0:
            continue

        multiplier = max(
            1,
            math.ceil(
                max_size / size
            ),
        )

        repeated = _repeat_dataset(
            source,
            multiplier,
        )

        # Don't blindly truncate based on character count because individual
        # examples can vary enormously in size. The multiplier is only a
        # balancing approximation.
        balanced.extend(
            repeated
        )

    return balanced


def _balance_equally(
    sources: List[Sequence[Any]],
) -> List[Any]:
    """
    Give each source approximately equal representation.

    Unlike the old implementation, this strategy does not accidentally make
    a large source dominate simply because it contains more examples.
    """
    non_empty = [
        source
        for source in sources
        if source
    ]

    if not non_empty:
        return []

    target = max(
        len(source)
        for source in non_empty
    )

    balanced: List[Any] = []

    for source in non_empty:
        if len(source) >= target:
            balanced.extend(
                source
            )
            continue

        multiplier = math.ceil(
            target / len(source)
        )

        repeated = _repeat_dataset(
            source,
            multiplier,
        )

        balanced.extend(
            repeated[:target]
        )

    return balanced


def balance_datasets(
    sources: Any,
    strategy: str = "tokens",
) -> List[Any]:
    """
    Balance multiple datasets.

    Parameters
    ----------
    sources:
        A sequence containing individual datasets.

    strategy:
        Supported strategies:

            ``"tokens"``
                Approximate balance based on textual size.

            ``"examples"``
                Balance based on number of examples.

            ``"samples"``
                Backward-compatible alias for ``"examples"``.

            ``"equal"``
                Give each non-empty source approximately equal example
                representation.

    Returns
    -------
    list
        Combined balanced dataset.

    Notes
    -----
    This function returns a normal Python list rather than attempting to
    preserve a particular Hugging Face Dataset class. The caller can convert
    it back to a Dataset afterward if required.
    """
    if sources is None:
        return []

    try:
        source_list = list(
            sources
        )
    except TypeError as exc:
        raise TypeError(
            "sources must be an iterable of datasets."
        ) from exc

    if not source_list:
        return []

    # Remove None sources while preserving ordering.
    normalized_sources: List[Sequence[Any]] = []

    for index, source in enumerate(
        source_list
    ):
        if source is None:
            logger.debug(
                "Ignoring empty dataset source at index %d.",
                index,
            )
            continue

        try:
            # Materialize generators so we can inspect them multiple times.
            if not isinstance(
                source,
                Sequence,
            ):
                source = list(source)

        except TypeError as exc:
            raise TypeError(
                f"Dataset source at index {index} is not iterable."
            ) from exc

        normalized_sources.append(
            source
        )

    if not normalized_sources:
        return []

    if len(normalized_sources) == 1:
        return list(
            normalized_sources[0]
        )

    normalized_strategy = str(
        strategy
    ).strip().lower()

    # Backward compatibility with the original implementation.
    if normalized_strategy == "samples":
        normalized_strategy = "examples"

    valid_strategies = {
        "tokens",
        "examples",
        "equal",
    }

    if normalized_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown balancing strategy {strategy!r}. "
            f"Expected one of: {sorted(valid_strategies)}."
        )

    if normalized_strategy == "tokens":
        result = _balance_by_tokens(
            normalized_sources
        )

    elif normalized_strategy == "examples":
        result = _balance_by_examples(
            normalized_sources
        )

    else:
        result = _balance_equally(
            normalized_sources
        )

    logger.info(
        "FTRAIN dataset balancing: %d sources -> %d examples "
        "(strategy=%s).",
        len(normalized_sources),
        len(result),
        normalized_strategy,
    )

    return result
