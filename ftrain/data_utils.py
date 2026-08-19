"""
FTRAIN Data Loading Utilities
=============================

Centralized dataset loading for the FTRAIN training pipeline.

Supported sources
-----------------

1. Python list:

    load_data([
        {"text": "Hello"},
        {"text": "World"},
    ])

2. JSON:

    load_data("dataset.json")

3. JSONL / NDJSON:

    load_data("dataset.jsonl")

4. Hugging Face datasets:

    load_data("hf://username/dataset")

    Optional configuration:

    load_data("hf://username/dataset/config")

    Optional split:

    load_data("hf://username/dataset?split=train")

Design goals
------------

• Keep the existing ``load_data(src)`` public API.
• Never leak file handles.
• Produce useful error messages.
• Validate loaded data before returning it.
• Avoid silently accepting malformed datasets.
• Support Hugging Face Dataset objects.
• Be defensive against malformed JSON/JSONL.
• Keep dataset records unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)

__all__ = [
    "load_data",
]


# =============================================================================
# Types
# =============================================================================

DataRecord = Dict[str, Any]

DataSource = Union[
    str,
    os.PathLike,
    List[Dict[str, Any]],
]


# =============================================================================
# Constants
# =============================================================================

_SUPPORTED_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".ndjson",
}

_DEFAULT_HF_SPLIT = "train"


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_records(
    data: Any,
    *,
    source: str = "dataset",
) -> List[Dict[str, Any]]:
    """
    Validate and normalize loaded dataset records.

    FTRAIN expects a list of dictionary-like examples.

    The original records are copied into normal Python dictionaries so
    downstream code can safely work with them.
    """
    if data is None:
        raise ValueError(
            f"{source} returned no data."
        )

    # Hugging Face Dataset and similar objects generally support iteration,
    # but we intentionally normalize them into a list here because FTRAIN's
    # public API promises List[Dict[str, Any]].
    try:
        records = list(data)
    except TypeError as exc:
        raise TypeError(
            f"{source} is not iterable and cannot be used as a dataset."
        ) from exc

    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(records):
        if isinstance(item, dict):
            normalized.append(
                dict(item)
            )
            continue

        if isinstance(item, Mapping):
            normalized.append(
                dict(item)
            )
            continue

        raise TypeError(
            f"Invalid dataset example at index {index} from {source}: "
            f"expected a dictionary, got {type(item).__name__}."
        )

    if not normalized:
        logger.warning(
            "FTRAIN loaded an empty dataset from %s.",
            source,
        )

    return normalized


# =============================================================================
# Local JSON
# =============================================================================


def _load_json(
    path: Path,
) -> List[Dict[str, Any]]:
    """
    Load a standard JSON dataset.

    Supported structures:

        [
            {"text": "hello"},
            {"text": "world"}
        ]

    And, for convenience:

        {
            "data": [
                {"text": "hello"}
            ]
        }

    The latter is useful with datasets exported from some tools.
    """
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(
                handle
            )

    except FileNotFoundError:
        raise

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in '{path}' at "
            f"line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"Failed to read JSON dataset '{path}': {exc}"
        ) from exc

    # Normal JSON dataset.
    if isinstance(payload, list):
        return _validate_records(
            payload,
            source=str(path),
        )

    # Common wrapped dataset format.
    if isinstance(payload, dict):
        for key in (
            "data",
            "examples",
            "records",
            "items",
        ):
            candidate = payload.get(key)

            if isinstance(candidate, list):
                logger.debug(
                    "Using JSON dataset field '%s' from %s.",
                    key,
                    path,
                )

                return _validate_records(
                    candidate,
                    source=f"{path}['{key}']",
                )

    raise ValueError(
        f"Unsupported JSON dataset structure in '{path}'. "
        "Expected a list of examples or an object containing "
        "'data', 'examples', 'records', or 'items'."
    )


# =============================================================================
# JSONL / NDJSON
# =============================================================================


def _load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    """
    Load JSON Lines / NDJSON data.

    Each non-empty line must contain one valid JSON object.
    """
    records: List[Dict[str, Any]] = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, raw_line in enumerate(
                handle,
                start=1,
            ):
                line = raw_line.strip()

                # Empty lines are harmless and commonly appear at EOF.
                if not line:
                    continue

                try:
                    item = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in '{path}' at line "
                        f"{line_number}, column {exc.colno}: "
                        f"{exc.msg}"
                    ) from exc

                if not isinstance(item, Mapping):
                    raise TypeError(
                        f"Invalid JSONL record in '{path}' at line "
                        f"{line_number}: expected an object/dictionary, "
                        f"got {type(item).__name__}."
                    )

                records.append(
                    dict(item)
                )

    except FileNotFoundError:
        raise

    except (ValueError, TypeError):
        raise

    except OSError as exc:
        raise RuntimeError(
            f"Failed to read JSONL dataset '{path}': {exc}"
        ) from exc

    if not records:
        logger.warning(
            "FTRAIN loaded an empty JSONL dataset from %s.",
            path,
        )

    return records


# =============================================================================
# Hugging Face datasets
# =============================================================================


def _parse_hf_source(
    source: str,
) -> tuple[str, Optional[str], str]:
    """
    Parse an FTRAIN ``hf://`` source.

    Examples
    --------

    hf://openai/gsm8k

        -> dataset="openai/gsm8k"
        -> config=None
        -> split="train"

    hf://openai/gsm8k/main

        -> dataset="openai/gsm8k"
        -> config="main"
        -> split="train"

    hf://openai/gsm8k?split=test

        -> dataset="openai/gsm8k"
        -> config=None
        -> split="test"
    """
    raw = source[5:].strip()

    if not raw:
        raise ValueError(
            "Invalid Hugging Face source: 'hf://' does not specify "
            "a dataset name."
        )

    parsed = urlparse(
        raw
    )

    # urlparse treats the first slash-containing part as a path.
    dataset_path = unquote(
        parsed.path
    ).strip("/")

    if not dataset_path:
        raise ValueError(
            f"Invalid Hugging Face dataset source: {source!r}."
        )

    query = parse_qs(
        parsed.query
    )

    split_values = query.get(
        "split"
    )

    split = (
        split_values[0].strip()
        if split_values
        else _DEFAULT_HF_SPLIT
    )

    if not split:
        split = _DEFAULT_HF_SPLIT

    # The normal Hugging Face dataset identifier is:
    #
    #     namespace/dataset
    #
    # Anything after the second component is interpreted as a config only
    # when explicitly provided using the conventional:
    #
    #     hf://namespace/dataset/config
    #
    parts = dataset_path.split("/")

    config: Optional[str] = None

    if len(parts) > 2:
        config = "/".join(
            parts[2:]
        )

        dataset_name = "/".join(
            parts[:2]
        )
    else:
        dataset_name = dataset_path

    return (
        dataset_name,
        config,
        split,
    )


def _load_huggingface(
    source: str,
) -> List[Dict[str, Any]]:
    """
    Load a Hugging Face dataset from an FTRAIN ``hf://`` source.
    """
    try:
        from datasets import load_dataset

    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load Hugging Face "
            "datasets. Install it with: pip install datasets"
        ) from exc

    dataset_name, config, split = _parse_hf_source(
        source
    )

    logger.info(
        "Loading Hugging Face dataset: %s "
        "(config=%s, split=%s)",
        dataset_name,
        config or "<default>",
        split,
    )

    try:
        kwargs: Dict[str, Any] = {
            "path": dataset_name,
            "split": split,
        }

        if config:
            kwargs["name"] = config

        dataset = load_dataset(
            **kwargs
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face dataset "
            f"'{dataset_name}'"
            f"{f' (config={config!r})' if config else ''}"
            f" with split '{split}': {exc}"
        ) from exc

    return _validate_records(
        dataset,
        source=source,
    )


# =============================================================================
# Public API
# =============================================================================


def load_data(
    src: DataSource,
) -> List[Dict[str, Any]]:
    """
    Load a FTRAIN dataset from a supported source.

    Parameters
    ----------
    src:
        One of:

        • ``list[dict]``
        • local ``.json`` file
        • local ``.jsonl`` / ``.ndjson`` file
        • ``hf://...`` Hugging Face dataset source
        • ``pathlib.Path`` / other PathLike object

    Returns
    -------
    list[dict]
        Dataset examples.

    Raises
    ------
    FileNotFoundError
        If a local path does not exist.

    ValueError
        If the dataset format is malformed.

    TypeError
        If dataset examples aren't dictionaries.

    RuntimeError
        If Hugging Face or filesystem loading fails.

    Examples
    --------
    ::

        data = load_data([
            {"text": "Hello"},
            {"text": "World"},
        ])

    ::

        data = load_data("train.json")

    ::

        data = load_data("train.jsonl")

    ::

        data = load_data("hf://username/dataset")

    ::

        data = load_data("hf://username/dataset/config?split=train")
    """
    # =========================================================================
    # In-memory data
    # =========================================================================

    if isinstance(
        src,
        list,
    ):
        # Validate and copy records. This protects FTRAIN from accidentally
        # mutating the user's original list later in the pipeline.
        return _validate_records(
            src,
            source="in-memory dataset",
        )

    # =========================================================================
    # Path-like sources
    # =========================================================================

    if isinstance(
        src,
        os.PathLike,
    ):
        src = os.fspath(
            src
        )

    if not isinstance(
        src,
        str,
    ):
        raise TypeError(
            "Data source must be a list of dictionaries, "
            "a string path/Hugging Face URI, or a PathLike object. "
            f"Got {type(src).__name__}."
        )

    source = src.strip()

    if not source:
        raise ValueError(
            "Data source cannot be empty."
        )

    # =========================================================================
    # Hugging Face source
    # =========================================================================

    if source.lower().startswith(
        "hf://"
    ):
        return _load_huggingface(
            source
        )

    # =========================================================================
    # Local filesystem source
    # =========================================================================

    path = Path(
        os.path.expanduser(
            source
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Data source not found: '{source}'."
        )

    if not path.is_file():
        raise ValueError(
            f"Data source is not a file: '{source}'."
        )

    extension = path.suffix.lower()

    logger.info(
        "Loading local FTRAIN dataset: %s",
        path,
    )

    if extension == ".json":
        data = _load_json(
            path
        )

    elif extension in {
        ".jsonl",
        ".ndjson",
    }:
        data = _load_jsonl(
            path
        )

    else:
        raise ValueError(
            f"Unsupported dataset file type '{extension or '<none>'}' "
            f"for '{source}'. Supported formats: "
            f"{', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )

    logger.info(
        "Loaded %d examples from %s.",
        len(data),
        path,
    )

    return data
