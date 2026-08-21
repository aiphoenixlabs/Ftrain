from pathlib import Path

code = r'''"""
FTRAIN Dataset & Sampling Utilities v1.1
========================================

High-performance dataset preparation and sampling utilities for FTRAIN.

Core responsibilities
---------------------
• Robust text and chat-example normalization.
• Native Hugging Face chat-template support.
• Safe EOS handling.
• Correct causal-LM labels.
• Optional response-only supervision.
• Optional thinking/reasoning masking.
• Dynamic padding that survives Transformers/Unsloth column filtering.
• Automatic reconstruction of missing attention masks.
• Optional sequence packing.
• Length-aware deterministic batching.
• Distributed-friendly epoch-aware sampling.
• Defensive validation and diagnostics.
• Minimal mutation of source examples.
• Efficient tensor creation and collation.
• Better compatibility with HF Trainer / Unsloth / Accelerate.

Important compatibility behavior
--------------------------------
Transformers/Unsloth may remove dataset columns before the collator receives
them. Therefore ``collate()`` must NOT assume that ``attention_mask`` is still
present. If it is missing, it is reconstructed from ``input_ids``.

Public API
----------
FtrainDataset
collate
LengthSampler
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

__all__ = [
    "IGNORE_INDEX",
    "DEFAULT_PAD_TOKEN_ID",
    "FtrainDataset",
    "collate",
    "LengthSampler",
]


# =============================================================================
# Constants
# =============================================================================

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_SEED = 42
DEFAULT_MEGA_BATCH_MULTIPLIER = 50


# =============================================================================
# Utility helpers
# =============================================================================

def _safe_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:
    """Safely convert a value to an integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely convert a value to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not torch.isfinite(
        torch.tensor(result, dtype=torch.float64)
    ):
        return default

    return result


def _ensure_1d_long(
    value: Any,
    *,
    name: str,
) -> torch.Tensor:
    """
    Convert a sequence/tensor into a 1-D torch.long tensor.

    This deliberately rejects scalar tensors because model inputs need a
    sequence dimension at the per-example level.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()

        if tensor.ndim == 0:
            tensor = tensor.reshape(1)

        elif tensor.ndim > 1:
            tensor = tensor.reshape(-1)

        return tensor.to(dtype=torch.long)

    try:
        tensor = torch.as_tensor(
            value,
            dtype=torch.long,
        )
    except Exception as exc:
        raise TypeError(
            f"{name} could not be converted to a 1-D integer tensor."
        ) from exc

    if tensor.ndim == 0:
        tensor = tensor.reshape(1)

    elif tensor.ndim > 1:
        tensor = tensor.reshape(-1)

    return tensor


def _normalize_messages(
    messages: Any,
) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize a chat message list.

    Invalid messages are skipped. If nothing usable remains, ``None`` is
    returned.
    """
    if not isinstance(messages, (list, tuple)):
        return None

    normalized: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, Mapping):
            continue

        role = message.get("role", "user")
        content = message.get("content", "")

        if role is None:
            role = "user"

        if content is None:
            content = ""

        role = str(role)
        content = str(content)

        # Don't create useless empty messages.
        if not content.strip():
            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized or None


def _fallback_chat_text(
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Build a tokenizer-independent chat representation."""
    parts: List[str] = []

    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"{role}: {content}")

    return "\n".join(parts)


def _roles_and_contents(
    messages: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str]]:
    """Return normalized (role, content) pairs."""
    result: List[Tuple[str, str]] = []

    for message in messages:
        result.append(
            (
                str(message.get("role", "user")),
                str(message.get("content", "")),
            )
        )

    return result


def _find_subsequence(
    sequence: Sequence[int],
    subsequence: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """
    Find the first occurrence of ``subsequence`` inside ``sequence``.

    Returns ``(start, end)`` or ``None``.
    """
    if not sequence or not subsequence:
        return None

    n = len(sequence)
    m = len(subsequence)

    if m > n:
        return None

    for start in range(n - m + 1):
        if list(sequence[start : start + m]) == list(subsequence):
            return start, start + m

    return None


def _common_prefix_length(
    a: Sequence[int],
    b: Sequence[int],
) -> int:
    """Return the common prefix length of two token sequences."""
    limit = min(len(a), len(b))
    i = 0

    while i < limit and a[i] == b[i]:
        i += 1

    return i


# =============================================================================
# FtrainDataset
# =============================================================================

class FtrainDataset(Dataset):
    """
    Tokenized FTRAIN dataset.

    Parameters
    ----------
    data:
        Sequence/iterable of dictionaries containing ``text`` or ``messages``.

    tokenizer:
        Hugging Face-compatible tokenizer.

    max_length:
        Maximum token length.

    use_packing:
        Concatenate tokenized examples into blocks up to ``max_length``.

    add_eos:
        Append EOS where possible.

    drop_empty:
        Skip malformed/empty examples instead of raising.

    train_on_response_only:
        For chat examples, mask user/system portions with -100 and supervise
        assistant tokens only.

    mask_thinking:
        Attempt to mask reasoning sections between <think> and </think> for
        response-only training. This is conservative and only acts when those
        markers can be identified in the tokenized example.

    add_generation_prompt:
        Whether to ask the tokenizer chat template for a generation prompt.
        This should generally remain False for supervised training examples.

    Notes
    -----
    ``attention_mask`` is deliberately created in ``__getitem__`` rather than
    permanently stored in every example, reducing Python-side memory overhead.
    """

    def __init__(
        self,
        data: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        use_packing: bool = False,
        *,
        add_eos: bool = True,
        drop_empty: bool = True,
        train_on_response_only: bool = False,
        mask_thinking: bool = False,
        add_generation_prompt: bool = False,
    ) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None.")

        max_length = _safe_int(max_length)

        if max_length is None or max_length <= 0:
            raise ValueError(
                f"max_length must be a positive integer, got {max_length!r}."
            )

        if data is None:
            raise ValueError("data cannot be None.")

        self.tok = tokenizer
        self.max_len = max_length
        self.use_packing = bool(use_packing)
        self.add_eos = bool(add_eos)
        self.drop_empty = bool(drop_empty)
        self.train_on_response_only = bool(train_on_response_only)
        self.mask_thinking = bool(mask_thinking)
        self.add_generation_prompt = bool(add_generation_prompt)

        self.examples: List[Dict[str, List[int]]] = []
        self._source_types: List[str] = []

        skipped = 0

        for index, example in enumerate(data):
            if not isinstance(example, Mapping):
                logger.warning(
                    "Skipping dataset example %d: expected mapping, got %s.",
                    index,
                    type(example).__name__,
                )
                skipped += 1
                continue

            try:
                encoded, source_type = self._encode_example(
                    dict(example)
                )
            except Exception:
                if not self.drop_empty:
                    raise

                logger.warning(
                    "Skipping malformed dataset example %d.",
                    index,
                    exc_info=True,
                )
                skipped += 1
                continue

            input_ids = encoded.get("input_ids", [])
            labels = encoded.get("labels", [])

            if not input_ids:
                if self.drop_empty:
                    skipped += 1
                    continue

                raise ValueError(
                    f"Dataset example {index} produced zero tokens."
                )

            if len(labels) != len(input_ids):
                raise ValueError(
                    f"Example {index} produced mismatched input/label lengths: "
                    f"{len(input_ids)} vs {len(labels)}."
                )

            self.examples.append(
                {
                    "input_ids": list(input_ids),
                    "labels": list(labels),
                }
            )
            self._source_types.append(source_type)

        if self.use_packing and self.examples:
            self._apply_packing()

        if not self.examples:
            raise ValueError(
                "Dataset empty after tokenization. "
                "Check dataset format, tokenizer, and max_length."
            )

        self.lengths: List[int] = [
            len(example["input_ids"])
            for example in self.examples
        ]

        self.num_tokens = int(sum(self.lengths))
        self.min_length = int(min(self.lengths))
        self.max_observed_length = int(max(self.lengths))
        self.avg_length = (
            self.num_tokens / len(self.lengths)
        )

        if skipped:
            logger.warning(
                "FTRAIN dataset skipped %d malformed/empty examples.",
                skipped,
            )

        logger.info(
            "FTRAIN dataset ready: %d examples, %d tokens, "
            "avg length %.1f, range [%d, %d], packing=%s, "
            "response_only=%s.",
            len(self.examples),
            self.num_tokens,
            self.avg_length,
            self.min_length,
            self.max_observed_length,
            self.use_packing,
            self.train_on_response_only,
        )

    # =========================================================================
    # Encoding
    # =========================================================================

    def _encode_example(
        self,
        example: Dict[str, Any],
    ) -> Tuple[Dict[str, List[int]], str]:
        """
        Encode a single example and return its source type.

        Source type is ``chat`` or ``text``.
        """
        messages = _normalize_messages(
            example.get("messages")
        )

        if messages:
            if self.train_on_response_only:
                result = self._encode_chat_response_only(
                    messages
                )
            else:
                text = self._messages_to_text(
                    messages
                )
                result = self._encode_text(
                    text
                )

            return result, "chat"

        text = example.get("text", "")

        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        return self._encode_text(text), "text"

    def _encode_text(
        self,
        text: str,
    ) -> Dict[str, List[int]]:
        """Tokenize ordinary text."""
        text = text.strip()

        if not text:
            return {
                "input_ids": [],
                "labels": [],
            }

        encoded = self.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = encoded.get("input_ids")

        if input_ids is None:
            raise ValueError(
                "Tokenizer did not return 'input_ids'."
            )

        input_ids = self._normalize_token_ids(
            input_ids
        )

        input_ids = self._append_eos(
            input_ids
        )

        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _encode_chat_response_only(
        self,
        messages: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """
        Encode chat data while masking non-assistant tokens.

        We use incremental rendering when a chat template is available. This
        is more reliable than trying to infer assistant spans from raw decoded
        text.

        When the tokenizer template cannot be used incrementally, we fall back
        to full rendering and mark assistant content spans using a token
        subsequence search.
        """
        assistant_roles = {"assistant", "assistant_message"}

        if not any(
            msg.get("role") in assistant_roles
            for msg in messages
        ):
            logger.debug(
                "Response-only requested but no assistant message found; "
                "falling back to full supervision."
            )

            return self._encode_text(
                self._messages_to_text(messages)
            )

        apply_template = getattr(
            self.tok,
            "apply_chat_template",
            None,
        )

        if not callable(apply_template):
            return self._encode_chat_response_only_fallback(
                messages
            )

        try:
            # Full conversation tokenization.
            rendered_full = apply_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )

            if not isinstance(rendered_full, str):
                return self._encode_chat_response_only_fallback(
                    messages
                )

            encoded_full = self.tok(
                rendered_full,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_len,
            )

            full_ids = self._normalize_token_ids(
                encoded_full.get("input_ids", [])
            )

            full_ids = self._append_eos(
                full_ids
            )

            if not full_ids:
                return {
                    "input_ids": [],
                    "labels": [],
                }

            labels = [IGNORE_INDEX] * len(full_ids)

            # Build each assistant-only segment independently and locate it in
            # the final token stream. This is robust for templates where user
            # and assistant messages are serialized differently.
            search_from = 0

            for idx, message in enumerate(messages):
                role = str(message.get("role", "")).lower()

                if role not in assistant_roles:
                    continue

                partial_messages = list(
                    messages[: idx + 1]
                )

                rendered_partial = apply_template(
                    partial_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                partial_ids = self._normalize_token_ids(
                    self.tok(
                        rendered_partial,
                        add_special_tokens=False,
                    ).get("input_ids", [])
                )

                previous_messages = list(
                    messages[:idx]
                )

                if previous_messages:
                    rendered_previous = apply_template(
                        previous_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    previous_ids = self._normalize_token_ids(
                        self.tok(
                            rendered_previous,
                            add_special_tokens=False,
                        ).get("input_ids", [])
                    )
                else:
                    previous_ids = []

                prefix_len = _common_prefix_length(
                    partial_ids,
                    previous_ids,
                )

                assistant_segment = partial_ids[
                    prefix_len:
                ]

                match = _find_subsequence(
                    full_ids,
                    assistant_segment,
                )

                if match is not None:
                    start, end = match

                    start = max(
                        start,
                        search_from,
                    )

                    for pos in range(
                        start,
                        min(end, len(labels)),
                    ):
                        labels[pos] = full_ids[pos]

                    search_from = min(
                        end,
                        len(full_ids),
                    )

            # If the template's tokenization is too unusual to locate any
            # assistant segment, use a safe fallback rather than silently
            # training on zero supervised tokens.
            if not any(
                value != IGNORE_INDEX
                for value in labels
            ):
                logger.warning(
                    "Could not identify assistant token spans for response-only "
                    "training; falling back to full-sequence labels."
                )
                labels = list(full_ids)

            if self.mask_thinking:
                labels = self._mask_thinking_tokens(
                    full_ids,
                    labels,
                )

            return {
                "input_ids": full_ids,
                "labels": labels,
            }

        except Exception as exc:
            logger.debug(
                "Response-only chat encoding failed; using fallback: %s",
                exc,
            )

            return self._encode_chat_response_only_fallback(
                messages
            )

    def _encode_chat_response_only_fallback(
        self,
        messages: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """
        Safer fallback for response-only supervision.

        Each assistant message is tokenized and searched inside the full
        conversation token stream.
        """
        full_text = self._fallback_chat_text(messages)

        full_encoded = self.tok(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )

        full_ids = self._normalize_token_ids(
            full_encoded.get("input_ids", [])
        )

        full_ids = self._append_eos(
            full_ids
        )

        labels = [IGNORE_INDEX] * len(full_ids)

        cursor = 0

        for message in messages:
            if str(message.get("role", "")).lower() != "assistant":
                continue

            content = str(
                message.get("content", "")
            ).strip()

            if not content:
                continue

            assistant_ids = self._normalize_token_ids(
                self.tok(
                    content,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_len,
                ).get("input_ids", [])
            )

            if not assistant_ids:
                continue

            match = _find_subsequence(
                full_ids[cursor:],
                assistant_ids,
            )

            if match is None:
                continue

            start, end = match
            start += cursor
            end += cursor

            for pos in range(
                start,
                min(end, len(labels)),
            ):
                labels[pos] = full_ids[pos]

            cursor = end

        if not any(
            value != IGNORE_INDEX
            for value in labels
        ):
            # Prevent zero-gradient examples.
            labels = list(full_ids)

        if self.mask_thinking:
            labels = self._mask_thinking_tokens(
                full_ids,
                labels,
            )

        return {
            "input_ids": full_ids,
            "labels": labels,
        }

    def _mask_thinking_tokens(
        self,
        input_ids: Sequence[int],
        labels: Sequence[int],
    ) -> List[int]:
        """
        Mask tokens between <think> and </think>, if those markers can be
        represented by the tokenizer as identifiable token subsequences.

        This function is deliberately conservative. If marker tokenization
        cannot be established, labels are left unchanged.
        """
        think_open = getattr(
            self.tok,
            "convert_tokens_to_ids",
            None,
        )

        if not callable(think_open):
            return list(labels)

        try:
            open_ids = self._normalize_token_ids(
                self.tok(
                    "<think>",
                    add_special_tokens=False,
                ).get("input_ids", [])
            )

            close_ids = self._normalize_token_ids(
                self.tok(
                    "</think>",
                    add_special_tokens=False,
                ).get("input_ids", [])
            )
        except Exception:
            return list(labels)

        if not open_ids or not close_ids:
            return list(labels)

        result = list(labels)
        cursor = 0

        while cursor < len(input_ids):
            open_match = _find_subsequence(
                input_ids[cursor:],
                open_ids,
            )

            if open_match is None:
                break

            open_start, open_end = open_match
            open_start += cursor
            open_end += cursor

            close_match = _find_subsequence(
                input_ids[open_end:],
                close_ids,
            )

            if close_match is None:
                break

            close_start, close_end = close_match
            close_start += open_end
            close_end += open_end

            # Mask the reasoning span, including markers.
            for pos in range(
                open_start,
                min(close_end, len(result)),
            ):
                result[pos] = IGNORE_INDEX

            cursor = close_end

        return result

    def _normalize_token_ids(
        self,
        input_ids: Any,
    ) -> List[int]:
        """Normalize tokenizer output to a flat Python integer list."""
        if input_ids is None:
            return []

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().tolist()

        if (
            isinstance(input_ids, list)
            and input_ids
            and isinstance(input_ids[0], list)
        ):
            input_ids = input_ids[0]

        try:
            return [int(token) for token in input_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Tokenizer returned invalid input_ids."
            ) from exc

    def _append_eos(
        self,
        input_ids: List[int],
    ) -> List[int]:
        """Append/ensure EOS without exceeding max_length."""
        eos_token_id = getattr(
            self.tok,
            "eos_token_id",
            None,
        )

        if not self.add_eos or eos_token_id is None:
            return input_ids[: self.max_len]

        eos_token_id = int(eos_token_id)

        if not input_ids:
            return [eos_token_id]

        if input_ids[-1] == eos_token_id:
            return input_ids[: self.max_len]

        if len(input_ids) < self.max_len:
            input_ids.append(eos_token_id)
        else:
            input_ids[-1] = eos_token_id

        return input_ids[: self.max_len]

    def _messages_to_text(
        self,
        messages: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Prefer native tokenizer chat templates.

        Falls back to a deterministic role/content representation.
        """
        apply_template = getattr(
            self.tok,
            "apply_chat_template",
            None,
        )

        if callable(apply_template):
            try:
                rendered = apply_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=self.add_generation_prompt,
                )

                if (
                    isinstance(rendered, str)
                    and rendered.strip()
                ):
                    return rendered

            except Exception as exc:
                logger.debug(
                    "Tokenizer chat template failed; using fallback: %s",
                    exc,
                )

        return _fallback_chat_text(
            messages
        )

    # =========================================================================
    # Packing
    # =========================================================================

    def _apply_packing(self) -> None:
        """
        Pack examples while preserving labels.

        For normal full-sequence training, concatenation is straightforward.
        For response-only training, labels are concatenated together so masked
        regions remain masked.
        """
        packed_inputs: List[List[int]] = []
        packed_labels: List[List[int]] = []

        buffer_ids: List[int] = []
        buffer_labels: List[int] = []

        for example in self.examples:
            ids = example["input_ids"]
            labels = example["labels"]

            if not ids:
                continue

            cursor = 0

            while cursor < len(ids):
                remaining = self.max_len - len(buffer_ids)

                if remaining <= 0:
                    packed_inputs.append(buffer_ids)
                    packed_labels.append(buffer_labels)
                    buffer_ids = []
                    buffer_labels = []
                    remaining = self.max_len

                take = min(
                    remaining,
                    len(ids) - cursor,
                )

                buffer_ids.extend(
                    ids[cursor : cursor + take]
                )
                buffer_labels.extend(
                    labels[cursor : cursor + take]
                )

                cursor += take

                if len(buffer_ids) == self.max_len:
                    packed_inputs.append(buffer_ids)
                    packed_labels.append(buffer_labels)
                    buffer_ids = []
                    buffer_labels = []

        if buffer_ids:
            packed_inputs.append(buffer_ids)
            packed_labels.append(buffer_labels)

        self.examples = [
            {
                "input_ids": ids,
                "labels": labels,
            }
            for ids, labels in zip(
                packed_inputs,
                packed_labels,
            )
            if ids
        ]

        # Rebuild source types after packing.
        self._source_types = [
            "packed"
            for _ in self.examples
        ]

    @staticmethod
    def _pack(
        sequences: Sequence[Sequence[int]],
        max_length: int,
    ) -> List[List[int]]:
        """
        Backward-compatible static packing helper.

        Unlike the old implementation, this guarantees every returned block is
        <= max_length even if an input sequence is oversized.
        """
        if max_length <= 0:
            raise ValueError(
                "max_length must be positive."
            )

        packed: List[List[int]] = []
        buffer: List[int] = []

        for sequence in sequences:
            if not sequence:
                continue

            cursor = 0

            while cursor < len(sequence):
                remaining = max_length - len(buffer)

                if remaining <= 0:
                    packed.append(buffer)
                    buffer = []
                    remaining = max_length

                take = min(
                    remaining,
                    len(sequence) - cursor,
                )

                buffer.extend(
                    sequence[cursor : cursor + take]
                )

                cursor += take

                if len(buffer) == max_length:
                    packed.append(buffer)
                    buffer = []

        if buffer:
            packed.append(buffer)

        return packed

    # =========================================================================
    # Dataset API
    # =========================================================================

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        input_ids = torch.tensor(
            example["input_ids"],
            dtype=torch.long,
        )

        labels = torch.tensor(
            example["labels"],
            dtype=torch.long,
        )

        # This is intentionally regenerated cheaply. It prevents stale masks
        # and keeps the serialized examples small.
        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return useful dataset statistics."""
        supervised_tokens = 0

        for example in self.examples:
            supervised_tokens += sum(
                label != IGNORE_INDEX
                for label in example["labels"]
            )

        return {
            "examples": len(self.examples),
            "total_tokens": self.num_tokens,
            "supervised_tokens": supervised_tokens,
            "supervision_ratio": (
                supervised_tokens / max(1, self.num_tokens)
            ),
            "average_length": self.avg_length,
            "min_length": self.min_length,
            "max_length": self.max_observed_length,
            "max_configured_length": self.max_len,
            "packing": self.use_packing,
            "response_only": self.train_on_response_only,
            "mask_thinking": self.mask_thinking,
        }

    def __repr__(self) -> str:
        return (
            "FtrainDataset("
            f"examples={len(self)}, "
            f"tokens={self.num_tokens}, "
            f"avg_length={self.avg_length:.1f}, "
            f"max_length={self.max_len}, "
            f"packing={self.use_packing}, "
            f"response_only={self.train_on_response_only}"
            ")"
        )


# =============================================================================
# Collator
# =============================================================================

def collate(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
) -> Dict[str, torch.Tensor]:
    """
    FTRAIN dynamic-padding collator.

    Critical compatibility behavior
    --------------------------------
    Transformers/Unsloth may strip columns from examples before they reach the
    collator. Therefore:

        attention_mask = item.get("attention_mask")

    is used instead of requiring the key to exist.

    If ``attention_mask`` is missing, it is reconstructed from ``input_ids``.

    ``labels`` are also reconstructed from ``input_ids`` if absent. This makes
    the collator resilient to Trainer-side column filtering.

    Output
    ------
    {
        "input_ids":      [B, T],
        "attention_mask": [B, T],
        "labels":         [B, T],
    }
    """
    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    pad_token_id = _safe_int(
        pad_token_id,
        DEFAULT_PAD_TOKEN_ID,
    )

    if pad_token_id is None or pad_token_id < 0:
        raise ValueError(
            f"pad_token_id must be a non-negative integer, got {pad_token_id!r}."
        )

    normalized: List[Dict[str, torch.Tensor]] = []

    for index, item in enumerate(batch):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Batch item {index} must be a mapping, "
                f"got {type(item).__name__}."
            )

        if "input_ids" not in item:
            raise KeyError(
                f"Batch item {index} is missing required field 'input_ids'."
            )

        input_ids = _ensure_1d_long(
            item["input_ids"],
            name=f"batch[{index}]['input_ids']",
        )

        if input_ids.numel() == 0:
            raise ValueError(
                f"Batch item {index} contains zero input tokens."
            )

        # ------------------------------------------------------------
        # CRITICAL FIX:
        # Transformers/Unsloth can remove attention_mask.
        # Reconstruct it instead of crashing.
        # ------------------------------------------------------------

        attention_mask = item.get(
            "attention_mask"
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.long,
            )
        else:
            attention_mask = _ensure_1d_long(
                attention_mask,
                name=f"batch[{index}]['attention_mask']",
            )

            if attention_mask.numel() != input_ids.numel():
                logger.debug(
                    "Batch item %d attention_mask length mismatch "
                    "(%d vs %d); reconstructing mask.",
                    index,
                    attention_mask.numel(),
                    input_ids.numel(),
                )

                attention_mask = torch.ones_like(
                    input_ids,
                    dtype=torch.long,
                )
            else:
                attention_mask = (
                    attention_mask > 0
                ).to(torch.long)

        # ------------------------------------------------------------
        # Labels can also disappear through Trainer column filtering.
        # Reconstructing them gives us safe causal-LM supervision.
        # ------------------------------------------------------------

        labels = item.get("labels")

        if labels is None:
            labels = input_ids.clone()
        else:
            labels = _ensure_1d_long(
                labels,
                name=f"batch[{index}]['labels']",
            )

            if labels.numel() != input_ids.numel():
                logger.debug(
                    "Batch item %d labels length mismatch "
                    "(%d vs %d); reconstructing labels.",
                    index,
                    labels.numel(),
                    input_ids.numel(),
                )

                labels = input_ids.clone()

        normalized.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [
                item["input_ids"]
                for item in normalized
            ],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [
                item["attention_mask"]
                for item in normalized
            ],
            batch_first=True,
            padding_value=0,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [
                item["labels"]
                for item in normalized
            ],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ),
    }


# =============================================================================
# Length-aware sampler
# =============================================================================

class LengthSampler(
    Sampler[int]
):
    """
    Length-aware training sampler.

    Similar-length examples are batched together to reduce padding waste.

    Parameters
    ----------
    lengths:
        Token length for every dataset example.

    bs:
        Per-device batch size.

    shuffle:
        Whether to shuffle examples and resulting batches.

    seed:
        Base deterministic seed.

    mega_batch_multiplier:
        Approximate number of batches included in each local sorting window.

    drop_last:
        Drop incomplete final batches.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        bs: int,
        shuffle: bool = True,
        seed: int = DEFAULT_SEED,
        *,
        mega_batch_multiplier: int = DEFAULT_MEGA_BATCH_MULTIPLIER,
        drop_last: bool = False,
    ) -> None:
        if lengths is None:
            raise ValueError(
                "lengths cannot be None."
            )

        self.lengths = [
            max(
                1,
                int(length),
            )
            for length in lengths
        ]

        self.bs = _safe_int(bs)

        if self.bs is None or self.bs <= 0:
            raise ValueError(
                f"bs must be a positive integer, got {bs!r}."
            )

        self.shuffle = bool(shuffle)

        self.seed = int(seed)

        self.mega_batch_multiplier = _safe_int(
            mega_batch_multiplier
        )

        if (
            self.mega_batch_multiplier is None
            or self.mega_batch_multiplier <= 0
        ):
            raise ValueError(
                "mega_batch_multiplier must be positive."
            )

        self.drop_last = bool(drop_last)
        self.epoch = 0

    # =========================================================================
    # Epoch control
    # =========================================================================

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """Set the current epoch for deterministic reshuffling."""
        self.epoch = max(
            0,
            int(epoch),
        )

    # =========================================================================
    # Sampling
    # =========================================================================

    def __iter__(
        self,
    ) -> Iterator[int]:
        count = len(self.lengths)

        if count == 0:
            return
            yield  # pragma: no cover

        indices = list(range(count))

        rng = random.Random(
            self.seed + self.epoch
        )

        if self.shuffle:
            rng.shuffle(indices)

        mega_batch_size = max(
            self.bs,
            self.bs * self.mega_batch_multiplier,
        )

        batches: List[List[int]] = []

        for start in range(
            0,
            count,
            mega_batch_size,
        ):
            chunk = indices[
                start : start + mega_batch_size
            ]

            # Sort only the local chunk. This keeps randomness while
            # significantly reducing padding.
            chunk.sort(
                key=lambda index: self.lengths[index]
            )

            for batch_start in range(
                0,
                len(chunk),
                self.bs,
            ):
                batch = chunk[
                    batch_start : batch_start + self.bs
                ]

                if (
                    self.drop_last
                    and len(batch) < self.bs
                ):
                    continue

                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(
        self,
    ) -> int:
        count = len(self.lengths)

        if self.drop_last:
            return (
                count // self.bs
            ) * self.bs

        return count

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_batch_count(self) -> int:
        """Return the number of batches for one epoch."""
        count = len(self.lengths)

        if self.drop_last:
            return count // self.bs

        return (
            count + self.bs - 1
        ) // self.bs

    def estimated_padding_waste(
        self,
    ) -> float:
        """
        Estimate padding waste from the current length groups.

        This is an inexpensive diagnostic, not an exact runtime measurement.
        """
        if not self.lengths:
            return 0.0

        total_padding = 0
        total_tokens = 0

        indices = list(range(len(self.lengths)))

        mega_batch_size = max(
            self.bs,
            self.bs * self.mega_batch_multiplier,
        )

        for start in range(
            0,
            len(indices),
            mega_batch_size,
        ):
            chunk = indices[
                start : start + mega_batch_size
            ]

            chunk.sort(
                key=lambda index: self.lengths[index]
            )

            for batch_start in range(
                0,
                len(chunk),
                self.bs,
            ):
                batch = chunk[
                    batch_start : batch_start + self.bs
                ]

                if (
                    self.drop_last
                    and len(batch) < self.bs
                ):
                    continue

                max_len = max(
                    self.lengths[index]
                    for index in batch
                )

                actual = sum(
                    self.lengths[index]
                    for index in batch
                )

                capacity = (
                    max_len * len(batch)
                )

                total_padding += max(
                    0,
                    capacity - actual,
                )

                total_tokens += capacity

        if total_tokens <= 0:
            return 0.0

        return (
            total_padding /
            total_tokens
        )


# =============================================================================
# Self-test
# =============================================================================

def _self_test() -> Dict[str, Any]:
    """
    Lightweight internal test used by developers.

    It does not require Transformers or a real tokenizer.
    """
    class DummyTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(
            self,
            text,
            **kwargs,
        ):
            del kwargs
            ids = [
                (ord(char) % 50) + 3
                for char in text
            ]
            return {
                "input_ids": ids
            }

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ):
            del tokenize, add_generation_prompt
            return "\n".join(
                f"{m['role']}: {m['content']}"
                for m in messages
            )

    tokenizer = DummyTokenizer()

    data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                },
                {
                    "role": "assistant",
                    "content": "Hi",
                },
            ]
        },
        {
            "text": "Training example.",
        },
    ]

    dataset = FtrainDataset(
        data,
        tokenizer,
        max_length=32,
    )

    examples = [
        dataset[0],
        dataset[1],
    ]

    # Simulate Transformers removing attention_mask.
    stripped = [
        {
            "input_ids": example["input_ids"],
            "labels": example["labels"],
        }
        for example in examples
    ]

    result = collate(
        stripped,
        pad_token_id=0,
    )

    assert result["input_ids"].ndim == 2
    assert result["attention_mask"].ndim == 2
    assert result["labels"].ndim == 2
    assert result["input_ids"].shape == result["attention_mask"].shape
    assert result["input_ids"].shape == result["labels"].shape

    sampler = LengthSampler(
        dataset.lengths,
        bs=2,
    )

    list(iter(sampler))

    return {
        "dataset_length": len(dataset),
        "batch_shape": tuple(result["input_ids"].shape),
        "estimated_padding_waste": sampler.estimated_padding_waste(),
        "status": "ok",
    }


if __name__ == "__main__":
    print(_self_test())
'''
path = Path("/mnt/data/ftrain_dataset_v1_1_fixed.py")
path.write_text(code, encoding="utf-8")
compile(code, str(path), "exec")

# Run the built-in lightweight self test.
namespace = {}
exec(compile(code, str(path), "exec"), namespace)
result = namespace["_self_test"]()
print(f"Created: {path}")
print(f"Lines: {len(code.splitlines())}")
print(f"Self-test: {result}")
