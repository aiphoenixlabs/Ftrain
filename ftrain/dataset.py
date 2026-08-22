"""
FTRAIN Dataset & Sampling Utilities v1.1
========================================

Robust dataset preparation, tokenization, collation, and length-aware sampling
for FTRAIN.

This module has NO import-time filesystem side effects.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

__all__ = [
    "IGNORE_INDEX",
    "DEFAULT_PAD_TOKEN_ID",
    "FtrainDataset",
    "collate",
    "LengthSampler",
]

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_SEED = 42
DEFAULT_MEGA_BATCH_MULTIPLIER = 50


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, (list, tuple)):
        return None

    result: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue

        role = str(message.get("role", "user") or "user").strip()
        content = str(message.get("content", "") or "")

        if not content.strip():
            continue

        result.append({"role": role, "content": content})

    return result or None


def _fallback_chat_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{str(message.get('role', 'user'))}: "
        f"{str(message.get('content', '') or '')}"
        for message in messages
    )


def _normalize_token_ids(value: Any) -> List[int]:
    if value is None:
        return []

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()

    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]

    try:
        return [int(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("Tokenizer returned invalid input_ids.") from exc


def _find_subsequence(
    sequence: Sequence[int],
    target: Sequence[int],
    start: int = 0,
) -> Optional[Tuple[int, int]]:
    if not target:
        return None

    n = len(sequence)
    m = len(target)
    if m > n:
        return None

    start = max(0, int(start))
    target_list = list(target)

    for i in range(start, n - m + 1):
        if list(sequence[i:i + m]) == target_list:
            return i, i + m

    return None


def _common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _tokenize_no_special(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    return _normalize_token_ids(encoded.get("input_ids", []))


class FtrainDataset(Dataset):
    """
    Tokenized FTRAIN dataset.

    Supports:
        {"text": "..."}
        {"messages": [{"role": "user", "content": "..."}, ...]}

    The per-example representation intentionally stores only input_ids and
    labels. attention_mask is generated in __getitem__, and collate() can
    reconstruct it if Transformers/Unsloth removes it before collation.
    """

    def __init__(
        self,
        data: Iterable[Dict[str, Any]],
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
        skipped = 0

        for index, source in enumerate(data):
            if not isinstance(source, Mapping):
                skipped += 1
                logger.warning(
                    "Skipping example %d: expected mapping, got %s",
                    index,
                    type(source).__name__,
                )
                continue

            try:
                encoded = self._encode_example(dict(source))
            except Exception as exc:
                if not self.drop_empty:
                    raise
                skipped += 1
                logger.warning(
                    "Skipping malformed example %d: %s",
                    index,
                    exc,
                    exc_info=True,
                )
                continue

            input_ids = encoded["input_ids"]
            labels = encoded["labels"]

            if not input_ids:
                if self.drop_empty:
                    skipped += 1
                    continue
                raise ValueError(f"Example {index} produced zero tokens.")

            if len(input_ids) != len(labels):
                raise ValueError(
                    f"Example {index} produced mismatched input/label lengths "
                    f"({len(input_ids)} vs {len(labels)})."
                )

            self.examples.append(
                {
                    "input_ids": list(input_ids),
                    "labels": list(labels),
                }
            )

        if self.use_packing and self.examples:
            self.examples = self._pack_examples(self.examples)

        if not self.examples:
            raise ValueError(
                "Dataset empty after tokenization. "
                "Check dataset format, tokenizer, and max_length."
            )

        self.lengths = [
            len(example["input_ids"]) for example in self.examples
        ]
        self.num_tokens = int(sum(self.lengths))
        self.min_length = int(min(self.lengths))
        self.max_observed_length = int(max(self.lengths))
        self.avg_length = self.num_tokens / len(self.examples)

        if skipped:
            logger.warning(
                "FTRAIN dataset skipped %d malformed/empty examples",
                skipped,
            )

        logger.info(
            "FTRAIN dataset ready: %d examples | %d tokens | avg %.1f | "
            "max %d | packing=%s | response_only=%s",
            len(self.examples),
            self.num_tokens,
            self.avg_length,
            self.max_observed_length,
            self.use_packing,
            self.train_on_response_only,
        )

    def _encode_example(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        messages = _normalize_messages(example.get("messages"))

        if messages:
            if self.train_on_response_only:
                return self._encode_response_only_chat(messages)

            return self._encode_text(self._render_messages(messages))

        text = example.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        return self._encode_text(text)

    def _encode_text(self, text: str) -> Dict[str, List[int]]:
        text = text.strip()
        if not text:
            return {"input_ids": [], "labels": []}

        encoded = self.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = self._truncate_and_eos(
            _normalize_token_ids(encoded.get("input_ids", []))
        )

        return {
            "input_ids": input_ids,
            "labels": list(input_ids),
        }

    def _render_messages(self, messages: Sequence[Mapping[str, Any]]) -> str:
        apply_template = getattr(self.tok, "apply_chat_template", None)

        if callable(apply_template):
            try:
                rendered = apply_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=self.add_generation_prompt,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception as exc:
                logger.debug(
                    "Chat template failed; using fallback: %s",
                    exc,
                )

        return _fallback_chat_text(messages)

    def _encode_response_only_chat(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Dict[str, List[int]]:
        assistant_indices = [
            i
            for i, message in enumerate(messages)
            if str(message.get("role", "")).lower()
            in {"assistant", "assistant_message"}
            and str(message.get("content", "")).strip()
        ]

        if not assistant_indices:
            return self._encode_text(self._render_messages(messages))

        apply_template = getattr(self.tok, "apply_chat_template", None)
        if not callable(apply_template):
            return self._encode_response_only_fallback(messages)

        try:
            full_text = apply_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )
            if not isinstance(full_text, str):
                return self._encode_response_only_fallback(messages)

            full_ids = self._truncate_and_eos(
                _tokenize_no_special(self.tok, full_text)
            )
            labels = [IGNORE_INDEX] * len(full_ids)

            search_cursor = 0
            supervised_tokens = 0

            for assistant_index in assistant_indices:
                prefix_messages = list(messages[:assistant_index])
                current_messages = list(messages[:assistant_index + 1])

                prefix_text = apply_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                current_text = apply_template(
                    current_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                prefix_ids = _tokenize_no_special(self.tok, prefix_text)
                current_ids = _tokenize_no_special(self.tok, current_text)
                prefix_len = _common_prefix_length(prefix_ids, current_ids)
                assistant_span = current_ids[prefix_len:]

                if not assistant_span:
                    assistant_span = _tokenize_no_special(
                        self.tok,
                        str(messages[assistant_index]["content"]),
                    )

                match = _find_subsequence(
                    full_ids,
                    assistant_span,
                    start=search_cursor,
                )

                if match is None:
                    content_ids = _tokenize_no_special(
                        self.tok,
                        str(messages[assistant_index]["content"]),
                    )
                    match = _find_subsequence(
                        full_ids,
                        content_ids,
                        start=search_cursor,
                    )

                if match is None:
                    continue

                start, end = match
                end = min(end, len(labels))

                for pos in range(start, end):
                    labels[pos] = full_ids[pos]

                supervised_tokens += max(0, end - start)
                search_cursor = max(search_cursor, end)

            if supervised_tokens == 0:
                logger.warning(
                    "Could not identify assistant spans; "
                    "falling back to full-sequence supervision."
                )
                labels = list(full_ids)

            if self.mask_thinking:
                labels = self._mask_thinking(full_ids, labels)

            return {"input_ids": full_ids, "labels": labels}

        except Exception:
            logger.debug(
                "Response-only template encoding failed; using fallback.",
                exc_info=True,
            )
            return self._encode_response_only_fallback(messages)

    def _encode_response_only_fallback(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Dict[str, List[int]]:
        text = self._render_messages(messages)
        full_ids = self._truncate_and_eos(
            _tokenize_no_special(self.tok, text)
        )
        labels = [IGNORE_INDEX] * len(full_ids)

        cursor = 0
        supervised_tokens = 0

        for message in messages:
            if str(message.get("role", "")).lower() != "assistant":
                continue

            content = str(message.get("content", "")).strip()
            if not content:
                continue

            content_ids = _tokenize_no_special(self.tok, content)
            match = _find_subsequence(
                full_ids,
                content_ids,
                start=cursor,
            )

            if match is None:
                continue

            start, end = match
            end = min(end, len(labels))

            for pos in range(start, end):
                labels[pos] = full_ids[pos]

            cursor = end
            supervised_tokens += max(0, end - start)

        if supervised_tokens == 0:
            labels = list(full_ids)

        if self.mask_thinking:
            labels = self._mask_thinking(full_ids, labels)

        return {"input_ids": full_ids, "labels": labels}

    def _mask_thinking(
        self,
        input_ids: Sequence[int],
        labels: Sequence[int],
    ) -> List[int]:
        try:
            open_ids = _tokenize_no_special(self.tok, "<think>")
            close_ids = _tokenize_no_special(self.tok, "</think>")
        except Exception:
            return list(labels)

        if not open_ids or not close_ids:
            return list(labels)

        result = list(labels)
        cursor = 0

        while cursor < len(input_ids):
            open_match = _find_subsequence(
                input_ids,
                open_ids,
                start=cursor,
            )
            if open_match is None:
                break

            open_start, open_end = open_match

            close_match = _find_subsequence(
                input_ids,
                close_ids,
                start=open_end,
            )
            if close_match is None:
                break

            _, close_end = close_match

            for pos in range(open_start, min(close_end, len(result))):
                result[pos] = IGNORE_INDEX

            cursor = close_end

        if not any(label != IGNORE_INDEX for label in result):
            return list(labels)

        return result

    def _truncate_and_eos(self, input_ids: List[int]) -> List[int]:
        input_ids = list(input_ids[:self.max_len])
        return self._ensure_eos(input_ids)

    def _ensure_eos(self, input_ids: List[int]) -> List[int]:
        eos_id = getattr(self.tok, "eos_token_id", None)

        if not self.add_eos or eos_id is None:
            return input_ids[:self.max_len]

        eos_id = int(eos_id)

        if not input_ids:
            return [eos_id]

        if input_ids[-1] == eos_id:
            return input_ids[:self.max_len]

        if len(input_ids) < self.max_len:
            input_ids.append(eos_id)
        else:
            input_ids[-1] = eos_id

        return input_ids[:self.max_len]

    def _pack_examples(
        self,
        examples: Sequence[Dict[str, List[int]]],
    ) -> List[Dict[str, List[int]]]:
        packed: List[Dict[str, List[int]]] = []
        ids_buffer: List[int] = []
        labels_buffer: List[int] = []

        def flush() -> None:
            nonlocal ids_buffer, labels_buffer
            if ids_buffer:
                packed.append(
                    {
                        "input_ids": ids_buffer,
                        "labels": labels_buffer,
                    }
                )
            ids_buffer = []
            labels_buffer = []

        for example in examples:
            ids = example["input_ids"]
            labels = example["labels"]
            cursor = 0

            while cursor < len(ids):
                remaining = self.max_len - len(ids_buffer)

                if remaining <= 0:
                    flush()
                    remaining = self.max_len

                take = min(remaining, len(ids) - cursor)
                ids_buffer.extend(ids[cursor:cursor + take])
                labels_buffer.extend(labels[cursor:cursor + take])
                cursor += take

                if len(ids_buffer) == self.max_len:
                    flush()

        flush()
        return packed

    @staticmethod
    def _pack(
        sequences: Sequence[Sequence[int]],
        max_length: int,
    ) -> List[List[int]]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        result: List[List[int]] = []
        buffer: List[int] = []

        for sequence in sequences:
            if not sequence:
                continue

            cursor = 0
            while cursor < len(sequence):
                remaining = max_length - len(buffer)

                if remaining <= 0:
                    result.append(buffer)
                    buffer = []
                    remaining = max_length

                take = min(remaining, len(sequence) - cursor)
                buffer.extend(sequence[cursor:cursor + take])
                cursor += take

                if len(buffer) == max_length:
                    result.append(buffer)
                    buffer = []

        if buffer:
            result.append(buffer)

        return result

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]

        input_ids = torch.tensor(
            example["input_ids"],
            dtype=torch.long,
        )
        labels = torch.tensor(
            example["labels"],
            dtype=torch.long,
        )

        # Normally present. The collator still reconstructs it defensively if
        # Transformers/Unsloth strips this field before collation.
        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_stats(self) -> Dict[str, Any]:
        supervised = sum(
            sum(label != IGNORE_INDEX for label in example["labels"])
            for example in self.examples
        )

        return {
            "examples": len(self.examples),
            "total_tokens": self.num_tokens,
            "supervised_tokens": supervised,
            "supervision_ratio": supervised / max(1, self.num_tokens),
            "average_length": self.avg_length,
            "min_length": self.min_length,
            "max_length": self.max_observed_length,
            "configured_max_length": self.max_len,
            "packing": self.use_packing,
            "response_only": self.train_on_response_only,
            "mask_thinking": self.mask_thinking,
        }

    def __repr__(self) -> str:
        return (
            "FtrainDataset("
            f"examples={len(self.examples)}, "
            f"tokens={self.num_tokens}, "
            f"avg_length={self.avg_length:.1f}, "
            f"max_length={self.max_len}, "
            f"packing={self.use_packing}, "
            f"response_only={self.train_on_response_only}"
            ")"
        )


def _ensure_1d_long(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        return tensor.to(dtype=torch.long)

    try:
        tensor = torch.as_tensor(value, dtype=torch.long)
    except Exception as exc:
        raise TypeError(
            f"{name} cannot be converted to a 1-D integer tensor"
        ) from exc

    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim > 1:
        tensor = tensor.reshape(-1)

    return tensor


def collate(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
) -> Dict[str, torch.Tensor]:
    """
    Dynamic-padding collator.

    CRITICAL:
    Transformers/Unsloth may remove attention_mask before this function. In
    that situation we reconstruct it from input_ids instead of raising.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    pad_token_id = _safe_int(
        pad_token_id,
        DEFAULT_PAD_TOKEN_ID,
    )
    if pad_token_id is None or pad_token_id < 0:
        raise ValueError(
            f"pad_token_id must be >= 0, got {pad_token_id!r}"
        )

    normalized: List[Dict[str, torch.Tensor]] = []

    for index, item in enumerate(batch):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Batch item {index} must be a mapping"
            )

        if "input_ids" not in item:
            raise KeyError(
                f"Batch item {index} is missing 'input_ids'"
            )

        input_ids = _ensure_1d_long(
            item["input_ids"],
            f"batch[{index}].input_ids",
        )

        if input_ids.numel() == 0:
            raise ValueError(
                f"Batch item {index} has zero input tokens"
            )

        attention_mask = item.get("attention_mask")

        # This directly fixes the error you encountered:
        # KeyError: Batch item 0 is missing fields: ['attention_mask']
        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.long,
            )
        else:
            attention_mask = _ensure_1d_long(
                attention_mask,
                f"batch[{index}].attention_mask",
            )

            if attention_mask.numel() != input_ids.numel():
                attention_mask = torch.ones_like(
                    input_ids,
                    dtype=torch.long,
                )
            else:
                attention_mask = (
                    attention_mask > 0
                ).to(torch.long)

        labels = item.get("labels")

        if labels is None:
            labels = input_ids.clone()
        else:
            labels = _ensure_1d_long(
                labels,
                f"batch[{index}].labels",
            )

            if labels.numel() != input_ids.numel():
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
            [x["input_ids"] for x in normalized],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in normalized],
            batch_first=True,
            padding_value=0,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in normalized],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ),
    }


class LengthSampler(Sampler[int]):
    """
    Length-aware deterministic sampler.

    Similar-length samples are grouped locally to reduce padding while keeping
    enough shuffle entropy for training.
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
            raise ValueError("lengths cannot be None")

        self.lengths = [max(1, int(x)) for x in lengths]

        self.bs = _safe_int(bs)
        if self.bs is None or self.bs <= 0:
            raise ValueError(
                f"bs must be positive, got {bs!r}"
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
                "mega_batch_multiplier must be positive"
            )

        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def _build_batches(self) -> List[List[int]]:
        count = len(self.lengths)
        if count == 0:
            return []

        indices = list(range(count))
        rng = random.Random(self.seed + self.epoch)

        if self.shuffle:
            rng.shuffle(indices)

        mega_size = max(
            self.bs,
            self.bs * self.mega_batch_multiplier,
        )

        batches: List[List[int]] = []

        for start in range(0, count, mega_size):
            chunk = indices[start:start + mega_size]
            chunk.sort(key=lambda i: self.lengths[i])

            for batch_start in range(0, len(chunk), self.bs):
                group = chunk[batch_start:batch_start + self.bs]

                if self.drop_last and len(group) < self.bs:
                    continue

                batches.append(group)

        if self.shuffle:
            rng.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[int]:
        for batch in self._build_batches():
            yield from batch

    def __len__(self) -> int:
        count = len(self.lengths)
        if self.drop_last:
            return (count // self.bs) * self.bs
        return count

    def get_batch_count(self) -> int:
        count = len(self.lengths)
        if self.drop_last:
            return count // self.bs
        return (count + self.bs - 1) // self.bs

    def estimated_padding_waste(self) -> float:
        batches = self._build_batches()
        if not batches:
            return 0.0

        padding = 0
        capacity = 0

        for batch in batches:
            longest = max(self.lengths[i] for i in batch)
            actual = sum(self.lengths[i] for i in batch)
            cap = longest * len(batch)
            padding += cap - actual
            capacity += cap

        return padding / capacity if capacity else 0.0


def _self_test() -> Dict[str, Any]:
    """Small local test, including the exact missing-attention-mask failure."""

    class DummyTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(
            self,
            text,
            add_special_tokens=True,
            truncation=False,
            max_length=None,
            **kwargs,
        ):
            del add_special_tokens, kwargs
            ids = [(ord(ch) % 50) + 3 for ch in str(text)]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return {"input_ids": ids}

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

    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {"text": "Normal training example."},
    ]

    dataset = FtrainDataset(
        samples,
        tokenizer,
        max_length=64,
    )

    # Normal batch.
    batch = collate(
        [dataset[0], dataset[1]],
        pad_token_id=0,
    )

    # Exact failure simulation: Transformers removes attention_mask.
    stripped = [
        {
            "input_ids": dataset[0]["input_ids"],
            "labels": dataset[0]["labels"],
        },
        {
            "input_ids": dataset[1]["input_ids"],
            "labels": dataset[1]["labels"],
        },
    ]

    recovered = collate(
        stripped,
        pad_token_id=0,
    )

    assert batch["input_ids"].ndim == 2
    assert recovered["input_ids"].shape == recovered["attention_mask"].shape
    assert recovered["input_ids"].shape == recovered["labels"].shape

    sampler = LengthSampler(
        dataset.lengths,
        2,
        seed=42,
    )
    list(iter(sampler))

    return {
        "dataset_length": len(dataset),
        "batch_shape": tuple(recovered["input_ids"].shape),
        "status": "PASS",
    }


if __name__ == "__main__":
    print(_self_test())
