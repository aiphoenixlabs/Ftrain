"""
FTRAIN Dataset & Sampling Utilities
====================================

High-performance dataset preparation for the FTRAIN training engine.

Provides
--------
FtrainDataset
    Tokenizes text/chat examples and optionally packs multiple examples into
    fixed-length training sequences.

collate
    Dynamic padding collator compatible with causal-language-model training.

LengthSampler
    Length-aware sampler that groups examples of similar sequence lengths to
    reduce padding waste.

Design goals
------------
• Robust text and chat-template handling.
• Correct EOS handling.
• Correct causal-LM labels.
• Safe padding with -100 labels.
• Efficient length-aware batching.
• Deterministic shuffling.
• Distributed-training friendly sampler interface.
• Defensive validation.
• No accidental mutation of source examples.
• Useful diagnostics for malformed data.
"""

from __future__ import annotations

import logging
import random
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
)

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

__all__ = [
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
    """Safely convert a value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_messages(
    messages: Any,
) -> Optional[List[Dict[str, Any]]]:
    """
    Validate a chat-message structure.

    Returns ``None`` when the supplied value is not a usable message list.
    """
    if not isinstance(messages, (list, tuple)):
        return None

    normalized: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        if role is None:
            role = "user"

        if content is None:
            content = ""

        normalized.append(
            {
                "role": str(role),
                "content": str(content),
            }
        )

    return normalized or None


def _fallback_chat_text(
    messages: Sequence[Dict[str, Any]],
) -> str:
    """
    Build a tokenizer-independent fallback representation of chat messages.

    This is only used when the tokenizer doesn't provide a usable chat
    template.
    """
    parts: List[str] = []

    for message in messages:
        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        parts.append(
            f"{role}: {content}"
        )

    return "\n".join(parts)


# =============================================================================
# FtrainDataset
# =============================================================================


class FtrainDataset(Dataset):
    """
    Tokenized FTRAIN dataset.

    Parameters
    ----------
    data:
        Iterable containing training examples.

    tokenizer:
        Hugging Face-compatible tokenizer.

    max_length:
        Maximum number of tokens per example.

    use_packing:
        If enabled, multiple tokenized examples are concatenated into larger
        sequences up to ``max_length``.

    add_eos:
        Whether to append EOS to examples when the tokenizer exposes an
        ``eos_token_id``.

    drop_empty:
        Whether empty examples should be skipped instead of raising.

    Notes
    -----
    Tokenization is performed once during initialization. This makes
    ``__getitem__`` extremely cheap during training.
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
    ) -> None:
        if tokenizer is None:
            raise ValueError(
                "tokenizer cannot be None."
            )

        max_length = _safe_int(
            max_length
        )

        if max_length is None or max_length <= 0:
            raise ValueError(
                f"max_length must be a positive integer, got {max_length!r}."
            )

        if data is None:
            raise ValueError(
                "data cannot be None."
            )

        self.tok = tokenizer
        self.max_len = max_length
        self.use_packing = bool(
            use_packing
        )
        self.add_eos = bool(
            add_eos
        )
        self.drop_empty = bool(
            drop_empty
        )

        examples: List[Dict[str, List[int]]] = []

        skipped = 0

        for index, example in enumerate(data):
            if not isinstance(example, dict):
                logger.warning(
                    "Skipping dataset example %d: expected dict, got %s.",
                    index,
                    type(example).__name__,
                )
                skipped += 1
                continue

            try:
                encoded = self._enc(
                    example
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

            input_ids = encoded.get(
                "input_ids",
                []
            )

            if not input_ids:
                if self.drop_empty:
                    skipped += 1
                    continue

                raise ValueError(
                    f"Dataset example {index} produced zero tokens."
                )

            examples.append(
                {
                    "input_ids": input_ids,
                    "labels": list(
                        input_ids
                    ),
                }
            )

        if self.use_packing and examples:
            packed_ids = self._pack(
                [
                    example["input_ids"]
                    for example in examples
                ],
                self.max_len,
            )

            examples = [
                {
                    "input_ids": sequence,
                    "labels": list(sequence),
                }
                for sequence in packed_ids
                if sequence
            ]

        if not examples:
            raise ValueError(
                "Dataset empty after tokenization. "
                "Check your dataset format, tokenizer, and max_length."
            )

        self.examples = examples

        self.lengths: List[int] = [
            len(example["input_ids"])
            for example in examples
        ]

        self.num_tokens = sum(
            self.lengths
        )

        self.min_length = min(
            self.lengths
        )

        self.max_observed_length = max(
            self.lengths
        )

        self.avg_length = (
            self.num_tokens / len(self.lengths)
        )

        if skipped:
            logger.warning(
                "FTRAIN dataset skipped %d malformed/empty examples.",
                skipped,
            )

        logger.info(
            "FTRAIN dataset ready: %d examples, "
            "%d total tokens, avg length %.1f, "
            "range [%d, %d].",
            len(self.examples),
            self.num_tokens,
            self.avg_length,
            self.min_length,
            self.max_observed_length,
        )

    # =========================================================================
    # Encoding
    # =========================================================================

    def _enc(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, List[int]]:
        """
        Encode one training example.

        Supports:

        ``{"text": "..."}``

        and:

        ``{"messages": [...]}``
        """
        messages = _normalize_messages(
            example.get("messages")
        )

        if messages:
            text = self._messages_to_text(
                messages
            )
        else:
            text = example.get(
                "text",
                ""
            )

            if text is None:
                text = ""

            if not isinstance(text, str):
                text = str(text)

        text = text.strip()

        if not text:
            return {
                "input_ids": [],
                "labels": [],
            }

        # Avoid passing unsupported tokenizer arguments to custom tokenizers.
        encoded = self.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = encoded.get(
            "input_ids"
        )

        if input_ids is None:
            raise ValueError(
                "Tokenizer did not return 'input_ids'."
            )

        # Some tokenizers return nested lists for unusual configurations.
        if input_ids and isinstance(
            input_ids[0],
            list,
        ):
            input_ids = input_ids[0]

        input_ids = [
            int(token)
            for token in input_ids
        ]

        # ---------------------------------------------------------------------
        # EOS handling
        # ---------------------------------------------------------------------

        eos_token_id = getattr(
            self.tok,
            "eos_token_id",
            None,
        )

        if (
            self.add_eos
            and eos_token_id is not None
        ):
            eos_token_id = int(
                eos_token_id
            )

            if not input_ids:
                input_ids = [
                    eos_token_id
                ]

            elif input_ids[-1] != eos_token_id:
                # Only append EOS if doing so doesn't exceed max_length.
                if len(input_ids) < self.max_len:
                    input_ids.append(
                        eos_token_id
                    )
                else:
                    # The tokenizer already truncated to max_length. Replacing
                    # the final token preserves the sequence length while
                    # guaranteeing a terminal EOS.
                    input_ids[-1] = eos_token_id

        # Absolute final safety boundary.
        input_ids = input_ids[
            : self.max_len
        ]

        return {
            "input_ids": input_ids,
            "labels": list(
                input_ids
            ),
        }

    def _messages_to_text(
        self,
        messages: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Convert chat messages into text.

        Prefer the tokenizer's native chat template. Fall back to a simple
        role/content representation when the tokenizer does not support one.
        """
        apply_template = getattr(
            self.tok,
            "apply_chat_template",
            None,
        )

        if callable(
            apply_template
        ):
            try:
                rendered = apply_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=False,
                )

                if isinstance(
                    rendered,
                    str,
                ) and rendered.strip():
                    return rendered

            except Exception as exc:
                logger.debug(
                    "Tokenizer chat template failed; "
                    "using fallback formatting: %s",
                    exc,
                )

        return _fallback_chat_text(
            messages
        )

    # =========================================================================
    # Packing
    # =========================================================================

    @staticmethod
    def _pack(
        sequences: Sequence[Sequence[int]],
        max_length: int,
    ) -> List[List[int]]:
        """
        Concatenate multiple sequences into max-length training blocks.

        Important:
        The original implementation could create sequences larger than the
        requested maximum if an individual sequence itself was oversized.
        This implementation guarantees:

            len(block) <= max_length

        for every packed block.
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

            # A sequence should normally already be <= max_length because
            # tokenization truncates it. This extra protection makes _pack()
            # independently safe.
            start = 0

            while start < len(sequence):
                remaining = max_length - len(buffer)

                if remaining <= 0:
                    packed.append(
                        buffer
                    )
                    buffer = []
                    remaining = max_length

                take = min(
                    remaining,
                    len(sequence) - start,
                )

                buffer.extend(
                    sequence[start : start + take]
                )

                start += take

                if len(buffer) == max_length:
                    packed.append(
                        buffer
                    )
                    buffer = []

        if buffer:
            packed.append(
                buffer
            )

        return packed

    # =========================================================================
    # Dataset API
    # =========================================================================

    def __len__(
        self,
    ) -> int:
        return len(
            self.examples
        )

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

        attention_mask = torch.ones_like(
            input_ids,
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

    def get_stats(
        self,
    ) -> Dict[str, Any]:
        """Return useful dataset statistics."""
        return {
            "examples": len(self.examples),
            "total_tokens": self.num_tokens,
            "average_length": self.avg_length,
            "min_length": self.min_length,
            "max_length": self.max_observed_length,
            "max_configured_length": self.max_len,
            "packing": self.use_packing,
        }


# =============================================================================
# Collator
# =============================================================================


def collate(
    batch: Sequence[Dict[str, torch.Tensor]],
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
) -> Dict[str, torch.Tensor]:
    """
    Dynamically pad a batch.

    Padding behavior
    ----------------

    input_ids:
        padded with tokenizer PAD token.

    attention_mask:
        padded with 0.

    labels:
        padded with -100 so PyTorch/Hugging Face loss functions ignore padding.

    The function also sorts nothing and therefore preserves the order supplied
    by the sampler.
    """
    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    pad_token_id = _safe_int(
        pad_token_id
    )

    if pad_token_id is None or pad_token_id < 0:
        raise ValueError(
            f"pad_token_id must be a non-negative integer, "
            f"got {pad_token_id!r}."
        )

    required = (
        "input_ids",
        "attention_mask",
        "labels",
    )

    for index, item in enumerate(batch):
        if not isinstance(item, dict):
            raise TypeError(
                f"Batch item {index} must be a dictionary."
            )

        missing = [
            key
            for key in required
            if key not in item
        ]

        if missing:
            raise KeyError(
                f"Batch item {index} is missing fields: {missing}."
            )

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [
                item["input_ids"]
                for item in batch
            ],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [
                item["attention_mask"]
                for item in batch
            ],
            batch_first=True,
            padding_value=0,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [
                item["labels"]
                for item in batch
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

    Examples with similar token lengths are grouped together before batches
    are formed. This reduces padding and therefore improves GPU utilization.

    Parameters
    ----------
    lengths:
        Token length for every dataset example.

    bs:
        Training batch size.

    shuffle:
        Whether to shuffle examples and batches.

    seed:
        Base random seed.

    mega_batch_multiplier:
        Number of batches contained approximately inside each sorting window.

    drop_last:
        Whether to discard an incomplete final batch.

    Notes
    -----
    Call ``set_epoch(epoch)`` at the beginning of every epoch when deterministic
    but changing shuffling is desired.
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

        self.bs = _safe_int(
            bs
        )

        if self.bs is None or self.bs <= 0:
            raise ValueError(
                f"bs must be a positive integer, got {bs!r}."
            )

        self.shuffle = bool(
            shuffle
        )

        self.seed = int(
            seed
        )

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

        self.drop_last = bool(
            drop_last
        )

        self.epoch = 0

    # =========================================================================
    # Epoch control
    # =========================================================================

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Set the current epoch.

        Compatible with the convention used by distributed samplers.
        """
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
        count = len(
            self.lengths
        )

        if count == 0:
            return iter(())

        indices = list(
            range(count)
        )

        rng = random.Random(
            self.seed + self.epoch
        )

        if self.shuffle:
            rng.shuffle(
                indices
            )

        # Sort locally inside mega-batches rather than globally. This keeps
        # enough randomness while still reducing padding dramatically.
        mega_batch_size = max(
            self.bs,
            self.bs
            * self.mega_batch_multiplier,
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

                batches.append(
                    batch
                )

        # Shuffle complete batches after length sorting. This gives us
        # length-efficient batches without forcing the entire epoch to be
        # ordered by sequence length.
        if self.shuffle:
            rng.shuffle(
                batches
            )

        for batch in batches:
            yield from batch

    def __len__(
        self,
    ) -> int:
        count = len(
            self.lengths
        )

        if self.drop_last:
            return (
                count // self.bs
            ) * self.bs

        return count

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_batch_count(
        self,
    ) -> int:
        """Return the number of batches generated by this sampler."""
        count = len(
            self.lengths
        )

        if self.drop_last:
            return count // self.bs

        return (
            count + self.bs - 1
        ) // self.bs
