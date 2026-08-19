"""
🔥 FTRAIN Phoenix Captain — Adaptive Training & Merge Intelligence
===================================================================

PhoenixCaptain is the decision-making layer used by FTRAIN.

Responsibilities
----------------
• Analyze the base model and dataset.
• Monitor training dynamics.
• Produce safe rule-based optimization advice.
• Optionally use a dedicated LLM as a higher-level training advisor.
• Combine asynchronous LLM reasoning with deterministic safety rules.
• Track training history and adaptive state.
• Analyze model-merge tensor statistics.
• Return normalized, validated recommendations.

Design goals
------------
1. Never let malformed LLM output corrupt training.
2. Never let an asynchronous worker overwrite newer decisions incorrectly.
3. Keep rule-based safety logic available even when the Captain LLM fails.
4. Avoid unnecessary GPU synchronization and repeated model-mode changes.
5. Keep all state mutations thread-safe.
6. Preserve the original public API:
       set_family_context()
       update_expert_imbalance()
       analyze_model()
       analyze_and_report_data()
       evaluate_improvement()
       analyze_data()
       inspect_training()
       inspect_merge()
       get_latest_advice()

The Captain is advisory. It should influence optimization, but it should
never become a single point of failure for the training pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = ["PhoenixCaptain"]


# ============================================================================
# Utility helpers
# ============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary numeric input into a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def _clamp(value: float, low: float, high: float) -> float:
    """Numerically safe clamp."""
    if low > high:
        low, high = high, low

    return max(low, min(high, value))


def _safe_text(value: Any, default: str = "") -> str:
    """Convert arbitrary values to clean strings."""
    if value is None:
        return default

    try:
        return str(value).strip()
    except Exception:
        return default


def _safe_len(value: Any) -> Optional[int]:
    """Return len(value), or None when the object is not sized."""
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


# ============================================================================
# Phoenix Captain
# ============================================================================


class PhoenixCaptain:
    """
    Adaptive FTRAIN training and merging advisor.

    The Captain has two layers:

        Rule Engine
            ↓
        Optional LLM Advisor

    Rules provide deterministic safety behavior. The LLM provides higher-level
    contextual reasoning when available.

    The LLM is intentionally treated as advisory rather than authoritative.
    """

    DEFAULT_MEMORY_SIZE = 20
    DEFAULT_LOSS_WINDOW = 5
    DEFAULT_REWARD_WINDOW = 10
    DEFAULT_LLM_MAX_LENGTH = 2048
    DEFAULT_LLM_MAX_NEW_TOKENS = 160

    DEFAULT_MULT_MIN = 0.25
    DEFAULT_MULT_MAX = 2.50

    DEFAULT_MIN_LLM_INTERVAL = 15.0

    VALID_LAYER_BOOSTS = frozenset(
        {
            "none",
            "all",
            "early",
            "late",
            "gate",
            "router",
            "other",
        }
    )

    def __init__(
        self,
        config: Any,
        answer_mode: Optional[str] = None,
    ) -> None:
        if config is None:
            raise ValueError("PhoenixCaptain requires a configuration object.")

        self.config = config

        # ------------------------------------------------------------------
        # Operating mode
        # ------------------------------------------------------------------

        configured_mode = _safe_text(
            getattr(config, "captain_mode", "rule")
        ).lower()

        if configured_mode not in {"rule", "llm", "disabled"}:
            logger.warning(
                "PhoenixCaptain: unknown captain_mode=%r; "
                "falling back to rule mode.",
                configured_mode,
            )
            configured_mode = "rule"

        self.mode = configured_mode
        self.async_mode = self.mode == "llm"

        # ------------------------------------------------------------------
        # Core state
        # ------------------------------------------------------------------

        self.previous_loss: Optional[float] = None
        self.previous_val_loss: Optional[float] = None

        self._last_result: Optional[Dict[str, Any]] = None
        self._last_applied: Optional[Dict[str, Any]] = None

        self._pending_llm_result: Optional[Dict[str, Any]] = None

        self._busy = False
        self._lock = threading.RLock()

        # Prevent stale async workers from publishing results from an older
        # inspection after a newer inspection has already started.
        self._generation = 0

        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Any = None

        self._last_call_ts = 0.0
        self._last_successful_llm_ts = 0.0

        # ------------------------------------------------------------------
        # Model/data context
        # ------------------------------------------------------------------

        self.family = "generic"
        self.is_moe = False
        self.expert_imbalance: Optional[float] = None

        self.model_profile: Optional[Dict[str, Any]] = None
        self.data_profile: Optional[Dict[str, Any]] = None

        self.answer_mode = (
            answer_mode
            if answer_mode is not None
            else getattr(config, "answer_mode", "auto_yes")
        )

        # ------------------------------------------------------------------
        # Adaptive memory
        # ------------------------------------------------------------------

        memory_size = max(
            1,
            int(
                getattr(
                    config,
                    "captain_memory_size",
                    self.DEFAULT_MEMORY_SIZE,
                )
            ),
        )

        loss_window = max(
            3,
            int(
                getattr(
                    config,
                    "captain_velocity_window",
                    self.DEFAULT_LOSS_WINDOW,
                )
            ),
        )

        reward_window = max(
            1,
            int(
                getattr(
                    config,
                    "captain_reward_window",
                    self.DEFAULT_REWARD_WINDOW,
                )
            ),
        )

        self.memory: deque[Dict[str, Any]] = deque(
            maxlen=memory_size
        )

        self.loss_history: deque[float] = deque(
            maxlen=loss_window
        )

        self.val_loss_history: deque[float] = deque(
            maxlen=loss_window
        )

        self.reward_history: deque[float] = deque(
            maxlen=reward_window
        )

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------

        self.inspection_count = 0
        self.llm_request_count = 0
        self.llm_success_count = 0
        self.llm_failure_count = 0

        self._last_error: Optional[str] = None

        # Model generation is guarded independently from state mutation.
        # This prevents concurrent calls to the same Captain model.
        self._generation_lock = threading.Lock()

        # ------------------------------------------------------------------
        # LLM initialization
        # ------------------------------------------------------------------

        if self.mode == "llm":
            captain_model = getattr(config, "captain_model", None)

            if captain_model:
                self._load_llm(captain_model)
            else:
                logger.warning(
                    "PhoenixCaptain: captain_mode='llm' but no "
                    "captain_model was configured. Falling back to rules."
                )
                self.mode = "rule"
                self.async_mode = False

    # ========================================================================
    # Configuration
    # ========================================================================

    def _multiplier_bounds(self) -> Tuple[float, float]:
        """
        Return safe Captain learning-rate multiplier bounds.

        Supports both:
            config.captain_clamp = (low, high)

        and:
            config.captain_mult_min
            config.captain_mult_max
        """
        clamp = getattr(self.config, "captain_clamp", None)

        if isinstance(clamp, Sequence) and not isinstance(clamp, str):
            if len(clamp) >= 2:
                low = _safe_float(clamp[0], self.DEFAULT_MULT_MIN)
                high = _safe_float(clamp[1], self.DEFAULT_MULT_MAX)
            else:
                low = self.DEFAULT_MULT_MIN
                high = self.DEFAULT_MULT_MAX
        else:
            low = _safe_float(
                getattr(
                    self.config,
                    "captain_mult_min",
                    self.DEFAULT_MULT_MIN,
                ),
                self.DEFAULT_MULT_MIN,
            )

            high = _safe_float(
                getattr(
                    self.config,
                    "captain_mult_max",
                    self.DEFAULT_MULT_MAX,
                ),
                self.DEFAULT_MULT_MAX,
            )

        if low <= 0.0:
            low = self.DEFAULT_MULT_MIN

        if high < low:
            high = max(self.DEFAULT_MULT_MAX, low)

        return low, high

    def _llm_interval(self) -> float:
        """Return the minimum interval between LLM advisor calls."""
        value = _safe_float(
            getattr(
                self.config,
                "captain_min_interval",
                self.DEFAULT_MIN_LLM_INTERVAL,
            ),
            self.DEFAULT_MIN_LLM_INTERVAL,
        )

        return max(0.0, value)

    # ========================================================================
    # LLM loading
    # ========================================================================

    def _load_llm(self, model_name: str) -> None:
        """
        Load the dedicated Captain LLM.

        Unsloth is preferred because this project already uses it, but failure
        is isolated and automatically falls back to rule mode.
        """
        try:
            from unsloth import FastLanguageModel

            max_seq_length = int(
                getattr(
                    self.config,
                    "captain_max_seq_length",
                    self.DEFAULT_LLM_MAX_LENGTH,
                )
            )

            max_seq_length = max(256, max_seq_length)

            load_in_4bit = bool(
                getattr(
                    self.config,
                    "captain_load_in_4bit",
                    True,
                )
            )

            # Let Unsloth select an appropriate dtype when possible.
            dtype = getattr(
                self.config,
                "captain_dtype",
                None,
            )

            self.model, self.tokenizer = (
                FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    dtype=dtype,
                )
            )

            if self.tokenizer is not None:
                if (
                    getattr(self.tokenizer, "pad_token_id", None)
                    is None
                ):
                    eos_token = getattr(
                        self.tokenizer,
                        "eos_token",
                        None,
                    )

                    if eos_token is not None:
                        self.tokenizer.pad_token = eos_token

                # Prefer left padding for generation-heavy causal LM usage.
                try:
                    self.tokenizer.padding_side = "left"
                except Exception:
                    pass

            logger.info(
                "🧠 Phoenix Captain LLM loaded successfully via Unsloth: %s",
                model_name,
            )

        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.mode = "rule"
            self.async_mode = False

            self._last_error = (
                f"Captain LLM loading failed: {exc}"
            )

            logger.exception(
                "⚠️ Phoenix Captain LLM failed to load. "
                "Falling back to rule-based mode."
            )

    # ========================================================================
    # Context
    # ========================================================================

    def set_family_context(
        self,
        family: str,
        is_moe: bool,
    ) -> None:
        """
        Store model-family and MoE context.

        The original implementation accepted ``family`` but did not actually
        persist it. That meant downstream reasoning could not use the family
        information supplied by the caller.
        """
        family_text = _safe_text(family, "generic").lower()

        if not family_text:
            family_text = "generic"

        with self._lock:
            self.family = family_text
            self.is_moe = bool(is_moe)

    def update_expert_imbalance(
        self,
        imb: Optional[float],
    ) -> None:
        """Update the latest MoE expert imbalance metric."""
        if imb is None:
            with self._lock:
                self.expert_imbalance = None
            return

        value = _safe_float(imb, 0.0)

        # Expert imbalance is conceptually a normalized signal.
        value = _clamp(value, 0.0, 1.0)

        with self._lock:
            self.expert_imbalance = value

    # ========================================================================
    # Model analysis
    # ========================================================================

    def analyze_model(
        self,
        model: Any,
    ) -> Dict[str, Any]:
        """
        Analyze the supplied model and cache a compact model profile.
        """
        profile: Dict[str, Any] = {
            "model_type": "unknown",
            "num_layers": 0,
            "hidden_size": 0,
            "family": self.family,
            "is_moe": self.is_moe,
        }

        if model is None:
            self.model_profile = profile
            return profile

        try:
            config = getattr(model, "config", None)

            if config is not None:
                profile["model_type"] = _safe_text(
                    getattr(
                        config,
                        "model_type",
                        "unknown",
                    ),
                    "unknown",
                )

                profile["architectures"] = list(
                    getattr(
                        config,
                        "architectures",
                        [],
                    )
                    or []
                )

                profile["num_layers"] = int(
                    getattr(
                        config,
                        "num_hidden_layers",
                        getattr(
                            config,
                            "num_layers",
                            0,
                        ),
                    )
                    or 0
                )

                profile["hidden_size"] = int(
                    getattr(
                        config,
                        "hidden_size",
                        0,
                    )
                    or 0
                )

                profile["num_attention_heads"] = int(
                    getattr(
                        config,
                        "num_attention_heads",
                        0,
                    )
                    or 0
                )

            try:
                from .model_utils import count_params

                profile["param_stats"] = count_params(model)
            except Exception as exc:
                logger.debug(
                    "PhoenixCaptain: count_params() failed.",
                    exc_info=True,
                )
                profile["param_stats"] = {}

            try:
                parameter_count = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )

                profile["parameter_count"] = int(
                    parameter_count
                )

                profile["trainable_parameter_count"] = int(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                )
            except Exception:
                pass

        except Exception as exc:
            logger.warning(
                "PhoenixCaptain: model analysis partially failed: %s",
                exc,
            )

        with self._lock:
            self.model_profile = profile

        return profile

    # ========================================================================
    # Data analysis
    # ========================================================================

    def analyze_data(
        self,
        dataset: Any,
        tokenizer: Any,
        max_samples: int = 200,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze sequence-length statistics without modifying the dataset.

        Supports:
            input_ids
            text
            content
        """
        if dataset is None:
            self.data_profile = None
            return None

        if tokenizer is None:
            logger.warning(
                "PhoenixCaptain: tokenizer is unavailable; "
                "cannot analyze raw-text dataset entries."
            )
            return None

        dataset_len = _safe_len(dataset)

        if dataset_len is None or dataset_len <= 0:
            self.data_profile = None
            return None

        sample_limit = max(
            1,
            int(max_samples),
        )

        count = min(
            dataset_len,
            sample_limit,
        )

        lengths: List[int] = []
        empty_samples = 0
        failed_samples = 0

        for index in range(count):
            try:
                item = dataset[index]

                if not isinstance(item, Mapping):
                    failed_samples += 1
                    continue

                ids = item.get("input_ids")

                if ids is None:
                    text = item.get("text")

                    if not isinstance(text, str):
                        text = item.get("content", "")

                    if not isinstance(text, str):
                        text = ""

                    if not text.strip():
                        empty_samples += 1
                        continue

                    ids = tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )

                try:
                    length = len(ids)
                except TypeError:
                    failed_samples += 1
                    continue

                if length <= 0:
                    empty_samples += 1
                    continue

                lengths.append(int(length))

            except Exception:
                failed_samples += 1
                logger.debug(
                    "PhoenixCaptain: dataset sample analysis failed at "
                    "index=%d.",
                    index,
                    exc_info=True,
                )

        if not lengths:
            self.data_profile = {
                "sampled": count,
                "valid_samples": 0,
                "empty_samples": empty_samples,
                "failed_samples": failed_samples,
            }
            return self.data_profile

        array = np.asarray(
            lengths,
            dtype=np.float64,
        )

        profile = {
            "sampled": count,
            "valid_samples": len(lengths),
            "empty_samples": empty_samples,
            "failed_samples": failed_samples,
            "avg_length": float(np.mean(array)),
            "median_length": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max_length": int(np.max(array)),
            "min_length": int(np.min(array)),
        }

        with self._lock:
            self.data_profile = profile

        return profile

    # ========================================================================
    # Reporting
    # ========================================================================

    def analyze_and_report_data(
        self,
        orig_len: int,
        new_len: int,
        changes: List[str],
    ) -> str:
        """
        Generate a data-quality report and optionally summarize it with the
        Captain LLM.
        """
        original = max(0, int(orig_len))
        final = max(0, int(new_len))

        safe_changes = [
            _safe_text(change)
            for change in (changes or [])
            if _safe_text(change)
        ]

        removed = max(0, original - final)

        lines = [
            f"🧹 Analyzed {original} samples.",
        ]

        if original == final and not safe_changes:
            lines.append(
                "✅ Data quality is excellent. No anomalies detected. "
                "Proceeding with raw dataset."
            )
        else:
            lines.append(
                "🔧 Actions taken to clean data:"
            )

            if safe_changes:
                lines.extend(
                    f"  - {change}"
                    for change in safe_changes
                )
            else:
                lines.append(
                    "  - Dataset size changed, but no textual changes "
                    "were reported."
                )

            lines.append(
                f"📊 Final dataset size: {final} samples "
                f"(Removed {removed})."
            )

        report = "\n".join(lines)

        if self._llm_available():
            prompt = (
                "You are a data quality expert.\n\n"
                "Summarize the following dataset cleaning report "
                "in exactly one concise sentence.\n\n"
                f"{report}"
            )

            try:
                summary = self._generate_llm_text(
                    prompt,
                    max_new_tokens=60,
                )

                if summary:
                    report += (
                        f"\n\n🧠 Captain's Verdict: "
                        f"{summary.strip()}"
                    )

            except Exception:
                logger.debug(
                    "PhoenixCaptain: data report LLM summary failed.",
                    exc_info=True,
                )

        try:
            from . import ui

            ui.print_captain_report(report)
        except Exception:
            logger.debug(
                "PhoenixCaptain: UI report failed.",
                exc_info=True,
            )

        return report

    def evaluate_improvement(
        self,
        prompt: str,
        before_text: str,
        after_text: str,
        correct_text: str,
    ) -> str:
        """
        Generate a human-readable pre/post-training evaluation.

        When an LLM is unavailable, a deterministic heuristic is used.
        """
        prompt_text = _safe_text(prompt)
        before = _safe_text(before_text)
        after = _safe_text(after_text)
        expected = _safe_text(correct_text)

        report = (
            "📊 Pre/Post Training Evaluation\n\n"
            f"❓ Prompt: {prompt_text[:300]}\n\n"
            f"❌ Before Training: {before[:500]}\n"
            f"✅ After Training:  {after[:500]}\n"
            f"🎯 Expected:        {expected[:500]}\n\n"
        )

        if self._llm_available():
            eval_prompt = (
                "You are an AI evaluator.\n\n"
                "Compare the Before and After answers against Expected.\n"
                "Return:\n"
                "Improvement Score: <0-100>%\n"
                "Reason: <one concise sentence>\n\n"
                f"{report}"
            )

            try:
                result = self._generate_llm_text(
                    eval_prompt,
                    max_new_tokens=100,
                )

                if result:
                    report += (
                        f"🧠 Captain's Evaluation: "
                        f"{result.strip()}"
                    )
                else:
                    raise RuntimeError(
                        "Captain LLM returned empty evaluation."
                    )

            except Exception:
                report += (
                    "🧠 Captain's Evaluation: "
                    "LLM evaluation failed; deterministic fallback used."
                )
                report += self._heuristic_improvement(
                    before,
                    after,
                    expected,
                )
        else:
            report += self._heuristic_improvement(
                before,
                after,
                expected,
            )

        try:
            from . import ui

            ui.print_captain_report(report)
        except Exception:
            logger.debug(
                "PhoenixCaptain: evaluation UI failed.",
                exc_info=True,
            )

        return report

    @staticmethod
    def _heuristic_improvement(
        before: str,
        after: str,
        expected: str,
    ) -> str:
        """
        Conservative fallback evaluation.

        This deliberately avoids claiming a literal "100% improvement" merely
        because one substring happens to exist in the answer.
        """
        expected_clean = expected.strip().lower()
        before_clean = before.strip().lower()
        after_clean = after.strip().lower()

        if not expected_clean:
            return (
                "\n🧠 Captain's Evaluation: "
                "Expected answer is empty; manual review required."
            )

        before_match = expected_clean in before_clean
        after_match = expected_clean in after_clean

        if after_match and not before_match:
            score = 100
            reason = "The expected answer is present after training but not before."
        elif after_match and before_match:
            score = 50
            reason = "The expected answer was already present before training."
        elif not after_match and before_match:
            score = 0
            reason = "The expected answer was present before training but is missing afterward."
        else:
            # Avoid fake precision.
            score = 0
            reason = "The expected answer was not found in either response."

        return (
            f"\n🧠 Captain's Evaluation: "
            f"{score}% — {reason}"
        )

    # ========================================================================
    # Rule-based training engine
    # ========================================================================

    def _rule_advice(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        brain_regions: Tuple[float, float, float],
        val_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produce deterministic training advice.

        Rules are intentionally conservative and ordered from strongest
        failure signals to weaker optimization signals.
        """
        loss = _safe_float(loss)
        lr = max(0.0, _safe_float(lr))
        grad_norm = max(0.0, _safe_float(grad_norm))

        early, late, gate = (
            _safe_float(brain_regions[0]),
            _safe_float(brain_regions[1]),
            _safe_float(brain_regions[2]),
        )

        early = max(0.0, early)
        late = max(0.0, late)
        gate = max(0.0, gate)

        self.loss_history.append(loss)

        if val_loss is not None:
            val_loss = _safe_float(val_loss)
            self.val_loss_history.append(val_loss)

        trend = "stable"

        if self.previous_loss is not None:
            diff = loss - self.previous_loss

            # Relative thresholds prevent the fixed 0.05 threshold from
            # becoming meaningless when losses are very small or very large.
            threshold = max(
                0.01,
                abs(self.previous_loss) * 0.02,
            )

            if diff > threshold:
                trend = "rising"
            elif diff < -threshold:
                trend = "falling"

        # Keep state update after calculating trend.
        self.previous_loss = loss

        if val_loss is not None:
            self.previous_val_loss = val_loss

        # --------------------------------------------------------------
        # 1. Gradient collapse
        # --------------------------------------------------------------

        collapse_threshold = _safe_float(
            getattr(
                self.config,
                "captain_gradient_collapse_threshold",
                1e-6,
            ),
            1e-6,
        )

        if grad_norm <= max(0.0, collapse_threshold):
            return self._make_advice(
                message="Gradient collapse detected",
                action="Boost LR & Unfreeze",
                mult=2.0,
                layer_boost="all",
                stop=False,
            )

        # --------------------------------------------------------------
        # 2. Strong loss instability
        # --------------------------------------------------------------

        if len(self.loss_history) >= 3:
            values = list(self.loss_history)

            d1 = values[-1] - values[-2]
            d0 = values[-2] - values[-3]

            acceleration = d1 - d0

            acceleration_threshold = max(
                0.02,
                abs(values[-2]) * 0.03,
            )

            if acceleration > acceleration_threshold:
                return self._make_advice(
                    message="Loss acceleration detected",
                    action="Aggressive LR Cut",
                    mult=0.40,
                    layer_boost="none",
                    stop=False,
                )

        # --------------------------------------------------------------
        # 3. Plateau detection
        # --------------------------------------------------------------

        plateau_window = max(
            3,
            int(
                getattr(
                    self.config,
                    "captain_plateau_window",
                    min(5, max(3, len(self.loss_history))),
                )
            ),
        )

        if len(self.loss_history) >= plateau_window:
            recent = np.asarray(
                list(self.loss_history)[-plateau_window:],
                dtype=np.float64,
            )

            recent_mean = float(np.mean(recent))
            recent_std = float(np.std(recent))

            plateau_threshold = max(
                1e-5,
                abs(recent_mean) * 0.001,
            )

            if recent_std < plateau_threshold:
                return self._make_advice(
                    message="Training plateau detected",
                    action="Boost LR",
                    mult=1.50,
                    layer_boost="all",
                    stop=False,
                )

        # --------------------------------------------------------------
        # 4. Validation degradation
        # --------------------------------------------------------------

        if (
            val_loss is not None
            and len(self.val_loss_history) >= 3
        ):
            previous_values = list(
                self.val_loss_history
            )[:-1]

            if previous_values:
                baseline = float(
                    np.mean(
                        previous_values[-3:]
                    )
                )

                degradation_threshold = max(
                    0.005,
                    abs(baseline) * 0.02,
                )

                if val_loss > baseline + degradation_threshold:
                    return self._make_advice(
                        message="Validation loss increasing",
                        action="Decrease LR",
                        mult=0.70,
                        layer_boost="none",
                        stop=False,
                    )

        # --------------------------------------------------------------
        # 5. MoE imbalance
        # --------------------------------------------------------------

        if (
            self.is_moe
            and self.expert_imbalance is not None
            and self.expert_imbalance > 0.60
        ):
            return self._make_advice(
                message="Expert imbalance high",
                action="Decrease LR",
                mult=0.70,
                layer_boost="none",
                stop=False,
            )

        # --------------------------------------------------------------
        # 6. Region-specific gradient imbalance
        # --------------------------------------------------------------

        region_threshold = _safe_float(
            getattr(
                self.config,
                "captain_brain_activity_threshold",
                0.20,
            ),
            0.20,
        )

        low_region_threshold = _safe_float(
            getattr(
                self.config,
                "captain_brain_low_threshold",
                0.05,
            ),
            0.05,
        )

        if late > region_threshold and early < low_region_threshold:
            return self._make_advice(
                message="Superficial learning",
                action="Deep Brain Stimulus",
                mult=1.50,
                layer_boost="early",
                stop=False,
            )

        if early > region_threshold and late < low_region_threshold:
            return self._make_advice(
                message="Perception overload",
                action="Cortex Stimulus",
                mult=1.50,
                layer_boost="late",
                stop=False,
            )

        # --------------------------------------------------------------
        # 7. High gradient + rising loss
        # --------------------------------------------------------------

        instability_threshold = _safe_float(
            getattr(
                self.config,
                "captain_instability_grad_norm",
                5.0,
            ),
            5.0,
        )

        if (
            trend == "rising"
            and grad_norm > instability_threshold
        ):
            return self._make_advice(
                message="Brain instability",
                action="Decrease LR",
                mult=0.50,
                layer_boost="none",
                stop=False,
            )

        # --------------------------------------------------------------
        # Stable
        # --------------------------------------------------------------

        return self._make_advice(
            message="Stable training",
            action="Keep LR",
            mult=1.0,
            layer_boost="none",
            stop=False,
        )

    # ========================================================================
    # Advice normalization
    # ========================================================================

    def _make_advice(
        self,
        *,
        message: str,
        action: str,
        mult: float,
        layer_boost: str,
        stop: bool = False,
    ) -> Dict[str, Any]:
        """Construct a normalized, validated Captain decision."""
        low, high = self._multiplier_bounds()

        multiplier = _clamp(
            _safe_float(mult, 1.0),
            low,
            high,
        )

        layer = _safe_text(
            layer_boost,
            "none",
        ).lower()

        if layer not in self.VALID_LAYER_BOOSTS:
            layer = "none"

        return {
            "message": _safe_text(
                message,
                "Captain recommendation",
            ),
            "action": _safe_text(
                action,
                "Keep LR",
            ),
            "mult": multiplier,
            "layer_boost": layer,
            "stop": bool(stop),
        }

    # ========================================================================
    # LLM output parsing
    # ========================================================================

    def _llm_parse(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """
        Parse Captain LLM output robustly.

        Accepted formats include the original text protocol:

            Diagnosis: ...
            Action: ...
            Multiplier: 1.25

        and basic JSON responses.
        """
        raw = _safe_text(response)

        if not raw:
            return self._make_advice(
                message="Empty LLM response",
                action="Keep LR",
                mult=1.0,
                layer_boost="none",
            )

        # --------------------------------------------------------------
        # JSON-first parsing
        # --------------------------------------------------------------

        json_candidates = [raw]

        fenced = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        json_candidates.extend(fenced)

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)

                if isinstance(parsed, Mapping):
                    diagnosis = parsed.get(
                        "diagnosis",
                        parsed.get(
                            "message",
                            "LLM advice",
                        ),
                    )

                    action = parsed.get(
                        "action",
                        "Keep LR",
                    )

                    multiplier = parsed.get(
                        "multiplier",
                        parsed.get(
                            "mult",
                            1.0,
                        ),
                    )

                    layer_boost = parsed.get(
                        "layer_boost",
                        "none",
                    )

                    stop = parsed.get(
                        "stop",
                        False,
                    )

                    return self._make_advice(
                        message=_safe_text(
                            diagnosis,
                            "LLM advice",
                        ),
                        action=_safe_text(
                            action,
                            "Keep LR",
                        ),
                        mult=_safe_float(
                            multiplier,
                            1.0,
                        ),
                        layer_boost=_safe_text(
                            layer_boost,
                            "none",
                        ),
                        stop=bool(stop),
                    )

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                continue

        # --------------------------------------------------------------
        # Original text protocol
        # --------------------------------------------------------------

        diagnosis_match = re.search(
            r"Diagnosis\s*:\s*(.+?)(?=\n\s*(?:Action|Multiplier|Layer|Stop)\s*:|$)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        action_match = re.search(
            r"Action\s*:\s*(.+?)(?=\n\s*(?:Multiplier|Layer|Stop|Diagnosis)\s*:|$)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        multiplier_match = re.search(
            r"Multiplier\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            raw,
            flags=re.IGNORECASE,
        )

        layer_match = re.search(
            r"Layer(?:[_ ]?Boost)?\s*:\s*([A-Za-z_]+)",
            raw,
            flags=re.IGNORECASE,
        )

        stop_match = re.search(
            r"Stop\s*:\s*(true|false|yes|no|1|0)",
            raw,
            flags=re.IGNORECASE,
        )

        diagnosis = (
            diagnosis_match.group(1).strip()
            if diagnosis_match
            else "LLM advice"
        )

        action = (
            action_match.group(1).strip()
            if action_match
            else "Keep LR"
        )

        multiplier = (
            _safe_float(
                multiplier_match.group(1),
                1.0,
            )
            if multiplier_match
            else 1.0
        )

        layer_boost = (
            layer_match.group(1).strip()
            if layer_match
            else "none"
        )

        stop = False

        if stop_match:
            stop = stop_match.group(1).lower() in {
                "true",
                "yes",
                "1",
            }

        return self._make_advice(
            message=diagnosis,
            action=action,
            mult=multiplier,
            layer_boost=layer_boost,
            stop=stop,
        )

    # ========================================================================
    # Prompt generation
    # ========================================================================

    def _build_prompt(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        brain_regions: Tuple[float, float, float],
        val_loss: Optional[float] = None,
    ) -> str:
        """Build a compact but information-rich Captain prompt."""
        early, late, gate = (
            _safe_float(brain_regions[0]),
            _safe_float(brain_regions[1]),
            _safe_float(brain_regions[2]),
        )

        model_info = (
            json.dumps(
                self.model_profile,
                ensure_ascii=False,
                default=str,
            )
            if self.model_profile
            else "{}"
        )

        data_info = (
            json.dumps(
                self.data_profile,
                ensure_ascii=False,
                default=str,
            )
            if self.data_profile
            else "{}"
        )

        with self._lock:
            history_items = list(self.memory)

            expert_imbalance = self.expert_imbalance
            family = self.family
            is_moe = self.is_moe

        history_lines = []

        for item in history_items[-10:]:
            history_lines.append(
                "Step {step}: loss={loss:.6f}, "
                "val_loss={val_loss}, action={action}, multiplier={mult:.3f}".format(
                    step=item.get("step", "?"),
                    loss=_safe_float(
                        item.get("loss", 0.0)
                    ),
                    val_loss=item.get(
                        "val_loss",
                        "N/A",
                    ),
                    action=item.get(
                        "action",
                        "",
                    ),
                    mult=_safe_float(
                        item.get("mult", 1.0),
                        1.0,
                    ),
                )
            )

        history = (
            "\n".join(history_lines)
            if history_lines
            else "No previous training history."
        )

        val_loss_text = (
            "N/A"
            if val_loss is None
            else f"{_safe_float(val_loss):.6f}"
        )

        expert_text = (
            "N/A"
            if expert_imbalance is None
            else f"{expert_imbalance:.4f}"
        )

        return f"""
You are Phoenix Captain, an adaptive deep-learning training advisor.

Your job is to recommend a SAFE optimization adjustment based only on the
observed training state.

Model family: {family}
MoE model: {is_moe}
Expert imbalance: {expert_text}

Model profile:
{model_info}

Dataset profile:
{data_info}

Current training state:
Step: {step}
Training loss: {loss:.6f}
Validation loss: {val_loss_text}
Learning rate: {lr:.6e}
Global gradient norm: {grad_norm:.6f}

Gradient activity:
Early layers: {early:.6f}
Late layers: {late:.6f}
Gate layers: {gate:.6f}

Recent history:
{history}

Return ONLY this structured format:

Diagnosis: <brief diagnosis>
Action: <brief action>
Multiplier: <number between {self._multiplier_bounds()[0]:.2f} and {self._multiplier_bounds()[1]:.2f}>
Layer Boost: <none|all|early|late|gate|router|other>
Stop: <true|false>
""".strip()

    # ========================================================================
    # LLM generation
    # ========================================================================

    def _llm_available(self) -> bool:
        """Whether a usable Captain LLM is currently available."""
        with self._lock:
            return (
                self.mode == "llm"
                and self.model is not None
                and self.tokenizer is not None
            )

    def _model_device(self) -> torch.device:
        """
        Determine an appropriate input device.

        Supports models with:
            .device
            .hf_device_map
        """
        if self.model is None:
            return torch.device("cpu")

        try:
            device = getattr(self.model, "device", None)

            if device is not None:
                return device
        except Exception:
            pass

        try:
            device_map = getattr(
                self.model,
                "hf_device_map",
                None,
            )

            if isinstance(device_map, Mapping):
                for device in device_map.values():
                    if isinstance(device, int):
                        return torch.device(
                            f"cuda:{device}"
                        )

                    if isinstance(device, str):
                        if device not in {
                            "cpu",
                            "disk",
                        }:
                            return torch.device(device)

        except Exception:
            pass

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def _generate_llm_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Synchronously generate text from the Captain LLM.

        A dedicated generation lock prevents concurrent calls against the
        same inference model.
        """
        if not self._llm_available():
            return ""

        model = self.model
        tokenizer = self.tokenizer

        if model is None or tokenizer is None:
            return ""

        if max_new_tokens is None:
            max_new_tokens = self.DEFAULT_LLM_MAX_NEW_TOKENS

        max_new_tokens = max(
            1,
            int(max_new_tokens),
        )

        with self._generation_lock:
            # Unsloth exposes this helper. Import only when LLM inference
            # actually occurs.
            try:
                from unsloth import FastLanguageModel

                FastLanguageModel.for_inference(model)
            except Exception:
                # Some model types do not need this helper.
                logger.debug(
                    "PhoenixCaptain: FastLanguageModel.for_inference() "
                    "was unavailable or unnecessary.",
                    exc_info=True,
                )

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max(
                    256,
                    int(
                        getattr(
                            self.config,
                            "captain_max_seq_length",
                            self.DEFAULT_LLM_MAX_LENGTH,
                        )
                    ),
                ),
            )

            device = self._model_device()

            # Move only tensors; BatchEncoding itself may vary between
            # Transformers versions.
            encoded = {
                key: value.to(device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in encoded.items()
            }

            input_length = int(
                encoded["input_ids"].shape[-1]
            )

            do_sample = bool(
                getattr(
                    self.config,
                    "captain_do_sample",
                    False,
                )
            )

            temperature = _safe_float(
                getattr(
                    self.config,
                    "captain_temperature",
                    0.2,
                ),
                0.2,
            )

            if temperature <= 0:
                temperature = 0.2

            generation_kwargs = {
                **encoded,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": getattr(
                    tokenizer,
                    "pad_token_id",
                    getattr(
                        tokenizer,
                        "eos_token_id",
                        None,
                    ),
                ),
            }

            if do_sample:
                generation_kwargs["temperature"] = temperature

            with torch.inference_mode():
                output = model.generate(
                    **generation_kwargs
                )

            generated_tokens = output[
                0,
                input_length:,
            ]

            return _safe_text(
                tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
            )

    # ========================================================================
    # Async LLM worker
    # ========================================================================

    def _start_async_llm_request(
        self,
        prompt: str,
        step: int,
        generation_id: int,
    ) -> None:
        """
        Start one asynchronous LLM request.

        The worker publishes its result only when its generation ID is still
        current. This prevents stale results from old inspections from
        replacing newer decisions.
        """
        with self._lock:
            if self._busy:
                return

            self._busy = True
            self.llm_request_count += 1

        def _worker() -> None:
            try:
                response = self._generate_llm_text(
                    prompt,
                    max_new_tokens=int(
                        getattr(
                            self.config,
                            "captain_max_new_tokens",
                            self.DEFAULT_LLM_MAX_NEW_TOKENS,
                        )
                    ),
                )

                if not response:
                    raise RuntimeError(
                        "Captain LLM returned an empty response."
                    )

                result = self._llm_parse(response)

                with self._lock:
                    # Reject stale responses.
                    if generation_id != self._generation:
                        logger.debug(
                            "PhoenixCaptain: ignoring stale LLM result "
                            "for step=%d.",
                            step,
                        )
                    else:
                        self._pending_llm_result = result
                        self._last_successful_llm_ts = time.time()
                        self.llm_success_count += 1

            except Exception as exc:
                with self._lock:
                    self.llm_failure_count += 1
                    self._last_error = (
                        f"Captain LLM generation failed: {exc}"
                    )

                logger.warning(
                    "⚠️ Phoenix Captain LLM generation failed: %s",
                    exc,
                )

            finally:
                with self._lock:
                    self._busy = False

        thread = threading.Thread(
            target=_worker,
            name=f"phoenix-captain-{step}",
            daemon=True,
        )

        thread.start()

    # ========================================================================
    # Training inspection
    # ========================================================================

    def _publish_result(
        self,
        result: Dict[str, Any],
        *,
        source: str,
        step: int,
        loss: float,
        val_loss: Optional[float],
    ) -> None:
        """Publish a validated decision and append a complete history entry."""
        normalized = self._make_advice(
            message=result.get(
                "message",
                "Captain recommendation",
            ),
            action=result.get(
                "action",
                "Keep LR",
            ),
            mult=result.get(
                "mult",
                1.0,
            ),
            layer_boost=result.get(
                "layer_boost",
                "none",
            ),
            stop=result.get(
                "stop",
                False,
            ),
        )

        with self._lock:
            self._last_result = normalized

            self.memory.append(
                {
                    "step": int(step),
                    "loss": _safe_float(loss),
                    "val_loss": (
                        None
                        if val_loss is None
                        else _safe_float(val_loss)
                    ),
                    "action": normalized["action"],
                    "message": normalized["message"],
                    "mult": normalized["mult"],
                    "layer_boost": normalized["layer_boost"],
                    "stop": normalized["stop"],
                    "source": source,
                }
            )

            self.inspection_count += 1

    def inspect_training(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        brain_regions: Tuple[float, float, float],
        val_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Inspect current training state and produce advice.

        Return value is always immediately available and deterministic.

        When LLM mode is active:
            • rule advice is generated immediately;
            • an LLM request may run asynchronously;
            • a completed newer LLM result can replace the rule decision;
            • the caller never has to wait for LLM generation.

        This fixes the original race where an asynchronous result could be
        written and then immediately overwritten by ``rule_res``.
        """
        step = max(0, int(step))

        loss = _safe_float(loss)
        lr = max(0.0, _safe_float(lr))
        grad_norm = max(0.0, _safe_float(grad_norm))

        if len(brain_regions) != 3:
            raise ValueError(
                "brain_regions must contain exactly three values: "
                "(early, late, gate)."
            )

        brain_regions = tuple(
            max(0.0, _safe_float(value))
            for value in brain_regions
        )  # type: ignore[assignment]

        if val_loss is not None:
            val_loss = _safe_float(val_loss)

        # --------------------------------------------------------------
        # Deterministic baseline
        # --------------------------------------------------------------

        rule_result = self._rule_advice(
            step=step,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            brain_regions=brain_regions,
            val_loss=val_loss,
        )

        # --------------------------------------------------------------
        # Consume a completed async LLM result first.
        # --------------------------------------------------------------

        completed_llm: Optional[Dict[str, Any]] = None

        with self._lock:
            if self._pending_llm_result is not None:
                completed_llm = self._pending_llm_result
                self._pending_llm_result = None

        if completed_llm is not None:
            self._publish_result(
                completed_llm,
                source="llm",
                step=step,
                loss=loss,
                val_loss=val_loss,
            )

            return completed_llm

        # --------------------------------------------------------------
        # LLM request scheduling
        # --------------------------------------------------------------

        if self._llm_available():
            now = time.monotonic()

            min_interval = self._llm_interval()

            should_request = (
                now - self._last_call_ts
                >= min_interval
            )

            if should_request:
                prompt = self._build_prompt(
                    step=step,
                    loss=loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    brain_regions=brain_regions,
                    val_loss=val_loss,
                )

                with self._lock:
                    self._last_call_ts = now
                    self._generation += 1
                    generation_id = self._generation

                self._start_async_llm_request(
                    prompt=prompt,
                    step=step,
                    generation_id=generation_id,
                )

        # --------------------------------------------------------------
        # Immediate safe result
        # --------------------------------------------------------------

        self._publish_result(
            rule_result,
            source="rule",
            step=step,
            loss=loss,
            val_loss=val_loss,
        )

        return rule_result

    # ========================================================================
    # Merge intelligence
    # ========================================================================

    def inspect_merge(
        self,
        info1: Dict[str, Any],
        info2: Dict[str, Any],
        tensor_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend a merge strategy from tensor similarity metadata.

        The current deterministic policy intentionally remains conservative:

            low similarity + ordinary tensor
                -> keep primary model

            high similarity
                -> weighted average

            moderate similarity
                -> SLERP

        ``norm`` and ``router`` tensors are handled more conservatively because
        those tensor categories can be disproportionately sensitive to naïve
        averaging.
        """
        tensor_analysis = (
            tensor_analysis
            if isinstance(tensor_analysis, Mapping)
            else {}
        )

        similarity = _clamp(
            _safe_float(
                tensor_analysis.get(
                    "similarity",
                    0.5,
                ),
                0.5,
            ),
            0.0,
            1.0,
        )

        category = _safe_text(
            tensor_analysis.get(
                "category",
                "other",
            ),
            "other",
        ).lower()

        # Normalize common aliases.
        category_aliases = {
            "layernorm": "norm",
            "layer_norm": "norm",
            "rmsnorm": "norm",
            "experts": "router",
            "routing": "router",
        }

        category = category_aliases.get(
            category,
            category,
        )

        # Optionally use supplied model metadata for stronger safety.
        model_a_name = _safe_text(
            info1.get("name")
            if isinstance(info1, Mapping)
            else ""
        )

        model_b_name = _safe_text(
            info2.get("name")
            if isinstance(info2, Mapping)
            else ""
        )

        if similarity < 0.30 and category not in {
            "norm",
            "router",
        }:
            return {
                "action": "keep_a",
                "alpha": 1.0,
                "reason": (
                    f"Low similarity ({similarity:.2f}) in "
                    f"{category} tensor. Retaining primary "
                    "model weights."
                ),
                "similarity": similarity,
                "category": category,
                "model_a": model_a_name,
                "model_b": model_b_name,
            }

        if similarity >= 0.85:
            return {
                "action": "weighted_average",
                "alpha": 0.5,
                "reason": (
                    f"High similarity ({similarity:.2f}) in "
                    f"{category} tensor. Standard linear "
                    "blending is considered appropriate."
                ),
                "similarity": similarity,
                "category": category,
                "model_a": model_a_name,
                "model_b": model_b_name,
            }

        return {
            "action": "slerp",
            "alpha": 0.5,
            "reason": (
                f"Moderate similarity ({similarity:.2f}) in "
                f"{category} tensor. Recommending spherical "
                "interpolation."
            ),
            "similarity": similarity,
            "category": category,
            "model_a": model_a_name,
            "model_b": model_b_name,
        }

    # ========================================================================
    # Public result retrieval
    # ========================================================================

    def get_latest_advice(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the newest decision that has not already been consumed.

        Returns a copy so callers cannot mutate Captain's internal state.
        """
        with self._lock:
            if (
                self._last_result is not None
                and self._last_result != self._last_applied
            ):
                result = dict(self._last_result)
                self._last_applied = dict(result)
                return result

        return None
