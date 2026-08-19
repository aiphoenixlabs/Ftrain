"""
FTRAIN Phoenix Captain Trainer Callback
========================================

Adaptive training controller used by FTRAIN to observe training dynamics and
apply Captain-generated optimization adjustments.

Responsibilities
----------------
- Monitor loss, validation loss, learning rate, and gradient statistics.
- Analyze parameter-group "brain activity" before optimizer updates.
- Ask PhoenixCaptain for adaptive training advice.
- Apply Captain learning-rate multipliers safely.
- Apply optional layer-group boosts.
- Keep adaptive multipliers stable across learning-rate scheduler updates.
- Integrate with the FTRAIN dashboard/UI without making them mandatory.
- Fail safely when optional Captain/UI/dashboard functionality encounters an
  error, unless strict Captain mode is explicitly enabled.

Important
---------
Hugging Face's ``on_pre_optimizer_step`` hook is called after gradient
clipping and immediately before ``optimizer.step()``. The callback therefore
uses that hook for collecting the gradient state and applying the current
Captain decision.

Because the Hugging Face scheduler may update optimizer learning rates after
the optimizer step, this implementation reapplies Captain's adaptive factor
at ``on_step_end`` so the decision is not silently lost on the next update.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional

import torch
from transformers import TrainerCallback

from . import ui
from .captain import PhoenixCaptain
from .model_utils import is_moe

__all__ = ["PhoenixCaptainCallback"]

LOGGER = logging.getLogger(__name__)


class PhoenixCaptainCallback(TrainerCallback):
    """
    Adaptive Phoenix Captain callback for Hugging Face Trainer.

    The callback observes the training process at regular intervals and asks
    ``PhoenixCaptain`` whether the optimization behavior should change.

    Parameter groups may optionally contain:

        group["name"] = "early"
        group["name"] = "late"
        group["name"] = "gate"
        group["name"] = "router"
        group["name"] = "other"

    Unknown or missing names are automatically treated as ``"other"``.
    """

    _GROUPS = ("early", "late", "gate", "router", "other")

    def __init__(
        self,
        cfg: Any,
        model: torch.nn.Module,
        tokenizer: Any,
        train_dataset: Any,
        dashboard: Any = None,
    ) -> None:
        super().__init__()

        if cfg is None:
            raise ValueError("PhoenixCaptainCallback requires a configuration object.")

        if model is None:
            raise ValueError("PhoenixCaptainCallback requires a model.")

        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.dashboard = dashboard

        self._enabled = True
        self._strict = bool(getattr(cfg, "captain_strict", False))

        self._captain_mult: float = 1.0
        self._captain_layer_boosts: Dict[str, float] = {
            group: 1.0 for group in self._GROUPS
        }

        # Latest trainer metrics are tracked through the actual callback
        # logging hooks instead of relying on state.log_history[-1], which may
        # contain unrelated/stale entries.
        self._latest_logs: Dict[str, Any] = {}
        self._latest_eval_metrics: Dict[str, Any] = {}

        # Keep the scheduler-independent base learning rates here.
        #
        # A Hugging Face scheduler may overwrite optimizer.param_groups["lr"]
        # after optimizer.step(). We therefore keep the Captain multiplier
        # separate and reapply it at the appropriate point in the step cycle.
        self._base_learning_rates: Dict[int, float] = {}

        # Useful diagnostics.
        self._last_inspection_step: int = -1
        self._inspection_count: int = 0
        self._last_error: Optional[str] = None

        # ------------------------------------------------------------------
        # Captain initialization
        # ------------------------------------------------------------------

        self.captain = PhoenixCaptain(cfg)

        try:
            family = getattr(cfg, "family", "auto")
            if family == "auto":
                family = "generic"

            try:
                moe = bool(is_moe(model))
            except Exception:
                LOGGER.exception(
                    "FTRAIN: failed to determine whether the model is MoE. "
                    "Assuming a non-MoE architecture."
                )
                moe = False

            self.captain.set_family_context(family, moe)

            # Model analysis should never accidentally construct gradients.
            with torch.no_grad():
                self.captain.analyze_model(model)

        except Exception as exc:
            self._handle_captain_error(
                "Captain model analysis failed during callback initialization.",
                exc,
            )

        # ``if train_dataset`` is unsafe for Dataset-like objects because
        # some implementations deliberately do not define truthiness.
        if train_dataset is not None:
            if tokenizer is None:
                LOGGER.warning(
                    "FTRAIN: training dataset was supplied but tokenizer/"
                    "processing class is None; skipping Captain data analysis."
                )
            else:
                try:
                    dataset_size = self._safe_len(train_dataset)

                    if dataset_size is not None and dataset_size == 0:
                        LOGGER.warning(
                            "FTRAIN: training dataset is empty; skipping "
                            "Captain data analysis."
                        )
                    else:
                        self.captain.analyze_data(
                            train_dataset,
                            tokenizer,
                        )

                except Exception as exc:
                    self._handle_captain_error(
                        "Captain training-data analysis failed.",
                        exc,
                    )

    # ======================================================================
    # Configuration helpers
    # ======================================================================

    def _get_interval(self) -> int:
        """Return a valid positive Captain inspection interval."""
        raw_interval = getattr(self.cfg, "captain_interval", 100)

        try:
            interval = int(raw_interval)
        except (TypeError, ValueError):
            LOGGER.warning(
                "FTRAIN: invalid captain_interval=%r; disabling "
                "periodic Captain inspections.",
                raw_interval,
            )
            return 0

        if interval <= 0:
            return 0

        return interval

    def _get_learning_rate(self) -> float:
        """Return configured learning rate with a safe fallback."""
        raw_lr = getattr(self.cfg, "learning_rate", 0.0)

        try:
            lr = float(raw_lr)
        except (TypeError, ValueError):
            return 0.0

        return lr if math.isfinite(lr) and lr >= 0.0 else 0.0

    def _get_multiplier_bounds(self) -> tuple[float, float]:
        """
        Return configurable bounds for Captain's learning-rate multiplier.

        Defaults are intentionally conservative to prevent a malformed or
        overly aggressive Captain response from multiplying the LR without
        bound.
        """
        raw_min = getattr(self.cfg, "captain_mult_min", 0.25)
        raw_max = getattr(self.cfg, "captain_mult_max", 2.0)

        try:
            minimum = float(raw_min)
        except (TypeError, ValueError):
            minimum = 0.25

        try:
            maximum = float(raw_max)
        except (TypeError, ValueError):
            maximum = 2.0

        if not math.isfinite(minimum) or minimum <= 0.0:
            minimum = 0.25

        if not math.isfinite(maximum) or maximum < minimum:
            maximum = max(2.0, minimum)

        return minimum, maximum

    def _get_layer_boost_value(self) -> float:
        """Return the configured layer boost factor."""
        raw_value = getattr(self.cfg, "captain_layer_boost", 2.0)

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 2.0

        if not math.isfinite(value) or value <= 0.0:
            value = 2.0

        return value

    # ======================================================================
    # Generic helpers
    # ======================================================================

    @staticmethod
    def _safe_len(value: Any) -> Optional[int]:
        """Return len(value) without allowing unusual objects to crash setup."""
        try:
            return len(value)
        except (TypeError, AttributeError):
            return None

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        """Convert arbitrary numeric-like values into a finite float."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default

        return number if math.isfinite(number) else default

    def _handle_captain_error(self, message: str, exc: BaseException) -> None:
        """
        Handle optional Captain failures.

        By default FTRAIN continues training instead of destroying a long
        training run because an advisory component failed. Strict mode turns
        those failures into hard exceptions.
        """
        self._last_error = f"{message}: {exc}"

        LOGGER.exception(message)

        if self._strict:
            raise RuntimeError(message) from exc

    # ======================================================================
    # Gradient / brain activity analysis
    # ======================================================================

    @classmethod
    def _normalize_group_name(cls, name: Any) -> str:
        """
        Normalize optimizer parameter-group names.

        Supports strings such as:
            "early"
            "late"
            "gate"
            "router"
            "other"

        Unknown values are safely mapped to "other".
        """
        if not isinstance(name, str):
            return "other"

        normalized = name.strip().lower()

        return normalized if normalized in cls._GROUPS else "other"

    @staticmethod
    def _gradient_squared_norm(grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute the squared L2 norm of a gradient safely.

        Handles dense and sparse gradients and avoids PyTorch's private
        ``torch._foreach_*`` APIs.
        """
        if grad is None:
            return None

        if not isinstance(grad, torch.Tensor):
            return None

        if grad.is_sparse:
            grad = grad.coalesce().values()

        if grad.numel() == 0:
            return None

        # Accumulating in float32 improves numerical stability when model
        # gradients are fp16/bf16.
        return torch.sum(
            grad.detach().float() * grad.detach().float()
        )

    def _compute_brain_activity(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> tuple[float, float, float]:
        """
        Calculate gradient activity for early, late and gating groups.

        The returned values are global L2 norms for each tracked region.

        Returns:
            (early_norm, late_norm, gate_norm)
        """
        if optimizer is None:
            return 0.0, 0.0, 0.0

        squared_totals: Dict[str, Optional[torch.Tensor]] = {
            "early": None,
            "late": None,
            "gate": None,
        }

        for param_group in optimizer.param_groups:
            group_name = self._normalize_group_name(
                param_group.get("name", "other")
            )

            if group_name not in squared_totals:
                continue

            params = param_group.get("params", ())

            for parameter in params:
                if parameter is None:
                    continue

                grad = getattr(parameter, "grad", None)

                if grad is None:
                    continue

                try:
                    squared_norm = self._gradient_squared_norm(grad)
                except Exception:
                    LOGGER.debug(
                        "FTRAIN: failed to calculate gradient norm for one "
                        "parameter.",
                        exc_info=True,
                    )
                    continue

                if squared_norm is None:
                    continue

                current = squared_totals[group_name]

                if current is None:
                    squared_totals[group_name] = squared_norm
                else:
                    # Keep accumulation on the same device.
                    squared_totals[group_name] = current + squared_norm

        results = []

        for group_name in ("early", "late", "gate"):
            value = squared_totals[group_name]

            if value is None:
                results.append(0.0)
                continue

            try:
                norm = torch.sqrt(value.clamp_min(0.0)).item()
            except Exception:
                norm = 0.0

            results.append(
                norm if math.isfinite(norm) else 0.0
            )

        return results[0], results[1], results[2]

    # ======================================================================
    # Learning-rate handling
    # ======================================================================

    def _ensure_base_learning_rates(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Capture scheduler-controlled/base learning rates for parameter groups.

        ``initial_lr`` is preferred when available because Hugging Face and
        PyTorch schedulers commonly store the original LR there.
        """
        for index, param_group in enumerate(optimizer.param_groups):
            group_id = id(param_group)

            if group_id in self._base_learning_rates:
                continue

            raw_lr = param_group.get("initial_lr", param_group.get("lr"))

            lr = self._finite_float(
                raw_lr,
                default=self._get_learning_rate(),
            )

            self._base_learning_rates[group_id] = max(0.0, lr)

    def _current_group_multiplier(self, group: Mapping[str, Any]) -> float:
        """Return the Captain multiplier for a specific optimizer group."""
        name = self._normalize_group_name(
            group.get("name", "other")
        )

        layer_boost = self._captain_layer_boosts.get(name, 1.0)

        multiplier = self._captain_mult * layer_boost

        return (
            multiplier
            if math.isfinite(multiplier) and multiplier > 0.0
            else 1.0
        )

    def _apply_captain_learning_rates(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        """
        Apply the current Captain multipliers to optimizer parameter groups.

        This is always calculated from the scheduler/base LR rather than from
        the already-multiplied LR, preventing accidental compounding such as:

            lr *= multiplier
            lr *= multiplier
            lr *= multiplier

        on successive Captain inspections.
        """
        if optimizer is None:
            return

        self._ensure_base_learning_rates(optimizer)

        for param_group in optimizer.param_groups:
            group_id = id(param_group)

            base_lr = self._base_learning_rates.get(
                group_id,
                self._finite_float(param_group.get("lr"), 0.0),
            )

            multiplier = self._current_group_multiplier(param_group)

            new_lr = base_lr * multiplier

            if not math.isfinite(new_lr) or new_lr < 0.0:
                LOGGER.warning(
                    "FTRAIN: calculated invalid adaptive learning rate "
                    "(base_lr=%r, multiplier=%r). Keeping base LR.",
                    base_lr,
                    multiplier,
                )
                new_lr = base_lr

            param_group["lr"] = new_lr

    def _synchronize_base_rates_after_scheduler(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        """
        Capture scheduler output after an optimizer/scheduler update.

        At ``on_step_end`` the optimizer LR should represent the scheduler's
        current base value. We store that value before applying Captain's
        multiplier for the next step.
        """
        if optimizer is None:
            return

        for param_group in optimizer.param_groups:
            group_id = id(param_group)

            current_lr = self._finite_float(
                param_group.get("lr"),
                default=self._get_learning_rate(),
            )

            # If Captain scaling is still present, recover the scheduler base
            # rather than treating the multiplied LR as the new base.
            multiplier = self._current_group_multiplier(param_group)

            if multiplier > 0.0:
                estimated_base = current_lr / multiplier
            else:
                estimated_base = current_lr

            if (
                not math.isfinite(estimated_base)
                or estimated_base < 0.0
            ):
                estimated_base = current_lr

            self._base_learning_rates[group_id] = estimated_base

    # ======================================================================
    # Captain advice
    # ======================================================================

    def _sanitize_advice(self, advice: Any) -> Optional[Dict[str, Any]]:
        """
        Validate Captain advice before allowing it to affect optimization.

        Expected structure:

            {
                "action": "...",
                "mult": 1.0,
                "layer_boost": "early"
            }

        Malformed advice is ignored rather than corrupting optimizer state.
        """
        if not isinstance(advice, Mapping):
            return None

        minimum, maximum = self._get_multiplier_bounds()

        raw_mult = advice.get("mult", 1.0)

        try:
            mult = float(raw_mult)
        except (TypeError, ValueError):
            mult = 1.0

        if not math.isfinite(mult):
            mult = 1.0

        mult = min(max(mult, minimum), maximum)

        action = advice.get("action", "hold")

        if action is None:
            action = "hold"

        action = str(action).strip() or "hold"

        raw_layer_boost = advice.get("layer_boost", "none")

        if raw_layer_boost is None:
            layer_boost = "none"
        else:
            layer_boost = str(raw_layer_boost).strip().lower()

        valid_layers = {"none", "all", *self._GROUPS}

        if layer_boost not in valid_layers:
            LOGGER.warning(
                "FTRAIN: Captain returned unknown layer_boost=%r; "
                "using 'none'.",
                raw_layer_boost,
            )
            layer_boost = "none"

        return {
            "action": action,
            "mult": mult,
            "layer_boost": layer_boost,
        }

    def _apply_advice(
        self,
        advice: Mapping[str, Any],
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
    ) -> None:
        """Apply a sanitized Captain decision."""
        clean_advice = self._sanitize_advice(advice)

        if clean_advice is None:
            LOGGER.warning(
                "FTRAIN: Captain returned malformed advice at step %d. "
                "Keeping previous optimization settings.",
                step,
            )
            return

        self._captain_mult = clean_advice["mult"]

        layer_boost = clean_advice["layer_boost"]

        # Always reset the layer map before applying a new decision.
        self._captain_layer_boosts = {
            group: 1.0 for group in self._GROUPS
        }

        boost_value = self._get_layer_boost_value()

        if layer_boost == "all":
            self._captain_layer_boosts = {
                group: boost_value for group in self._GROUPS
            }
        elif layer_boost in self._captain_layer_boosts:
            self._captain_layer_boosts[layer_boost] = boost_value

        self._apply_captain_learning_rates(optimizer)

    # ======================================================================
    # Metrics hooks
    # ======================================================================

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the newest trainer log values."""
        if logs:
            self._latest_logs.update(logs)

        return control

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the newest evaluation metrics separately from train logs."""
        if metrics:
            self._latest_eval_metrics.update(metrics)

        return control

    # ======================================================================
    # Training lifecycle
    # ======================================================================

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Initialize optimizer LR tracking once the actual Trainer optimizer
        becomes available.
        """
        optimizer = optimizer or kwargs.get("optimizer")

        if optimizer is not None:
            self._ensure_base_learning_rates(optimizer)

        return control

    def on_pre_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Inspect gradients and apply Captain advice before optimizer.step().

        Hugging Face invokes this hook after gradient clipping and before the
        optimizer update. :contentReference[oaicite:1]{index=1}
        """
        if not self._enabled:
            return control

        optimizer = optimizer or kwargs.get("optimizer")

        if optimizer is None:
            LOGGER.debug(
                "FTRAIN: on_pre_optimizer_step received no optimizer."
            )
            return control

        interval = self._get_interval()

        if interval <= 0:
            return control

        try:
            current_global_step = int(getattr(state, "global_step", 0))
        except (TypeError, ValueError):
            current_global_step = 0

        # The pre-optimizer hook occurs before the optimizer increments the
        # trainer's global step, so the upcoming optimizer update is +1.
        upcoming_step = current_global_step + 1

        if upcoming_step % interval != 0:
            # Keep the current Captain decision active even when the scheduler
            # has just changed the base learning rate.
            self._apply_captain_learning_rates(optimizer)
            return control

        if upcoming_step == self._last_inspection_step:
            return control

        self._last_inspection_step = upcoming_step
        self._inspection_count += 1

        try:
            # Ensure the gradient state is analyzed exactly as it exists at
            # this point in the training lifecycle.
            brain = self._compute_brain_activity(optimizer)

            loss = self._finite_float(
                self._latest_logs.get(
                    "loss",
                    getattr(state, "loss", 0.0),
                ),
                default=0.0,
            )

            lr = self._finite_float(
                self._latest_logs.get(
                    "learning_rate",
                    self._get_learning_rate(),
                ),
                default=self._get_learning_rate(),
            )

            grad_norm = self._finite_float(
                self._latest_logs.get("grad_norm", 0.0),
                default=0.0,
            )

            val_loss_raw = self._latest_eval_metrics.get(
                "eval_loss",
                self._latest_logs.get("eval_loss"),
            )

            val_loss = (
                None
                if val_loss_raw is None
                else self._finite_float(val_loss_raw, default=0.0)
            )

            self.captain.inspect_training(
                upcoming_step,
                loss,
                lr,
                grad_norm,
                brain,
                val_loss,
            )

            advice = self.captain.get_latest_advice()

            if not advice:
                # Still make sure the previously selected Captain multiplier
                # remains active after scheduler updates.
                self._apply_captain_learning_rates(optimizer)
                return control

            clean_advice = self._sanitize_advice(advice)

            if clean_advice is None:
                self._apply_captain_learning_rates(optimizer)
                return control

            self._apply_advice(
                clean_advice,
                optimizer,
                upcoming_step,
            )

            action_text = (
                f"{clean_advice['action']} "
                f"(x{clean_advice['mult']:.2f})"
            )

            # UI is optional. A broken progress renderer should never destroy
            # a multi-hour training run.
            try:
                ui.print_train_table(
                    upcoming_step,
                    getattr(
                        self.cfg,
                        "max_steps",
                        getattr(args, "max_steps", upcoming_step),
                    ),
                    loss,
                    val_loss,
                    lr,
                    grad_norm,
                    action_text,
                )
            except Exception as exc:
                LOGGER.warning(
                    "FTRAIN: training UI update failed; continuing training: %s",
                    exc,
                )

            if self.dashboard is not None:
                try:
                    self.dashboard.log_metric(
                        upcoming_step,
                        loss,
                        lr,
                        val_loss,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "FTRAIN: dashboard logging failed; continuing "
                        "training: %s",
                        exc,
                    )

        except Exception as exc:
            self._handle_captain_error(
                f"Captain inspection failed at training step {upcoming_step}.",
                exc,
            )

        finally:
            # Never leave an unintended raw/scheduler LR after this callback.
            self._apply_captain_learning_rates(optimizer)

        return control

    def on_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Observe the post-optimizer state.

        This hook is intentionally lightweight. It provides a stable place
        for diagnostics without performing another expensive Captain
        inspection.
        """
        return control

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Reconcile the scheduler's new LR with Captain's persistent multiplier.

        This is the key protection against the scheduler silently wiping out
        the Captain's adaptive learning-rate decision.
        """
        if not self._enabled:
            return control

        optimizer = optimizer or kwargs.get("optimizer")

        if optimizer is None:
            return control

        try:
            self._synchronize_base_rates_after_scheduler(optimizer)
            self._apply_captain_learning_rates(optimizer)
        except Exception as exc:
            self._handle_captain_error(
                "Failed to synchronize Captain learning rates after "
                "the optimizer/scheduler step.",
                exc,
            )

        return control

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        """Emit a final diagnostic summary without changing trainer state."""
        LOGGER.info(
            "FTRAIN Captain training observer finished: inspections=%d, "
            "last_step=%d, multiplier=%.4f",
            self._inspection_count,
            self._last_inspection_step,
            self._captain_mult,
        )

        if self._last_error is not None:
            LOGGER.warning(
                "FTRAIN Captain encountered a recoverable error during "
                "training: %s",
                self._last_error,
            )

        return control
