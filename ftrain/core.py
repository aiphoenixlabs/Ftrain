"""
FTRAIN Core Training Engine
===========================

Production-oriented training orchestration for FTRAIN.

This module coordinates:

    Model loading
        ↓
    Dataset preparation
        ↓
    PEFT / LoRA / DoRA
        ↓
    Optimizer
        ↓
    Scheduler
        ↓
    Hugging Face / Unsloth / Custom training
        ↓
    Captain monitoring
        ↓
    Evaluation
        ↓
    Checkpointing
        ↓
    Final model export

Important design principles
---------------------------
• Never silently lose gradients.
• Never assume CUDA exists.
• Never use model/dataset truthiness when ``is not None`` is required.
• Keep checkpoint state separate from model weights.
• Make resume behavior deterministic.
• Avoid duplicate adaptive-LR logic.
• Keep optional integrations optional.
• Restore training mode after evaluation.
• Avoid swallowing unexpected exceptions.
"""

from __future__ import annotations

import io
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from unsloth import FastLanguageModel

from . import ui
from .captain import PhoenixCaptain
from .config import TrainConfig
from .data_quality import (
    balance_datasets,
    deduplicate,
    filter_by_perplexity,
)
from .dataset import FtrainDataset, LengthSampler, collate
from .families import get_preset
from .lora import inject as inject_lora
from .lora_dora import inject_dora
from .model_utils import (
    count_params,
    get_family,
    get_num_layers,
    is_moe,
    seed_everything,
)
from .speed import flash_mode
from .train_optim import (
    LRFinder,
    adaptive_accumulation,
    cosine_restart_scheduler,
)

logger = logging.getLogger(__name__)

__all__ = ["Ftrain"]


# =============================================================================
# Utility helpers
# =============================================================================


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _is_empty(value: Any) -> bool:
    size = _safe_len(value)
    return size == 0 if size is not None else False


# =============================================================================
# Core engine
# =============================================================================


class Ftrain:
    """
    Main FTRAIN training engine.

    The object is intentionally stateful because training requires persistent
    optimizer, scheduler, checkpoint and Captain state.
    """

    def __init__(
        self,
        config: TrainConfig,
        train_data: Any,
        val_data: Any = None,
    ) -> None:
        if config is None:
            raise ValueError("Ftrain requires a TrainConfig.")

        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------

        self.loss_history: List[float] = []

        self.step: int = 0
        self.epoch: int = 0

        self._last_loss: Optional[float] = None
        self._last_val_loss: Optional[float] = None

        self._captain_mult: float = 1.0

        self._captain_layer_boosts: Dict[str, float] = {
            "early": 1.0,
            "late": 1.0,
            "gate": 1.0,
            "router": 1.0,
            "other": 1.0,
        }

        self.device = self._resolve_device()

        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Any = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Any = None

        self.train_dataset: Any = None
        self.val_dataset: Any = None

        self.dashboard: Any = None
        self.captain: Optional[PhoenixCaptain] = None

        self.total_steps: int = int(config.max_steps)

        self._checkpoint_lock = threading.Lock()
        self._dashboard_started = False

        # Number of gradient accumulation units currently active.
        self._current_accumulation_steps = max(
            1,
            int(config.gradient_accumulation_steps),
        )

        # ---------------------------------------------------------------------
        # Environment initialization
        # ---------------------------------------------------------------------

        self._configure_runtime()

        # ---------------------------------------------------------------------
        # Resume discovery
        # ---------------------------------------------------------------------

        self._resolve_auto_resume()

        # ---------------------------------------------------------------------
        # Model family / preset
        # ---------------------------------------------------------------------

        self.family = (
            config.family
            if config.family != "auto"
            else get_family(config.model_name)
        )

        self.preset = get_preset(self.family) or {}

        if (
            not getattr(config, "lora_target_modules", None)
            and self.preset.get("lora_targets")
        ):
            config.lora_target_modules = list(
                self.preset["lora_targets"]
            )

        # ---------------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------------

        self._load_model()

        # ---------------------------------------------------------------------
        # Captain
        # ---------------------------------------------------------------------

        if config.captain_enabled:
            self.captain = PhoenixCaptain(config)
            self.captain.set_family_context(
                self.family,
                is_moe(self.model),
            )
            self.captain.analyze_model(self.model)

        # ---------------------------------------------------------------------
        # Dataset quality pipeline
        # ---------------------------------------------------------------------

        self._prepare_data()

        # ---------------------------------------------------------------------
        # Optional automatic LoRA target discovery
        # ---------------------------------------------------------------------

        if config.auto_lora_targets:
            self._discover_lora_targets()

        # ---------------------------------------------------------------------
        # PEFT
        # ---------------------------------------------------------------------

        self._apply_adapters()

        self._print_parameter_summary()

        # ---------------------------------------------------------------------
        # Trainer datasets
        # ---------------------------------------------------------------------

        self._build_datasets()

        # ---------------------------------------------------------------------
        # Dashboard
        # ---------------------------------------------------------------------

        self._start_dashboard()

        Path(config.output_dir).expanduser().mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # Environment
    # =========================================================================

    def _resolve_device(self) -> torch.device:
        """Select the safest available training device."""
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _configure_runtime(self) -> None:
        """Configure kernels, reproducibility and optional GRPO integration."""
        try:
            flash_mode(
                enabled=True,
                tf32=self.device.type == "cuda",
            )
        except Exception:
            logger.warning(
                "FTRAIN: optimized kernel configuration failed; "
                "continuing with defaults.",
                exc_info=True,
            )

        seed_everything(self.config.seed)

        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                logger.debug(
                    "FTRAIN: TF32 configuration unavailable.",
                    exc_info=True,
                )

        if self.config.use_grpo:
            try:
                from unsloth import PatchFastRL

                PatchFastRL(
                    "GRPO",
                    FastLanguageModel,
                )

                logger.info(
                    "🧠 Unsloth patched for GRPO."
                )

            except Exception:
                # GRPO import/setup can legitimately vary by installed
                # Unsloth/TRL version. The actual GRPO path will surface a
                # precise failure if it cannot run.
                logger.warning(
                    "FTRAIN: GRPO patching was unavailable.",
                    exc_info=True,
                )

    # =========================================================================
    # Resume handling
    # =========================================================================

    def _resolve_auto_resume(self) -> None:
        """Select the newest valid FTRAIN or Transformers checkpoint."""
        if not self.config.auto_resume:
            return
        root = Path(self.config.output_dir).expanduser()
        if not root.exists():
            return
        candidates: List[Tuple[int, Path, str]] = []
        seen = set()
        for search_root in (root / "checkpoints", root):
            if not search_root.is_dir():
                continue
            for entry in search_root.iterdir():
                if not entry.is_dir():
                    continue
                resolved = entry.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                match = re.fullmatch(r"step_(\d+)", entry.name)
                kind = "ftrain"
                if match is None:
                    match = re.fullmatch(r"checkpoint-(\d+)", entry.name)
                    kind = "trainer"
                if match is None:
                    continue
                has_state = any(path.exists() for path in (
                    entry / "ftrain_state.json", entry / "trainer_state.json",
                    entry / "optimizer.pt", entry / "optimizer.bin",
                    entry / "model.safetensors", entry / "pytorch_model.bin",
                    entry / "config.json", entry / "adapter_config.json",
                )) or any(entry.glob("*.safetensors"))
                if has_state:
                    candidates.append((int(match.group(1)), entry, kind))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            step, checkpoint, kind = candidates[-1]
            self.config.resume_from_checkpoint = str(checkpoint)
            logger.info("FTRAIN: auto-resume selected %s (step=%d, kind=%s).", checkpoint, step, kind)

    # =========================================================================
    # Model loading
    # =========================================================================

    @contextmanager
    def _quiet_stdout(self):
        """
        Temporarily capture noisy model-loader stdout.

        stderr is deliberately left untouched so low-level CUDA/model-loading
        failures remain visible.
        """
        previous = sys.stdout
        buffer = io.StringIO()

        try:
            sys.stdout = buffer
            yield buffer
        finally:
            sys.stdout = previous

    def _load_model(self) -> None:
        """Load model/tokenizer with Unsloth-first, Transformers fallback."""
        cfg = self.config

        bar = ui.LoadingBar(
            message=f"Loading {cfg.model_name}",
            real_progress=cfg.show_model_progress,
        )

        bar.start()

        try:
            kwargs: Dict[str, Any] = {
                "model_name": cfg.model_name,
                "max_seq_length": cfg.max_seq_length,
                "load_in_4bit": cfg.load_in_4bit,
            }

            if not cfg.load_in_4bit:
                kwargs["dtype"] = self._preferred_model_dtype()

            attention_impl = self.preset.get(
                "attn_implementation"
            )

            if attention_impl:
                kwargs["attn_implementation"] = attention_impl

            logger.info(
                "Loading model through Unsloth: %s",
                cfg.model_name,
            )

            try:
                with self._quiet_stdout():
                    self.model, self.tokenizer = (
                        FastLanguageModel.from_pretrained(
                            **kwargs
                        )
                    )

            except Exception as unsloth_error:
                logger.warning(
                    "⚠️ Unsloth model loading failed: %s",
                    unsloth_error,
                )

                self._load_model_transformers()

            if self.model is None or self.tokenizer is None:
                raise RuntimeError(
                    "Model loading completed without both model and tokenizer."
                )

            self._prepare_tokenizer()
            self._prepare_model()

        finally:
            bar.done()

    def _preferred_model_dtype(self) -> torch.dtype:
        """Choose a dtype supported by the current device."""
        if self.device.type == "cuda":
            if (
                torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16

            return torch.float16

        if self.device.type == "cpu":
            return torch.float32

        # MPS has broad fp16 support but fp32 is the safest default when
        # loading through the generic Transformers path.
        return torch.float32

    def _load_model_transformers(self) -> None:
        """Fallback Hugging Face Transformers model loader."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        cfg = self.config

        dtype = self._preferred_model_dtype()

        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
        }

        # Avoid forcing device_map="auto" for CPU/MPS.
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"

        logger.info(
            "Loading model through Transformers fallback: %s",
            cfg.model_name,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            **kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name
        )

    def _prepare_tokenizer(self) -> None:
        """Ensure tokenizer has a usable padding token."""
        tokenizer = self.tokenizer

        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer was not loaded."
            )

        if getattr(
            tokenizer,
            "pad_token_id",
            None,
        ) is None:
            eos_token = getattr(
                tokenizer,
                "eos_token",
                None,
            )

            if eos_token is not None:
                tokenizer.pad_token = eos_token
            else:
                logger.warning(
                    "Tokenizer has no pad or EOS token."
                )

        try:
            tokenizer.padding_side = "right"
        except Exception:
            pass

    def _prepare_model(self) -> None:
        """Move/prepare the model when appropriate."""
        if self.model is None:
            raise RuntimeError(
                "Model is unavailable."
            )

        # device_map="auto" models are already dispatched and should not be
        # moved wholesale.
        if self.device.type in {"cuda", "cpu", "mps"}:
            try:
                has_device_map = bool(
                    getattr(
                        self.model,
                        "hf_device_map",
                        None,
                    )
                )

                if not has_device_map:
                    self.model.to(self.device)
            except Exception:
                logger.debug(
                    "FTRAIN: model device preparation was skipped.",
                    exc_info=True,
                )

    # =========================================================================
    # Dataset preparation
    # =========================================================================

    def _prepare_data(self) -> None:
        """Run configured dataset-quality transformations."""
        cfg = self.config

        if self.train_data is None:
            raise ValueError(
                "Training data cannot be None."
            )

        train_len = _safe_len(self.train_data)

        if train_len == 0:
            raise ValueError(
                "Training dataset is empty."
            )

        needs_processing = (
            cfg.data_perplexity_filter
            or cfg.data_dedup
            or bool(cfg.data_sources)
        )

        if not needs_processing:
            return

        original_length = train_len
        changes: List[str] = []

        # ---------------------------------------------------------------
        # Deduplication
        # ---------------------------------------------------------------

        if cfg.data_dedup:
            before = _safe_len(self.train_data) or 0

            self.train_data = deduplicate(
                self.train_data
            )

            after = _safe_len(self.train_data) or 0

            if after < before:
                changes.append(
                    f"Deduplication removed {before - after} duplicates."
                )

        # ---------------------------------------------------------------
        # Perplexity filtering
        # ---------------------------------------------------------------

        if cfg.data_perplexity_filter:
            before = _safe_len(self.train_data) or 0

            self.train_data = filter_by_perplexity(
                self.train_data,
                self.model,
                self.tokenizer,
                self.device,
                cfg.data_perplexity_keep_pct,
            )

            after = _safe_len(self.train_data) or 0

            if after < before:
                changes.append(
                    f"Perplexity filter removed "
                    f"{before - after} samples."
                )

        # ---------------------------------------------------------------
        # Additional data sources
        # ---------------------------------------------------------------

        if cfg.data_sources:
            from .data_utils import load_data

            datasets = [self.train_data]

            for source in cfg.data_sources:
                try:
                    datasets.append(
                        load_data(source)
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load additional data source "
                        f"{source!r}."
                    ) from exc

            self.train_data = balance_datasets(
                datasets,
                cfg.data_balance_strategy,
            )

            changes.append(
                "Balanced multiple data sources."
            )

        final_length = _safe_len(
            self.train_data
        ) or 0

        if final_length == 0:
            raise ValueError(
                "Dataset processing removed all training samples."
            )

        if self.captain:
            self.captain.analyze_and_report_data(
                original_length,
                final_length,
                changes,
            )

    # =========================================================================
    # Automatic LoRA target discovery
    # =========================================================================

    def _discover_lora_targets(self) -> None:
        """
        Discover useful LoRA targets from actual gradient activity.

        The old implementation cleared ``p.grad`` before collecting it,
        making the resulting gradient map empty. This implementation reads
        gradient norms first and clears gradients only afterward.
        """
        if self.model is None:
            raise RuntimeError(
                "Cannot discover LoRA targets without a model."
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "Cannot discover LoRA targets without a tokenizer."
            )

        logger.info(
            "🔍 Automatically discovering LoRA target modules..."
        )

        self.model.train()
        self._clear_gradients()

        encoded = self.tokenizer(
            "Test",
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in encoded.items()
        }

        try:
            with self._autocast_context():
                output = self.model(
                    **encoded,
                    labels=encoded["input_ids"],
                )

            loss = output.loss

            if loss is None:
                raise RuntimeError(
                    "Model returned no loss during LoRA target discovery."
                )

            loss.backward()

            gradient_scores: Dict[str, float] = {}

            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue

                if not parameter.requires_grad:
                    continue

                # Remove adapter parameters from discovery; target selection
                # should be based on the base model's actual trainable signal.
                if "lora_" in name.lower():
                    continue

                module_name = self._extract_target_module_name(
                    name
                )

                if not module_name:
                    continue

                try:
                    score = parameter.grad.detach().float().norm().item()
                except Exception:
                    continue

                if not math.isfinite(score):
                    continue

                gradient_scores[module_name] = (
                    gradient_scores.get(module_name, 0.0)
                    + score
                )

            if not gradient_scores:
                logger.warning(
                    "Automatic LoRA discovery found no usable gradient "
                    "signals. Keeping configured/preset targets."
                )
                return

            target_count = max(
                1,
                int(self.config.lora_target_count),
            )

            ranked = sorted(
                gradient_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            selected = [
                name
                for name, _ in ranked[:target_count]
            ]

            if selected:
                self.config.lora_target_modules = selected

                logger.info(
                    "🎯 Auto-selected LoRA targets: %s",
                    ", ".join(selected),
                )

        finally:
            self._clear_gradients()

    @staticmethod
    def _extract_target_module_name(
        parameter_name: str,
    ) -> Optional[str]:
        """
        Extract the module component immediately preceding a parameter name.

        Examples:
            self_attn.q_proj.weight -> q_proj
            mlp.gate_proj.weight    -> gate_proj
        """
        parts = parameter_name.split(".")

        if len(parts) < 2:
            return None

        candidate = parts[-2]

        if candidate.isdigit():
            return None

        return candidate

    # =========================================================================
    # Adapter injection
    # =========================================================================

    def _apply_adapters(self) -> None:
        """Inject LoRA/DoRA with compatibility across Unsloth releases."""
        cfg=self.config
        if self.model is None:
            raise RuntimeError("Cannot apply adapters without a model.")
        targets=list(cfg.lora_target_modules) if cfg.lora_target_modules else None
        if not targets:
            raise ValueError("No LoRA target modules are configured.")
        try:
            if cfg.use_unsloth_lora:
                fn=getattr(FastLanguageModel,"get_peft_model")
                supported=self._supported_parameters(fn)
                kwargs={"r":cfg.lora_r,"lora_alpha":cfg.lora_alpha,"target_modules":targets}
                if cfg.use_dora and "use_dora" in supported:
                    kwargs["use_dora"]=True
                if "lora_dropout" in supported:
                    kwargs["lora_dropout"]=0.0
                if "bias" in supported:
                    kwargs["bias"]="none"
                self.model=fn(self.model,**kwargs)
            elif cfg.use_custom_lora:
                if cfg.use_dora:
                    self.model=inject_dora(self.model,targets,cfg.lora_r,cfg.lora_alpha)
                else:
                    self.model=inject_lora(self.model,targets,cfg.lora_r,cfg.lora_alpha)
            else:
                logger.info("No adapter backend enabled; training full parameters where supported.")
        except Exception as exc:
            raise RuntimeError(f"Adapter initialization failed: {exc}") from exc

    def _print_parameter_summary(self) -> None:
        """Log trainable/total parameter statistics."""
        if self.model is None:
            return

        try:
            stats = count_params(self.model)

            total = _safe_float(
                stats.get("total", 0)
            )

            trainable = _safe_float(
                stats.get("trainable", 0)
            )

            logger.info(
                "Trainable parameters: %.2fM / %.2fM",
                trainable / 1e6,
                total / 1e6,
            )

            print(
                f"Trainable params: "
                f"{trainable / 1e6:.2f}M / "
                f"{total / 1e6:.2f}M"
            )

        except Exception:
            logger.warning(
                "Unable to compute parameter statistics.",
                exc_info=True,
            )

    # =========================================================================
    # Dataset wrappers
    # =========================================================================

    def _build_datasets(self) -> None:
        """Build FTRAIN dataset wrappers for the selected backend."""
        cfg = self.config

        if cfg.use_grpo:
            self.train_dataset = self.train_data
        else:
            self.train_dataset = FtrainDataset(
                self.train_data,
                self.tokenizer,
                cfg.max_seq_length,
                cfg.use_packing,
            )

        if (
            self.val_data is not None
            and not _is_empty(self.val_data)
        ):
            self.val_dataset = FtrainDataset(
                self.val_data,
                self.tokenizer,
                cfg.max_seq_length,
            )
        else:
            self.val_dataset = None

        if _is_empty(self.train_dataset):
            raise ValueError(
                "Training dataset wrapper is empty."
            )

        self.total_steps = cfg.max_steps

    # =========================================================================
    # Dashboard
    # =========================================================================

    def _start_dashboard(self) -> None:
        """Start optional dashboard in a daemon thread."""
        cfg = self.config

        if not cfg.use_dashboard:
            self.dashboard = None
            return

        from .dashboard import TrainingDashboard

        self.dashboard = TrainingDashboard(
            port=cfg.dashboard_port
        )

        thread = threading.Thread(
            target=self.dashboard.start,
            name="ftrain-dashboard",
            daemon=True,
        )

        thread.start()

        self._dashboard_started = True

    def _stop_dashboard(self) -> None:
        """Stop optional dashboard safely."""
        if self.dashboard is None:
            return

        try:
            self.dashboard.stop()
        except Exception:
            logger.warning(
                "FTRAIN: dashboard shutdown failed.",
                exc_info=True,
            )
        finally:
            self.dashboard = None
            self._dashboard_started = False

    # =========================================================================
    # Evaluation / inference
    # =========================================================================

    def _evaluate_model(
        self,
        prompt: str,
    ) -> str:
        """Generate a deterministic response from the current model."""
        if self.model is None or self.tokenizer is None:
            return "Evaluation unavailable: model/tokenizer missing."

        was_training = self.model.training

        try:
            self.model.eval()

            try:
                from unsloth import FastLanguageModel

                FastLanguageModel.for_inference(
                    self.model
                )
            except Exception:
                logger.debug(
                    "FTRAIN: Unsloth inference preparation unavailable.",
                    exc_info=True,
                )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(
                    self.config.max_seq_length,
                    512,
                ),
            )

            inputs = {
                key: value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in inputs.items()
            }

            input_length = int(
                inputs["input_ids"].shape[-1]
            )

            with torch.inference_mode():
                with self._autocast_context():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            generated = output[
                0,
                input_length:,
            ]

            return self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()

        except Exception as exc:
            logger.warning(
                "FTRAIN: evaluation generation failed: %s",
                exc,
            )
            return "Evaluation failed."

        finally:
            if was_training:
                try:
                    FastLanguageModel.for_training(self.model)
                except Exception:
                    pass
                self.model.train()

    # =========================================================================
    # Optimizer
    # =========================================================================

    def _build_opt(self) -> torch.optim.Optimizer:
        """Build compact semantic groups with correct no-decay handling."""
        cfg=self.config
        if self.model is None: raise RuntimeError("Cannot build optimizer without a model.")
        roles=("early","late","gate","router","lora_a","lora_b","other")
        decay={r:[] for r in roles}; no_decay={r:[] for r in roles}
        layers=max(1,get_num_layers(self.model)); early_cutoff=max(1,layers//3); late_cutoff=max(early_cutoff,(2*layers)//3)
        pattern=re.compile(r"(?:layers|h|block|blocks)\.(\d+)\.")
        mult={"early":float(cfg.layerwise_lr_decay),"late":1.0,"gate":float(cfg.swiglu_gate_boost),"router":float(cfg.moe_router_lr_multiplier),"lora_a":float(cfg.lora_a_lr_mult),"lora_b":float(cfg.lora_b_lr_mult),"other":1.0}
        for name,param in self.model.named_parameters():
            if not param.requires_grad: continue
            lname=name.lower(); role="other"
            if "lora_a" in lname: role="lora_a"
            elif "lora_b" in lname: role="lora_b"
            elif any(x in lname for x in ("router","router_logits","gate_logits","expert_gate")): role="router"
            elif "gate_proj" in lname or ".gate." in lname or lname.endswith(".gate.weight"): role="gate"
            else:
                m=pattern.search(name)
                if m:
                    idx=_safe_int(m.group(1),0)
                    if idx<early_cutoff: role="early"
                    elif idx>=late_cutoff: role="late"
            is_no_decay=(param.ndim<2 or lname.endswith(".bias") or "norm" in lname or "layernorm" in lname or "rmsnorm" in lname)
            (no_decay if is_no_decay else decay)[role].append(param)
        if not any(decay.values()) and not any(no_decay.values()): raise RuntimeError("No trainable parameters were found after adapter setup.")
        groups=[]
        for role in roles:
            lr=cfg.learning_rate*max(0.0,mult[role])
            if decay[role]: groups.append({"params":decay[role],"lr":lr,"initial_lr":lr,"name":role,"weight_decay":0.01,"captain_multiplier":1.0})
            if no_decay[role]: groups.append({"params":no_decay[role],"lr":lr,"initial_lr":lr,"name":role,"weight_decay":0.0,"captain_multiplier":1.0})
        kwargs={"lr":float(cfg.learning_rate),"betas":(0.9,0.999),"eps":1e-8}
        try: self.optimizer=torch.optim.AdamW(groups,**kwargs,**({"fused":True} if self.device.type=="cuda" else {}))
        except (TypeError,RuntimeError): self.optimizer=torch.optim.AdamW(groups,**kwargs)
        return self.optimizer

    @staticmethod
    def _find_layer_index(
        name: str,
        patterns: Sequence[re.Pattern[str]],
    ) -> Optional[int]:
        for pattern in patterns:
            match = pattern.search(name)

            if match:
                try:
                    return int(
                        match.group(1)
                    )
                except (TypeError, ValueError):
                    return None

        return None

    # =========================================================================
    # Autocast
    # =========================================================================

    @contextmanager
    def _autocast_context(self):
        if self.device.type=="cuda":
            try:
                dtype=torch.float16 if self.config.load_in_4bit else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                with torch.autocast(device_type="cuda",dtype=dtype,enabled=True): yield
                return
            except Exception: logger.debug("FTRAIN: CUDA autocast unavailable.",exc_info=True)
        elif self.device.type=="mps":
            try:
                with torch.autocast(device_type="mps",dtype=torch.float16,enabled=True): yield
                return
            except Exception: logger.debug("FTRAIN: MPS autocast unavailable.",exc_info=True)
        yield

    def _amp_enabled(self) -> bool:
        return bool(self.device.type=="cuda" and self.config.load_in_4bit and not torch.cuda.is_bf16_supported())

    def _get_scaler(self):
        if not self._amp_enabled(): return None
        if self._scaler is not None: return self._scaler
        try: self._scaler=torch.amp.GradScaler("cuda",enabled=True)
        except (AttributeError,TypeError): self._scaler=torch.cuda.amp.GradScaler(enabled=True)
        return self._scaler

    # =========================================================================
    # Gradient handling
    # =========================================================================

    def _clear_gradients(self) -> None:
        """Clear model gradients safely."""
        if self.model is None:
            return

        self.model.zero_grad(
            set_to_none=True
        )

        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(
                    set_to_none=True
                )
            except Exception:
                pass

    def _compute_brain_activity(
        self,
    ) -> Tuple[float, float, float]:
        """
        Compute early/late/gate gradient norms.

        This runs before ``optimizer.zero_grad()`` in the custom loop.
        """
        if self.optimizer is None:
            return 0.0, 0.0, 0.0

        squared = {
            "early": 0.0,
            "late": 0.0,
            "gate": 0.0,
        }

        for parameter_group in self.optimizer.param_groups:
            name = parameter_group.get(
                "name",
                "other",
            )

            if name not in squared:
                continue

            for parameter in parameter_group.get(
                "params",
                (),
            ):
                gradient = getattr(
                    parameter,
                    "grad",
                    None,
                )

                if gradient is None:
                    continue

                try:
                    if gradient.is_sparse:
                        gradient = gradient.coalesce().values()

                    value = (
                        gradient.detach()
                        .float()
                    )

                    squared[name] += float(
                        torch.sum(
                            value * value
                        ).item()
                    )

                except Exception:
                    logger.debug(
                        "FTRAIN: gradient activity calculation failed "
                        "for one parameter.",
                        exc_info=True,
                    )

        return (
            math.sqrt(
                max(0.0, squared["early"])
            ),
            math.sqrt(
                max(0.0, squared["late"])
            ),
            math.sqrt(
                max(0.0, squared["gate"])
            ),
        )

    # =========================================================================
    # Training entry point
    # =========================================================================

    def train(self) -> Any:
        cfg=self.config
        if self.model is None: raise RuntimeError("Training cannot begin without a model.")
        if self.train_dataset is None: raise RuntimeError("Training cannot begin without a dataset.")
        self._train_started_at=time.time()
        self.model.train()
        ui.fire_header()
        print(f"🧬 Model: {cfg.model_name} | Steps: {self.total_steps} | Mode: {'GRPO' if cfg.use_grpo else 'SFT'} | Backend: {'HF/Unsloth' if cfg.use_hf_trainer else 'Custom'}")
        eval_prompt,correct_answer=self._select_evaluation_example()
        before_answer=""
        if not cfg.use_grpo and self.captain is not None and eval_prompt:
            print("\n🧠 Captain is asking the model a question before training...")
            before_answer=self._evaluate_model(eval_prompt)
        result=None
        try:
            if cfg.use_grpo: result=self._train_grpo()
            elif cfg.use_hf_trainer: result=self._train_hf()
            else: result=self._train_custom()
        finally:
            self._stop_dashboard()
        if self.captain is not None and eval_prompt and before_answer:
            print("\n🧠 Captain is asking the model the same question after training...")
            after_answer=self._evaluate_model(eval_prompt)
            try:
                self.captain.evaluate_improvement(eval_prompt,before_answer,after_answer,correct_answer)
            except Exception:
                logger.debug("FTRAIN: Captain improvement evaluation failed.",exc_info=True)
        return result

    def _select_evaluation_example(self) -> Tuple[str,str]:
        """Select a deterministic validation/training example for Captain evaluation."""
        source=self.val_data if self.val_data is not None and not _is_empty(self.val_data) else self.train_data
        if source is None: return "",""
        try: length=len(source)
        except TypeError: return "",""
        if length<=0: return "",""
        indices=list(range(length))
        random.Random(int(self.config.seed)+99173).shuffle(indices)
        for index in indices[:min(32,length)]:
            sample=source[index]
            if not isinstance(sample,Mapping): continue
            messages=sample.get("messages")
            if isinstance(messages,Sequence) and not isinstance(messages,(str,bytes)):
                user_message=None; answer=None
                for message in messages:
                    if not isinstance(message,Mapping): continue
                    role=str(message.get("role","")).lower()
                    content=str(message.get("content","")).strip()
                    if role=="user" and content and user_message is None: user_message={"role":"user","content":content}
                    elif role=="assistant" and content and answer is None: answer=content
                    if user_message is not None and answer is not None: break
                if user_message is None: continue
                prompt=str(user_message["content"])
                try: prompt=self.tokenizer.apply_chat_template([user_message],tokenize=False,add_generation_prompt=True)
                except Exception: pass
                return prompt,str(answer or "")
            prompt=sample.get("prompt") or sample.get("question") or sample.get("query")
            answer=sample.get("answer") or sample.get("response") or sample.get("solution")
            if prompt: return str(prompt),str(answer or "")
            text=sample.get("text")
            if isinstance(text,str) and text.strip(): return text.strip(),""
        return "",""

    # =========================================================================
    # GRPO
    # =========================================================================

    def _train_grpo(self) -> Any:
        cfg=self.config
        if self.model is None: raise RuntimeError("GRPO training requires a model.")
        from trl import GRPOConfig,GRPOTrainer
        if not cfg.grpo_reward_funcs: raise ValueError("GRPO training requires at least one reward function.")
        grpo_data=self._build_grpo_dataset()
        if not grpo_data: raise ValueError("No valid GRPO training examples were produced.")
        config_supported=self._supported_parameters(GRPOConfig)
        values={
            "output_dir":cfg.output_dir,"max_steps":cfg.max_steps,"learning_rate":cfg.learning_rate,
            "logging_steps":max(1,cfg.captain_interval),"save_steps":max(1,cfg.checkpoint_interval),
            "per_device_train_batch_size":cfg.per_device_batch_size,"gradient_accumulation_steps":cfg.gradient_accumulation_steps,
            "num_generations":cfg.grpo_num_generations,"max_prompt_length":512,"max_completion_length":1024,
            "temperature":0.7,"beta":0.01,"report_to":"none","remove_unused_columns":False,
            "bf16":bool(self.device.type=="cuda" and not cfg.load_in_4bit and torch.cuda.is_bf16_supported()),
            "fp16":bool(self.device.type=="cuda" and cfg.load_in_4bit and not torch.cuda.is_bf16_supported()),
            "gradient_checkpointing":cfg.gradient_checkpointing_enable,
        }
        args=GRPOConfig(**{k:v for k,v in values.items() if k in config_supported})
        trainer_supported=self._supported_parameters(GRPOTrainer)
        kwargs={"model":self.model,"args":args,"reward_funcs":cfg.grpo_reward_funcs,"train_dataset":grpo_data}
        if "processing_class" in trainer_supported: kwargs["processing_class"]=self.tokenizer
        elif "tokenizer" in trainer_supported: kwargs["tokenizer"]=self.tokenizer
        trainer=GRPOTrainer(**{k:v for k,v in kwargs.items() if k in trainer_supported})
        self._trainer=trainer; self._backend="grpo"
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
        self._sync_trainer_state(trainer)
        return self._finalize_model(trainer.model,mode="GRPO")

    def _build_grpo_dataset(self) -> List[Dict[str, Any]]:
        """Normalize FTRAIN examples to the format expected by GRPO."""
        result: List[Dict[str, Any]] = []

        for example in self.train_data:
            if not isinstance(example, Mapping):
                continue

            if "messages" in example:
                messages = example["messages"]

                if not isinstance(
                    messages,
                    Sequence,
                ):
                    continue

                prompt_messages = [
                    message
                    for message in messages
                    if isinstance(message, Mapping)
                    and message.get("role") != "assistant"
                ]

                if not prompt_messages:
                    continue

                try:
                    prompt = (
                        self.tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                except Exception:
                    prompt = "\n".join(
                        (
                            f"{message.get('role', 'user')}: "
                            f"{message.get('content', '')}"
                        )
                        for message in prompt_messages
                    )

                result.append(
                    {
                        "prompt": prompt,
                        "solution": example.get(
                            "solution",
                            "",
                        ),
                    }
                )

            elif "prompt" in example:
                result.append(
                    dict(example)
                )

        return result

    # =========================================================================
    # Hugging Face / Unsloth Trainer
    # =========================================================================

    def _supported_parameters(self, callable_object: Any) -> set[str]:
        """Return accepted constructor/call parameters across classes and callables."""
        try:
            target = callable_object.__init__ if inspect.isclass(callable_object) else callable_object
            signature = inspect.signature(target)
            names = set(signature.parameters)
            names.discard("self")
            names.discard("args")
            names.discard("kwargs")
            return names
        except Exception:
            return set()

    def _build_training_arguments(self, argument_class: Any) -> Any:
        cfg=self.config
        supported=self._supported_parameters(argument_class)
        kwargs={
            "output_dir":cfg.output_dir,"max_steps":cfg.max_steps,
            "per_device_train_batch_size":cfg.per_device_batch_size,
            "gradient_accumulation_steps":cfg.gradient_accumulation_steps,
            "learning_rate":cfg.learning_rate,"warmup_ratio":cfg.warmup_ratio,
            "warmup_steps":cfg.warmup_steps,"logging_steps":max(1,cfg.captain_interval),
            "save_strategy":"steps","save_steps":max(1,cfg.checkpoint_interval),
            "save_total_limit":max(1,cfg.save_total_limit),
            "gradient_checkpointing":cfg.gradient_checkpointing_enable,
            "dataloader_num_workers":max(0,cfg.dataloader_num_workers),
            "report_to":cfg.report_to,"remove_unused_columns":False,
            "max_grad_norm":cfg.max_grad_norm,"seed":cfg.seed,
            "group_by_length":cfg.group_by_length,
            "bf16":bool(self.device.type=="cuda" and not cfg.load_in_4bit and torch.cuda.is_bf16_supported()),
            "fp16":bool(self.device.type=="cuda" and cfg.load_in_4bit and not torch.cuda.is_bf16_supported()),
        }
        if self.val_dataset is not None:
            if "eval_strategy" in supported: kwargs["eval_strategy"]="steps"
            elif "evaluation_strategy" in supported: kwargs["evaluation_strategy"]="steps"
            if "eval_steps" in supported: kwargs["eval_steps"]=max(1,cfg.eval_interval)
        else:
            if "eval_strategy" in supported: kwargs["eval_strategy"]="no"
            elif "evaluation_strategy" in supported: kwargs["evaluation_strategy"]="no"
        if "gradient_checkpointing_kwargs" in supported:
            kwargs["gradient_checkpointing_kwargs"]={"use_reentrant":False}
        if "optim" in supported:
            kwargs["optim"]="adamw_torch_fused" if self.device.type=="cuda" else "adamw_torch"
        return argument_class(**{k:v for k,v in kwargs.items() if k in supported})

    def _build_unsloth_trainer(self, callback: Any) -> Any:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        supported=self._supported_parameters(UnslothTrainer)
        args=self._build_training_arguments(UnslothTrainingArguments)
        kwargs={
            "model":self.model,"args":args,"train_dataset":self.train_dataset,
            "eval_dataset":self.val_dataset,
            "data_collator":partial(collate,pad_token_id=getattr(self.tokenizer,"pad_token_id",None) or 0),
        }
        if "processing_class" in supported: kwargs["processing_class"]=self.tokenizer
        elif "tokenizer" in supported: kwargs["tokenizer"]=self.tokenizer
        if "optimizers" in supported:
            if self.optimizer is None: self._build_opt()
            kwargs["optimizers"]=(self.optimizer,None)
        if callback is not None and "callbacks" in supported: kwargs["callbacks"]=[callback]
        return UnslothTrainer(**{k:v for k,v in kwargs.items() if k in supported})

    def _build_transformers_trainer(self, callback: Any) -> Any:
        from transformers import Trainer, TrainingArguments
        supported=self._supported_parameters(Trainer)
        args=self._build_training_arguments(TrainingArguments)
        kwargs={
            "model":self.model,"args":args,"train_dataset":self.train_dataset,
            "eval_dataset":self.val_dataset,
            "data_collator":partial(collate,pad_token_id=getattr(self.tokenizer,"pad_token_id",None) or 0),
        }
        if "processing_class" in supported: kwargs["processing_class"]=self.tokenizer
        elif "tokenizer" in supported: kwargs["tokenizer"]=self.tokenizer
        if self.optimizer is None: self._build_opt()
        if "optimizers" in supported: kwargs["optimizers"]=(self.optimizer,None)
        if callback is not None and "callbacks" in supported: kwargs["callbacks"]=[callback]
        return Trainer(**{k:v for k,v in kwargs.items() if k in supported})

    def _sync_trainer_state(self, trainer: Any) -> None:
        state=getattr(trainer,"state",None)
        if state is not None:
            self.step=max(self.step,_safe_int(getattr(state,"global_step",self.step),self.step))
            try: self.epoch=max(self.epoch,int(float(getattr(state,"epoch",self.epoch))))
            except Exception: pass
        self.optimizer=getattr(trainer,"optimizer",self.optimizer)
        self.scheduler=getattr(trainer,"lr_scheduler",self.scheduler)

    def _train_hf(self) -> Any:
        cfg=self.config
        self._build_opt()
        callback=None
        if cfg.captain_enabled:
            try:
                from .callbacks import PhoenixCaptainCallback
                callback=PhoenixCaptainCallback(cfg,self.model,self.tokenizer,self.train_dataset,self.dashboard)
            except Exception:
                logger.warning("FTRAIN: Captain callback unavailable; continuing without it.",exc_info=True)
        trainer=None
        if cfg.use_unsloth_trainer:
            try:
                trainer=self._build_unsloth_trainer(callback)
                self._backend="unsloth"
            except (ImportError,AttributeError,TypeError,ValueError,RuntimeError) as exc:
                logger.warning("FTRAIN: Unsloth Trainer unavailable/incompatible: %s",exc)
        if trainer is None:
            trainer=self._build_transformers_trainer(callback)
            self._backend="transformers"
        self._trainer=trainer
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
        self._sync_trainer_state(trainer)
        return self._finalize_model(trainer.model,mode="SFT")

    # =========================================================================
    # Custom training
    # =========================================================================

    def _train_custom(self) -> torch.nn.Module:
        cfg=self.config; self._build_opt(); self._build_sched()
        if cfg.resume_from_checkpoint: self.load_training_state(cfg.resume_from_checkpoint)
        loader=self._dataloader(self.train_dataset,shuffle=True); iterator=iter(loader)
        accumulated_loss=0.0; accumulated_micro_steps=0; latest_val_loss=self._last_val_loss; latest_grad_norm=0.0; status_message=""
        scaler=self._get_scaler(); self._clear_gradients()
        while self.step<self.total_steps:
            try: batch=next(iterator)
            except StopIteration:
                self.epoch+=1; sampler=getattr(loader,"sampler",None)
                if hasattr(sampler,"set_epoch"): sampler.set_epoch(self.epoch)
                iterator=iter(loader); batch=next(iterator)
            if accumulated_micro_steps==0:
                if cfg.use_adaptive_accumulation:
                    try: self._accumulation_target=max(1,int(adaptive_accumulation(cfg.gradient_accumulation_steps,int(batch["input_ids"].numel()),cfg.target_batch_tokens)))
                    except Exception: self._accumulation_target=max(1,int(cfg.gradient_accumulation_steps))
                else: self._accumulation_target=max(1,int(cfg.gradient_accumulation_steps))
                self._current_accumulation_steps=self._accumulation_target
            input_ids=self._move_to_device(batch.get("input_ids")); attention_mask=self._move_to_device(batch.get("attention_mask")); labels=self._move_to_device(batch.get("labels"))
            if input_ids is None: raise RuntimeError("Training batch contains no input_ids.")
            if attention_mask is None: attention_mask=torch.ones_like(input_ids)
            if labels is None: labels=input_ids
            try:
                with self._autocast_context(): raw_loss=getattr(self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels),"loss",None)
                if raw_loss is None: raise RuntimeError("Model returned no loss during custom training.")
                if not torch.isfinite(raw_loss).all():
                    self._invalid_loss_count+=1; self._skipped_steps+=1; self._clear_gradients(); accumulated_loss=0.0; accumulated_micro_steps=0; continue
                loss=raw_loss/self._accumulation_target
                if scaler is not None: scaler.scale(loss).backward()
                else: loss.backward()
                accumulated_loss+=_safe_float(raw_loss.detach().item()); accumulated_micro_steps+=1
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self._oom_count+=1; self._clear_gradients();
                    if self.device.type=="cuda": torch.cuda.empty_cache()
                    raise RuntimeError("FTRAIN ran out of memory during custom training; reduce batch/sequence length or increase accumulation.") from exc
                raise
            if accumulated_micro_steps<self._accumulation_target: continue
            assert self.optimizer is not None
            if scaler is not None: scaler.unscale_(self.optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg.max_grad_norm)
            latest_grad_norm=_safe_float(grad_norm.item() if isinstance(grad_norm,torch.Tensor) else grad_norm)
            brain_activity=self._compute_brain_activity(); current_lr=self._current_learning_rate()
            if self.captain is not None and self.step%max(1,cfg.captain_interval)==max(1,cfg.captain_interval)-1:
                try:
                    advice=self.captain.inspect_training(self.step+1,accumulated_loss/max(1,accumulated_micro_steps),current_lr,latest_grad_norm,brain_activity,latest_val_loss)
                    if advice:
                        self._apply_captain_advice(advice); status_message=f"{advice.get('action','Keep LR')} (x{_safe_float(advice.get('mult',1.0),1.0):.2f})"
                except Exception: logger.warning("FTRAIN: Captain inspection failed.",exc_info=True)
            if scaler is not None: scaler.step(self.optimizer); scaler.update()
            else: self.optimizer.step()
            if self.scheduler is not None: self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True); self.step+=1
            average_loss=accumulated_loss/max(1,accumulated_micro_steps); self.loss_history.append(average_loss); self._last_loss=average_loss; latest_lr=self._current_learning_rate(); self.lr_history.append(latest_lr)
            if cfg.eval_interval>0 and self.step%cfg.eval_interval==0 and self.val_dataset is not None: latest_val_loss=self.validate()
            ui.print_train_table(self.step,self.total_steps,average_loss,latest_val_loss,latest_lr,latest_grad_norm,status_message)
            if self.dashboard is not None:
                try: self.dashboard.log_metric(self.step,average_loss,latest_lr,latest_val_loss)
                except Exception: logger.debug("FTRAIN: dashboard metric logging failed.",exc_info=True)
            if cfg.checkpoint_interval>0 and self.step%cfg.checkpoint_interval==0: self.save_checkpoint(self.step)
            accumulated_loss=0.0; accumulated_micro_steps=0
        return self._finalize_model(self.model,mode="SFT")

    # =========================================================================
    # Captain / optimizer interaction
    # =========================================================================

    def _apply_captain_advice(
        self,
        advice: Mapping[str, Any],
    ) -> None:
        """
        Store Captain state.

        The custom scheduler reads these values when constructing LR factors.
        This is intentionally kept separate from direct optimizer mutation so
        we do not create two competing LR controllers.
        """
        multiplier = _safe_float(
            advice.get(
                "mult",
                1.0,
            ),
            1.0,
        )

        clamp = getattr(self.config, "captain_clamp", [0.25, 2.5])
        if isinstance(clamp, (list, tuple)) and len(clamp) >= 2:
            low = _safe_float(clamp[0], 0.25)
            high = _safe_float(clamp[1], 2.5)
        else:
            low = _safe_float(getattr(self.config, "captain_mult_min", 0.25), 0.25)
            high = _safe_float(getattr(self.config, "captain_mult_max", 2.5), 2.5)

        multiplier = max(
            low,
            min(
                high,
                multiplier,
            ),
        )

        self._captain_mult = multiplier

        layer = str(
            advice.get(
                "layer_boost",
                "none",
            )
        ).strip().lower()

        boost = _safe_float(
            self.config.captain_layer_boost,
            2.0,
        )

        self._captain_layer_boosts = {
            group: 1.0
            for group in self._captain_layer_boosts
        }

        if layer == "all":
            self._captain_layer_boosts = {
                group: boost
                for group in self._captain_layer_boosts
            }

        elif layer in self._captain_layer_boosts:
            self._captain_layer_boosts[
                layer
            ] = boost

    # =========================================================================
    # Scheduler
    # =========================================================================

    def _build_sched(self) -> Any:
        """
        Build the custom scheduler.

        Captain scaling is represented inside LambdaLR rather than manually
        changing optimizer LR from two different places.
        """
        cfg = self.config

        if self.optimizer is None:
            raise RuntimeError(
                "Cannot build scheduler before optimizer."
            )

        total_steps = max(
            1,
            self.total_steps,
        )

        warmup = (
            cfg.warmup_steps
            if cfg.warmup_steps > 0
            else int(
                cfg.warmup_ratio
                * total_steps
            )
        )

        warmup = max(
            0,
            min(
                warmup,
                total_steps,
            ),
        )

        if cfg.use_cosine_restarts:
            self.scheduler = cosine_restart_scheduler(
                self.optimizer,
                cfg.learning_rate,
                cfg.learning_rate
                * cfg.min_lr_ratio,
                warmup,
                total_steps,
                cfg.restart_interval,
            )

            return self.scheduler

        lr_lambdas = []

        for parameter_group in self.optimizer.param_groups:
            group_name = parameter_group.get(
                "name",
                "other",
            )

            def make_lambda(
                name: str,
            ):
                def lr_lambda(
                    current_step: int,
                ) -> float:
                    if current_step < warmup:
                        base_factor = (
                            current_step
                            / max(
                                1,
                                warmup,
                            )
                        )

                    else:
                        progress = min(
                            1.0,
                            max(
                                0.0,
                                (
                                    current_step
                                    - warmup
                                )
                                / max(
                                    1,
                                    total_steps
                                    - warmup,
                                ),
                            ),
                        )

                        base_factor = (
                            cfg.min_lr_ratio
                            + (
                                1.0
                                - cfg.min_lr_ratio
                            )
                            * 0.5
                            * (
                                1.0
                                + math.cos(
                                    math.pi
                                    * progress
                                )
                            )
                        )

                    captain_factor = (
                        self._captain_mult
                    )

                    layer_factor = (
                        self._captain_layer_boosts.get(
                            name,
                            1.0,
                        )
                    )

                    return (
                        base_factor
                        * captain_factor
                        * layer_factor
                    )

                return lr_lambda

            lr_lambdas.append(
                make_lambda(
                    group_name
                )
            )

        self.scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambdas,
            )
        )

        return self.scheduler

    # =========================================================================
    # Dataloader
    # =========================================================================

    def _dataloader(
        self,
        dataset: Any,
        shuffle: bool = True,
    ) -> DataLoader:
        """Construct the FTRAIN dataloader."""
        if dataset is None:
            raise ValueError(
                "Cannot create DataLoader from None."
            )

        lengths = getattr(
            dataset,
            "lengths",
            None,
        )

        if lengths is None:
            # Fallback to sequential/random sampling when Dataset does not
            # expose the optimized FTRAIN length metadata.
            generator = torch.Generator()

            generator.manual_seed(
                self.config.seed
                + self.epoch
            )

            workers = max(0, int(self.config.dataloader_num_workers))
            return DataLoader(
                dataset,
                batch_size=(
                    max(1, int(self.config.per_device_batch_size))
                ),
                shuffle=shuffle,
                collate_fn=partial(
                    collate,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        or 0
                    ),
                ),
                num_workers=workers,
                persistent_workers=workers > 0,
                pin_memory=(
                    self.config.pin_memory
                    and self.device.type == "cuda"
                ),
                generator=generator,
            )

        workers = max(0, int(self.config.dataloader_num_workers))
        sampler = LengthSampler(
            lengths,
            max(1, int(self.config.per_device_batch_size)),
            shuffle,
            self.config.seed
            + self.epoch,
        )

        return DataLoader(
            dataset,
            batch_size=max(1, int(self.config.per_device_batch_size)),
            sampler=sampler,
            collate_fn=partial(
                collate,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    or 0
                ),
            ),
            num_workers=self.config.dataloader_num_workers,
            pin_memory=(
                self.config.pin_memory
                and self.device.type == "cuda"
            ),
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> Optional[float]:
        """Evaluate average validation loss without modifying training state."""
        if self.val_dataset is None:
            return None

        if self.model is None:
            return None

        was_training = self.model.training

        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        try:
            loader = self._dataloader(
                self.val_dataset,
                shuffle=False,
            )

            with torch.inference_mode():
                for batch in loader:
                    input_ids = self._move_to_device(
                        batch.get(
                            "input_ids"
                        )
                    )

                    if input_ids is None:
                        continue

                    attention_mask = batch.get(
                        "attention_mask"
                    )

                    if attention_mask is None:
                        attention_mask = (
                            torch.ones_like(
                                input_ids
                            )
                        )

                    else:
                        attention_mask = (
                            self._move_to_device(
                                attention_mask
                            )
                        )

                    labels = batch.get(
                        "labels"
                    )

                    if labels is None:
                        labels = input_ids

                    else:
                        labels = self._move_to_device(
                            labels
                        )

                    with self._autocast_context():
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                    if output.loss is None:
                        continue

                    loss = _safe_float(
                        output.loss.item()
                    )

                    if not math.isfinite(loss):
                        logger.warning(
                            "FTRAIN: non-finite validation loss ignored."
                        )
                        continue

                    total_loss += loss
                    batch_count += 1

            result = (
                total_loss
                / max(
                    1,
                    batch_count,
                )
            )

            self._last_val_loss = result

            return result

        finally:
            if was_training:
                self.model.train()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(
        self,
        step: int,
        final: bool = False,
    ) -> str:
        """
        Save model, tokenizer and training state.

        Unlike the original implementation, optimizer/scheduler state and
        runtime metadata are also persisted so resume can actually restore the
        training process rather than merely reload model weights.
        """
        if self.model is None:
            raise RuntimeError(
                "Cannot checkpoint without a model."
            )

        cfg = self.config

        tag = (
            "final"
            if final
            else f"step_{int(step)}"
        )

        root = (
            Path(cfg.output_dir).expanduser()
            / "checkpoints"
        )

        path = root / tag

        with self._checkpoint_lock:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.model.save_pretrained(
                str(path)
            )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(
                    str(path)
                )

            # -----------------------------------------------------------
            # Optimizer / scheduler state
            # -----------------------------------------------------------

            if self.optimizer is not None:
                torch.save(
                    self.optimizer.state_dict(),
                    path / "optimizer.pt",
                )

            if self.scheduler is not None:
                try:
                    torch.save(
                        self.scheduler.state_dict(),
                        path / "scheduler.pt",
                    )
                except Exception:
                    logger.warning(
                        "FTRAIN: scheduler state could not be serialized.",
                        exc_info=True,
                    )

            if self._scaler is not None:
                try:
                    torch.save(self._scaler.state_dict(), path / "scaler.pt")
                except Exception:
                    logger.debug("FTRAIN: AMP scaler state could not be serialized.", exc_info=True)

            # -----------------------------------------------------------
            # FTRAIN runtime state
            # -----------------------------------------------------------

            runtime_state = {
                "version": "core-v2",
                "step": int(self.step),
                "epoch": int(self.epoch),
                "loss_history": list(
                    self.loss_history[-1000:]
                ),
                "last_loss": self._last_loss,
                "last_val_loss": self._last_val_loss,
                "captain_mult": self._captain_mult,
                "captain_layer_boosts": dict(
                    self._captain_layer_boosts
                ),
                "current_accumulation_steps": (
                    self._current_accumulation_steps
                ),
                "model_name": cfg.model_name,
                "family": self.family,
                "backend": self._backend,
                "invalid_loss_count": self._invalid_loss_count,
                "oom_count": self._oom_count,
            }

            with (
                path / "ftrain_state.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    runtime_state,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
                file.write("\n")

            # -----------------------------------------------------------
            # Checkpoint retention
            # -----------------------------------------------------------

            if not final:
                self._prune_checkpoints(
                    root
                )

            self._last_checkpoint_step = int(step)
            self._last_checkpoint_time = time.time()
            print(
                f"💾 checkpoint → {path}"
            )

        return str(path)

    def _prune_checkpoints(
        self,
        checkpoint_root: Path,
    ) -> None:
        """Remove oldest step checkpoints according to save_total_limit."""
        candidates: List[Tuple[int, Path]] = []

        for entry in checkpoint_root.iterdir():
            if not entry.is_dir():
                continue

            match = re.fullmatch(
                r"step_(\d+)",
                entry.name,
            )

            if match:
                candidates.append(
                    (
                        int(
                            match.group(1)
                        ),
                        entry,
                    )
                )

        candidates.sort(
            key=lambda item: item[0]
        )

        while len(candidates) > self.config.save_total_limit:
            _, old_path = candidates.pop(0)

            try:
                shutil.rmtree(
                    old_path
                )
            except OSError:
                logger.warning(
                    "FTRAIN: failed to remove old checkpoint %s.",
                    old_path,
                    exc_info=True,
                )

    # =========================================================================
    # Resume state
    # =========================================================================

    def load_training_state(
        self,
        checkpoint: Optional[str] = None,
    ) -> bool:
        """
        Restore optimizer/scheduler/FTRAIN state from a checkpoint.

        Model weights are normally loaded by the Trainer/model loader itself;
        this method restores the training-specific state.
        """
        checkpoint_path = Path(
            checkpoint
            or self.config.resume_from_checkpoint
            or ""
        ).expanduser()

        if not checkpoint_path.is_dir():
            return False

        state_file = (
            checkpoint_path
            / "ftrain_state.json"
        )

        if state_file.exists():
            try:
                with state_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    state = json.load(file)

                self.step = max(
                    0,
                    _safe_int(
                        state.get(
                            "step",
                            0,
                        )
                    ),
                )

                self.epoch = max(
                    0,
                    _safe_int(
                        state.get(
                            "epoch",
                            0,
                        )
                    ),
                )

                loss_history = state.get(
                    "loss_history",
                    [],
                )

                if isinstance(
                    loss_history,
                    list,
                ):
                    self.loss_history = [
                        _safe_float(value)
                        for value in loss_history
                        if _is_finite(value)
                    ]

                self._last_loss = state.get(
                    "last_loss"
                )

                self._last_val_loss = state.get(
                    "last_val_loss"
                )
                self._backend = str(state.get("backend", self._backend))
                self._invalid_loss_count = max(0, _safe_int(state.get("invalid_loss_count", 0), 0))
                self._oom_count = max(0, _safe_int(state.get("oom_count", 0), 0))

                self._captain_mult = _safe_float(
                    state.get(
                        "captain_mult",
                        1.0,
                    ),
                    1.0,
                )

                boosts = state.get(
                    "captain_layer_boosts",
                    {},
                )

                if isinstance(
                    boosts,
                    Mapping,
                ):
                    for key in self._captain_layer_boosts:
                        self._captain_layer_boosts[
                            key
                        ] = _safe_float(
                            boosts.get(
                                key,
                                1.0,
                            ),
                            1.0,
                        )

            except Exception:
                logger.warning(
                    "FTRAIN: checkpoint runtime state could not be restored.",
                    exc_info=True,
                )

        optimizer_file = (
            checkpoint_path
            / "optimizer.pt"
        )

        if (
            optimizer_file.exists()
            and self.optimizer is not None
        ):
            try:
                state_dict = torch.load(
                    optimizer_file,
                    map_location=self.device,
                )

                self.optimizer.load_state_dict(
                    state_dict
                )

            except Exception:
                logger.warning(
                    "FTRAIN: optimizer state could not be restored.",
                    exc_info=True,
                )

        scheduler_file = (
            checkpoint_path
            / "scheduler.pt"
        )

        if (
            scheduler_file.exists()
            and self.scheduler is not None
        ):
            try:
                state_dict = torch.load(
                    scheduler_file,
                    map_location="cpu",
                )

                self.scheduler.load_state_dict(
                    state_dict
                )

            except Exception:
                logger.warning(
                    "FTRAIN: scheduler state could not be restored.",
                    exc_info=True,
                )

        scaler_file = checkpoint_path / "scaler.pt"
        if scaler_file.exists() and self._scaler is not None:
            try:
                self._scaler.load_state_dict(torch.load(scaler_file, map_location="cpu"))
            except Exception:
                logger.debug("FTRAIN: AMP scaler state could not be restored.", exc_info=True)

        self._last_checkpoint_step = self.step
        return True

    def _optimizer_state_to_device(self) -> None:
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for key,value in list(state.items()):
                if isinstance(value,torch.Tensor):
                    state[key]=value.to(self.device)

    # =========================================================================
    # Finalization
    # =========================================================================

    def _finalize_model(
        self,
        model: torch.nn.Module,
        mode: str,
    ) -> torch.nn.Module:
        """Save final model and produce the standard FTRAIN summary."""
        self.model = model

        final_path = (
            Path(
                self.config.output_dir
            ).expanduser()
            / "final"
        )

        final_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            str(final_path)
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(
                str(final_path)
            )

        metadata = {
            "version": "ftrain-core-v4",
            "mode": mode,
            "backend": self._backend,
            "step": self.step,
            "epoch": self.epoch,
            "final_loss": self._last_loss,
            "validation_loss": self._last_val_loss,
            "best_validation_loss": self._best_val_loss,
            "skipped_steps": self._skipped_steps,
            "invalid_loss_count": self._invalid_loss_count,
            "oom_count": self._oom_count,
        }
        with (final_path / "ftrain_final_state.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
            file.write("\n")

        # Also maintain FTRAIN's checkpoint structure.
        self.save_checkpoint(self.step, final=True)

        ui.print_final_summary(
            {
                "Model": self.config.model_name,
                "Steps": self.total_steps,
                "Mode": mode,
                "Dir": str(final_path),
            }
        )

        return self.model

    # =========================================================================
    # Device helpers
    # =========================================================================

    def _move_to_device(
        self,
        value: Any,
    ) -> Any:
        """Move tensors recursively to the engine device."""
        if value is None:
            return None

        if isinstance(
            value,
            torch.Tensor,
        ):
            return value.to(
                self.device,
                non_blocking=(
                    self.config.pin_memory
                    and self.device.type == "cuda"
                ),
            )

        if isinstance(
            value,
            Mapping,
        ):
            return {
                key: self._move_to_device(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            converted = [
                self._move_to_device(item)
                for item in value
            ]

            return (
                type(value)(converted)
                if isinstance(
                    value,
                    tuple,
                )
                else converted
            )

        return value

    def _current_learning_rate(self) -> float:
        """Get a representative current optimizer learning rate."""
        if self.optimizer is None:
            return float(
                self.config.learning_rate
            )

        values = [
            _safe_float(
                group.get(
                    "lr",
                    self.config.learning_rate,
                ),
                self.config.learning_rate,
            )
            for group in self.optimizer.param_groups
        ]

        return (
            sum(values) / len(values)
            if values
            else float(self.config.learning_rate)
        )
