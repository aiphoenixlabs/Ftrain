"""
FTRAIN Core Training Engine
===========================

Production-oriented training orchestration for FTRAIN.

This version focuses on correctness first, then performance, resilience,
and adaptive training intelligence.

Backends
--------
- Unsloth Trainer (preferred when available)
- Hugging Face Trainer fallback
- Native FTRAIN custom loop
- GRPO through TRL/Unsloth

Key guarantees
--------------
- No missing evaluation method / unreachable method definitions.
- Safe CPU/CUDA/MPS behavior.
- Explicit train/eval mode restoration.
- Stable gradient accumulation, including adaptive accumulation.
- Grouped optimizer parameter groups instead of one optimizer group per tensor.
- Captain advice is represented without creating competing LR controllers.
- Version-tolerant Trainer construction using runtime signature inspection.
- Real runtime metrics and checkpoint state.
- Deterministic validation and evaluation where possible.
- Optional automatic LoRA target discovery.
- Safer mixed precision and finite-loss/gradient handling.
- Clean final model export under ``output_dir/final``.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers is optional until training
    TrainerCallback = object

from unsloth import FastLanguageModel

from . import ui
from .captain import PhoenixCaptain
from .config import TrainConfig
from .data_quality import balance_datasets, deduplicate, filter_by_perplexity
from .dataset import FtrainDataset, LengthSampler, collate
from .families import get_preset
from .lora import inject as inject_lora
from .lora_dora import inject_dora
from .model_utils import count_params, get_family, get_num_layers, is_moe, seed_everything
from .speed import flash_mode
from .train_optim import adaptive_accumulation, cosine_restart_scheduler

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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, _safe_float(value, low)))


# =============================================================================
# Core engine
# =============================================================================


class Ftrain:
    """Main FTRAIN training engine."""

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

        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.lr_history: List[float] = []

        self.step = 0
        self.epoch = 0
        self._last_loss: Optional[float] = None
        self._last_val_loss: Optional[float] = None
        self._captain_mult = 1.0
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
        self._hf_trainer: Any = None
        self._hf_scheduler: Any = None
        self._hf_optimizer: Optional[torch.optim.Optimizer] = None
        self.train_dataset: Any = None
        self.val_dataset: Any = None
        self.dashboard: Any = None
        self.captain: Optional[PhoenixCaptain] = None
        self.total_steps = max(1, int(config.max_steps))
        self._checkpoint_lock = threading.Lock()
        self._dashboard_started = False
        self._current_accumulation_steps = max(1, int(config.gradient_accumulation_steps))
        self._accumulation_target = self._current_accumulation_steps
        self._best_val_loss: Optional[float] = None
        self._bad_eval_count = 0
        self._skipped_steps = 0

        self._configure_runtime()
        self._resolve_auto_resume()

        self.family = config.family if config.family != "auto" else get_family(config.model_name)
        self.preset = get_preset(self.family) or {}

        if not getattr(config, "lora_target_modules", None) and self.preset.get("lora_targets"):
            config.lora_target_modules = list(self.preset["lora_targets"])

        self._load_model()

        if config.captain_enabled:
            try:
                self.captain = PhoenixCaptain(config)
                self.captain.set_family_context(self.family, is_moe(self.model))
                self.captain.analyze_model(self.model)
            except Exception:
                logger.warning("FTRAIN: Captain initialization failed; continuing without Captain.", exc_info=True)
                self.captain = None

        self._prepare_data()

        if config.auto_lora_targets:
            self._discover_lora_targets()

        self._apply_adapters()
        self._print_parameter_summary()
        self._build_datasets()
        self._start_dashboard()

        Path(config.output_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Environment
    # =========================================================================

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _configure_runtime(self) -> None:
        try:
            flash_mode(enabled=True, tf32=self.device.type == "cuda")
        except Exception:
            logger.debug("FTRAIN: optimized kernel configuration unavailable.", exc_info=True)

        seed_everything(self.config.seed)

        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                logger.debug("FTRAIN: TF32 configuration unavailable.", exc_info=True)

        if self.config.use_grpo:
            try:
                from unsloth import PatchFastRL
                PatchFastRL("GRPO", FastLanguageModel)
                logger.info("FTRAIN: Unsloth GRPO patch enabled.")
            except Exception:
                logger.warning("FTRAIN: GRPO patching unavailable.", exc_info=True)

    # =========================================================================
    # Resume
    # =========================================================================

    def _resolve_auto_resume(self) -> None:
        """Find the newest compatible FTRAIN/HF-style checkpoint."""
        if not self.config.auto_resume:
            return

        roots = [
            Path(self.config.output_dir).expanduser() / "checkpoints",
            Path(self.config.output_dir).expanduser(),
        ]

        candidates: List[Tuple[int, Path]] = []
        seen: set[str] = set()

        for root in roots:
            if not root.is_dir():
                continue

            try:
                entries = list(root.iterdir())
            except OSError:
                continue

            for entry in entries:
                if not entry.is_dir():
                    continue

                name = entry.name
                match = re.fullmatch(r"(?:step_|checkpoint-)(\d+)", name)
                if not match:
                    continue

                step = int(match.group(1))
                resolved = str(entry.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)

                has_model = (
                    (entry / "config.json").exists()
                    or (entry / "adapter_config.json").exists()
                    or (entry / "model.safetensors").exists()
                    or (entry / "pytorch_model.bin").exists()
                    or any(entry.glob("*.safetensors"))
                )

                # HF/Trainer checkpoints may contain only sharded model files
                # and trainer_state.json; accept those as valid resumable state.
                has_trainer_state = (entry / "trainer_state.json").exists()

                if has_model or has_trainer_state:
                    candidates.append((step, entry))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            step, checkpoint = candidates[-1]
            self.config.resume_from_checkpoint = str(checkpoint)
            logger.info(
                "FTRAIN: auto-resume selected %s (step %d).",
                checkpoint,
                step,
            )

    # =========================================================================
    # Model loading
    # =========================================================================

    @contextmanager
    def _quiet_stdout(self):
        previous = sys.stdout
        buffer = io.StringIO()
        try:
            sys.stdout = buffer
            yield buffer
        finally:
            sys.stdout = previous

    def _preferred_model_dtype(self) -> torch.dtype:
        if self.device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _load_model(self) -> None:
        cfg = self.config
        bar = ui.LoadingBar(message=f"Loading {cfg.model_name}", real_progress=cfg.show_model_progress)
        bar.start()
        try:
            kwargs: Dict[str, Any] = {
                "model_name": cfg.model_name,
                "max_seq_length": cfg.max_seq_length,
                "load_in_4bit": cfg.load_in_4bit,
            }
            if not cfg.load_in_4bit:
                kwargs["dtype"] = self._preferred_model_dtype()

            attention_impl = self.preset.get("attn_implementation")
            if attention_impl:
                kwargs["attn_implementation"] = attention_impl

            try:
                with self._quiet_stdout():
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(**kwargs)
            except Exception as unsloth_error:
                logger.warning("FTRAIN: Unsloth loader failed: %s", unsloth_error)
                self._load_model_transformers()

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model loading did not produce both model and tokenizer.")

            self._prepare_tokenizer()
            self._prepare_model()
        finally:
            bar.done()

    def _load_model_transformers(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {
            "torch_dtype": self._preferred_model_dtype(),
        }
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def _prepare_tokenizer(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer was not loaded.")

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos
            else:
                logger.warning("Tokenizer has neither pad_token nor eos_token.")

        try:
            self.tokenizer.padding_side = "right"
        except Exception:
            pass

    def _prepare_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is unavailable.")

        try:
            has_device_map = bool(getattr(self.model, "hf_device_map", None))
            if not has_device_map:
                self.model.to(self.device)
        except Exception:
            logger.debug("FTRAIN: model device preparation skipped.", exc_info=True)

        if self.config.gradient_checkpointing_enable:
            try:
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
            except Exception:
                logger.warning("FTRAIN: gradient checkpointing could not be enabled.", exc_info=True)

    # =========================================================================
    # Dataset processing
    # =========================================================================

    def _prepare_data(self) -> None:
        cfg = self.config
        if self.train_data is None:
            raise ValueError("Training data cannot be None.")

        train_len = _safe_len(self.train_data)
        if train_len == 0:
            raise ValueError("Training dataset is empty.")

        needs_processing = bool(cfg.data_perplexity_filter or cfg.data_dedup or cfg.data_sources)
        if not needs_processing:
            return

        original_length = train_len
        changes: List[str] = []

        if cfg.data_dedup:
            before = _safe_len(self.train_data) or 0
            self.train_data = deduplicate(self.train_data)
            after = _safe_len(self.train_data) or 0
            if after < before:
                changes.append(f"Deduplication removed {before - after} duplicates.")

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
                changes.append(f"Perplexity filter removed {before - after} samples.")

        if cfg.data_sources:
            from .data_utils import load_data
            datasets = [self.train_data]
            for source in cfg.data_sources:
                datasets.append(load_data(source))
            self.train_data = balance_datasets(datasets, cfg.data_balance_strategy)
            changes.append("Balanced multiple data sources.")

        final_length = _safe_len(self.train_data) or 0
        if final_length == 0:
            raise ValueError("Dataset processing removed all training samples.")

        if self.captain is not None:
            try:
                self.captain.analyze_and_report_data(original_length, final_length, changes)
            except Exception:
                logger.debug("FTRAIN: Captain data report failed.", exc_info=True)

    # =========================================================================
    # LoRA target discovery
    # =========================================================================

    def _discover_lora_targets(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LoRA target discovery requires model and tokenizer.")

        self.model.train()
        self._clear_gradients()
        encoded = self.tokenizer("Test", return_tensors="pt")
        encoded = self._move_to_device(encoded)

        try:
            with self._autocast_context():
                output = self.model(**encoded, labels=encoded["input_ids"])
            loss = getattr(output, "loss", None)
            if loss is None or not torch.isfinite(loss):
                return

            loss.backward()
            scores: Dict[str, float] = {}

            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                lname = name.lower()
                if "lora_" in lname or "dora" in lname or "magnitude" in lname:
                    continue

                module_name = self._extract_target_module_name(name)
                if not module_name:
                    continue

                try:
                    score = parameter.grad.detach().float().norm().item()
                except Exception:
                    continue
                if not math.isfinite(score):
                    continue
                scores[module_name] = scores.get(module_name, 0.0) + score

            if scores:
                target_count = max(1, int(self.config.lora_target_count))
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected = [name for name, _ in ranked[:target_count]]
                if selected:
                    self.config.lora_target_modules = selected
                    logger.info("FTRAIN: auto LoRA targets: %s", ", ".join(selected))
        finally:
            self._clear_gradients()

    @staticmethod
    def _extract_target_module_name(parameter_name: str) -> Optional[str]:
        parts = parameter_name.split(".")
        if len(parts) < 2:
            return None
        candidate = parts[-2]
        if candidate.isdigit():
            return None
        return candidate

    # =========================================================================
    # Adapter setup
    # =========================================================================

    def _apply_adapters(self) -> None:
        cfg = self.config
        if self.model is None:
            raise RuntimeError("Cannot apply adapters without a model.")

        targets = list(cfg.lora_target_modules) if cfg.lora_target_modules else None
        if not targets:
            raise ValueError(
                "No LoRA target modules are configured. Provide lora_target_modules, "
                "enable auto_lora_targets, or use a model family with presets."
            )

        try:
            if cfg.use_unsloth_lora:
                kwargs = {
                    "r": cfg.lora_r,
                    "lora_alpha": cfg.lora_alpha,
                    "target_modules": targets,
                }
                if cfg.use_dora:
                    kwargs["use_dora"] = True
                self.model = FastLanguageModel.get_peft_model(self.model, **kwargs)
            elif cfg.use_custom_lora:
                if cfg.use_dora:
                    self.model = inject_dora(self.model, targets, cfg.lora_r, cfg.lora_alpha)
                else:
                    self.model = inject_lora(self.model, targets, cfg.lora_r, cfg.lora_alpha)
            else:
                logger.info("FTRAIN: adapter backend disabled; full-parameter training selected.")
        except Exception as exc:
            raise RuntimeError(f"Adapter initialization failed: {exc}") from exc

    def _print_parameter_summary(self) -> None:
        if self.model is None:
            return
        try:
            stats = count_params(self.model)
            total = _safe_float(stats.get("total", 0))
            trainable = _safe_float(stats.get("trainable", 0))
            percent = 100.0 * trainable / max(total, 1.0)
            print(
                f"Trainable params: {trainable / 1e6:.2f}M / "
                f"{total / 1e6:.2f}M ({percent:.2f}%)"
            )
        except Exception:
            logger.warning("FTRAIN: parameter summary failed.", exc_info=True)

    # =========================================================================
    # Dataset wrappers / dashboard
    # =========================================================================

    def _build_datasets(self) -> None:
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

        if self.val_data is not None and not _is_empty(self.val_data):
            self.val_dataset = FtrainDataset(
                self.val_data,
                self.tokenizer,
                cfg.max_seq_length,
            )
        else:
            self.val_dataset = None

        if _is_empty(self.train_dataset):
            raise ValueError("Training dataset wrapper is empty.")
        self.total_steps = max(1, int(cfg.max_steps))

    def _start_dashboard(self) -> None:
        if not self.config.use_dashboard:
            return
        try:
            from .dashboard import TrainingDashboard
            self.dashboard = TrainingDashboard(port=self.config.dashboard_port)
            thread = threading.Thread(target=self.dashboard.start, name="ftrain-dashboard", daemon=True)
            thread.start()
            self._dashboard_started = True
        except Exception:
            self.dashboard = None
            logger.warning("FTRAIN: dashboard could not start.", exc_info=True)

    def _stop_dashboard(self) -> None:
        if self.dashboard is None:
            return
        try:
            self.dashboard.stop()
        except Exception:
            logger.debug("FTRAIN: dashboard shutdown failed.", exc_info=True)
        finally:
            self.dashboard = None
            self._dashboard_started = False

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _select_evaluation_example(self) -> Tuple[str, str]:
        """
        Select a deterministic, usable evaluation example.

        IMPORTANT FIX:
        This method is intentionally defined at class scope. The previous
        file accidentally placed it inside ``train()`` after ``return result``,
        which made it unreachable and caused AttributeError.
        """
        source = self.val_data
        if source is None or _is_empty(source):
            source = self.train_data

        if source is None:
            return "", ""

        try:
            length = len(source)
        except TypeError:
            return "", ""

        if length <= 0:
            return "", ""

        indices = list(range(length))
        rng = random.Random(int(self.config.seed) + 99173)
        rng.shuffle(indices)

        for index in indices[: min(32, length)]:
            sample = source[index]
            if not isinstance(sample, Mapping):
                continue

            messages = sample.get("messages")
            if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
                user_message = None
                assistant_message = None
                for message in messages:
                    if not isinstance(message, Mapping):
                        continue
                    role = str(message.get("role", "")).lower()
                    content = str(message.get("content", "")).strip()
                    if role == "user" and content and user_message is None:
                        user_message = {"role": "user", "content": content}
                    elif role == "assistant" and content and assistant_message is None:
                        assistant_message = content
                    if user_message is not None and assistant_message is not None:
                        break

                if user_message is None:
                    continue

                prompt = user_message["content"]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        [user_message],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass

                return str(prompt), str(assistant_message or "")

            prompt = sample.get("prompt") or sample.get("question") or sample.get("query")
            answer = sample.get("answer") or sample.get("response") or sample.get("solution")
            if prompt:
                return str(prompt), str(answer or "")

            text = sample.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip(), ""

        return "", ""

    def _evaluate_model(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            return "Evaluation unavailable: model/tokenizer missing."

        was_training = bool(self.model.training)
        try:
            self.model.eval()
            try:
                FastLanguageModel.for_inference(self.model)
            except Exception:
                pass

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(int(self.config.max_seq_length), 512),
            )
            inputs = self._move_to_device(inputs)
            input_length = int(inputs["input_ids"].shape[-1])

            with torch.inference_mode(), self._autocast_context():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = output[0, input_length:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:
            logger.warning("FTRAIN: evaluation generation failed: %s", exc)
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
        """Build grouped AdamW with semantic LR groups and proper no-decay handling."""
        cfg = self.config
        if self.model is None:
            raise RuntimeError("Cannot build optimizer without a model.")

        buckets: Dict[str, Dict[str, List[torch.nn.Parameter]]] = {
            role: {"decay": [], "nodecay": []}
            for role in ("early", "late", "gate", "router", "lora_a", "lora_b", "other")
        }

        layers = max(1, get_num_layers(self.model))
        early_cutoff = max(1, layers // 3)
        late_cutoff = max(early_cutoff, (2 * layers) // 3)
        layer_pattern = re.compile(r"(?:layers|h|block|blocks)\.(\d+)\.")

        def role_for(name: str) -> str:
            lname = name.lower()

            if "lora_a" in lname:
                return "lora_a"
            if "lora_b" in lname:
                return "lora_b"
            if "magnitude" in lname:
                return "other"
            if (
                "router" in lname
                or "router_logits" in lname
                or "gate_logits" in lname
                or "expert_gate" in lname
            ):
                return "router"
            if "gate_proj" in lname or ".gate." in lname or lname.endswith(".gate.weight"):
                return "gate"

            match = layer_pattern.search(name)
            if match:
                idx = _safe_int(match.group(1), 0)
                if idx < early_cutoff:
                    return "early"
                if idx >= late_cutoff:
                    return "late"

            return "other"

        def needs_decay(name: str, parameter: torch.nn.Parameter) -> bool:
            lname = name.lower()
            if parameter.ndim < 2:
                return False
            if lname.endswith(".bias") or ".bias" in lname:
                return False
            if any(token in lname for token in ("norm", "layernorm", "rmsnorm", "ln_f")):
                return False
            return True

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            role = role_for(name)
            bucket = "decay" if needs_decay(name, parameter) else "nodecay"
            buckets[role][bucket].append(parameter)

        if not any(params for role in buckets.values() for params in role.values()):
            raise RuntimeError("No trainable parameters were found after adapter setup.")

        base_lr = float(cfg.learning_rate)
        role_lr = {
            "early": base_lr * float(cfg.layerwise_lr_decay),
            "late": base_lr,
            "gate": base_lr * float(cfg.swiglu_gate_boost),
            "router": base_lr * float(cfg.moe_router_lr_multiplier),
            "lora_a": base_lr * float(cfg.lora_a_lr_mult),
            "lora_b": base_lr * float(cfg.lora_b_lr_mult),
            "other": base_lr,
        }

        groups: List[Dict[str, Any]] = []

        def add_group(role: str, decay_kind: str) -> None:
            params = buckets[role][decay_kind]
            if not params:
                return
            groups.append(
                {
                    "params": params,
                    "lr": role_lr[role],
                    "initial_lr": role_lr[role],
                    "name": f"{role}:{decay_kind}",
                    "captain_multiplier": 1.0,
                    "weight_decay": 0.01 if decay_kind == "decay" else 0.0,
                }
            )

        # Keep stable ordering so checkpoints restore deterministically.
        for role in ("early", "late", "gate", "router", "lora_a", "lora_b", "other"):
            add_group(role, "decay")
            add_group(role, "nodecay")

        optimizer_kwargs: Dict[str, Any] = {
            "lr": base_lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }

        if self.device.type == "cuda":
            optimizer_kwargs["fused"] = True

        try:
            self.optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)

        return self.optimizer

    # =========================================================================
    # Scheduler
    # =========================================================================

    def _apply_captain_factor(self, group_name: str) -> float:
        role = str(group_name).split(":", 1)[0]
        return self._captain_mult * self._captain_layer_boosts.get(role, 1.0)

    def _build_sched(self) -> Any:
        cfg = self.config
        if self.optimizer is None:
            raise RuntimeError("Cannot build scheduler before optimizer.")

        total_steps = max(1, self.total_steps)
        warmup = cfg.warmup_steps if cfg.warmup_steps > 0 else int(cfg.warmup_ratio * total_steps)
        warmup = max(0, min(warmup, total_steps))

        if cfg.use_cosine_restarts:
            self.scheduler = cosine_restart_scheduler(
                self.optimizer,
                cfg.learning_rate,
                cfg.learning_rate * cfg.min_lr_ratio,
                warmup,
                total_steps,
                cfg.restart_interval,
            )
            return self.scheduler

        def make_lambda(group_name: str):
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup:
                    base_factor = current_step / max(1, warmup)
                else:
                    progress = (current_step - warmup) / max(1, total_steps - warmup)
                    progress = max(0.0, min(1.0, progress))
                    base_factor = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )

                return base_factor * self._apply_captain_factor(group_name)

            return lr_lambda

        lambdas = [make_lambda(pg.get("name", "other")) for pg in self.optimizer.param_groups]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdas)
        return self.scheduler

    # =========================================================================
    # Autocast / gradients
    # =========================================================================

    @contextmanager
    def _autocast_context(self):
        """Use safe automatic mixed precision for the active device."""
        if self.device.type == "cuda":
            try:
                dtype = (
                    torch.float16
                    if self.config.load_in_4bit
                    else (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                )
                with torch.autocast(device_type="cuda", dtype=dtype):
                    yield
                return
            except Exception:
                logger.debug("FTRAIN: CUDA autocast failed; using full precision for this operation.", exc_info=True)
                yield
                return

        if self.device.type == "mps":
            try:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    yield
                return
            except Exception:
                logger.debug("FTRAIN: MPS autocast unavailable.", exc_info=True)
                yield
                return

        yield

    def _clear_gradients(self) -> None:
        if self.model is not None:
            self.model.zero_grad(set_to_none=True)
        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass

    def _compute_brain_activity(self) -> Tuple[float, float, float]:
        if self.optimizer is None:
            return 0.0, 0.0, 0.0

        squared = {"early": 0.0, "late": 0.0, "gate": 0.0}
        for group in self.optimizer.param_groups:
            name = group.get("name", "other")
            if name not in squared:
                continue
            for parameter in group.get("params", ()):
                grad = parameter.grad
                if grad is None:
                    continue
                try:
                    if grad.is_sparse:
                        grad = grad.coalesce().values()
                    value = grad.detach().float()
                    squared[name] += float((value * value).sum().item())
                except Exception:
                    continue

        return (
            math.sqrt(max(0.0, squared["early"])),
            math.sqrt(max(0.0, squared["late"])),
            math.sqrt(max(0.0, squared["gate"])),
        )

    # =========================================================================
    # Training entry point
    # =========================================================================

    def train(self) -> Any:
        cfg = self.config
        if self.model is None:
            raise RuntimeError("Training cannot begin without a model.")
        if self.train_dataset is None:
            raise RuntimeError("Training cannot begin without a dataset.")

        self.model.train()
        ui.fire_header()
        print(
            f"🧬 Model: {cfg.model_name} | Steps: {self.total_steps} | "
            f"Mode: {'GRPO' if cfg.use_grpo else 'SFT'} | "
            f"Backend: {'HF/Unsloth' if cfg.use_hf_trainer else 'Custom'}"
        )

        eval_prompt, correct_answer = "", ""
        before_answer = ""

        if not cfg.use_grpo and self.captain is not None:
            eval_prompt, correct_answer = self._select_evaluation_example()
            if eval_prompt:
                print("\n🧠 Captain is asking the model a question before training...")
                before_answer = self._evaluate_model(eval_prompt)

        try:
            if cfg.use_grpo:
                result = self._train_grpo()
            elif cfg.use_hf_trainer:
                result = self._train_hf()
            else:
                result = self._train_custom()
        finally:
            self._stop_dashboard()

        if self.captain is not None and eval_prompt:
            print("\n🧠 Captain is asking the model the same question after training...")
            after_answer = self._evaluate_model(eval_prompt)
            try:
                self.captain.evaluate_improvement(
                    eval_prompt,
                    before_answer,
                    after_answer,
                    correct_answer,
                )
            except Exception:
                logger.debug("FTRAIN: Captain improvement evaluation failed.", exc_info=True)

        return result

    # =========================================================================
    # Captain
    # =========================================================================

    def _apply_captain_advice(self, advice: Mapping[str, Any]) -> None:
        multiplier = _clamp(
            _safe_float(advice.get("mult", 1.0), 1.0),
            _safe_float(getattr(self.config, "captain_mult_min", 0.25), 0.25),
            _safe_float(getattr(self.config, "captain_mult_max", 2.5), 2.5),
        )
        self._captain_mult = multiplier

        layer = str(advice.get("layer_boost", "none")).strip().lower()
        boost = max(0.0, _safe_float(getattr(self.config, "captain_layer_boost", 2.0), 2.0))
        self._captain_layer_boosts = {group: 1.0 for group in self._captain_layer_boosts}
        if layer == "all":
            self._captain_layer_boosts = {group: boost for group in self._captain_layer_boosts}
        elif layer in self._captain_layer_boosts:
            self._captain_layer_boosts[layer] = boost

        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                role = str(group.get("name", "other")).split(":", 1)[0]
                group["captain_multiplier"] = self._apply_captain_factor(role)

    # =========================================================================
    # Dataloader / validation
    # =========================================================================

    def _dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        if dataset is None:
            raise ValueError("Cannot create DataLoader from None.")

        collator = partial(
            collate,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or 0,
        )

        lengths = getattr(dataset, "lengths", None)
        if lengths is None:
            generator = torch.Generator()
            generator.manual_seed(int(self.config.seed) + int(self.epoch))
            return DataLoader(
                dataset,
                batch_size=max(1, int(self.config.per_device_batch_size)),
                shuffle=shuffle,
                collate_fn=collator,
                num_workers=max(0, int(self.config.dataloader_num_workers)),
                pin_memory=bool(self.config.pin_memory and self.device.type == "cuda"),
                persistent_workers=(
                    self.config.dataloader_num_workers > 0
                ),
                generator=generator,
            )

        sampler = LengthSampler(
            lengths,
            max(1, int(self.config.per_device_batch_size)),
            shuffle,
            int(self.config.seed) + int(self.epoch),
        )
        return DataLoader(
            dataset,
            batch_size=max(1, int(self.config.per_device_batch_size)),
            sampler=sampler,
            collate_fn=collator,
            num_workers=max(0, int(self.config.dataloader_num_workers)),
            pin_memory=bool(self.config.pin_memory and self.device.type == "cuda"),
            persistent_workers=(
                self.config.dataloader_num_workers > 0
            ),
        )

    def validate(self, max_batches: Optional[int] = None) -> Optional[float]:
        if self.val_dataset is None or self.model is None:
            return None

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        batches = 0

        try:
            loader = self._dataloader(self.val_dataset, shuffle=False)
            with torch.inference_mode():
                for batch in loader:
                    if max_batches is not None and batches >= max_batches:
                        break

                    input_ids = self._move_to_device(batch.get("input_ids"))
                    if input_ids is None:
                        continue
                    attention_mask = self._move_to_device(batch.get("attention_mask"))
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                    labels = self._move_to_device(batch.get("labels"))
                    if labels is None:
                        labels = input_ids

                    with self._autocast_context():
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                    loss = getattr(output, "loss", None)
                    if loss is None or not torch.isfinite(loss):
                        continue
                    total_loss += float(loss.item())
                    batches += 1

            result = total_loss / max(1, batches)
            self._last_val_loss = result
            self.val_loss_history.append(result)
            if self._best_val_loss is None or result < self._best_val_loss:
                self._best_val_loss = result
                self._bad_eval_count = 0
            else:
                self._bad_eval_count += 1
            return result
        finally:
            if was_training:
                self.model.train()

    # =========================================================================
    # HF / Unsloth training
    # =========================================================================

    def _build_training_arguments(self, argument_class: Any) -> Any:
        cfg = self.config
        kwargs: Dict[str, Any] = {
            "output_dir": cfg.output_dir,
            "max_steps": cfg.max_steps,
            "per_device_train_batch_size": cfg.per_device_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "logging_steps": max(1, cfg.captain_interval),
            "save_strategy": "steps",
            "save_steps": max(1, cfg.checkpoint_interval),
            "save_total_limit": cfg.save_total_limit,
            "gradient_checkpointing": cfg.gradient_checkpointing_enable,
            "dataloader_num_workers": max(0, cfg.dataloader_num_workers),
            "report_to": cfg.report_to,
            "remove_unused_columns": False,
            "max_grad_norm": cfg.max_grad_norm,
            "seed": cfg.seed,
            "bf16": bool(
                self.device.type == "cuda"
                and not cfg.load_in_4bit
                and torch.cuda.is_bf16_supported()
            ),
            "fp16": bool(
                self.device.type == "cuda"
                and cfg.load_in_4bit
                and not torch.cuda.is_bf16_supported()
            ),
            "group_by_length": bool(cfg.group_by_length),
        }

        if self.val_dataset is not None:
            if "eval_strategy" in self._supported_parameters(argument_class):
                kwargs["eval_strategy"] = "steps"
            elif "evaluation_strategy" in self._supported_parameters(argument_class):
                kwargs["evaluation_strategy"] = "steps"
            kwargs["eval_steps"] = max(1, int(cfg.eval_interval))
        else:
            if "eval_strategy" in self._supported_parameters(argument_class):
                kwargs["eval_strategy"] = "no"
            elif "evaluation_strategy" in self._supported_parameters(argument_class):
                kwargs["evaluation_strategy"] = "no"

        if "gradient_checkpointing_kwargs" in self._supported_parameters(argument_class):
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        supported = self._supported_parameters(argument_class)
        filtered = {key: value for key, value in kwargs.items() if key in supported}
        return argument_class(**filtered)

    @staticmethod
    def _supported_parameters(callable_object: Any) -> set[str]:
        try:
            signature = inspect.signature(callable_object.__init__)
            return {name for name in signature.parameters if name != "self"}
        except Exception:
            return set()

    def _build_unsloth_trainer(self, callback: Any) -> Any:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        args = self._build_training_arguments(UnslothTrainingArguments)
        supported = self._supported_parameters(UnslothTrainer)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "args": args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.val_dataset,
            "data_collator": partial(
                collate,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or 0,
            ),
        }

        if "processing_class" in supported:
            kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in supported:
            kwargs["tokenizer"] = self.tokenizer

        if "optimizers" in supported:
            if self.optimizer is None:
                self._build_opt()
            kwargs["optimizers"] = (self.optimizer, None)

        if callback is not None and "callbacks" in supported:
            kwargs["callbacks"] = [callback]

        return UnslothTrainer(**{k: v for k, v in kwargs.items() if k in supported})

    def _build_transformers_trainer(self, callback: Any) -> Any:
        from transformers import Trainer, TrainingArguments
        args = self._build_training_arguments(TrainingArguments)
        supported = self._supported_parameters(Trainer)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "args": args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.val_dataset,
            "data_collator": partial(
                collate,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or 0,
            ),
        }
        if "processing_class" in supported:
            kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in supported:
            kwargs["tokenizer"] = self.tokenizer
        if self.optimizer is None:
            self._build_opt()
        if "optimizers" in supported:
            kwargs["optimizers"] = (self.optimizer, None)
        if callback is not None and "callbacks" in supported:
            kwargs["callbacks"] = [callback]
        return Trainer(**{k: v for k, v in kwargs.items() if k in supported})

    def _sync_trainer_state(self, trainer: Any) -> None:
        state = getattr(trainer, "state", None)
        if state is not None:
            self.step = max(self.step, _safe_int(getattr(state, "global_step", self.step), self.step))
            self.epoch = max(self.epoch, _safe_int(getattr(state, "epoch", self.epoch), self.epoch))

    def _make_hf_captain_callback(self) -> Any:
        """Build a Trainer callback that is the single Captain LR controller."""
        if not self.config.captain_enabled or self.captain is None:
            return None

        engine = self

        class _CaptainCallback(TrainerCallback):
            def _latest_metrics(self, state: Any) -> Tuple[float, Optional[float], float]:
                history = getattr(state, "log_history", None) or []
                latest: Mapping[str, Any] = history[-1] if history else {}
                loss = _safe_float(latest.get("loss", engine._last_loss or 0.0), 0.0)
                val_loss = latest.get("eval_loss", engine._last_val_loss)
                if val_loss is not None:
                    val_loss = _safe_float(val_loss, float("nan"))
                lr = _safe_float(latest.get("learning_rate", engine._current_learning_rate()), engine._current_learning_rate())
                return loss, val_loss, lr

            def _apply_runtime_multiplier(self) -> None:
                if engine.optimizer is None:
                    return
                for group in engine.optimizer.param_groups:
                    role = str(group.get("name", "other")).split(":", 1)[0]
                    multiplier = engine._apply_captain_factor(role)
                    group["captain_multiplier"] = multiplier
                    # Trainer's scheduler owns the base LR. Captain only
                    # applies a multiplicative overlay after the scheduler step.
                    scheduler_base = group.get("_ftrain_scheduler_lr")
                    if scheduler_base is not None:
                        group["lr"] = float(scheduler_base) * multiplier

            def on_step_end(self, args, state, control, **kwargs):
                engine._sync_trainer_state(kwargs.get("model") and kwargs.get("trainer") or engine._hf_trainer)
                trainer = kwargs.get("trainer") or engine._hf_trainer
                optimizer = getattr(trainer, "optimizer", None) if trainer is not None else None
                if optimizer is not None:
                    engine.optimizer = optimizer
                scheduler = getattr(trainer, "lr_scheduler", None) if trainer is not None else None
                if scheduler is not None:
                    engine.scheduler = scheduler

                if engine.optimizer is not None:
                    for group in engine.optimizer.param_groups:
                        group["_ftrain_scheduler_lr"] = float(group.get("lr", engine.config.learning_rate))

                interval = max(1, int(engine.config.captain_interval))
                if engine.captain is None or state.global_step <= 0:
                    self._apply_runtime_multiplier()
                    return
                if state.global_step % interval != 0:
                    self._apply_runtime_multiplier()
                    return

                loss, val_loss, lr = self._latest_metrics(state)
                brain = engine._compute_brain_activity()
                try:
                    advice = engine.captain.inspect_training(
                        state.global_step,
                        loss,
                        lr,
                        _safe_float(getattr(state, "grad_norm", 0.0), 0.0),
                        brain,
                        val_loss,
                    )
                    if advice:
                        engine._apply_captain_advice(advice)
                        action = advice.get("action", "Keep LR")
                        mult = _safe_float(advice.get("mult", 1.0), 1.0)
                        try:
                            ui.print_train_table(
                                state.global_step,
                                engine.total_steps,
                                loss,
                                val_loss,
                                lr,
                                _safe_float(getattr(state, "grad_norm", 0.0), 0.0),
                                f"{action} (x{mult:.2f})",
                            )
                        except Exception:
                            pass
                except Exception:
                    logger.warning("FTRAIN: Captain inspection failed in Trainer callback.", exc_info=True)

                self._apply_runtime_multiplier()

        return _CaptainCallback()

    def _train_hf(self) -> Any:
        cfg = self.config
        self._build_opt()

        callback = self._make_hf_captain_callback()

        trainer = None
        if cfg.use_unsloth_trainer:
            try:
                trainer = self._build_unsloth_trainer(callback)
                logger.info("FTRAIN: using UnslothTrainer backend.")
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.warning("FTRAIN: UnslothTrainer unavailable/incompatible: %s", exc)

        if trainer is None:
            trainer = self._build_transformers_trainer(callback)
            logger.info("FTRAIN: using Transformers Trainer fallback.")

        self._hf_trainer = trainer
        self.optimizer = getattr(trainer, "optimizer", self.optimizer)
        self.scheduler = getattr(trainer, "lr_scheduler", self.scheduler)

        resume = cfg.resume_from_checkpoint or None
        trainer.train(resume_from_checkpoint=resume)

        # Trainer owns the actual scheduler/optimizer on this path. Keep exact
        # references so diagnostics and final checkpoint metadata reflect the
        # real state rather than FTRAIN's pre-construction placeholders.
        self.optimizer = getattr(trainer, "optimizer", self.optimizer)
        self.scheduler = getattr(trainer, "lr_scheduler", self.scheduler)
        self._hf_optimizer = self.optimizer
        self._hf_scheduler = self.scheduler
        self._sync_trainer_state(trainer)

        # Persist the Trainer's state in its native format as well as FTRAIN's
        # compact runtime metadata. This makes resume much more reliable.
        try:
            trainer.save_state()
        except Exception:
            logger.debug("FTRAIN: trainer.save_state() failed.", exc_info=True)

        return self._finalize_model(trainer.model, mode="SFT")

    # =========================================================================
    # GRPO
    # =========================================================================

    def _build_grpo_dataset(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for example in self.train_data:
            if not isinstance(example, Mapping):
                continue
            if "messages" in example:
                messages = example["messages"]
                if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
                    continue
                prompt_messages = [
                    message
                    for message in messages
                    if isinstance(message, Mapping) and message.get("role") != "assistant"
                ]
                if not prompt_messages:
                    continue
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt = "\n".join(
                        f"{message.get('role', 'user')}: {message.get('content', '')}"
                        for message in prompt_messages
                    )
                result.append({"prompt": prompt, "solution": example.get("solution", "")})
            elif "prompt" in example:
                result.append(dict(example))
        return result

    def _train_grpo(self) -> Any:
        from trl import GRPOConfig, GRPOTrainer
        cfg = self.config
        if self.model is None:
            raise RuntimeError("GRPO training requires a model.")
        if not cfg.grpo_reward_funcs:
            raise ValueError("GRPO training requires at least one reward function.")

        grpo_data = self._build_grpo_dataset()
        if not grpo_data:
            raise ValueError("No valid GRPO training examples were produced.")

        supported = self._supported_parameters(GRPOConfig)
        kwargs: Dict[str, Any] = {
            "output_dir": cfg.output_dir,
            "max_steps": cfg.max_steps,
            "learning_rate": cfg.learning_rate,
            "logging_steps": max(1, cfg.captain_interval),
            "save_steps": max(1, cfg.checkpoint_interval),
            "per_device_train_batch_size": cfg.per_device_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_generations": cfg.grpo_num_generations,
            "max_prompt_length": 512,
            "max_completion_length": 1024,
            "temperature": 0.7,
            "beta": 0.01,
            "report_to": "none",
            "bf16": bool(self.device.type == "cuda" and not cfg.load_in_4bit and torch.cuda.is_bf16_supported()),
            "fp16": bool(self.device.type == "cuda" and cfg.load_in_4bit and not torch.cuda.is_bf16_supported()),
            "gradient_checkpointing": cfg.gradient_checkpointing_enable,
            "remove_unused_columns": False,
        }
        trainer_config = GRPOConfig(**{k: v for k, v in kwargs.items() if k in supported})

        trainer_supported = self._supported_parameters(GRPOTrainer)
        trainer_kwargs: Dict[str, Any] = {
            "model": self.model,
            "args": trainer_config,
            "reward_funcs": cfg.grpo_reward_funcs,
            "train_dataset": grpo_data,
        }
        if "processing_class" in trainer_supported:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_supported:
            trainer_kwargs["tokenizer"] = self.tokenizer

        trainer = GRPOTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_supported})
        self._hf_trainer = trainer
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
        self.optimizer = getattr(trainer, "optimizer", self.optimizer)
        self.scheduler = getattr(trainer, "lr_scheduler", self.scheduler)
        self._hf_optimizer = self.optimizer
        self._hf_scheduler = self.scheduler
        self._sync_trainer_state(trainer)
        try:
            trainer.save_state()
        except Exception:
            logger.debug("FTRAIN: GRPO trainer.save_state() failed.", exc_info=True)
        return self._finalize_model(trainer.model, mode="GRPO")

    # =========================================================================
    # Native custom loop
    # =========================================================================

    def _train_custom(self) -> torch.nn.Module:
        cfg = self.config
        self._build_opt()
        self._build_sched()

        if cfg.resume_from_checkpoint:
            try:
                self.load_training_state(cfg.resume_from_checkpoint)
            except Exception:
                logger.warning("FTRAIN: resume state restoration failed; starting optimizer state fresh.", exc_info=True)

        if self.captain is not None:
            try:
                self.captain.set_family_context(self.family, is_moe(self.model))
                self.captain.analyze_data(self.train_dataset, self.tokenizer)
            except Exception:
                logger.debug("FTRAIN: Captain custom-loop setup failed.", exc_info=True)

        loader = self._dataloader(self.train_dataset, shuffle=True)
        iterator = iter(loader)
        accumulated_loss = 0.0
        accumulated_micro_steps = 0
        latest_val_loss: Optional[float] = self._last_val_loss
        latest_grad_norm = 0.0
        status_message = ""
        self._clear_gradients()

        while self.step < self.total_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                self.epoch += 1
                sampler = getattr(loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch)
                iterator = iter(loader)
                batch = next(iterator)

            # Choose accumulation target ONCE per optimizer cycle. This avoids
            # changing the denominator halfway through an accumulated gradient.
            if accumulated_micro_steps == 0:
                if cfg.use_adaptive_accumulation:
                    try:
                        token_count = int(batch["input_ids"].numel())
                        self._accumulation_target = max(
                            1,
                            int(
                                adaptive_accumulation(
                                    cfg.gradient_accumulation_steps,
                                    token_count,
                                    cfg.target_batch_tokens,
                                )
                            ),
                        )
                    except Exception:
                        self._accumulation_target = max(1, int(cfg.gradient_accumulation_steps))
                else:
                    self._accumulation_target = max(1, int(cfg.gradient_accumulation_steps))
                self._current_accumulation_steps = self._accumulation_target

            input_ids = self._move_to_device(batch.get("input_ids"))
            if input_ids is None:
                raise RuntimeError("Training batch contains no input_ids.")

            attention_mask = self._move_to_device(batch.get("attention_mask"))
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            labels = self._move_to_device(batch.get("labels"))
            if labels is None:
                labels = input_ids

            try:
                with self._autocast_context():
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    raw_loss = getattr(output, "loss", None)

                if raw_loss is None:
                    raise RuntimeError("Model returned no loss during custom training.")
                if not torch.isfinite(raw_loss):
                    self._skipped_steps += 1
                    self._clear_gradients()
                    continue

                scaled_loss = raw_loss / self._accumulation_target
                scaled_loss.backward()

                raw_loss_value = _safe_float(raw_loss.detach().item())
                accumulated_loss += raw_loss_value
                accumulated_micro_steps += 1

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self._clear_gradients()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "FTRAIN ran out of memory during the custom training forward/backward pass. "
                        "Reduce per_device_batch_size/max_seq_length or increase gradient accumulation."
                    ) from exc
                raise

            if accumulated_micro_steps < self._accumulation_target:
                continue

            assert self.optimizer is not None

            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                cfg.max_grad_norm,
            )
            latest_grad_norm = _safe_float(
                grad_norm_tensor.item() if isinstance(grad_norm_tensor, torch.Tensor) else grad_norm_tensor
            )

            if self.captain is not None and self.step % max(1, cfg.captain_interval) == max(1, cfg.captain_interval) - 1:
                brain_activity = self._compute_brain_activity()
                current_lr = self._current_learning_rate()
                try:
                    advice = self.captain.inspect_training(
                        self.step + 1,
                        accumulated_loss / max(1, accumulated_micro_steps),
                        current_lr,
                        latest_grad_norm,
                        brain_activity,
                        latest_val_loss,
                    )
                    if advice:
                        self._apply_captain_advice(advice)
                        status_message = (
                            f"{advice.get('action', 'Keep LR')} "
                            f"(x{_safe_float(advice.get('mult', 1.0), 1.0):.2f})"
                        )
                except Exception:
                    logger.warning("FTRAIN: Captain inspection failed during training.", exc_info=True)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1

            average_loss = accumulated_loss / max(1, accumulated_micro_steps)
            self.loss_history.append(average_loss)
            self._last_loss = average_loss
            self.lr_history.append(self._current_learning_rate())

            if cfg.eval_interval > 0 and self.step % cfg.eval_interval == 0 and self.val_dataset is not None:
                latest_val_loss = self.validate()

            ui.print_train_table(
                self.step,
                self.total_steps,
                average_loss,
                latest_val_loss,
                self._current_learning_rate(),
                latest_grad_norm,
                status_message,
            )

            if self.dashboard is not None:
                try:
                    self.dashboard.log_metric(
                        self.step,
                        average_loss,
                        self._current_learning_rate(),
                        latest_val_loss,
                    )
                except Exception:
                    logger.debug("FTRAIN: dashboard metric logging failed.", exc_info=True)

            if cfg.checkpoint_interval > 0 and self.step % cfg.checkpoint_interval == 0:
                self.save_checkpoint(self.step)

            accumulated_loss = 0.0
            accumulated_micro_steps = 0

        self.save_checkpoint(self.step, final=True)
        return self._finalize_model(self.model, mode="SFT")

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self, step: int, final: bool = False) -> str:
        if self.model is None:
            raise RuntimeError("Cannot checkpoint without a model.")

        tag = "final" if final else f"step_{int(step)}"
        root = Path(self.config.output_dir).expanduser() / "checkpoints"
        path = root / tag

        with self._checkpoint_lock:
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(path))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(path))

            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
            if self.scheduler is not None:
                try:
                    torch.save(self.scheduler.state_dict(), path / "scheduler.pt")
                except Exception:
                    logger.debug("FTRAIN: scheduler state save failed.", exc_info=True)

            runtime_state = {
                "version": "core-v4",
                "step": int(self.step),
                "epoch": int(self.epoch),
                "loss_history": list(self.loss_history[-2000:]),
                "val_loss_history": list(self.val_loss_history[-500:]),
                "lr_history": list(self.lr_history[-500:]),
                "last_loss": self._last_loss,
                "last_val_loss": self._last_val_loss,
                "best_val_loss": self._best_val_loss,
                "captain_mult": self._captain_mult,
                "captain_layer_boosts": dict(self._captain_layer_boosts),
                "current_accumulation_steps": self._current_accumulation_steps,
                "model_name": self.config.model_name,
                "family": self.family,
            }
            with (path / "ftrain_state.json").open("w", encoding="utf-8") as file:
                json.dump(runtime_state, file, indent=2, ensure_ascii=False)
                file.write("\n")

            if not final:
                self._prune_checkpoints(root)

        logger.info("FTRAIN checkpoint saved: %s", path)
        print(f"💾 checkpoint → {path}")
        return str(path)

    def _prune_checkpoints(self, checkpoint_root: Path) -> None:
        limit = max(1, int(self.config.save_total_limit))
        candidates: List[Tuple[int, Path]] = []
        for entry in checkpoint_root.iterdir():
            if not entry.is_dir():
                continue
            match = re.fullmatch(r"step_(\d+)", entry.name)
            if match:
                candidates.append((int(match.group(1)), entry))
        candidates.sort(key=lambda x: x[0])
        while len(candidates) > limit:
            _, old_path = candidates.pop(0)
            try:
                shutil.rmtree(old_path)
            except OSError:
                logger.debug("FTRAIN: failed to remove checkpoint %s", old_path, exc_info=True)

    def load_training_state(self, checkpoint: Optional[str] = None) -> bool:
        checkpoint_path = Path(
            checkpoint or self.config.resume_from_checkpoint or ""
        ).expanduser()
        if not checkpoint_path.is_dir():
            return False

        state_file = checkpoint_path / "ftrain_state.json"
        if state_file.exists():
            try:
                with state_file.open("r", encoding="utf-8") as file:
                    state = json.load(file)
                self.step = max(0, _safe_int(state.get("step", 0)))
                self.epoch = max(0, _safe_int(state.get("epoch", 0)))
                self.loss_history = [
                    _safe_float(v)
                    for v in state.get("loss_history", [])
                    if _is_finite(v)
                ]
                self.val_loss_history = [
                    _safe_float(v)
                    for v in state.get("val_loss_history", [])
                    if _is_finite(v)
                ]
                self.lr_history = [
                    _safe_float(v)
                    for v in state.get("lr_history", [])
                    if _is_finite(v)
                ]
                self._last_loss = state.get("last_loss")
                self._last_val_loss = state.get("last_val_loss")
                self._best_val_loss = state.get("best_val_loss")
                self._captain_mult = _safe_float(state.get("captain_mult", 1.0), 1.0)
                boosts = state.get("captain_layer_boosts", {})
                if isinstance(boosts, Mapping):
                    for name in self._captain_layer_boosts:
                        self._captain_layer_boosts[name] = _safe_float(boosts.get(name, 1.0), 1.0)
                self._current_accumulation_steps = max(
                    1,
                    _safe_int(state.get("current_accumulation_steps", 1), 1),
                )
            except Exception:
                logger.warning("FTRAIN: runtime state restore failed.", exc_info=True)

        optimizer_file = checkpoint_path / "optimizer.pt"
        if optimizer_file.exists() and self.optimizer is not None:
            try:
                state = torch.load(optimizer_file, map_location="cpu")
                self.optimizer.load_state_dict(state)
                self._optimizer_state_to_device()
            except Exception:
                logger.warning("FTRAIN: optimizer state restore failed.", exc_info=True)

        scheduler_file = checkpoint_path / "scheduler.pt"
        if scheduler_file.exists() and self.scheduler is not None:
            try:
                state = torch.load(scheduler_file, map_location="cpu")
                self.scheduler.load_state_dict(state)
            except Exception:
                logger.warning("FTRAIN: scheduler state restore failed.", exc_info=True)

        return True

    def _optimizer_state_to_device(self) -> None:
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    # =========================================================================
    # Finalization
    # =========================================================================

    def _finalize_model(self, model: torch.nn.Module, mode: str) -> torch.nn.Module:
        """Export the final model once and persist final runtime metadata."""
        self.model = model
        final_path = Path(self.config.output_dir).expanduser() / "final"
        final_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(final_path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(final_path))

        final_state = {
            "version": "core-v4",
            "step": int(self.step),
            "epoch": int(self.epoch),
            "mode": mode,
            "loss_history": list(self.loss_history[-2000:]),
            "val_loss_history": list(self.val_loss_history[-500:]),
            "lr_history": list(self.lr_history[-500:]),
            "last_loss": self._last_loss,
            "last_val_loss": self._last_val_loss,
            "best_val_loss": self._best_val_loss,
            "skipped_steps": int(self._skipped_steps),
            "captain_mult": float(self._captain_mult),
            "captain_layer_boosts": dict(self._captain_layer_boosts),
            "current_accumulation_steps": int(self._current_accumulation_steps),
            "model_name": self.config.model_name,
            "family": self.family,
        }

        try:
            with (final_path / "ftrain_final_state.json").open("w", encoding="utf-8") as file:
                json.dump(final_state, file, indent=2, ensure_ascii=False)
                file.write("\n")
        except Exception:
            logger.warning("FTRAIN: final runtime state save failed.", exc_info=True)

        try:
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), final_path / "optimizer.pt")
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), final_path / "scheduler.pt")
        except Exception:
            logger.warning("FTRAIN: final optimizer/scheduler state save failed.", exc_info=True)

        ui.print_final_summary(
            {
                "Model": self.config.model_name,
                "Steps": self.step,
                "Mode": mode,
                "Final Loss": (
                    f"{self._last_loss:.4f}" if self._last_loss is not None else "N/A"
                ),
                "Validation Loss": (
                    f"{self._last_val_loss:.4f}" if self._last_val_loss is not None else "N/A"
                ),
                "Best Validation Loss": (
                    f"{self._best_val_loss:.4f}" if self._best_val_loss is not None else "N/A"
                ),
                "Skipped Steps": self._skipped_steps,
                "Dir": str(final_path),
            }
        )
        return self.model

    # =========================================================================
    # Device helpers
    # =========================================================================

    def _move_to_device(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(
                self.device,
                non_blocking=bool(self.config.pin_memory and self.device.type == "cuda"),
            )
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    def _current_learning_rate(self) -> float:
        if self.optimizer is None:
            return float(self.config.learning_rate)

        weighted_sum = 0.0
        total_params = 0
        fallback = float(self.config.learning_rate)

        for group in self.optimizer.param_groups:
            lr = _safe_float(group.get("lr", fallback), fallback)
            try:
                count = sum(int(p.numel()) for p in group.get("params", ()))
            except Exception:
                count = 0
            if count <= 0:
                continue
            weighted_sum += lr * count
            total_params += count

        if total_params <= 0:
            return fallback
        return weighted_sum / total_params
