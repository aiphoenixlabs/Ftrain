from __future__ import annotations

import inspect
import io
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
# Helpers
# =============================================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_len(
    value: Any,
) -> Optional[int]:
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _is_empty(
    value: Any,
) -> bool:
    length = _safe_len(value)
    return length == 0 if length is not None else False


def _is_finite(
    value: Any,
) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# =============================================================================
# FTRAIN
# =============================================================================

class Ftrain:
    """
    Main stateful FTRAIN training engine.
    """

    def __init__(
        self,
        config: TrainConfig,
        train_data: Any,
        val_data: Any = None,
    ) -> None:
        if config is None:
            raise ValueError(
                "Ftrain requires a TrainConfig."
            )

        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        # ------------------------------------------------------------------
        # Persistent runtime state
        # ------------------------------------------------------------------

        self.loss_history: List[float] = []
        self.lr_history: List[float] = []

        self.step = 0
        self.epoch = 0

        self._last_loss: Optional[float] = None
        self._last_val_loss: Optional[float] = None
        self._best_val_loss: Optional[float] = None

        self._captain_mult = 1.0
        self._captain_layer_boosts = {
            "early": 1.0,
            "late": 1.0,
            "gate": 1.0,
            "router": 1.0,
            "lora_a": 1.0,
            "lora_b": 1.0,
            "other": 1.0,
        }

        self._current_accumulation_steps = max(
            1,
            int(getattr(config, "gradient_accumulation_steps", 1)),
        )

        self._train_started_at: Optional[float] = None
        self._last_checkpoint_step = 0
        self._last_checkpoint_time = 0.0

        self._invalid_loss_count = 0
        self._skipped_steps = 0
        self._oom_count = 0

        self._scaler = None
        self._trainer = None
        self._backend = "none"

        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Any = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Any = None

        self.train_dataset: Any = None
        self.val_dataset: Any = None

        self.dashboard: Any = None
        self.captain: Optional[PhoenixCaptain] = None

        self.total_steps = max(
            1,
            int(config.max_steps),
        )

        self.device = self._resolve_device()

        self._checkpoint_lock = threading.Lock()
        self._dashboard_started = False

        # ------------------------------------------------------------------
        # Runtime
        # ------------------------------------------------------------------

        self._configure_runtime()

        # ------------------------------------------------------------------
        # Resume / family
        # ------------------------------------------------------------------

        self._resolve_auto_resume()

        self.family = (
            config.family
            if config.family != "auto"
            else get_family(config.model_name)
        )

        self.preset = get_preset(
            self.family
        ) or {}

        if (
            not getattr(
                config,
                "lora_target_modules",
                None,
            )
            and self.preset.get("lora_targets")
        ):
            config.lora_target_modules = list(
                self.preset["lora_targets"]
            )

        # ------------------------------------------------------------------
        # Load model / Captain / data / adapters
        # ------------------------------------------------------------------

        self._load_model()

        if config.captain_enabled:
            try:
                self.captain = PhoenixCaptain(
                    config
                )
                self.captain.set_family_context(
                    self.family,
                    is_moe(self.model),
                )
                if self.model is not None:
                    self.captain.analyze_model(
                        self.model
                    )
            except Exception:
                logger.warning(
                    "FTRAIN: Captain initialization failed; "
                    "continuing without Captain.",
                    exc_info=True,
                )
                self.captain = None

        self._prepare_data()

        if config.auto_lora_targets:
            self._discover_lora_targets()

        self._apply_adapters()
        self._print_parameter_summary()
        self._build_datasets()
        self._start_dashboard()

        Path(
            config.output_dir
        ).expanduser().mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # Device/runtime
    # =========================================================================

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    def _configure_runtime(self) -> None:
        try:
            flash_mode(
                enabled=True,
                tf32=self.device.type == "cuda",
            )
        except Exception:
            logger.debug(
                "FTRAIN: optimized kernel setup unavailable.",
                exc_info=True,
            )

        seed_everything(
            self.config.seed
        )

        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        if self.config.use_grpo:
            try:
                from unsloth import PatchFastRL

                PatchFastRL(
                    "GRPO",
                    FastLanguageModel,
                )
            except Exception:
                logger.warning(
                    "FTRAIN: GRPO patching unavailable.",
                    exc_info=True,
                )

    # =========================================================================
    # Resume discovery
    # =========================================================================

    def _resolve_auto_resume(self) -> None:
        if not self.config.auto_resume:
            return

        root = Path(
            self.config.output_dir
        ).expanduser()

        if not root.exists():
            return

        candidates: List[Tuple[int, Path]] = []
        seen = set()

        for search_root in (
            root / "checkpoints",
            root,
        ):
            if not search_root.is_dir():
                continue

            for entry in search_root.iterdir():
                if not entry.is_dir():
                    continue

                resolved = entry.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)

                match = re.fullmatch(
                    r"step_(\d+)",
                    entry.name,
                )

                if match is None:
                    match = re.fullmatch(
                        r"checkpoint-(\d+)",
                        entry.name,
                    )

                if match is None:
                    continue

                has_state = (
                    (entry / "ftrain_state.json").exists()
                    or (entry / "trainer_state.json").exists()
                    or (entry / "optimizer.pt").exists()
                    or (entry / "optimizer.bin").exists()
                    or any(entry.glob("*.safetensors"))
                    or (entry / "pytorch_model.bin").exists()
                )

                if has_state:
                    candidates.append(
                        (
                            int(match.group(1)),
                            entry,
                        )
                    )

        if candidates:
            candidates.sort(
                key=lambda x: x[0]
            )

            step, checkpoint = candidates[-1]

            self.config.resume_from_checkpoint = str(
                checkpoint
            )

            logger.info(
                "FTRAIN auto-resume selected %s at step %d.",
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
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16

        return torch.float32

    def _load_model(self) -> None:
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
                # Newer Unsloth versions may expose dtype while older versions
                # use torch_dtype internally; only pass the public parameter.
                kwargs["dtype"] = self._preferred_model_dtype()

            attention_impl = self.preset.get(
                "attn_implementation"
            )

            if attention_impl:
                kwargs["attn_implementation"] = attention_impl

            try:
                with self._quiet_stdout():
                    self.model, self.tokenizer = (
                        FastLanguageModel.from_pretrained(
                            **kwargs
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Unsloth model loading failed; "
                    "falling back to Transformers: %s",
                    exc,
                )
                self._load_model_transformers()

            if self.model is None or self.tokenizer is None:
                raise RuntimeError(
                    "Model loading produced no model/tokenizer."
                )

            self._prepare_tokenizer()
            self._prepare_model()

        finally:
            bar.done()

    def _load_model_transformers(self) -> None:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        kwargs: Dict[str, Any] = {
            "torch_dtype": self._preferred_model_dtype(),
        }

        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )

    def _prepare_tokenizer(self) -> None:
        tokenizer = self.tokenizer

        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer is unavailable."
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
                    "Tokenizer has neither pad_token nor eos_token."
                )

        try:
            tokenizer.padding_side = "right"
        except Exception:
            pass

    def _prepare_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model is unavailable."
            )

        try:
            device_map = getattr(
                self.model,
                "hf_device_map",
                None,
            )

            if device_map:
                return

            self.model.to(
                self.device
            )

        except Exception:
            logger.debug(
                "FTRAIN: model .to(device) skipped.",
                exc_info=True,
            )

    # =========================================================================
    # Data
    # =========================================================================

    def _prepare_data(self) -> None:
        cfg = self.config

        if self.train_data is None:
            raise ValueError(
                "Training data cannot be None."
            )

        if _safe_len(self.train_data) == 0:
            raise ValueError(
                "Training data is empty."
            )

        processing_needed = (
            cfg.data_perplexity_filter
            or cfg.data_dedup
            or bool(cfg.data_sources)
        )

        if not processing_needed:
            return

        original_length = _safe_len(
            self.train_data
        ) or 0

        changes: List[str] = []

        if cfg.data_dedup:
            before = _safe_len(
                self.train_data
            ) or 0

            self.train_data = deduplicate(
                self.train_data
            )

            after = _safe_len(
                self.train_data
            ) or 0

            changes.append(
                f"deduplicated: {before - after} removed"
            )

        if cfg.data_perplexity_filter:
            before = _safe_len(
                self.train_data
            ) or 0

            self.train_data = filter_by_perplexity(
                self.train_data,
                self.model,
                self.tokenizer,
                self.device,
                cfg.data_perplexity_keep_pct,
            )

            after = _safe_len(
                self.train_data
            ) or 0

            changes.append(
                f"perplexity filter: {before - after} removed"
            )

        if cfg.data_sources:
            from .data_utils import load_data

            sources = [
                self.train_data
            ]

            for source in cfg.data_sources:
                sources.append(
                    load_data(
                        source
                    )
                )

            self.train_data = balance_datasets(
                sources,
                cfg.data_balance_strategy,
            )

            changes.append(
                "balanced multiple data sources"
            )

        final_length = _safe_len(
            self.train_data
        ) or 0

        if final_length == 0:
            raise ValueError(
                "Dataset processing removed all examples."
            )

        if self.captain is not None:
            try:
                self.captain.analyze_and_report_data(
                    original_length,
                    final_length,
                    changes,
                )
            except Exception:
                logger.debug(
                    "Captain data analysis failed.",
                    exc_info=True,
                )

    # =========================================================================
    # LoRA target discovery
    # =========================================================================

    @staticmethod
    def _extract_target_module_name(
        parameter_name: str,
    ) -> Optional[str]:
        parts = parameter_name.split(".")

        if len(parts) < 2:
            return None

        candidate = parts[-2]

        if candidate.isdigit():
            return None

        return candidate

    def _discover_lora_targets(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LoRA discovery requires model and tokenizer."
            )

        logger.info(
            "FTRAIN: discovering LoRA targets..."
        )

        was_training = self.model.training

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

            loss = getattr(
                output,
                "loss",
                None,
            )

            if loss is None:
                raise RuntimeError(
                    "Model returned no loss for target discovery."
                )

            if not torch.isfinite(loss).all():
                raise RuntimeError(
                    "Discovery loss was non-finite."
                )

            loss.backward()

            scores: Dict[str, float] = {}

            for name, parameter in self.model.named_parameters():
                if (
                    parameter.grad is None
                    or not parameter.requires_grad
                    or "lora_" in name.lower()
                ):
                    continue

                module_name = self._extract_target_module_name(
                    name
                )

                if not module_name:
                    continue

                try:
                    score = float(
                        parameter.grad.detach()
                        .float()
                        .norm()
                        .item()
                    )
                except Exception:
                    continue

                if not math.isfinite(score):
                    continue

                scores[module_name] = (
                    scores.get(
                        module_name,
                        0.0,
                    )
                    + score
                )

            if scores:
                target_count = max(
                    1,
                    int(self.config.lora_target_count),
                )

                ranked = sorted(
                    scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                self.config.lora_target_modules = [
                    name
                    for name, _ in ranked[:target_count]
                ]

                logger.info(
                    "FTRAIN: auto LoRA targets: %s",
                    ", ".join(
                        self.config.lora_target_modules
                    ),
                )

        finally:
            self._clear_gradients()

            try:
                self.model.train(
                    was_training
                )
            except Exception:
                pass

    # =========================================================================
    # Adapters
    # =========================================================================

    def _apply_adapters(self) -> None:
        cfg = self.config

        if self.model is None:
            raise RuntimeError(
                "Cannot apply adapters without a model."
            )

        targets = list(
            cfg.lora_target_modules or []
        )

        if not targets:
            raise ValueError(
                "No LoRA target modules are configured."
            )

        try:
            if cfg.use_unsloth_lora:
                fn = getattr(
                    FastLanguageModel,
                    "get_peft_model",
                )

                kwargs: Dict[str, Any] = {
                    "r": cfg.lora_r,
                    "lora_alpha": cfg.lora_alpha,
                    "target_modules": targets,
                }

                supported = self._supported_parameters(
                    fn
                )

                if (
                    cfg.use_dora
                    and (
                        not supported
                        or "use_dora" in supported
                    )
                ):
                    kwargs["use_dora"] = True

                if (
                    not supported
                    or "lora_dropout" in supported
                ):
                    kwargs["lora_dropout"] = 0.0

                if (
                    not supported
                    or "bias" in supported
                ):
                    kwargs["bias"] = "none"

                self.model = fn(
                    self.model,
                    **kwargs,
                )

            elif cfg.use_custom_lora:
                if cfg.use_dora:
                    self.model = inject_dora(
                        self.model,
                        targets,
                        cfg.lora_r,
                        cfg.lora_alpha,
                    )
                else:
                    self.model = inject_lora(
                        self.model,
                        targets,
                        cfg.lora_r,
                        cfg.lora_alpha,
                    )

            else:
                logger.info(
                    "FTRAIN: adapter backend disabled."
                )

        except Exception as exc:
            raise RuntimeError(
                f"Adapter initialization failed: {exc}"
            ) from exc

    def _print_parameter_summary(self) -> None:
        if self.model is None:
            return

        try:
            stats = count_params(
                self.model
            )

            total = int(
                stats.get(
                    "total",
                    0,
                )
            )

            trainable = int(
                stats.get(
                    "trainable",
                    0,
                )
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
        cfg = self.config

        if cfg.use_grpo:
            self.train_dataset = self.train_data
        else:
            self.train_dataset = FtrainDataset(
                self.train_data,
                self.tokenizer,
                cfg.max_seq_length,
                cfg.use_packing,
                train_on_response_only=cfg.train_on_response_only,
                mask_thinking=cfg.mask_thinking,
            )

        if (
            self.val_data is not None
            and not _is_empty(
                self.val_data
            )
        ):
            self.val_dataset = FtrainDataset(
                self.val_data,
                self.tokenizer,
                cfg.max_seq_length,
            )
        else:
            self.val_dataset = None

        if (
            self.train_dataset is None
            or _safe_len(self.train_dataset) == 0
        ):
            raise ValueError(
                "Training dataset wrapper is empty."
            )

        self.total_steps = max(
            1,
            int(cfg.max_steps),
        )

    # =========================================================================
    # Dashboard
    # =========================================================================

    def _start_dashboard(self) -> None:
        if not self.config.use_dashboard:
            self.dashboard = None
            return

        try:
            from .dashboard import TrainingDashboard

            self.dashboard = TrainingDashboard(
                port=self.config.dashboard_port
            )

            thread = threading.Thread(
                target=self.dashboard.start,
                name="ftrain-dashboard",
                daemon=True,
            )

            thread.start()
            self._dashboard_started = True

        except Exception:
            logger.warning(
                "FTRAIN: dashboard startup failed.",
                exc_info=True,
            )
            self.dashboard = None

    def _stop_dashboard(self) -> None:
        if self.dashboard is None:
            return

        try:
            self.dashboard.stop()
        except Exception:
            logger.debug(
                "FTRAIN dashboard shutdown failed.",
                exc_info=True,
            )
        finally:
            self.dashboard = None
            self._dashboard_started = False

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _select_evaluation_example(
        self,
    ) -> Tuple[str, str]:
        source = (
            self.val_data
            if self.val_data is not None
            and not _is_empty(self.val_data)
            else self.train_data
        )

        if source is None:
            return "", ""

        try:
            length = len(source)
        except TypeError:
            return "", ""

        if length <= 0:
            return "", ""

        indices = list(
            range(length)
        )

        random.Random(
            int(self.config.seed)
            + 99173
        ).shuffle(
            indices
        )

        for index in indices[: min(32, length)]:
            sample = source[index]

            if not isinstance(
                sample,
                Mapping,
            ):
                continue

            messages = sample.get(
                "messages"
            )

            if isinstance(
                messages,
                Sequence,
            ) and not isinstance(
                messages,
                (str, bytes),
            ):
                user = None
                answer = None

                for message in messages:
                    if not isinstance(
                        message,
                        Mapping,
                    ):
                        continue

                    role = str(
                        message.get(
                            "role",
                            "",
                        )
                    ).lower()

                    content = str(
                        message.get(
                            "content",
                            "",
                        )
                        or ""
                    ).strip()

                    if (
                        role == "user"
                        and content
                        and user is None
                    ):
                        user = content

                    elif (
                        role == "assistant"
                        and content
                        and answer is None
                    ):
                        answer = content

                if user is not None:
                    prompt = user

                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            [
                                {
                                    "role": "user",
                                    "content": user,
                                }
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        pass

                    return (
                        str(prompt),
                        str(answer or ""),
                    )

            prompt = (
                sample.get("prompt")
                or sample.get("question")
                or sample.get("query")
            )

            answer = (
                sample.get("answer")
                or sample.get("response")
                or sample.get("solution")
            )

            if prompt:
                return (
                    str(prompt),
                    str(answer or ""),
                )

            text = sample.get(
                "text"
            )

            if isinstance(
                text,
                str,
            ) and text.strip():
                return (
                    text.strip(),
                    "",
                )

        return "", ""

    def _evaluate_model(
        self,
        prompt: str,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return (
                "Evaluation unavailable: model/tokenizer missing."
            )

        was_training = self.model.training

        try:
            self.model.eval()

            try:
                FastLanguageModel.for_inference(
                    self.model
                )
            except Exception:
                pass

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
                if isinstance(
                    value,
                    torch.Tensor,
                )
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
                        pad_token_id=(
                            self.tokenizer.eos_token_id
                            or self.tokenizer.pad_token_id
                            or 0
                        ),
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
                "FTRAIN evaluation failed: %s",
                exc,
            )
            return "Evaluation failed."

        finally:
            if was_training:
                try:
                    FastLanguageModel.for_training(
                        self.model
                    )
                except Exception:
                    pass

                self.model.train()

    # =========================================================================
    # Optimizer
    # =========================================================================

    def _build_opt(self) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError(
                "Cannot build optimizer without a model."
            )

        cfg = self.config

        roles = (
            "early",
            "late",
            "gate",
            "router",
            "lora_a",
            "lora_b",
            "other",
        )

        decay = {
            role: []
            for role in roles
        }

        no_decay = {
            role: []
            for role in roles
        }

        layers = max(
            1,
            get_num_layers(
                self.model
            ),
        )

        early_cutoff = max(
            1,
            layers // 3,
        )

        late_cutoff = max(
            early_cutoff,
            (2 * layers) // 3,
        )

        pattern = re.compile(
            r"(?:layers|h|block|blocks)\.(\d+)\."
        )

        multipliers = {
            "early": float(
                cfg.layerwise_lr_decay
            ),
            "late": 1.0,
            "gate": float(
                cfg.swiglu_gate_boost
            ),
            "router": float(
                cfg.moe_router_lr_multiplier
            ),
            "lora_a": float(
                cfg.lora_a_lr_mult
            ),
            "lora_b": float(
                cfg.lora_b_lr_mult
            ),
            "other": 1.0,
        }

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            lname = name.lower()
            role = "other"

            if "lora_a" in lname:
                role = "lora_a"
            elif "lora_b" in lname:
                role = "lora_b"
            elif any(
                marker in lname
                for marker in (
                    "router",
                    "router_logits",
                    "gate_logits",
                    "expert_gate",
                )
            ):
                role = "router"
            elif (
                "gate_proj" in lname
                or ".gate." in lname
                or lname.endswith(
                    ".gate.weight"
                )
            ):
                role = "gate"
            else:
                match = pattern.search(
                    name
                )

                if match:
                    index = _safe_int(
                        match.group(1),
                        0,
                    )

                    if index < early_cutoff:
                        role = "early"
                    elif index >= late_cutoff:
                        role = "late"

            is_no_decay = (
                parameter.ndim < 2
                or lname.endswith(".bias")
                or "norm" in lname
                or "layernorm" in lname
                or "rmsnorm" in lname
            )

            target = (
                no_decay
                if is_no_decay
                else decay
            )

            target[role].append(
                parameter
            )

        groups = []

        for role in roles:
            lr = (
                float(cfg.learning_rate)
                * max(
                    0.0,
                    multipliers[role],
                )
            )

            if decay[role]:
                groups.append(
                    {
                        "params": decay[role],
                        "lr": lr,
                        "initial_lr": lr,
                        "name": role,
                        "weight_decay": 0.01,
                    }
                )

            if no_decay[role]:
                groups.append(
                    {
                        "params": no_decay[role],
                        "lr": lr,
                        "initial_lr": lr,
                        "name": role,
                        "weight_decay": 0.0,
                    }
                )

        if not groups:
            raise RuntimeError(
                "No trainable parameters were found after adapter setup."
            )

        kwargs: Dict[str, Any] = {
            "lr": float(
                cfg.learning_rate
            ),
            "betas": (
                0.9,
                0.999,
            ),
            "eps": 1e-8,
        }

        try:
            if self.device.type == "cuda":
                self.optimizer = torch.optim.AdamW(
                    groups,
                    fused=True,
                    **kwargs,
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    groups,
                    **kwargs,
                )
        except (TypeError, RuntimeError):
            self.optimizer = torch.optim.AdamW(
                groups,
                **kwargs,
            )

        return self.optimizer

    # =========================================================================
    # Compatibility / signature handling
    # =========================================================================

    @staticmethod
    def _supported_parameters(
        callable_object: Any,
    ) -> set[str]:
        """
        Return accepted parameters across modern and older APIs.

        CRITICAL FIX
        ------------
        If a class accepts **kwargs, signature inspection alone is insufficient.
        Modern Unsloth/TRL wrappers often expose base TrainingArguments through
        **kwargs. We therefore also inspect MRO/base classes and dataclass
        fields.
        """
        try:
            target = (
                callable_object.__init__
                if inspect.isclass(
                    callable_object
                )
                else callable_object
            )

            signature = inspect.signature(
                target
            )

            parameters = signature.parameters

            names = {
                name
                for name, parameter in parameters.items()
                if (
                    name not in {
                        "self",
                        "args",
                        "kwargs",
                    }
                    and parameter.kind
                    in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                )
            }

            has_var_kwargs = any(
                parameter.kind
                == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if has_var_kwargs:
                # Dataclass fields.
                try:
                    import dataclasses

                    if dataclasses.is_dataclass(
                        callable_object
                    ):
                        names.update(
                            field.name
                            for field in dataclasses.fields(
                                callable_object
                            )
                            if field.init
                        )
                except Exception:
                    pass

                # Walk MRO so wrappers such as
                # UnslothTrainingArguments -> SFTConfig ->
                # TrainingArguments retain their configuration surface.
                try:
                    for cls in callable_object.__mro__:
                        try:
                            base_signature = inspect.signature(
                                cls.__init__
                            )

                            for name, parameter in (
                                base_signature.parameters.items()
                            ):
                                if name in {
                                    "self",
                                    "args",
                                    "kwargs",
                                }:
                                    continue

                                if parameter.kind in {
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    inspect.Parameter.KEYWORD_ONLY,
                                }:
                                    names.add(
                                        name
                                    )

                        except Exception:
                            continue
                except Exception:
                    pass

            return names

        except Exception:
            logger.debug(
                "FTRAIN: unable to inspect %r.",
                callable_object,
                exc_info=True,
            )
            return set()

    @classmethod
    def _filter_kwargs_for_callable(
        cls,
        callable_object: Any,
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Filter kwargs only when the callable truly does not accept **kwargs.

        This is the core compatibility fix that prevents max_steps and other
        valid TrainingArguments from silently disappearing in Unsloth.
        """
        try:
            target = (
                callable_object.__init__
                if inspect.isclass(
                    callable_object
                )
                else callable_object
            )

            signature = inspect.signature(
                target
            )

            accepts_kwargs = any(
                parameter.kind
                == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )

            if accepts_kwargs:
                return dict(
                    kwargs
                )

        except Exception:
            pass

        supported = cls._supported_parameters(
            callable_object
        )

        return {
            key: value
            for key, value in kwargs.items()
            if key in supported
        }

    # =========================================================================
    # Trainer arguments
    # =========================================================================

    def _build_training_arguments(
        self,
        argument_class: Any,
    ) -> Any:
        cfg = self.config

        requested: Dict[str, Any] = {
            "output_dir": cfg.output_dir,
            "max_steps": int(
                cfg.max_steps
            ),
            "num_train_epochs": 1,
            "per_device_train_batch_size": max(
                1,
                int(
                    cfg.per_device_batch_size
                ),
            ),
            "gradient_accumulation_steps": max(
                1,
                int(
                    cfg.gradient_accumulation_steps
                ),
            ),
            "learning_rate": float(
                cfg.learning_rate
            ),
            "warmup_ratio": float(
                cfg.warmup_ratio
            ),
            "warmup_steps": max(
                0,
                int(
                    cfg.warmup_steps
                ),
            ),
            "logging_steps": max(
                1,
                int(
                    cfg.captain_interval
                ),
            ),
            "save_strategy": "steps",
            "save_steps": max(
                1,
                int(
                    cfg.checkpoint_interval
                ),
            ),
            "save_total_limit": max(
                1,
                int(
                    cfg.save_total_limit
                ),
            ),
            "gradient_checkpointing": bool(
                cfg.gradient_checkpointing_enable
            ),
            "dataloader_num_workers": max(
                0,
                int(
                    cfg.dataloader_num_workers
                ),
            ),
            "report_to": cfg.report_to,
            "remove_unused_columns": False,
            "max_grad_norm": float(
                cfg.max_grad_norm
            ),
            "seed": int(
                cfg.seed
            ),
            "group_by_length": False,
        }

        # Keeping num_train_epochs=1 is deliberate: max_steps is FTRAIN's
        # primary control. When max_steps is positive, HF Trainer should stop
        # at exactly that number of optimizer steps.
        if self.val_dataset is not None:
            requested["eval_steps"] = max(
                1,
                int(
                    cfg.eval_interval
                ),
            )

            if hasattr(
                argument_class,
                "eval_strategy",
            ):
                requested["eval_strategy"] = "steps"

            elif hasattr(
                argument_class,
                "evaluation_strategy",
            ):
                requested["evaluation_strategy"] = "steps"

        else:
            if hasattr(
                argument_class,
                "eval_strategy",
            ):
                requested["eval_strategy"] = "no"

            elif hasattr(
                argument_class,
                "evaluation_strategy",
            ):
                requested["evaluation_strategy"] = "no"

        if self.device.type == "cuda":
            bf16 = False
            fp16 = False

            try:
                bf16 = (
                    torch.cuda.is_bf16_supported()
                    and not cfg.load_in_4bit
                )
            except Exception:
                pass

            fp16 = (
                bool(
                    cfg.load_in_4bit
                )
                and not bf16
            )

            requested["bf16"] = bf16
            requested["fp16"] = fp16

        else:
            requested["bf16"] = False
            requested["fp16"] = False

        supported = self._supported_parameters(
            argument_class
        )

        actual = self._filter_kwargs_for_callable(
            argument_class,
            requested,
        )

        # ------------------------------------------------------------------
        # Critical guard: max_steps may NEVER silently disappear.
        # ------------------------------------------------------------------
        if (
            int(cfg.max_steps) > 0
            and "max_steps" not in actual
        ):
            raise RuntimeError(
                "FTRAIN compatibility error: the selected training "
                "arguments class rejected max_steps. "
                "This would make Steps=... unreliable. "
                f"Argument class: {argument_class!r}. "
                f"Detected parameters: {sorted(supported)[:80]}"
            )

        # Explicit compatibility for older classes.
        try:
            if "gradient_checkpointing_kwargs" in supported:
                actual[
                    "gradient_checkpointing_kwargs"
                ] = {
                    "use_reentrant": False
                }

            if "optim" in supported:
                actual["optim"] = (
                    "adamw_torch_fused"
                    if self.device.type == "cuda"
                    else "adamw_torch"
                )

        except Exception:
            pass

        args = argument_class(
            **actual
        )

        # Verify the actual object, not merely our kwargs.
        actual_max_steps = getattr(
            args,
            "max_steps",
            None,
        )

        if (
            int(cfg.max_steps) > 0
            and actual_max_steps is not None
            and int(actual_max_steps) != int(
                cfg.max_steps
            )
        ):
            raise RuntimeError(
                "FTRAIN refused to start because the trainer arguments "
                f"reported max_steps={actual_max_steps!r}, while FTRAIN "
                f"requested {cfg.max_steps!r}."
            )

        return args

    # =========================================================================
    # Trainers
    # =========================================================================

    def _build_unsloth_trainer(
        self,
        callback: Any,
    ) -> Any:
        from unsloth import (
            UnslothTrainer,
            UnslothTrainingArguments,
        )

        args = self._build_training_arguments(
            UnslothTrainingArguments
        )

        requested: Dict[str, Any] = {
            "model": self.model,
            "args": args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.val_dataset,
            "data_collator": partial(
                collate,
                pad_token_id=(
                    getattr(
                        self.tokenizer,
                        "pad_token_id",
                        None,
                    )
                    or 0
                ),
            ),
        }

        supported = self._supported_parameters(
            UnslothTrainer
        )

        if (
            callback is not None
            and (
                not supported
                or "callbacks" in supported
            )
        ):
            requested["callbacks"] = [
                callback
            ]

        if (
            not supported
            or "processing_class" in supported
        ):
            requested["processing_class"] = self.tokenizer

        elif "tokenizer" in supported:
            requested["tokenizer"] = self.tokenizer

        kwargs = self._filter_kwargs_for_callable(
            UnslothTrainer,
            requested,
        )

        return UnslothTrainer(
            **kwargs
        )

    def _build_transformers_trainer(
        self,
        callback: Any,
    ) -> Any:
        from transformers import (
            Trainer,
            TrainingArguments,
        )

        args = self._build_training_arguments(
            TrainingArguments
        )

        requested: Dict[str, Any] = {
            "model": self.model,
            "args": args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.val_dataset,
            "data_collator": partial(
                collate,
                pad_token_id=(
                    getattr(
                        self.tokenizer,
                        "pad_token_id",
                        None,
                    )
                    or 0
                ),
            ),
        }

        supported = self._supported_parameters(
            Trainer
        )

        if (
            callback is not None
            and "callbacks" in supported
        ):
            requested["callbacks"] = [
                callback
            ]

        if "processing_class" in supported:
            requested["processing_class"] = self.tokenizer

        elif "tokenizer" in supported:
            requested["tokenizer"] = self.tokenizer

        return Trainer(
            **self._filter_kwargs_for_callable(
                Trainer,
                requested,
            )
        )

    # =========================================================================
    # GRPO
    # =========================================================================

    def _build_grpo_dataset(
        self,
    ) -> List[Dict[str, Any]]:
        result = []

        for example in self.train_data:
            if not isinstance(
                example,
                Mapping,
            ):
                continue

            messages = example.get(
                "messages"
            )

            if isinstance(
                messages,
                Sequence,
            ) and not isinstance(
                messages,
                (str, bytes),
            ):
                prompt_messages = [
                    message
                    for message in messages
                    if isinstance(
                        message,
                        Mapping,
                    )
                    and message.get(
                        "role"
                    ) != "assistant"
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
                        f"{message.get('role', 'user')}: "
                        f"{message.get('content', '')}"
                        for message in prompt_messages
                    )

                result.append(
                    {
                        "prompt": prompt,
                        "solution": (
                            example.get(
                                "solution",
                                "",
                            )
                            or example.get(
                                "response",
                                "",
                            )
                        ),
                    }
                )

            elif "prompt" in example:
                result.append(
                    dict(
                        example
                    )
                )

        return result

    def _train_grpo(self) -> Any:
        if self.model is None:
            raise RuntimeError(
                "GRPO requires a model."
            )

        if not self.config.grpo_reward_funcs:
            raise ValueError(
                "GRPO requires reward functions."
            )

        from trl import (
            GRPOConfig,
            GRPOTrainer,
        )

        data = self._build_grpo_dataset()

        if not data:
            raise ValueError(
                "No valid GRPO examples were produced."
            )

        requested = {
            "output_dir": self.config.output_dir,
            "max_steps": self.config.max_steps,
            "learning_rate": self.config.learning_rate,
            "logging_steps": max(
                1,
                self.config.captain_interval,
            ),
            "save_steps": max(
                1,
                self.config.checkpoint_interval,
            ),
            "per_device_train_batch_size": max(
                1,
                self.config.per_device_batch_size,
            ),
            "gradient_accumulation_steps": max(
                1,
                self.config.gradient_accumulation_steps,
            ),
            "num_generations": self.config.grpo_num_generations,
            "max_prompt_length": 512,
            "max_completion_length": 1024,
            "temperature": 0.7,
            "beta": 0.01,
            "report_to": "none",
            "remove_unused_columns": False,
            "gradient_checkpointing": self.config.gradient_checkpointing_enable,
        }

        if self.device.type == "cuda":
            bf16 = False
            try:
                bf16 = (
                    not self.config.load_in_4bit
                    and torch.cuda.is_bf16_supported()
                )
            except Exception:
                pass

            requested["bf16"] = bf16
            requested["fp16"] = bool(
                self.config.load_in_4bit
                and not bf16
            )

        args = GRPOConfig(
            **self._filter_kwargs_for_callable(
                GRPOConfig,
                requested,
            )
        )

        kwargs = {
            "model": self.model,
            "args": args,
            "reward_funcs": self.config.grpo_reward_funcs,
            "train_dataset": data,
        }

        supported = self._supported_parameters(
            GRPOTrainer
        )

        if "processing_class" in supported:
            kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in supported:
            kwargs["tokenizer"] = self.tokenizer

        trainer = GRPOTrainer(
            **self._filter_kwargs_for_callable(
                GRPOTrainer,
                kwargs,
            )
        )

        self._trainer = trainer
        self._backend = "grpo"

        trainer.train(
            resume_from_checkpoint=(
                self.config.resume_from_checkpoint
                or None
            )
        )

        self._sync_trainer_state(
            trainer
        )

        return self._finalize_model(
            trainer.model,
            mode="GRPO",
        )

    # =========================================================================
    # Trainer state
    # =========================================================================

    def _sync_trainer_state(
        self,
        trainer: Any,
    ) -> None:
        state = getattr(
            trainer,
            "state",
            None,
        )

        if state is not None:
            self.step = max(
                self.step,
                _safe_int(
                    getattr(
                        state,
                        "global_step",
                        self.step,
                    ),
                    self.step,
                ),
            )

            try:
                self.epoch = max(
                    self.epoch,
                    int(
                        float(
                            getattr(
                                state,
                                "epoch",
                                self.epoch,
                            )
                        )
                    ),
                )
            except Exception:
                pass

        self.optimizer = getattr(
            trainer,
            "optimizer",
            self.optimizer,
        )

        self.scheduler = getattr(
            trainer,
            "lr_scheduler",
            self.scheduler,
        )

    # =========================================================================
    # HF/Unsloth SFT
    # =========================================================================

    def _train_hf(self) -> Any:
        cfg = self.config

        # Build our optimizer only when requested/needed by the custom argument
        # path. Unsloth may provide its own optimizer defaults.
        callback = None

        if cfg.captain_enabled:
            try:
                from .callbacks import (
                    PhoenixCaptainCallback
                )

                callback = PhoenixCaptainCallback(
                    cfg,
                    self.model,
                    self.tokenizer,
                    self.train_dataset,
                    self.dashboard,
                )

            except Exception:
                logger.warning(
                    "FTRAIN: Captain callback unavailable.",
                    exc_info=True,
                )

        trainer = None

        if cfg.use_unsloth_trainer:
            try:
                trainer = self._build_unsloth_trainer(
                    callback
                )
                self._backend = "unsloth"

            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                logger.warning(
                    "FTRAIN: Unsloth Trainer unavailable; "
                    "falling back to Transformers: %s",
                    exc,
                )

        if trainer is None:
            trainer = self._build_transformers_trainer(
                callback
            )
            self._backend = "transformers"

        self._trainer = trainer

        # ------------------------------------------------------------------
        # HARD SAFETY CHECK
        # ------------------------------------------------------------------
        trainer_args = getattr(
            trainer,
            "args",
            None,
        )

        actual_max_steps = getattr(
            trainer_args,
            "max_steps",
            None,
        )

        if (
            actual_max_steps is not None
            and int(cfg.max_steps) > 0
            and int(actual_max_steps)
            != int(cfg.max_steps)
        ):
            raise RuntimeError(
                "FTRAIN REFUSED TO START: "
                f"requested max_steps={cfg.max_steps}, "
                f"trainer has max_steps={actual_max_steps}."
            )

        logger.info(
            "FTRAIN: starting %s trainer for exactly %s configured max steps.",
            self._backend,
            cfg.max_steps,
        )

        trainer.train(
            resume_from_checkpoint=(
                cfg.resume_from_checkpoint
                or None
            )
        )

        self._sync_trainer_state(
            trainer
        )

        return self._finalize_model(
            trainer.model,
            mode="SFT",
        )

    # =========================================================================
    # Custom training
    # =========================================================================

    def _train_custom(self) -> torch.nn.Module:
        cfg = self.config

        if self.model is None:
            raise RuntimeError(
                "Custom training requires a model."
            )

        self._build_opt()
        self._build_sched()

        if cfg.resume_from_checkpoint:
            self.load_training_state(
                cfg.resume_from_checkpoint
            )

        loader = self._dataloader(
            self.train_dataset,
            shuffle=True,
        )

        iterator = iter(
            loader
        )

        accumulated_loss = 0.0
        accumulated_micro_steps = 0
        latest_val_loss = self._last_val_loss
        latest_grad_norm = 0.0
        status_message = ""

        scaler = self._get_scaler()

        self._clear_gradients()

        while self.step < self.total_steps:
            try:
                batch = next(
                    iterator
                )
            except StopIteration:
                self.epoch += 1

                sampler = getattr(
                    loader,
                    "sampler",
                    None,
                )

                if hasattr(
                    sampler,
                    "set_epoch",
                ):
                    sampler.set_epoch(
                        self.epoch
                    )

                iterator = iter(
                    loader
                )

                batch = next(
                    iterator
                )

            if accumulated_micro_steps == 0:
                if cfg.use_adaptive_accumulation:
                    try:
                        self._accumulation_target = max(
                            1,
                            int(
                                adaptive_accumulation(
                                    cfg.gradient_accumulation_steps,
                                    int(
                                        batch[
                                            "input_ids"
                                        ].numel()
                                    ),
                                    cfg.target_batch_tokens,
                                )
                            ),
                        )
                    except Exception:
                        self._accumulation_target = max(
                            1,
                            int(
                                cfg.gradient_accumulation_steps
                            ),
                        )
                else:
                    self._accumulation_target = max(
                        1,
                        int(
                            cfg.gradient_accumulation_steps
                        ),
                    )

                self._current_accumulation_steps = (
                    self._accumulation_target
                )

            input_ids = self._move_to_device(
                batch.get(
                    "input_ids"
                )
            )

            attention_mask = self._move_to_device(
                batch.get(
                    "attention_mask"
                )
            )

            labels = self._move_to_device(
                batch.get(
                    "labels"
                )
            )

            if input_ids is None:
                raise RuntimeError(
                    "Training batch contains no input_ids."
                )

            if attention_mask is None:
                attention_mask = torch.ones_like(
                    input_ids
                )

            if labels is None:
                labels = input_ids

            try:
                with self._autocast_context():
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    raw_loss = getattr(
                        output,
                        "loss",
                        None,
                    )

                if raw_loss is None:
                    raise RuntimeError(
                        "Model returned no loss."
                    )

                if not torch.isfinite(
                    raw_loss
                ).all():
                    self._invalid_loss_count += 1
                    self._skipped_steps += 1
                    self._clear_gradients()
                    accumulated_loss = 0.0
                    accumulated_micro_steps = 0
                    continue

                loss = (
                    raw_loss
                    / self._accumulation_target
                )

                if scaler is not None:
                    scaler.scale(
                        loss
                    ).backward()
                else:
                    loss.backward()

                accumulated_loss += _safe_float(
                    raw_loss.detach().item()
                )

                accumulated_micro_steps += 1

            except RuntimeError as exc:
                if "out of memory" in str(
                    exc
                ).lower():
                    self._oom_count += 1
                    self._clear_gradients()

                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                    raise RuntimeError(
                        "FTRAIN ran out of memory during custom training. "
                        "Reduce batch/sequence length or increase accumulation."
                    ) from exc

                raise

            if accumulated_micro_steps < self._accumulation_target:
                continue

            assert self.optimizer is not None

            if scaler is not None:
                scaler.unscale_(
                    self.optimizer
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                cfg.max_grad_norm,
            )

            latest_grad_norm = _safe_float(
                grad_norm.item()
                if isinstance(
                    grad_norm,
                    torch.Tensor,
                )
                else grad_norm
            )

            brain_activity = self._compute_brain_activity()
            current_lr = self._current_learning_rate()

            if (
                self.captain is not None
                and (
                    self.step
                    % max(
                        1,
                        cfg.captain_interval,
                    )
                    == max(
                        1,
                        cfg.captain_interval,
                    )
                    - 1
                )
            ):
                try:
                    advice = (
                        self.captain.inspect_training(
                            self.step + 1,
                            accumulated_loss
                            / max(
                                1,
                                accumulated_micro_steps,
                            ),
                            current_lr,
                            latest_grad_norm,
                            brain_activity,
                            latest_val_loss,
                        )
                    )

                    if advice:
                        self._apply_captain_advice(
                            advice
                        )

                        status_message = (
                            f"{advice.get('action', 'Keep LR')} "
                            f"(x{_safe_float(advice.get('mult', 1.0), 1.0):.2f})"
                        )

                except Exception:
                    logger.warning(
                        "FTRAIN: Captain training inspection failed.",
                        exc_info=True,
                    )

            if scaler is not None:
                scaler.step(
                    self.optimizer
                )
                scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(
                set_to_none=True
            )

            self.step += 1

            average_loss = (
                accumulated_loss
                / max(
                    1,
                    accumulated_micro_steps,
                )
            )

            self.loss_history.append(
                average_loss
            )

            self._last_loss = (
                average_loss
            )

            latest_lr = (
                self._current_learning_rate()
            )

            self.lr_history.append(
                latest_lr
            )

            if (
                cfg.eval_interval > 0
                and self.step
                % cfg.eval_interval == 0
                and self.val_dataset is not None
            ):
                latest_val_loss = self.validate()

            ui.print_train_table(
                self.step,
                self.total_steps,
                average_loss,
                latest_val_loss,
                latest_lr,
                latest_grad_norm,
                status_message,
            )

            if self.dashboard is not None:
                try:
                    self.dashboard.log_metric(
                        self.step,
                        average_loss,
                        latest_lr,
                        latest_val_loss,
                    )
                except Exception:
                    logger.debug(
                        "FTRAIN dashboard metric failed.",
                        exc_info=True,
                    )

            if (
                cfg.checkpoint_interval > 0
                and self.step
                % cfg.checkpoint_interval == 0
            ):
                self.save_checkpoint(
                    self.step
                )

            accumulated_loss = 0.0
            accumulated_micro_steps = 0

        return self._finalize_model(
            self.model,
            mode="SFT",
        )

    # =========================================================================
    # Captain
    # =========================================================================

    def _apply_captain_advice(
        self,
        advice: Mapping[str, Any],
    ) -> None:
        multiplier = _safe_float(
            advice.get(
                "mult",
                1.0,
            ),
            1.0,
        )

        clamp = getattr(
            self.config,
            "captain_clamp",
            [0.25, 2.5],
        )

        if (
            isinstance(
                clamp,
                (list, tuple),
            )
            and len(clamp) >= 2
        ):
            low = _safe_float(
                clamp[0],
                0.25,
            )

            high = _safe_float(
                clamp[1],
                2.5,
            )
        else:
            low = 0.25
            high = 2.5

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
            getattr(
                self.config,
                "captain_layer_boost",
                2.0,
            ),
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
        if self.optimizer is None:
            raise RuntimeError(
                "Cannot build scheduler before optimizer."
            )

        cfg = self.config
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

        lambdas = []

        for group in self.optimizer.param_groups:
            name = group.get(
                "name",
                "other",
            )

            def make_lambda(
                group_name: str,
            ):
                def lr_lambda(
                    current_step: int,
                ) -> float:
                    if current_step < warmup:
                        base = (
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

                        base = (
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

                    return (
                        base
                        * self._captain_mult
                        * self._captain_layer_boosts.get(
                            group_name,
                            1.0,
                        )
                    )

                return lr_lambda

            lambdas.append(
                make_lambda(
                    name
                )
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambdas,
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
        if dataset is None:
            raise ValueError(
                "Cannot create DataLoader from None."
            )

        workers = max(
            0,
            int(
                self.config.dataloader_num_workers
            ),
        )

        collator = partial(
            collate,
            pad_token_id=(
                getattr(
                    self.tokenizer,
                    "pad_token_id",
                    None,
                )
                or 0
            ),
        )

        common = {
            "batch_size": max(
                1,
                int(
                    self.config.per_device_batch_size
                ),
            ),
            "collate_fn": collator,
            "num_workers": workers,
            "pin_memory": bool(
                self.config.pin_memory
                and self.device.type == "cuda"
            ),
        }

        lengths = getattr(
            dataset,
            "lengths",
            None,
        )

        if lengths is None:
            generator = torch.Generator()

            generator.manual_seed(
                int(
                    self.config.seed
                )
                + self.epoch
            )

            return DataLoader(
                dataset,
                shuffle=shuffle,
                generator=generator,
                persistent_workers=bool(
                    workers > 0
                ),
                **common,
            )

        sampler = LengthSampler(
            lengths,
            max(
                1,
                int(
                    self.config.per_device_batch_size
                ),
            ),
            shuffle=shuffle,
            seed=(
                int(
                    self.config.seed
                )
                + self.epoch
            ),
        )

        return DataLoader(
            dataset,
            sampler=sampler,
            **common,
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> Optional[float]:
        if self.val_dataset is None or self.model is None:
            return None

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        batches = 0

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

                    attention_mask = self._move_to_device(
                        batch.get(
                            "attention_mask"
                        )
                    )

                    labels = self._move_to_device(
                        batch.get(
                            "labels"
                        )
                    )

                    if attention_mask is None:
                        attention_mask = torch.ones_like(
                            input_ids
                        )

                    if labels is None:
                        labels = input_ids

                    with self._autocast_context():
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                    loss = getattr(
                        output,
                        "loss",
                        None,
                    )

                    if loss is None:
                        continue

                    value = _safe_float(
                        loss.item(),
                        float("nan"),
                    )

                    if not math.isfinite(
                        value
                    ):
                        continue

                    total_loss += value
                    batches += 1

            if batches == 0:
                return None

            result = (
                total_loss
                / batches
            )

            self._last_val_loss = result

            if (
                self._best_val_loss is None
                or result < self._best_val_loss
            ):
                self._best_val_loss = result

            return result

        finally:
            if was_training:
                self.model.train()

    # =========================================================================
    # AMP / gradients
    # =========================================================================

    @contextmanager
    def _autocast_context(self):
        if self.device.type == "cuda":
            try:
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    and not self.config.load_in_4bit
                    else torch.float16
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=True,
                ):
                    yield

                return

            except Exception:
                logger.debug(
                    "FTRAIN: CUDA autocast unavailable.",
                    exc_info=True,
                )

        elif self.device.type == "mps":
            try:
                with torch.autocast(
                    device_type="mps",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    yield

                return

            except Exception:
                logger.debug(
                    "FTRAIN: MPS autocast unavailable.",
                    exc_info=True,
                )

        yield

    def _amp_enabled(self) -> bool:
        if self.device.type != "cuda":
            return False

        try:
            return bool(
                self.config.load_in_4bit
                and not torch.cuda.is_bf16_supported()
            )
        except Exception:
            return bool(
                self.config.load_in_4bit
            )

    def _get_scaler(self):
        if not self._amp_enabled():
            return None

        if self._scaler is not None:
            return self._scaler

        try:
            self._scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
            )
        except (
            AttributeError,
            TypeError,
        ):
            self._scaler = torch.cuda.amp.GradScaler(
                enabled=True
            )

        return self._scaler

    def _clear_gradients(self) -> None:
        if self.model is not None:
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
        if self.optimizer is None:
            return (
                0.0,
                0.0,
                0.0,
            )

        values = {
            "early": 0.0,
            "late": 0.0,
            "gate": 0.0,
        }

        for group in self.optimizer.param_groups:
            name = group.get(
                "name",
                "other",
            )

            if name not in values:
                continue

            for parameter in group.get(
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

                    value = gradient.detach().float()

                    values[name] += float(
                        torch.sum(
                            value * value
                        ).item()
                    )

                except Exception:
                    logger.debug(
                        "FTRAIN gradient activity failed for one parameter.",
                        exc_info=True,
                    )

        return (
            math.sqrt(
                max(
                    0.0,
                    values["early"],
                )
            ),
            math.sqrt(
                max(
                    0.0,
                    values["late"],
                )
            ),
            math.sqrt(
                max(
                    0.0,
                    values["gate"],
                )
            ),
        )

    # =========================================================================
    # Training entry point
    # =========================================================================

    def train(self) -> Any:
        if self.model is None:
            raise RuntimeError(
                "Training cannot begin without a model."
            )

        if self.train_dataset is None:
            raise RuntimeError(
                "Training cannot begin without a dataset."
            )

        self._train_started_at = time.time()

        self.model.train()

        ui.fire_header()

        print(
            f"🧬 Model: {self.config.model_name} | "
            f"Steps: {self.total_steps} | "
            f"Mode: "
            f"{'GRPO' if self.config.use_grpo else 'SFT'} | "
            f"Backend: "
            f"{'HF/Unsloth' if self.config.use_hf_trainer else 'Custom'}"
        )

        eval_prompt, correct_answer = (
            self._select_evaluation_example()
        )

        before_answer = ""

        if (
            not self.config.use_grpo
            and self.captain is not None
            and eval_prompt
        ):
            print(
                "\n🧠 Captain is asking the model a question before training..."
            )

            before_answer = self._evaluate_model(
                eval_prompt
            )

        result = None

        try:
            if self.config.use_grpo:
                result = self._train_grpo()
            elif self.config.use_hf_trainer:
                result = self._train_hf()
            else:
                result = self._train_custom()

        finally:
            self._stop_dashboard()

        if (
            self.captain is not None
            and eval_prompt
            and before_answer
        ):
            print(
                "\n🧠 Captain is asking the model the same question after training..."
            )

            after_answer = self._evaluate_model(
                eval_prompt
            )

            try:
                self.captain.evaluate_improvement(
                    eval_prompt,
                    before_answer,
                    after_answer,
                    correct_answer,
                )
            except Exception:
                logger.debug(
                    "FTRAIN: Captain improvement evaluation failed.",
                    exc_info=True,
                )

        return result

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(
        self,
        step: int,
        final: bool = False,
    ) -> str:
        if self.model is None:
            raise RuntimeError(
                "Cannot checkpoint without a model."
            )

        root = (
            Path(
                self.config.output_dir
            ).expanduser()
            / "checkpoints"
        )

        tag = (
            "final"
            if final
            else f"step_{int(step)}"
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
                    logger.debug(
                        "FTRAIN: scheduler state save failed.",
                        exc_info=True,
                    )

            if self._scaler is not None:
                try:
                    torch.save(
                        self._scaler.state_dict(),
                        path / "scaler.pt",
                    )
                except Exception:
                    logger.debug(
                        "FTRAIN: scaler state save failed.",
                        exc_info=True,
                    )

            state = {
                "version": "ftrain-core-v1.1",
                "step": int(
                    self.step
                ),
                "epoch": int(
                    self.epoch
                ),
                "loss_history": list(
                    self.loss_history[-1000:]
                ),
                "lr_history": list(
                    self.lr_history[-1000:]
                ),
                "last_loss": self._last_loss,
                "last_val_loss": self._last_val_loss,
                "best_val_loss": self._best_val_loss,
                "captain_mult": self._captain_mult,
                "captain_layer_boosts": dict(
                    self._captain_layer_boosts
                ),
                "current_accumulation_steps": (
                    self._current_accumulation_steps
                ),
                "model_name": self.config.model_name,
                "family": self.family,
                "backend": self._backend,
                "invalid_loss_count": self._invalid_loss_count,
                "skipped_steps": self._skipped_steps,
                "oom_count": self._oom_count,
            }

            with (
                path / "ftrain_state.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    state,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            if not final:
                self._prune_checkpoints(
                    root
                )

            self._last_checkpoint_step = int(
                step
            )

            self._last_checkpoint_time = time.time()

            print(
                f"💾 checkpoint → {path}"
            )

        return str(path)

    def _prune_checkpoints(
        self,
        root: Path,
    ) -> None:
        limit = max(
            1,
            int(
                self.config.save_total_limit
            ),
        )

        candidates = []

        if not root.exists():
            return

        for entry in root.iterdir():
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

        while len(candidates) > limit:
            _, old_path = candidates.pop(0)

            try:
                shutil.rmtree(
                    old_path
                )
            except OSError:
                logger.debug(
                    "Could not prune checkpoint %s.",
                    old_path,
                    exc_info=True,
                )

    def load_training_state(
        self,
        checkpoint: Optional[str] = None,
    ) -> bool:
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
                ) as handle:
                    state = json.load(
                        handle
                    )

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

                self.loss_history = [
                    _safe_float(
                        value
                    )
                    for value in state.get(
                        "loss_history",
                        [],
                    )
                    if _is_finite(
                        value
                    )
                ]

                self.lr_history = [
                    _safe_float(
                        value
                    )
                    for value in state.get(
                        "lr_history",
                        [],
                    )
                    if _is_finite(
                        value
                    )
                ]

                self._last_loss = state.get(
                    "last_loss"
                )

                self._last_val_loss = state.get(
                    "last_val_loss"
                )

                self._best_val_loss = state.get(
                    "best_val_loss"
                )

                self._backend = str(
                    state.get(
                        "backend",
                        self._backend,
                    )
                )

                self._invalid_loss_count = max(
                    0,
                    _safe_int(
                        state.get(
                            "invalid_loss_count",
                            0,
                        )
                    ),
                )

                self._skipped_steps = max(
                    0,
                    _safe_int(
                        state.get(
                            "skipped_steps",
                            0,
                        )
                    ),
                )

                self._oom_count = max(
                    0,
                    _safe_int(
                        state.get(
                            "oom_count",
                            0,
                        )
                    ),
                )

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
                    "FTRAIN runtime state restore failed.",
                    exc_info=True,
                )

        if (
            self.optimizer is not None
            and (
                checkpoint_path
                / "optimizer.pt"
            ).exists()
        ):
            try:
                self.optimizer.load_state_dict(
                    torch.load(
                        checkpoint_path
                        / "optimizer.pt",
                        map_location=self.device,
                    )
                )
            except Exception:
                logger.warning(
                    "FTRAIN optimizer restore failed.",
                    exc_info=True,
                )

        if (
            self.scheduler is not None
            and (
                checkpoint_path
                / "scheduler.pt"
            ).exists()
        ):
            try:
                self.scheduler.load_state_dict(
                    torch.load(
                        checkpoint_path
                        / "scheduler.pt",
                        map_location="cpu",
                    )
                )
            except Exception:
                logger.warning(
                    "FTRAIN scheduler restore failed.",
                    exc_info=True,
                )

        if (
            self._scaler is not None
            and (
                checkpoint_path
                / "scaler.pt"
            ).exists()
        ):
            try:
                self._scaler.load_state_dict(
                    torch.load(
                        checkpoint_path
                        / "scaler.pt",
                        map_location="cpu",
                    )
                )
            except Exception:
                logger.debug(
                    "FTRAIN scaler restore failed.",
                    exc_info=True,
                )

        self._last_checkpoint_step = self.step

        return True

    # =========================================================================
    # Finalization
    # =========================================================================

    def _finalize_model(
        self,
        model: torch.nn.Module,
        mode: str,
    ) -> torch.nn.Module:
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
            "version": "ftrain-core-v1.1",
            "mode": mode,
            "backend": self._backend,
            "step": self.step,
            "requested_max_steps": self.total_steps,
            "epoch": self.epoch,
            "final_loss": self._last_loss,
            "validation_loss": self._last_val_loss,
            "best_validation_loss": self._best_val_loss,
            "skipped_steps": self._skipped_steps,
            "invalid_loss_count": self._invalid_loss_count,
            "oom_count": self._oom_count,
        }

        with (
            final_path
            / "ftrain_final_state.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        # Preserve FTRAIN checkpoint compatibility.
        self.save_checkpoint(
            self.step,
            final=True,
        )

        ui.print_final_summary(
            {
                "Model": self.config.model_name,
                "Steps": self.step,
                "Requested Steps": self.total_steps,
                "Mode": mode,
                "Backend": self._backend,
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
            tuple,
        ):
            return tuple(
                self._move_to_device(
                    item
                )
                for item in value
            )

        if isinstance(
            value,
            list,
        ):
            return [
                self._move_to_device(
                    item
                )
                for item in value
            ]

        return value

    def _current_learning_rate(
        self,
    ) -> float:
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
            sum(values)
            / len(values)
            if values
            else float(
                self.config.learning_rate
            )
        )


def _self_test() -> Dict[str, Any]:
    """
    Static core-level compatibility test.

    This does not load a model. It verifies that the crucial argument-filtering
    code preserves values when a wrapper exposes **kwargs.
    """
    class FakeArguments:
        def __init__(
            self,
            special=None,
            **kwargs,
        ):
            self.special = special
            for key, value in kwargs.items():
                setattr(self, key, value)

    supported = Ftrain._supported_parameters(
        FakeArguments
    )

    values = Ftrain._filter_kwargs_for_callable(
        FakeArguments,
        {
            "max_steps": 100,
            "learning_rate": 2e-4,
            "remove_unused_columns": False,
        },
    )

    assert "max_steps" in values
    assert values["max_steps"] == 100
    assert values["remove_unused_columns"] is False

    return {
        "kwargs_passthrough": "PASS",
        "max_steps_preserved": values["max_steps"],
        "remove_unused_columns": values[
            "remove_unused_columns"
        ],
    }


if __name__ == "__main__":
    print(_self_test())
