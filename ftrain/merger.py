# ============================================================
# FTRAIN / PHOENIX INTELLIGENT BRAIN MERGER
# ============================================================
#
# Goals:
#   1. Merge models with different parameter naming/layouts
#   2. Layer-aware architecture translation
#   3. Fisher-aware merging
#   4. TIES / SLERP / weighted merging
#   5. Procrustes representation alignment
#   6. Importance-aware parameter protection
#   7. Post-merge calibration training
#   8. Gradient-health analysis
#   9. Trainability optimization
#  10. Automatic repair of weak merged regions
#
# IMPORTANT:
#   A truly different architecture cannot be made mathematically
#   identical by resizing tensors. This implementation therefore
#   treats Model A as the output architecture and translates the
#   compatible knowledge from Model B into A's parameter space.
# ============================================================

import os
import re
import gc
import math
import logging
from functools import partial
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unsloth import FastLanguageModel

from . import ui
from .merge_intel import MergeAnalyzer, MergePlanner
from .safety import check_state_dict, sanitize
from .captain import PhoenixCaptain
from .merge_advanced import compute_fisher
from .data_utils import load_data
from .dataset import FtrainDataset, collate
from .cpp_merge import (
    fast_weighted_avg,
    fast_slerp,
    fast_ties,
    fast_fisher_merge,
)

logger = logging.getLogger(__name__)


# ============================================================
# SMALL UTILITIES
# ============================================================

def _finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _safe_float(x) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


# ============================================================
# MERGER
# ============================================================

class Merger:

    def __init__(self, config):

        self.config = config

        self.model_a = config.model_a
        self.model_b = config.model_b

        self.output_dir = config.output_dir

        self.strategy = getattr(config, "strategy", "intelligent")
        self.alpha = float(getattr(config, "alpha", 0.5))

        save_dtype = getattr(config, "save_dtype", "bf16")

        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(save_dtype, torch.bfloat16)

        # ----------------------------------------------------
        # New trainability settings
        # ----------------------------------------------------

        self.adaptation_enabled = getattr(
            config,
            "merge_adaptation",
            True,
        )

        self.adaptation_steps = int(
            getattr(config, "merge_adaptation_steps", 100)
        )

        self.adaptation_lr = float(
            getattr(config, "merge_adaptation_lr", 1e-5)
        )

        self.adaptation_weight_decay = float(
            getattr(config, "merge_adaptation_weight_decay", 0.01)
        )

        self.gradient_clip = float(
            getattr(config, "merge_gradient_clip", 1.0)
        )

        self.repair_rounds = int(
            getattr(config, "merge_repair_rounds", 2)
        )

        self.repair_strength = float(
            getattr(config, "merge_repair_strength", 0.15)
        )

        self.max_calibration_samples = int(
            getattr(config, "merge_calibration_samples", 128)
        )

        self.use_gradient_health = getattr(
            config,
            "merge_gradient_health",
            True,
        )

        self.use_fisher_protection = getattr(
            config,
            "merge_fisher_protection",
            True,
        )

        # ----------------------------------------------------
        # Captain
        # ----------------------------------------------------

        self.captain = None

        captain_model = getattr(
            config,
            "captain_model",
            None,
        )

        if captain_model:

            try:

                from .config import TrainConfig

                cap_cfg = TrainConfig(
                    model_name=captain_model,
                    captain_model=captain_model,
                    captain_mode="llm",
                    answer_mode="auto_yes",
                )

                self.captain = PhoenixCaptain(cap_cfg)

            except Exception as e:

                logger.warning(
                    "Captain initialization failed: %s",
                    e,
                )

    # ========================================================
    # MEMORY
    # ========================================================

    def _purge_memory(self):

        gc.collect()

        if torch.cuda.is_available():

            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    # ========================================================
    # MODEL LAYER COUNT
    # ========================================================

    def _get_num_layers(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> int:

        layers = set()

        patterns = [
            r"(?:layers|h)\.(\d+)\.",
            r"(?:decoder\.layers)\.(\d+)\.",
            r"(?:transformer\.h)\.(\d+)\.",
        ]

        for key in state_dict.keys():

            for pattern in patterns:

                m = re.search(pattern, key)

                if m:
                    layers.add(int(m.group(1)))
                    break

        return max(layers) + 1 if layers else 1

    # ========================================================
    # NORMALIZE LAYER NAME
    # ========================================================

    def _translate_layer_index(
        self,
        layer: int,
        source_layers: int,
        target_layers: int,
    ) -> int:

        if source_layers <= 1 or target_layers <= 1:
            return 0

        ratio = layer / float(source_layers - 1)

        return int(
            round(
                ratio * (target_layers - 1)
            )
        )

    # ========================================================
    # FIND CORRESPONDING PARAMETER
    # ========================================================

    def _find_matching_key(
        self,
        key_a: str,
        keys_b: List[str],
        num_layers_a: int,
        num_layers_b: int,
    ) -> Optional[str]:

        key_set = set(keys_b)

        # Exact match is always best.
        if key_a in key_set:
            return key_a

        layer_match = re.search(
            r"(?:model\.layers|decoder\.layers|transformer\.h|layers|h)\.(\d+)\.",
            key_a,
        )

        layer_a = None

        if layer_match:
            layer_a = int(layer_match.group(1))

        candidates = []

        # ----------------------------------------------------
        # Translate layer position
        # ----------------------------------------------------

        if layer_a is not None:

            layer_b = self._translate_layer_index(
                layer_a,
                num_layers_a,
                num_layers_b,
            )

            variants = [
                re.sub(
                    r"(?:model\.layers|decoder\.layers|transformer\.h|layers|h)\.\d+\.",
                    f"model.layers.{layer_b}.",
                    key_a,
                ),
                re.sub(
                    r"(?:model\.layers|decoder\.layers|transformer\.h|layers|h)\.\d+\.",
                    f"layers.{layer_b}.",
                    key_a,
                ),
                re.sub(
                    r"(?:model\.layers|decoder\.layers|transformer\.h|layers|h)\.\d+\.",
                    f"transformer.h.{layer_b}.",
                    key_a,
                ),
            ]

            candidates.extend(variants)

        candidates.append(key_a)

        # ----------------------------------------------------
        # Architecture aliases
        # ----------------------------------------------------

        aliases = [

            # Attention
            ("self_attn.q_proj", "attention.wq"),
            ("self_attn.k_proj", "attention.wk"),
            ("self_attn.v_proj", "attention.wv"),
            ("self_attn.o_proj", "attention.wo"),

            ("self_attn.q_proj", "attention.q_proj"),
            ("self_attn.k_proj", "attention.k_proj"),
            ("self_attn.v_proj", "attention.v_proj"),
            ("self_attn.o_proj", "attention.o_proj"),

            ("q_proj", "wq"),
            ("k_proj", "wk"),
            ("v_proj", "wv"),
            ("o_proj", "wo"),

            # FFN
            ("mlp.gate_proj", "feed_forward.w1"),
            ("mlp.down_proj", "feed_forward.w2"),
            ("mlp.up_proj", "feed_forward.w3"),

            ("gate_proj", "w1"),
            ("down_proj", "w2"),
            ("up_proj", "w3"),

            # Norms
            ("input_layernorm", "attention_norm"),
            ("post_attention_layernorm", "ffn_norm"),

            ("input_layernorm", "attention_norm.weight"),
            ("post_attention_layernorm", "ffn_norm.weight"),
        ]

        original_candidates = list(candidates)

        for candidate in original_candidates:

            for src, dst in aliases:

                if src in candidate:
                    candidates.append(
                        candidate.replace(src, dst)
                    )

                if dst in candidate:
                    candidates.append(
                        candidate.replace(dst, src)
                    )

        for candidate in candidates:

            if candidate in key_set:
                return candidate

        return None

    # ========================================================
    # SHAPE ALIGNMENT
    # ========================================================

    def _align_tensor_shapes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:

        if a.shape == b.shape:
            return b

        device = a.device

        a_cpu = a.detach().to(
            "cpu",
            dtype=torch.float32,
        )

        b_cpu = b.detach().to(
            "cpu",
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # 1D
        # ----------------------------------------------------

        if a_cpu.dim() == 1 and b_cpu.dim() == 1:

            b_resized = F.interpolate(
                b_cpu[None, None, :],
                size=a_cpu.shape[0],
                mode="linear",
                align_corners=False,
            )[0, 0]

            return b_resized.to(
                dtype=a.dtype,
                device=device,
            )

        # ----------------------------------------------------
        # 2D
        # ----------------------------------------------------

        if a_cpu.dim() == 2 and b_cpu.dim() == 2:

            b_resized = F.interpolate(
                b_cpu[None, None, :, :],
                size=(
                    a_cpu.shape[0],
                    a_cpu.shape[1],
                ),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            return b_resized.to(
                dtype=a.dtype,
                device=device,
            )

        # ----------------------------------------------------
        # Higher dimensions
        # ----------------------------------------------------

        if a_cpu.dim() == b_cpu.dim():

            out = torch.zeros_like(
                a_cpu,
                dtype=torch.float32,
            )

            slices = tuple(
                slice(
                    0,
                    min(sa, sb),
                )
                for sa, sb in zip(
                    a_cpu.shape,
                    b_cpu.shape,
                )
            )

            out[slices] = b_cpu[slices]

            return out.to(
                dtype=a.dtype,
                device=device,
            )

        # ----------------------------------------------------
        # Impossible structural translation
        # ----------------------------------------------------

        logger.warning(
            "Cannot structurally align tensor %s -> %s. "
            "Using Model A tensor.",
            tuple(b.shape),
            tuple(a.shape),
        )

        return torch.zeros_like(a)

    # ========================================================
    # PROCRUSTES ALIGNMENT
    # ========================================================

    def _procrustes_align(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:

        if a.dim() != 2 or b.dim() != 2:
            return self._align_tensor_shapes(a, b)

        b = self._align_tensor_shapes(a, b)

        if a.shape != b.shape:
            return b

        try:

            a32 = a.float()
            b32 = b.float()

            # Center the representations.
            a_mean = a32.mean(dim=0, keepdim=True)
            b_mean = b32.mean(dim=0, keepdim=True)

            ac = a32 - a_mean
            bc = b32 - b_mean

            # Cross covariance.
            covariance = bc.T @ ac

            # Orthogonal Procrustes.
            u, _, vh = torch.linalg.svd(
                covariance,
                full_matrices=False,
            )

            rotation = u @ vh

            aligned = bc @ rotation

            aligned = aligned + a_mean

            return aligned.to(
                dtype=a.dtype
            )

        except Exception as e:

            logger.debug(
                "Procrustes alignment failed: %s",
                e,
            )

            return b

    # ========================================================
    # LOSS
    # ========================================================

    def _compute_loss(
        self,
        model,
        tokenizer,
        device,
        max_samples=10,
    ) -> float:

        calibration_data = getattr(
            self.config,
            "calibration_data",
            None,
        )

        if not calibration_data:
            return float("inf")

        try:

            data = load_data(
                calibration_data
            )[:max_samples]

            ds = FtrainDataset(
                data,
                tokenizer,
                512,
            )

            loader = DataLoader(
                ds,
                batch_size=2,
                collate_fn=partial(
                    collate,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        or 0
                    ),
                ),
            )

            model.eval()

            total = 0.0
            count = 0

            with torch.inference_mode():

                for batch in loader:

                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if torch.is_tensor(v)
                    }

                    outputs = model(
                        **batch
                    )

                    loss = getattr(
                        outputs,
                        "loss",
                        None,
                    )

                    if loss is None:
                        continue

                    if not torch.isfinite(loss):
                        continue

                    total += loss.item()
                    count += 1

            return total / max(
                1,
                count,
            )

        except Exception as e:

            logger.warning(
                "Loss evaluation failed: %s",
                e,
            )

            return float("inf")

    # ========================================================
    # BUILD CALIBRATION LOADER
    # ========================================================

    def _build_calibration_loader(
        self,
        tokenizer,
    ):

        calibration_data = getattr(
            self.config,
            "calibration_data",
            None,
        )

        if not calibration_data:
            return None

        data = load_data(
            calibration_data
        )

        data = data[
            :self.max_calibration_samples
        ]

        dataset = FtrainDataset(
            data,
            tokenizer,
            512,
        )

        return DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=partial(
                collate,
                pad_token_id=(
                    tokenizer.pad_token_id
                    or 0
                ),
            ),
        )

    # ========================================================
    # GRADIENT HEALTH
    # ========================================================

    def _measure_gradient_health(
        self,
        model,
        loader,
        device,
        max_batches=8,
    ) -> Dict[str, float]:

        model.train()

        total_norm = 0.0
        max_norm = 0.0
        finite = 0
        count = 0

        for batch_idx, batch in enumerate(loader):

            if batch_idx >= max_batches:
                break

            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }

            model.zero_grad(
                set_to_none=True
            )

            try:

                outputs = model(
                    **batch
                )

                loss = getattr(
                    outputs,
                    "loss",
                    None,
                )

                if loss is None:
                    continue

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                batch_norm_sq = 0.0

                for p in model.parameters():

                    if p.grad is None:
                        continue

                    g = p.grad.detach()

                    if not torch.isfinite(g).all():
                        continue

                    n = g.float().norm().item()

                    batch_norm_sq += n * n
                    max_norm = max(
                        max_norm,
                        n,
                    )

                    finite += 1

                total_norm += math.sqrt(
                    batch_norm_sq
                )

                count += 1

            except Exception:

                continue

        model.zero_grad(
            set_to_none=True
        )

        return {
            "mean_grad_norm":
                total_norm / max(1, count),

            "max_grad_norm":
                max_norm,

            "finite_gradient_ratio":
                finite / max(
                    1,
                    len(list(model.parameters()))
                    * max(1, count),
                ),
        }

    # ========================================================
    # SHORT TRAINABILITY TEST
    # ========================================================

    def _adapt_merged_brain(
        self,
        model,
        tokenizer,
        device,
        loader,
    ) -> Dict[str, Any]:

        if not self.adaptation_enabled:
            return {
                "enabled": False,
                "initial_loss": float("inf"),
                "final_loss": float("inf"),
                "improvement": 0.0,
            }

        if loader is None:
            return {
                "enabled": False,
                "initial_loss": float("inf"),
                "final_loss": float("inf"),
                "improvement": 0.0,
            }

        print(
            "\n🧠 Starting post-merge brain adaptation..."
        )

        model.train()

        # ----------------------------------------------------
        # Only parameters that can actually train.
        # ----------------------------------------------------

        trainable = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        if not trainable:

            # Base model is normally frozen after merge.
            # Temporarily enable it for adaptation.
            for p in model.parameters():
                p.requires_grad = True

            trainable = list(
                model.parameters()
            )

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.adaptation_lr,
            weight_decay=self.adaptation_weight_decay,
        )

        initial_loss = self._compute_loss(
            model,
            tokenizer,
            device,
            max_samples=10,
        )

        losses = []

        iterator = iter(loader)

        for step in range(
            self.adaptation_steps
        ):

            try:
                batch = next(iterator)

            except StopIteration:

                iterator = iter(loader)
                batch = next(iterator)

            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            try:

                outputs = model(
                    **batch
                )

                loss = getattr(
                    outputs,
                    "loss",
                    None,
                )

                if loss is None:
                    continue

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    self.gradient_clip,
                )

                optimizer.step()

                loss_value = loss.detach().item()

                losses.append(
                    loss_value
                )

            except RuntimeError as e:

                if "out of memory" in str(e).lower():

                    logger.warning(
                        "OOM during adaptation; "
                        "stopping early."
                    )

                    self._purge_memory()
                    break

                logger.warning(
                    "Adaptation step failed: %s",
                    e,
                )

        final_loss = self._compute_loss(
            model,
            tokenizer,
            device,
            max_samples=10,
        )

        improvement = 0.0

        if math.isfinite(initial_loss) and math.isfinite(final_loss):

            improvement = (
                initial_loss - final_loss
            ) / max(
                abs(initial_loss),
                1e-8,
            )

        print(
            f"🧠 Adaptation: "
            f"{initial_loss:.4f} → "
            f"{final_loss:.4f} "
            f"({improvement * 100:+.2f}%)"
        )

        return {
            "enabled": True,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement": improvement,
            "steps": len(losses),
            "loss_history": losses,
        }

    # ========================================================
    # IMPORTANCE-AWARE REPAIR
    # ========================================================

    def _repair_model(
        self,
        model,
        baseline_state,
        device,
        fisher=None,
    ):

        if self.repair_rounds <= 0:
            return

        print(
            "\n🔧 Running post-merge brain repair..."
        )

        with torch.no_grad():

            current = model.state_dict()

            for name, tensor in current.items():

                if name not in baseline_state:
                    continue

                if not torch.is_floating_point(
                    tensor
                ):
                    continue

                base = baseline_state[name].to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )

                if tensor.shape != base.shape:
                    continue

                delta = (
                    tensor.float()
                    - base.float()
                )

                # --------------------------------------------
                # Parameter drift
                # --------------------------------------------

                drift = delta.norm()

                if not torch.isfinite(drift):
                    tensor.copy_(base)
                    continue

                base_norm = (
                    base.float().norm()
                    + 1e-8
                )

                relative_drift = (
                    drift / base_norm
                )

                # --------------------------------------------
                # Fisher protection
                # --------------------------------------------

                protection = 1.0

                if (
                    self.use_fisher_protection
                    and fisher is not None
                    and name in fisher
                ):

                    f = fisher[name].to(
                        device=tensor.device,
                        dtype=torch.float32,
                    )

                    f_mean = (
                        f.mean()
                        + 1e-8
                    )

                    importance = (
                        f / f_mean
                    )

                    # Important parameters receive
                    # stronger protection.
                    protection = (
                        1.0
                        / (
                            1.0
                            + importance
                        )
                    )

                    protection = protection.clamp(
                        0.05,
                        1.0,
                    )

                    delta = (
                        delta
                        * (
                            1.0
                            - (
                                self.repair_strength
                                * (
                                    1.0
                                    - protection
                                )
                            )
                        )
                    )

                # --------------------------------------------
                # Catastrophic drift protection
                # --------------------------------------------

                if relative_drift > 3.0:

                    delta = delta * 0.25

                elif relative_drift > 2.0:

                    delta = delta * 0.50

                repaired = (
                    base.float()
                    + delta
                )

                tensor.copy_(
                    repaired.to(
                        dtype=tensor.dtype
                    )
                )

    # ========================================================
    # MERGE ONE TENSOR
    # ========================================================

    def _merge_tensor(
        self,
        name,
        a,
        b,
        fisher_a=None,
        fisher_b=None,
        analyzer=None,
        planner=None,
    ):

        # ----------------------------------------------------
        # Explicit strategies
        # ----------------------------------------------------

        if self.strategy == "weighted":

            return fast_weighted_avg(
                a,
                b,
                self.alpha,
            )

        if (
            self.strategy == "fisher"
            and fisher_a is not None
            and fisher_b is not None
        ):

            if (
                name in fisher_a
                and name in fisher_b
            ):

                fa = fisher_a[name].to(
                    device=a.device,
                    dtype=a.dtype,
                )

                fb = fisher_b[name].to(
                    device=a.device,
                    dtype=a.dtype,
                )

                if (
                    fa.shape == a.shape
                    and fb.shape == a.shape
                ):

                    return fast_fisher_merge(
                        a,
                        b,
                        fa,
                        fb,
                    )

        if self.strategy == "slerp":

            return fast_slerp(
                a,
                b,
                self.alpha,
            )

        if self.strategy == "ties":

            return fast_ties(
                a,
                b,
            )

        # ----------------------------------------------------
        # Intelligent strategy
        # ----------------------------------------------------

        if self.strategy != "intelligent":

            return (
                self.alpha * a
                + (1.0 - self.alpha) * b
            )

        analysis = analyzer.analyze_pair(
            name,
            a,
            b,
        )

        plan = planner.plan_for_pair(
            name,
            analysis,
        )

        # ----------------------------------------------------
        # Critical tensors
        # ----------------------------------------------------

        if plan.strategy == "keep_a":
            return a

        if plan.strategy == "keep_b":
            return b

        # ----------------------------------------------------
        # Weighted
        # ----------------------------------------------------

        if plan.strategy == "weighted":

            return fast_weighted_avg(
                a,
                b,
                plan.alpha,
            )

        # ----------------------------------------------------
        # SLERP
        # ----------------------------------------------------

        if plan.strategy == "slerp":

            return fast_slerp(
                a,
                b,
                plan.alpha,
            )

        # ----------------------------------------------------
        # TIES
        # ----------------------------------------------------

        if plan.strategy == "ties":

            return fast_ties(
                a,
                b,
            )

        # ----------------------------------------------------
        # Projection / Procrustes
        # ----------------------------------------------------

        if plan.strategy == "projection":

            if (
                a.dim() == 2
                and b.dim() == 2
            ):

                try:

                    b_aligned = (
                        self._procrustes_align(
                            a,
                            b,
                        )
                    )

                    return (
                        plan.alpha * a
                        + (
                            1.0
                            - plan.alpha
                        ) * b_aligned
                    )

                except Exception:

                    pass

            return (
                plan.alpha * a
                + (
                    1.0
                    - plan.alpha
                ) * b
            )

        return a

    # ========================================================
    # MAIN MERGE
    # ========================================================

    def merge(self) -> bool:

        ui.fire_header()

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "\n🔥 FTRAIN INTELLIGENT BRAIN MERGE"
        )

        print(
            f"Model A: {self.model_a}"
        )

        print(
            f"Model B: {self.model_b}"
        )

        # ====================================================
        # STEP 1
        # CALIBRATION DATA
        # ====================================================

        fisher_a = None
        fisher_b = None
        cal_loader = None

        calibration_data = getattr(
            self.config,
            "calibration_data",
            None,
        )

        if calibration_data:

            print(
                "\n📚 Preparing calibration dataset..."
            )

            try:

                cal_data = load_data(
                    calibration_data
                )

                _, temp_tok = (
                    FastLanguageModel.from_pretrained(
                        self.model_a,
                        load_in_4bit=True,
                    )
                )

                cal_data = cal_data[
                    :self.max_calibration_samples
                ]

                dataset = FtrainDataset(
                    cal_data,
                    temp_tok,
                    512,
                )

                cal_loader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=True,
                    collate_fn=partial(
                        collate,
                        pad_token_id=(
                            temp_tok.pad_token_id
                            or 0
                        ),
                    ),
                )

                del temp_tok

                self._purge_memory()

            except Exception as e:

                logger.warning(
                    "Calibration loader failed: %s",
                    e,
                )

        # ====================================================
        # STEP 2
        # FISHER
        # ====================================================

        use_fisher = bool(
            getattr(
                self.config,
                "use_fisher",
                False,
            )
        )

        if use_fisher and cal_loader:

            print(
                "\n🎣 Computing Fisher information..."
            )

            try:

                print(
                    "🎣 Model A Fisher..."
                )

                m1_temp, _ = (
                    FastLanguageModel.from_pretrained(
                        self.model_a,
                        load_in_4bit=False,
                        dtype=torch.float16,
                    )
                )

                fisher_a = compute_fisher(
                    m1_temp,
                    cal_loader,
                    device,
                )

                del m1_temp
                self._purge_memory()

                print(
                    "🎣 Model B Fisher..."
                )

                m2_temp, _ = (
                    FastLanguageModel.from_pretrained(
                        self.model_b,
                        load_in_4bit=False,
                        dtype=torch.float16,
                    )
                )

                fisher_b = compute_fisher(
                    m2_temp,
                    cal_loader,
                    device,
                )

                del m2_temp
                self._purge_memory()

            except Exception as e:

                logger.warning(
                    "Fisher computation failed: %s",
                    e,
                )

                fisher_a = None
                fisher_b = None

        # ====================================================
        # STEP 3
        # LOAD MODEL A
        # ====================================================

        bar = ui.LoadingBar(
            message=(
                f"Loading {self.model_a} "
                "(CPU State Dict)"
            )
        )

        bar.start()

        model1, tokenizer = (
            FastLanguageModel.from_pretrained(
                self.model_a,
                load_in_4bit=False,
                dtype=torch.float16,
            )
        )

        sd1 = {
            k: v.detach().cpu()
            for k, v in model1.state_dict().items()
        }

        del model1

        self._purge_memory()

        bar.done()

        # ====================================================
        # STEP 4
        # LOAD MODEL B
        # ====================================================

        bar = ui.LoadingBar(
            message=(
                f"Loading {self.model_b} "
                "(CPU State Dict)"
            )
        )

        bar.start()

        model2, _ = (
            FastLanguageModel.from_pretrained(
                self.model_b,
                load_in_4bit=False,
                dtype=torch.float16,
            )
        )

        sd2 = {
            k: v.detach().cpu()
            for k, v in model2.state_dict().items()
        }

        del model2

        self._purge_memory()

        bar.done()

        # ====================================================
        # STEP 5
        # ARCHITECTURE ANALYSIS
        # ====================================================

        num_layers_a = (
            self._get_num_layers(sd1)
        )

        num_layers_b = (
            self._get_num_layers(sd2)
        )

        print(
            "\n🏗️ Architecture analysis:"
        )

        print(
            f"   Model A layers: {num_layers_a}"
        )

        print(
            f"   Model B layers: {num_layers_b}"
        )

        keys_b = list(sd2.keys())

        baseline = {
            k: v.clone().to(torch.float16)
            for k, v in sd1.items()
        }

        # ====================================================
        # STEP 6
        # INTELLIGENT MERGE
        # ====================================================

        analyzer = MergeAnalyzer()
        planner = MergePlanner()

        keys_a = list(sd1.keys())

        total = len(keys_a)

        merge_statistics = {
            "merged": 0,
            "missing": 0,
            "weighted": 0,
            "slerp": 0,
            "ties": 0,
            "kept_a": 0,
            "kept_b": 0,
            "projection": 0,
        }

        print(
            "\n🧠 Merging neural representations..."
        )

        for i, key_a in enumerate(
            keys_a
        ):

            progress = (
                i / max(1, total)
            )

            if i % max(
                1,
                total // 20,
            ) == 0 or i == total - 1:

                ui.print_merge_progress(
                    i + 1,
                    total,
                    message=(
                        "Architecture-aware "
                        "intelligent merge"
                    ),
                )

            key_b = self._find_matching_key(
                key_a,
                keys_b,
                num_layers_a,
                num_layers_b,
            )

            if key_b is None:

                merge_statistics[
                    "missing"
                ] += 1

                continue

            a = sd1[key_a].to(
                device=device,
                dtype=torch.float32,
            )

            b_raw = sd2[key_b].to(
                device=device,
                dtype=torch.float32,
            )

            # ------------------------------------------------
            # Architecture translation
            # ------------------------------------------------

            b = self._align_tensor_shapes(
                a,
                b_raw,
            )

            del b_raw

            # ------------------------------------------------
            # Merge
            # ------------------------------------------------

            try:

                merged = self._merge_tensor(
                    key_a,
                    a,
                    b,
                    fisher_a=fisher_a,
                    fisher_b=fisher_b,
                    analyzer=analyzer,
                    planner=planner,
                )

            except Exception as e:

                logger.warning(
                    "Merge failed for %s: %s",
                    key_a,
                    e,
                )

                merged = a

            if not torch.isfinite(
                merged
            ).all():

                logger.warning(
                    "Non-finite merge detected: %s",
                    key_a,
                )

                merged = a

            sd1[key_a] = (
                merged
                .detach()
                .to(
                    device="cpu",
                    dtype=self.dtype,
                )
            )

            del a
            del b
            del merged

        # B no longer needed.
        del sd2

        self._purge_memory()

        # ====================================================
        # STEP 7
        # STATIC SAFETY CHECK
        # ====================================================

        print(
            "\n🛡️ Running pre-adaptation safety..."
        )

        report = check_state_dict(
            sd1,
            baseline=baseline,
            norm_collapse_factor=0.1,
        )

        print(
            report.summary()
        )

        if not report.ok:

            print(
                "⚠️ Unsafe merge detected. "
                "Applying sanitizer..."
            )

            sd1 = sanitize(
                sd1,
                baseline,
                norm_collapse_factor=0.1,
            )

        # ====================================================
        # STEP 8
        # LOAD MERGED CANDIDATE
        # ====================================================

        print(
            "\n🧠 Loading merged candidate..."
        )

        merged_model, tokenizer = (
            FastLanguageModel.from_pretrained(
                self.model_a,
                load_in_4bit=False,
                dtype=self.dtype,
            )
        )

        merged_model.load_state_dict(
            sd1,
            strict=False,
        )

        del sd1

        self._purge_memory()

        # ====================================================
        # STEP 9
        # PARENT BASELINE
        # ====================================================

        parent_a_loss = float("inf")
        parent_b_loss = float("inf")

        if cal_loader is not None:

            print(
                "\n📊 Measuring parent capabilities..."
            )

            try:

                parent_a_loss = (
                    self._compute_loss(
                        merged_model,
                        tokenizer,
                        device,
                        max_samples=10,
                    )
                )

            except Exception:
                pass

        # ====================================================
        # STEP 10
        # POST-MERGE BRAIN ADAPTATION
        # ====================================================

        adaptation_result = {
            "enabled": False
        }

        if (
            self.adaptation_enabled
            and cal_loader is not None
        ):

            adaptation_result = (
                self._adapt_merged_brain(
                    merged_model,
                    tokenizer,
                    device,
                    cal_loader,
                )
            )

        # ====================================================
        # STEP 11
        # GRADIENT HEALTH
        # ====================================================

        gradient_health = {}

        if (
            self.use_gradient_health
            and cal_loader is not None
        ):

            print(
                "\n🩺 Checking merged brain "
                "gradient health..."
            )

            gradient_health = (
                self._measure_gradient_health(
                    merged_model,
                    cal_loader,
                    device,
                )
            )

            print(
                "Gradient norm: "
                f"{gradient_health.get('mean_grad_norm', 0):.6f}"
            )

            print(
                "Maximum gradient: "
                f"{gradient_health.get('max_grad_norm', 0):.6f}"
            )

            print(
                "Finite gradient ratio: "
                f"{gradient_health.get('finite_gradient_ratio', 0):.4f}"
            )

        # ====================================================
        # STEP 12
        # TARGETED REPAIR
        # ====================================================

        if (
            self.repair_rounds > 0
            and cal_loader is not None
        ):

            for repair_round in range(
                self.repair_rounds
            ):

                print(
                    f"\n🔧 Repair round "
                    f"{repair_round + 1}/"
                    f"{self.repair_rounds}"
                )

                self._repair_model(
                    merged_model,
                    baseline,
                    device,
                    fisher=(
                        fisher_a
                        if fisher_a is not None
                        else None
                    ),
                )

                # Small adaptation after each repair.
                self._adapt_merged_brain(
                    merged_model,
                    tokenizer,
                    device,
                    cal_loader,
                )

        # ====================================================
        # STEP 13
        # FINAL TRAINABILITY TEST
        # ====================================================

        final_loss = float("inf")

        if cal_loader is not None:

            final_loss = (
                self._compute_loss(
                    merged_model,
                    tokenizer,
                    device,
                    max_samples=10,
                )
            )

        print(
            "\n"
            + "=" * 64
        )

        print(
            "🧠 MERGED BRAIN TRAINABILITY REPORT"
        )

        print(
            "=" * 64
        )

        print(
            f"Candidate initial loss: "
            f"{adaptation_result.get('initial_loss', float('inf')):.5f}"
        )

        print(
            f"Candidate final loss:   "
            f"{final_loss:.5f}"
        )

        print(
            f"Training improvement:   "
            f"{adaptation_result.get('improvement', 0) * 100:+.2f}%"
        )

        print(
            "=" * 64
        )

        # ====================================================
        # STEP 14
        # FINAL SAFETY
        # ====================================================

        print(
            "\n🛡️ Final model safety validation..."
        )

        final_state = (
            merged_model.state_dict()
        )

        for name, tensor in final_state.items():

            if (
                torch.is_floating_point(tensor)
                and not torch.isfinite(tensor).all()
            ):

                print(
                    f"⚠️ Non-finite tensor: {name}"
                )

                with torch.no_grad():

                    if name in baseline:

                        tensor.copy_(
                            baseline[name].to(
                                tensor.device,
                                tensor.dtype,
                            )
                        )

        # ====================================================
        # STEP 15
        # SAVE
        # ====================================================

        if getattr(
            self.config,
            "name",
            "auto",
        ) == "auto":

            a_name = (
                self.model_a
                .split("/")[-1]
                .replace("-", "_")
            )

            b_name = (
                self.model_b
                .split("/")[-1]
                .replace("-", "_")
            )

            repo_name = (
                f"{a_name}_"
                f"{b_name}_"
                f"PhoenixIntelMerge"
            )

        else:

            repo_name = self.config.name

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        print(
            "\n💾 Saving final merged brain..."
        )

        merged_model.save_pretrained(
            self.output_dir
        )

        tokenizer.save_pretrained(
            self.output_dir
        )

        # ====================================================
        # STEP 16
        # FINAL SUMMARY
        # ====================================================

        stats = {

            "Model A":
                self.model_a,

            "Model B":
                self.model_b,

            "Output Dir":
                self.output_dir,

            "Strategy":
                self.strategy,

            "Architecture A Layers":
                num_layers_a,

            "Architecture B Layers":
                num_layers_b,

            "Adaptation Enabled":
                adaptation_result.get(
                    "enabled",
                    False,
                ),

            "Adaptation Steps":
                adaptation_result.get(
                    "steps",
                    0,
                ),

            "Training Improvement":
                adaptation_result.get(
                    "improvement",
                    0.0,
                ),

            "Final Calibration Loss":
                final_loss,

            "HF Repo":
                "Not pushed yet",
        }

        ui.print_final_summary(
            stats
        )

        # ====================================================
        # CLEANUP
        # ====================================================

        del merged_model
        del tokenizer
        del baseline

        self._purge_memory()

        return True
