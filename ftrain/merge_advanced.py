
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Iterator, Optional, Union, List

logger = logging.getLogger(__name__)

def compute_fisher(
    model: nn.Module,
    loader: Iterator,
    device: str = "cuda",
    num_samples: int = 50,
    use_amp: bool = True
) -> Optional[Dict[str, torch.Tensor]]:
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros(param.shape, dtype=torch.float32, device="cpu")

    if not fisher:
        logger.warning("⚠️ No trainable parameters found with requires_grad=True.")
        return None

    samples_processed = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        for batch in loader:
            if samples_processed >= num_samples:
                break

            input_ids = batch.get("input_ids")
            if input_ids is None:
                continue

            batch_size = input_ids.size(0)
            batch_device = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            model.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and "cuda" in device):
                outputs = model(**batch_device)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in fisher:
                        grad_sq = (param.grad.detach().float() ** 2).cpu()
                        fisher[name].add_(grad_sq)

            samples_processed += batch_size

        if samples_processed == 0:
            logger.error("❌ Zero valid samples processed during Fisher computation.")
            return None

        actual_samples = float(max(1, samples_processed))
        with torch.no_grad():
            for name in fisher:
                fisher[name].div_(actual_samples)

        logger.info(f"✅ Successfully computed Fisher diagonals over {samples_processed} samples.")
        return fisher

    except Exception as e:
        logger.warning(f"⚠️ Fisher computation failed: {str(e)}. Falling back to safe merge.")
        return None

def dare_merge(
    da: torch.Tensor,
    db: torch.Tensor,
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None
) -> torch.Tensor:
    if da.shape != db.shape:
        raise ValueError(f"Tensor shape mismatch in DARE merge: {da.shape} vs {db.shape}")

    if drop_rate <= 0.0:
        return db
    if drop_rate >= 1.0:
        return da

    with torch.no_grad():
        target_dtype = da.dtype
        device = da.device

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        delta = (db - da).to(torch.float32)
        
        keep_prob = 1.0 - drop_rate
        mask = torch.bernoulli(
            torch.full_like(delta, keep_prob, device=device),
            generator=generator
        )

        if rescale:
            mask.div_(keep_prob)

        res = da.to(torch.float32) + (delta * mask)
        return res.to(dtype=target_dtype)

def task_arithmetic(
    ma: Dict[str, torch.Tensor],
    mb: Dict[str, torch.Tensor],
    base: Dict[str, torch.Tensor],
    scaling: float = 0.5,
    max_norm_ratio: float = 2.5
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for k, p_base in base.items():
            if k not in ma:
                continue

            ta = (ma[k] - p_base).to(torch.float32)
            tb = (mb[k] - p_base).to(torch.float32) if k in mb else torch.zeros_like(ta)

            combined_delta = scaling * (ta + tb)
            base_fp32 = p_base.to(torch.float32)
            candidate = base_fp32 + combined_delta

            base_norm = float(base_fp32.norm()) + 1e-8
            cand_norm = float(candidate.norm())
            ratio = cand_norm / base_norm

            if ratio > max_norm_ratio:
                combined_delta.mul_(max_norm_ratio * base_norm / cand_norm)
                candidate = base_fp32 + combined_delta

            merged[k] = candidate.to(dtype=p_base.dtype)

    return merged

def ties_merge_state_dict(
    models: List[Dict[str, torch.Tensor]],
    base: Dict[str, torch.Tensor],
    density: float = 0.2,
    scaling: float = 1.0
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for k, p_base in base.items():
            deltas = []
            for m in models:
                if k in m:
                    deltas.append((m[k] - p_base).to(torch.float32))

            if not deltas:
                merged[k] = p_base
                continue

            stacked_deltas = torch.stack(deltas, dim=0)

            if density < 1.0:
                k_val = max(1, int(stacked_deltas.numel() / len(models) * density))
                flat_deltas = stacked_deltas.abs().view(len(models), -1)
                thresholds = torch.topk(flat_deltas, k_val, dim=1).values[:, -1].view(-1, *([1] * (stacked_deltas.dim() - 1)))
                mask = stacked_deltas.abs() >= thresholds
                stacked_deltas.mul_(mask.float())

            sign_votes = torch.sign(stacked_deltas).sum(dim=0)
            elected_sign = torch.sign(sign_votes)

            matching_mask = (torch.sign(stacked_deltas) == elected_sign.unsqueeze(0)) & (elected_sign.unsqueeze(0) != 0)
            filtered_deltas = stacked_deltas * matching_mask.float()

            counts = matching_mask.sum(dim=0).clamp(min=1)
            final_delta = (filtered_deltas.sum(dim=0) / counts) * scaling

            merged[k] = (p_base.to(torch.float32) + final_delta).to(dtype=p_base.dtype)

    return merged
