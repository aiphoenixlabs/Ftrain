"""
🔥 FTRAIN Advanced Merging (v8.0.6) 🔥
"""

import torch
import logging
from typing import Dict, Any, Iterator

logger = logging.getLogger(__name__)

def compute_fisher(model: torch.nn.Module, loader: Iterator, device: str, num_samples: int = 50) -> Dict[str, torch.Tensor]:
    fisher = {}
    model.eval()
    for n, p in model.named_parameters():
        if p.requires_grad: fisher[n] = torch.zeros_like(p, device=device)
    samples_processed = 0
    
    try:
        for batch in loader:
            batch_size = batch['input_ids'].size(0)
            if samples_processed >= num_samples: break
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None: continue
            loss.backward()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None: fisher[n] += p.grad.data ** 2
            samples_processed += batch_size
        actual_samples = max(1, samples_processed)
        with torch.no_grad():
            for n in fisher: fisher[n] /= actual_samples
        return fisher
    except RuntimeError as e:
        logger.warning(f"⚠️ Fisher computation failed due to an in-place operation (likely Unsloth patching). Falling back to safe merge.")
        return None

def dare_merge(da: torch.Tensor, db: torch.Tensor, drop_rate: float = 0.9, rescale: bool = True) -> torch.Tensor:
    if da.shape != db.shape:
        raise ValueError(f"Tensor shape mismatch in DARE merge: {da.shape} vs {db.shape}")
    with torch.no_grad():
        mask = torch.bernoulli(torch.full_like(da, 1.0 - drop_rate))
        if rescale and drop_rate < 1.0: mask /= (1.0 - drop_rate)
        return da + mask * (db - da)

def task_arithmetic(ma: Dict[str, torch.Tensor], mb: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor], scaling: float = 0.5) -> Dict[str, torch.Tensor]:
    merged = {}
    with torch.no_grad():
        for k in ma:
            if k not in base: continue
            ta = ma[k] - base[k]
            if k in mb: tb = mb[k] - base[k]
            else: tb = torch.zeros_like(ta)
            mt = scaling * (ta + tb)
            merged[k] = base[k] + mt
    return merged
