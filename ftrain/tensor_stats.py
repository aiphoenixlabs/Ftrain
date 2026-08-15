import math, torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TensorStats:
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    mean: float = 0.0
    std: float = 0.0
    l2_norm: float = 0.0
    spectral_norm: float = 0.0
    effective_rank: float = 0.0
    entropy: float = 0.0
    sparsity: float = 0.0
    dead: bool = False
    singular_values: List[float] = field(default_factory=list)

def compute_tensor_stats(name, t):
    s = TensorStats(name, tuple(t.shape), t.numel(), str(t.dtype))
    if t.numel() == 0:
        s.dead = True
        return s
    tf = t.detach().float()
    if torch.isnan(tf).any() or torch.isinf(tf).any():
        s.dead = True
        return s
    s.mean = float(tf.mean())
    s.std = float(tf.std())
    s.l2_norm = float(torch.linalg.norm(tf))
    if s.l2_norm < 1e-12:
        s.dead = True
        return s
    try:
        two = tf if tf.dim() == 2 else (tf.reshape(tf.shape[0], -1) if tf.dim() > 2 else tf.unsqueeze(0))
        sv = torch.linalg.svdvals(two).clamp_min(1e-12)
        s.spectral_norm = float(sv[0])
        p = sv / sv.sum()
        h = -(p * p.log()).sum().item()
        s.entropy = h / math.log(max(2, sv.numel()))
        s.effective_rank = math.exp(h) / max(1.0, float(sv.numel()))
        s.singular_values = sv.clamp_min(1e-12)[:16].tolist()
        s.sparsity = float((tf.abs() < max(1e-5, s.l2_norm * 1e-3 / math.sqrt(s.numel))).float().mean())
    except:
        pass
    return s
