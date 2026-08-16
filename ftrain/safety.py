
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

@dataclass
class SafetyReport:
    ok: bool = True
    total_tensors: int = 0
    safe_tensors: int = 0
    nan_keys: List[str] = field(default_factory=list)
    inf_keys: List[str] = field(default_factory=list)
    exploded_keys: List[str] = field(default_factory=list)
    collapsed_keys: List[str] = field(default_factory=list)
    zero_keys: List[str] = field(default_factory=list)
    max_ratio: float = 1.0
    min_ratio: float = 1.0

    def summary(self) -> str:
        if self.ok:
            return f"✅ SAFE ({self.safe_tensors}/{self.total_tensors} Tensors Clean)"
        
        issues = []
        if self.nan_keys: issues.append(f"NaN: {len(self.nan_keys)}")
        if self.inf_keys: issues.append(f"Inf: {len(self.inf_keys)}")
        if self.exploded_keys: issues.append(f"Explosion: {len(self.exploded_keys)}")
        if self.collapsed_keys: issues.append(f"Collapse: {len(self.collapsed_keys)}")
        if self.zero_keys: issues.append(f"Zeroed: {len(self.zero_keys)}")
        
        return f"❌ UNSAFE ({', '.join(issues)}) | Max Ratio: {self.max_ratio:.2f}x, Min Ratio: {self.min_ratio:.2f}x"


def check_state_dict(
    sd: dict,
    baseline: Optional[dict] = None,
    norm_explosion_factor: float = 5.0,
    norm_collapse_factor: float = 0.2
) -> SafetyReport:
    """
    Performs memory-efficient numerical health checks across all model weights.
    """
    rep = SafetyReport()
    tensor_keys = [k for k, v in sd.items() if torch.is_tensor(v)]
    rep.total_tensors = len(tensor_keys)
    
    for k in tensor_keys:
        v = sd[k]
        
        # Memory-efficient NaN / Inf checks
        if torch.isnan(v).any().item():
            rep.ok = False
            rep.nan_keys.append(k)
            continue
            
        if torch.isinf(v).any().item():
            rep.ok = False
            rep.inf_keys.append(k)
            continue

        # Low-memory Norm calculation (prevents FP16 overflow without full float copy)
        curr_norm = float(v.norm(dtype=torch.float32))
        
        if curr_norm == 0.0:
            rep.ok = False
            rep.zero_keys.append(k)
            continue

        if baseline and k in baseline and torch.is_tensor(baseline[k]):
            base_norm = float(baseline[k].norm(dtype=torch.float32)) + 1e-9
            ratio = curr_norm / base_norm
            
            rep.max_ratio = max(rep.max_ratio, ratio)
            rep.min_ratio = min(rep.min_ratio, ratio)
            
            if ratio > norm_explosion_factor:
                rep.ok = False
                rep.exploded_keys.append(k)
            elif ratio < norm_collapse_factor:
                rep.ok = False
                rep.collapsed_keys.append(k)

    bad_count = len(rep.nan_keys) + len(rep.inf_keys) + len(rep.exploded_keys) + len(rep.collapsed_keys) + len(rep.zero_keys)
    rep.safe_tensors = rep.total_tensors - bad_count
    return rep


def sanitize(
    sd: dict,
    baseline: Optional[dict] = None,
    norm_explosion_factor: float = 5.0,
    norm_collapse_factor: float = 0.2,
    mode: str = "rescale",
    inplace: bool = True
) -> dict:
    """
    Sanitizes damaged model state dicts.

    Parameters:
    - mode: 
      - "rescale": Preserves weight direction/knowledge while scaling norm to safe boundaries.
      - "fallback": Replaces damaged weights completely with baseline weights.
    - inplace: Mutates the dictionary in-place to prevent RAM spikes.
    """
    out = sd if inplace else {}
    
    for k, v in list(sd.items()):
        if not torch.is_tensor(v):
            if not inplace: out[k] = v
            continue
            
        # 1. Critical Failures (NaN / Inf) -> Fallback to Baseline
        if torch.isnan(v).any() or torch.isinf(v).any():
            if baseline and k in baseline:
                out[k] = baseline[k].to(dtype=v.dtype, device=v.device)
            else:
                out[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
            continue

        # 2. Norm Anomalies Check
        if baseline and k in baseline and torch.is_tensor(baseline[k]):
            curr_norm = float(v.norm(dtype=torch.float32))
            base_norm = float(baseline[k].norm(dtype=torch.float32)) + 1e-9
            ratio = curr_norm / base_norm

            if ratio > norm_explosion_factor or ratio < norm_collapse_factor or curr_norm == 0.0:
                if mode == "rescale" and curr_norm > 0.0:
                    # Smart Norm Rescaling: Preserve direction, adjust magnitude
                    target_ratio = norm_explosion_factor if ratio > norm_explosion_factor else norm_collapse_factor
                    target_norm = base_norm * target_ratio
                    scale_factor = target_norm / curr_norm
                    out[k] = (v * scale_factor).to(v.dtype)
                else:
                    # Fallback Mode: Revert directly to baseline tensor
                    out[k] = baseline[k].to(dtype=v.dtype, device=v.device)
            else:
                if not inplace: out[k] = v
        else:
            if not inplace: out[k] = v

    return out
