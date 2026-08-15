
from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class SafetyReport:
    ok: bool = True
    nan_keys: List[str] = field(default_factory=list)
    inf_keys: List[str] = field(default_factory=list)
    exploded_keys: List[str] = field(default_factory=list)
    collapsed_keys: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        if self.ok: return "✅ SAFE"
        msg = "❌ UNSAFE ("
        if self.nan_keys: msg += "NaN "
        if self.inf_keys: msg += "Inf "
        if self.exploded_keys: msg += "Explosion "
        if self.collapsed_keys: msg += "Collapse"
        return msg + ")"

def check_state_dict(sd: dict, baseline: dict = None, norm_explosion_factor: float = 10.0, norm_collapse_factor: float = 0.1) -> SafetyReport:
    rep = SafetyReport()
    for k, v in sd.items():
        if not torch.is_tensor(v): continue
        vf = v.detach().float()
        if torch.isnan(vf).any(): rep.ok = False; rep.nan_keys.append(k); continue
        if torch.isinf(vf).any(): rep.ok = False; rep.inf_keys.append(k); continue
        if baseline and k in baseline:
            curr_norm = float(vf.norm())
            base_norm = float(baseline[k].detach().float().norm()) + 1e-9
            ratio = curr_norm / base_norm
            if ratio > norm_explosion_factor: rep.ok = False; rep.exploded_keys.append(k)
            elif ratio < norm_collapse_factor: rep.ok = False; rep.collapsed_keys.append(k)
    return rep

def sanitize(sd: dict, baseline: dict, norm_explosion_factor: float = 10.0, norm_collapse_factor: float = 0.1) -> dict:
    out = {}
    for k, v in sd.items():
        vf = v.detach().float()
        if torch.isnan(vf).any() or torch.isinf(vf).any():
            out[k] = baseline.get(k, v); continue
        if k in baseline:
            curr_norm = float(vf.norm())
            base_norm = float(baseline[k].float().norm()) + 1e-9
            ratio = curr_norm / base_norm
            if ratio > norm_explosion_factor or ratio < norm_collapse_factor:
                out[k] = baseline[k]; continue
        out[k] = v
    return out
