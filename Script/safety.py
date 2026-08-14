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

    def summary(self):
        if self.ok:
            return "✅ SAFE"
        m = "❌ UNSAFE ("
        if self.nan_keys: m += "NaN "
        if self.inf_keys: m += "Inf "
        if self.exploded_keys: m += "Explosion "
        if self.collapsed_keys: m += "Collapse"
        return m + ")"

def check_state_dict(sd, baseline=None, ne=10.0, nc=0.1):
    r = SafetyReport()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        vf = v.detach().float()
        if torch.isnan(vf).any():
            r.ok = False
            r.nan_keys.append(k)
            continue
        if torch.isinf(vf).any():
            r.ok = False
            r.inf_keys.append(k)
            continue
        if baseline and k in baseline:
            rt = float(vf.norm()) / (float(baseline[k].detach().float().norm()) + 1e-9)
            if rt > ne:
                r.ok = False
                r.exploded_keys.append(k)
            elif rt < nc:
                r.ok = False
                r.collapsed_keys.append(k)
    return r

def sanitize(sd, baseline, ne=10.0, nc=0.1):
    out = {}
    for k, v in sd.items():
        vf = v.detach().float()
        if torch.isnan(vf).any() or torch.isinf(vf).any():
            out[k] = baseline.get(k, v)
            continue
        if k in baseline:
            rt = float(vf.norm()) / (float(baseline[k].float().norm()) + 1e-9)
            if rt > ne or rt < nc:
                out[k] = baseline[k]
                continue
        out[k] = v
    return out
