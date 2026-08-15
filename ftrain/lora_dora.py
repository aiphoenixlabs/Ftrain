import math, torch, torch.nn as nn
from typing import List
from .kernels_dora import DoraFusedFunction

class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        with torch.no_grad():
            self.magnitude = nn.Parameter(base.weight.norm(p=2, dim=1).to(torch.float32))
        self.scale = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if self.training:
            if isinstance(self.dropout, nn.Dropout) and self.dropout.p > 0:
                x = self.dropout(x)
            l = self.lora_B(self.lora_A(x)) * self.scale
            b = self.base(x)
            c = b + l
            return DoraFusedFunction.apply(c, self.magnitude)
        else:
            b = self.base(x)
            l = self.lora_B(self.lora_A(x)) * self.scale
            c = b + l
            c_f32 = c.to(torch.float32)
            n = c_f32.norm(p=2, dim=-1, keepdim=True) + 1e-8
            return ((c_f32 / n) * self.magnitude).to(c.dtype)

def inject_dora(model, targets, r, alpha, dropout=0.0):
    c = 0
    for name, mod in list(model.named_modules()):
        if any(name.endswith("." + t) or name == t for t in targets) and isinstance(mod, nn.Linear):
            if isinstance(mod, DoRALinear):
                continue
            pn, _, cn = name.rpartition('.')
            p = model.get_submodule(pn) if pn else model
            setattr(p, cn, DoRALinear(mod, r, alpha, dropout))
            c += 1
    return c
