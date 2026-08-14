import os, random, numpy as np
from typing import Dict, Union, Any

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass

def get_family(name: str) -> str:
    n = (name or "").lower()
    for f in ["qwen", "deepseek", "llama", "gemma", "phi", "mistral"]:
        if f in n:
            return f
    return "generic"

def get_num_layers(model: Any) -> int:
    c = getattr(model, "config", model)
    for a in ("num_hidden_layers", "num_layers", "n_layer"):
        if hasattr(c, a):
            return int(getattr(c, a))
    return 0

def is_moe(model: Any) -> bool:
    cfg = getattr(model, "config", None)
    if not cfg:
        return False
    for a in ("num_local_experts", "num_experts"):
        if hasattr(cfg, a) and int(getattr(cfg, a)) > 1:
            return True
    return False

def count_params(model: Any) -> Dict[str, Union[int, float, str]]:
    t = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": t, "trainable": tr, "pct_trainable": 100.0 * tr / max(1, t)}
