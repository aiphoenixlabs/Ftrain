
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

def _flatten_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() <= 2:
        return t.reshape(-1, t.shape[-1]) if t.dim() == 2 else t.unsqueeze(0)
    return t.reshape(t.shape[0], -1)

def _safe_subsample(a: torch.Tensor, b: torch.Tensor, max_rows: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    a_f = _flatten_2d(a.detach().to(torch.float32))
    b_f = _flatten_2d(b.detach().to(torch.float32))

    if a_f.shape[1] != b_f.shape[1]:
        n_cols = min(a_f.shape[1], b_f.shape[1])
        a_f, b_f = a_f[:, :n_cols], b_f[:, :n_cols]

    if a_f.shape[0] > max_rows:
        a_f = a_f[:max_rows]
        b_f = b_f[:max_rows]

    return a_f, b_f

def cosine_similarity_weights(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = _safe_subsample(a, b, max_rows=4096)
    a_flat = a_f.reshape(-1)
    b_flat = b_f.reshape(-1)

    norm_a = torch.linalg.norm(a_flat)
    norm_b = torch.linalg.norm(b_flat)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = torch.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-12)

    if torch.isnan(sim) or torch.isinf(sim):
        return 0.0

    return float(torch.clamp(sim, -1.0, 1.0).item())

def cka(a: torch.Tensor, b: torch.Tensor, kernel: str = "linear") -> float:
    a_f, b_f = _safe_subsample(a, b, max_rows=2048)
    n = a_f.shape[0]
    
    if n < 2:
        return 0.0

    if kernel == "linear":
        ga = a_f @ a_f.T
        gb = b_f @ b_f.T
    else:
        dist_a = torch.cdist(a_f, a_f).pow(2)
        dist_b = torch.cdist(b_f, b_f).pow(2)

        sig_a = torch.median(dist_a)
        sig_b = torch.median(dist_b)

        sig_a = sig_a if sig_a > 1e-8 else torch.tensor(1.0, device=a.device)
        sig_b = sig_b if sig_b > 1e-8 else torch.tensor(1.0, device=b.device)

        ga = torch.exp(-dist_a / (2.0 * sig_a))
        gb = torch.exp(-dist_b / (2.0 * sig_b))

    h = torch.eye(n, device=a_f.device, dtype=torch.float32) - (1.0 / n)
    k_c = h @ ga @ h
    l_c = h @ gb @ h

    hsic_kl = (k_c * l_c).sum()
    hsic_kk = (k_c * k_c).sum()
    hsic_ll = (l_c * l_c).sum()

    denom = torch.sqrt(hsic_kk * hsic_ll) + 1e-12
    score = hsic_kl / denom

    if torch.isnan(score) or torch.isinf(score):
        return 0.0

    return float(torch.clamp(score, 0.0, 1.0).item())

def neuron_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = _safe_subsample(a, b, max_rows=2048)

    a_norm = F.normalize(a_f, p=2, dim=1, eps=1e-12)
    b_norm = F.normalize(b_f, p=2, dim=1, eps=1e-12)

    row_sims = (a_norm * b_norm).sum(dim=1).abs()
    score = row_sims.mean()

    if torch.isnan(score) or torch.isinf(score):
        return 0.0

    return float(score.item())

def sv_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 64) -> float:
    try:
        a_f, b_f = _safe_subsample(a, b, max_rows=2048)
        min_dim = min(a_f.shape)
        
        if min_dim == 0:
            return 0.0

        q = min(k, min_dim)
        if min_dim > 128:
            _, sa, _ = torch.svd_lowrank(a_f, q=q)
            _, sb, _ = torch.svd_lowrank(b_f, q=q)
        else:
            sa = torch.linalg.svdvals(a_f)
            sb = torch.linalg.svdvals(b_f)

        n = min(sa.numel(), sb.numel())
        sa, sb = sa[:n], sb[:n]
        return cosine_similarity_weights(sa, sb)
    except Exception:
        return 0.0

def similarity_bundle(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    return {
        "cosine": cosine_similarity_weights(a, b),
        "cka_linear": cka(a, b, "linear"),
        "cka_rbf": cka(a, b, "rbf"),
        "neuron_overlap": neuron_overlap(a, b),
        "sv_overlap": sv_overlap(a, b)
    }

def aggregate_similarity(b: Dict[str, float]) -> float:
    w = {
        "cosine": 0.35,
        "cka_linear": 0.25,
        "cka_rbf": 0.15,
        "neuron_overlap": 0.15,
        "sv_overlap": 0.10
    }
    total_weight = 0.0
    weighted_sum = 0.0

    for key, weight in w.items():
        val = b.get(key, 0.0)
        if not (math.isnan(val) or math.isinf(val)):
            weighted_sum += weight * max(0.0, min(1.0, abs(val)))
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0
