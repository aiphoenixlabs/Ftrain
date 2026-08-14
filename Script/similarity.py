import torch

def _f(t):
    return t.unsqueeze(0) if t.dim() == 1 else (t if t.dim() == 2 else t.reshape(t.shape[0], -1))

def cos_w(a, b):
    if a.shape != b.shape:
        if a.numel() == b.numel():
            a, b = a.reshape(-1), b.reshape(-1)
        else:
            return 0.0
    af, bf = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return float(torch.dot(af, bf) / ((af.norm() + 1e-12) * (bf.norm() + 1e-12)))

def cka(a, b, k="linear"):
    a, b = _f(a.detach().float()), _f(b.detach().float())
    if a.shape[1] != b.shape[1]:
        n = min(a.shape[1], b.shape[1])
        a, b = a[:,:n], b[:,:n]
    ga = a @ a.t() if k == "linear" else torch.exp(-torch.cdist(a,a).pow(2))
    gb = b @ b.t() if k == "linear" else torch.exp(-torch.cdist(b,b).pow(2))
    n = a.shape[0]
    if n < 2:
        return 0.0
    h = torch.eye(n, device=a.device) - 1.0/n
    k_, l_ = h @ ga @ h, h @ gb @ h
    hsic = (k_ * l_).sum() / ((n - 1) ** 2)
    return float(hsic / (hsic * hsic + 1e-12).sqrt())

def no(a, b):
    a, b = _f(a.detach().float()), _f(b.detach().float())
    if a.shape != b.shape:
        return 0.0
    na, nb = a.norm(dim=1, keepdim=True).clamp_min(1e-12), b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float(((a/na) * (b/nb)).sum(dim=1).abs().mean())

def svo(a, b):
    try:
        sa, sb = torch.linalg.svdvals(_f(a.detach().float())).float(), torch.linalg.svdvals(_f(b.detach().float())).float()
        n = min(sa.numel(), sb.numel())
        sa, sb = sa[:n], sb[:n]
        return cos_w(sa, sb)
    except:
        return 0.0

def similarity_bundle(a, b):
    return {"cosine": cos_w(a,b), "cka_linear": cka(a,b,"linear"), "cka_rbf": cka(a,b,"rbf"), "neuron_overlap": no(a,b), "sv_overlap": svo(a,b)}

def aggregate_similarity(b):
    w = {"cosine": 0.4, "cka_linear": 0.25, "cka_rbf": 0.15, "neuron_overlap": 0.1, "sv_overlap": 0.1}
    return sum(w[k] * max(0.0, min(1.0, abs(b.get(k, 0.0)))) for k in w if k in w)
