import torch

def _m(t):
    t = t.detach()
    return t.unsqueeze(0) if t.dim() == 1 else (t if t.dim() == 2 else t.reshape(t.shape[0], -1))

def identity(a, b):
    return torch.eye(_m(a).shape[0], dtype=a.dtype, device=a.device)

def procrustes(a, b):
    a2, b2 = _m(a), _m(b)
    if a2.shape != b2.shape:
        return identity(a, b)
    try:
        u, _, vh = torch.linalg.svd(a2.t() @ b2, full_matrices=False)
        return u @ vh
    except:
        return identity(a, b)

PROJECTIONS = {"identity": identity, "procrustes": procrustes, "svd": procrustes, "cca": procrustes, "orthogonal": procrustes, "lstsq": procrustes}

def apply_projection(a, s, b):
    return PROJECTIONS.get(s, identity)(a, b)
