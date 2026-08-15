import torch

def compute_fisher(model, loader, dev, n=50):
    f = {}
    model.eval()
    for nm, p in model.named_parameters():
        if p.requires_grad:
            f[nm] = torch.zeros_like(p, device=dev)
    sp = 0
    for b in loader:
        if sp >= n:
            break
        b = {k: v.to(dev) for k, v in b.items()}
        model.zero_grad()
        batch_size = b['input_ids'].size(0)
        model(**b).loss.backward()
        with torch.no_grad():
            for nm, p in model.named_parameters():
                if p.grad is not None:
                    f[nm] += p.grad.data ** 2
        sp += batch_size
    for nm in f:
        f[nm] /= max(1, sp)
    return f

def dare_merge(da, db, dr=0.9, rescale=True):
    m = torch.bernoulli(torch.full_like(da, 1.0 - dr))
    if rescale and dr < 1.0:
        m /= (1.0 - dr)
    return da + m * (db - da)

def task_arithmetic(ma, mb, base, sc=0.5):
    m = {}
    for k in ma:
        if k not in base:
            continue
        ta = ma[k] - base[k]
        tb = mb[k] - base[k] if k in mb else torch.zeros_like(ta)
        m[k] = base[k] + sc * (ta + tb)
    return m
