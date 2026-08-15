import os, math, torch
from torch.utils.cpp_extension import load
from typing import Optional, Any

_here = os.path.dirname(os.path.abspath(__file__))
_kernel = os.path.join(_here, "merge_kernel.cu")
_ext = None
_failed = False

def _mod():
    global _ext, _failed
    if _ext is not None:
        return _ext
    if _failed:
        return None
    try:
        _ext = load(name="ftrain_merge", sources=[_kernel], extra_cflags=['-O3', '-fopenmp'], extra_cuda_cflags=['-O3', '-use_fast_math'], extra_ldflags=['-fopenmp'], verbose=False)
        print("✅ C++ merge kernel compiled")
    except:
        _failed = True
        _ext = None
    return _ext

@torch.jit.script
def _jit_w(a: torch.Tensor, b: torch.Tensor, a_: float) -> torch.Tensor:
    return a_ * a + (1.0 - a_) * b

@torch.jit.script
def _jit_f(a: torch.Tensor, b: torch.Tensor, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    fa = fa + 1e-8
    fb = fb + 1e-8
    return (fa * a + fb * b) / (fa + fb)

@torch.jit.script
def _jit_s(a: torch.Tensor, b: torch.Tensor, ta: float, tb: float) -> torch.Tensor:
    return ta * a + tb * b

def _cuda(a, b):
    return a.is_cuda and b.is_cuda

def fast_weighted_avg(a, b, alpha=0.5, use_cuda=True):
    m = _mod()
    if m and _cuda(a, b) and use_cuda:
        return m.weighted_avg_cuda(a.contiguous().float(), b.contiguous().float(), alpha).to(a.dtype)
    elif m:
        return m.weighted_avg_cpu(a.contiguous().float(), b.contiguous().float(), alpha).to(a.dtype)
    return _jit_w(a.float(), b.float(), alpha).to(a.dtype)

def fast_fisher_merge(a, b, fa, fb, use_cuda=True):
    m = _mod()
    if m and _cuda(a, b) and use_cuda:
        return m.fisher_merge_cuda(a.contiguous().float(), b.contiguous().float(), fa.contiguous().float(), fb.contiguous().float()).to(a.dtype)
    elif m:
        return m.fisher_merge_cpu(a.contiguous().float(), b.contiguous().float(), fa.contiguous().float(), fb.contiguous().float()).to(a.dtype)
    return _jit_f(a.float(), b.float(), fa.float(), fb.float()).to(a.dtype)

def fast_slerp(a, b, alpha=0.5, use_cuda=True):
    m = _mod()
    t1, t2 = a.float().flatten(), b.float().flatten()
    n1, n2 = t1.norm(), t2.norm()
    if n1 < 1e-8 or n2 < 1e-8:
        return (alpha * a.float() + (1 - alpha) * b.float()).to(a.dtype)
    dot = (t1 * t2).sum()
    c = (dot / (n1 * n2)).clamp(-1.0, 1.0)
    omega = torch.acos(c).item()
    sin_om = math.sin(omega) if math.sin(omega) > 1e-6 else 1e-6
    ta = math.sin((1 - alpha) * omega) / sin_om
    tb = math.sin(alpha * omega) / sin_om
    if m and _cuda(a, b) and use_cuda:
        return m.slerp_merge_cuda(a.contiguous().float(), b.contiguous().float(), alpha, omega, sin_om).to(a.dtype)
    return _jit_s(a.float(), b.float(), ta, tb).to(a.dtype)

def fast_ties(a, b, density=0.5, use_cuda=True):
    m = _mod()
    k = max(1, int(density * a.numel()))
    ma = (a.abs() >= torch.topk(a.abs().flatten(), k).values.min()).float()
    mb = (b.abs() >= torch.topk(b.abs().flatten(), k).values.min()).float()
    if m and _cuda(a, b) and use_cuda:
        return m.ties_merge_cuda(a.contiguous().float(), b.contiguous().float(), ma.contiguous().float(), mb.contiguous().float()).to(a.dtype)
    sa, sb = a.sign(), b.sign()
    cons = (sa == sb) & (sa != 0) & (ma > 0.5) & (mb > 0.5)
    return torch.where(cons, (a + b) * 0.5, a).to(a.dtype)
