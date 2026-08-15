import torch, triton, triton.language as tl
from typing import Tuple

@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64})],
    key=['M', 'N']
)
@triton.jit
def _fwd(cp, mag, out, M, N, sc_m, sc_n, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mm = rm[:, None] < M
    mn = rn[None, :] < N
    c = tl.load(cp + rm[:, None] * sc_m + rn[None, :] * sc_n, mask=mm & mn, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(c * c, axis=1, keepdim=True)) + 1e-8
    m = tl.load(mag + rn, mask=mn, other=0.0)
    tl.store(out + rm[:, None] * sc_m + rn[None, :] * sc_n, (c / n * m).to(c.dtype), mask=mm & mn)

@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64})],
    key=['M', 'N']
)
@triton.jit
def _bwd(go, cp, mag, gc, gm, M, N, sg_m, sg_n, sc_m, sc_n, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mm = rm[:, None] < M
    mn = rn[None, :] < N
    g = tl.load(go + rm[:, None] * sg_m + rn[None, :] * sg_n, mask=mm & mn, other=0.0).to(tl.float32)
    c = tl.load(cp + rm[:, None] * sc_m + rn[None, :] * sc_n, mask=mm & mn, other=0.0).to(tl.float32)
    m = tl.load(mag + rn, mask=mn, other=0.0).to(tl.float32)
    ns = tl.sum(c * c, axis=1, keepdim=True) + 1e-8
    inv = 1.0 / tl.sqrt(ns)
    inv3 = inv * inv * inv
    dc = (m * inv) * g - (m * inv3) * tl.sum(c * g, axis=1, keepdim=True) * c
    dm = tl.sum(c * g * inv, axis=0)
    tl.store(gc + rm[:, None] * sc_m + rn[None, :] * sc_n, dc.to(c.dtype), mask=mm & mn)
    tl.atomic_add(gm + rn, dm.to(m.dtype), mask=mn)

class DoraFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comb, mag):
        comb, mag = comb.contiguous(), mag.contiguous()
        M, N = comb.shape
        out = torch.empty_like(comb)
        grid = lambda m: (triton.cdiv(M, m['BLOCK_M']),)
        _fwd[grid](comb, mag, out, M, N, comb.stride(0), comb.stride(1))
        ctx.save_for_backward(comb, mag)
        return out

    @staticmethod
    def backward(ctx, go):
        go = go.contiguous()
        comb, mag = ctx.saved_tensors
        M, N = comb.shape
        gc = torch.empty_like(comb)
        gm = torch.zeros(N, dtype=mag.dtype, device=mag.device)
        grid = lambda m: (triton.cdiv(M, m['BLOCK_M']),)
        _bwd[grid](go, comb, mag, gc, gm, M, N, go.stride(0), go.stride(1), comb.stride(0), comb.stride(1))
        return gc, gm
