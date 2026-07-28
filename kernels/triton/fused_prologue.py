#!/usr/bin/env python3
"""Fused prologue for FlashFFN: (optional per-head RMSNorm) + fp32 router
(sigmoid-normalized) + layout transforms, in ONE pass over Q.
Replaces: q_norm (2 full passes) + Q.float() copy + einsum + sigmoid/norm
+ Q transpose-contiguous + R transpose-contiguous.
Numerics match the eager path: norm in fp32, round to bf16, router products on
bf16-exact values with fp32 accumulation (same as cuBLAS fp32 GEMM on upcast bf16).
"""
import torch, triton
import triton.language as tl

@triton.jit
def _prologue(X, W, G, Q, R,
              L, EPS,
              sxb, sxl, sxh,
              sqb, sqh, sql,
              srb, srh, sre,
              swh,
              HAS_NORM: tl.constexpr, BL: tl.constexpr, DH: tl.constexpr,
              E: tl.constexpr, EP: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    offs_l = pid_l * BL + tl.arange(0, BL)
    mask_l = offs_l < L
    offs_d = tl.arange(0, DH)
    x = tl.load(X + b*sxb + offs_l[:, None]*sxl + h*sxh + offs_d[None, :],
                mask=mask_l[:, None], other=0.).to(tl.float32)
    if HAS_NORM:
        var = tl.sum(x * x, 1) / DH
        x = x * tl.rsqrt(var + EPS)[:, None]
        g = tl.load(G + h*DH + offs_d).to(tl.float32)
        x = x * g[None, :]
    qb = x.to(tl.bfloat16)
    tl.store(Q + b*sqb + h*sqh + offs_l[:, None]*sql + offs_d[None, :],
             qb, mask=mask_l[:, None])
    offs_e = tl.arange(0, EP)
    w = tl.load(W + h*swh + offs_d[:, None]*E + offs_e[None, :],
                mask=(offs_e[None, :] < E), other=0.)
    logits = tl.dot(qb, w, out_dtype=tl.float32)          # bf16 x bf16 -> fp32 acc
    sig = tl.sigmoid(logits)
    sig = tl.where(offs_e[None, :] < E, sig, 0.)
    den = tl.maximum(tl.sum(sig, 1), 1e-20)
    r = sig / den[:, None]
    tl.store(R + b*srb + h*srh + offs_e[None, :]*sre + offs_l[:, None],
             r, mask=mask_l[:, None] & (offs_e[None, :] < E))

def fused_prologue(x_blhd, router_w, gamma=None, eps=1e-5, BL=64, num_warps=4):
    """x_blhd: [B,L,H,dh] bf16 (a view of the W_in output). router_w: [H,dh,E] (bf16 ok).
    Returns Q [B,H,L,dh] bf16 contiguous, R_t [B,H,E,L] fp32 contiguous (TK layout)."""
    B, L, H, DH = x_blhd.shape
    E = router_w.shape[-1]
    EP = max(16, triton.next_power_of_2(E))
    Q = torch.empty(B, H, L, DH, device=x_blhd.device, dtype=torch.bfloat16)
    R = torch.empty(B, H, E, L, device=x_blhd.device, dtype=torch.float32)
    w = router_w.to(torch.bfloat16).contiguous()
    grid = (triton.cdiv(L, BL), H, B)
    _prologue[grid](x_blhd, w, gamma if gamma is not None else w, Q, R,
                    L, eps,
                    x_blhd.stride(0), x_blhd.stride(1), x_blhd.stride(2),
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    R.stride(0), R.stride(1), R.stride(2),
                    w.stride(0),
                    HAS_NORM=gamma is not None, BL=BL, DH=DH, E=E, EP=EP,
                    num_warps=num_warps)
    return Q, R

def tk_forward_rt(ext, Q, K, U, V, R_t):
    """Direct TK call with pre-transposed R (skips wrapper's transpose+contiguous).
    Q must be [B,H,L,dh] with L % 192 == 0."""
    O = torch.empty_like(Q)
    ext.flashffn_moe_forward(Q, K, U, V, R_t.contiguous(), O)
    return O
