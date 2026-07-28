#!/usr/bin/env python3
"""Dedicated decode-step kernel for the FlashFFN block (L=1).
Fuses per-head RMSNorm + fp32 router + all E gated sub-networks into one kernel.
Grid = (H*E,); each block streams its K/U/V slice once; fp32 atomicAdd output."""
import torch, triton
import triton.language as tl

@triton.jit
def _dec(Q, G, WR, K, U, V, O, B,
         sqb, sqh, skh, ske, skr, sob, soh, swrh,
         HAS_NORM: tl.constexpr, EPS: tl.constexpr,
         DH: tl.constexpr, DE: tl.constexpr, E: tl.constexpr,
         EP: tl.constexpr, CH: tl.constexpr, BP: tl.constexpr):
    pid = tl.program_id(0)
    h = pid // E
    e = pid % E
    offs_b = tl.arange(0, BP)
    mb = offs_b < B
    offs_d = tl.arange(0, DH)
    q = tl.load(Q + offs_b[:, None]*sqb + h*sqh + offs_d[None, :],
                mask=mb[:, None], other=0.).to(tl.float32)
    if HAS_NORM:
        var = tl.sum(q*q, 1) / DH
        q = q * tl.rsqrt(var + EPS)[:, None]
        g = tl.load(G + h*DH + offs_d).to(tl.float32)
        q = q * g[None, :]
    qb = q.to(tl.bfloat16)
    offs_e = tl.arange(0, EP)
    w = tl.load(WR + h*swrh + offs_d[:, None]*E + offs_e[None, :],
                mask=(offs_e[None, :] < E), other=0.)
    logits = tl.dot(qb, w, out_dtype=tl.float32)
    sig = tl.sigmoid(logits)
    sig = tl.where(offs_e[None, :] < E, sig, 0.)
    den = tl.maximum(tl.sum(sig, 1), 1e-20)
    r_e = tl.sum(tl.where(offs_e[None, :] == e, sig, 0.), 1) / den
    acc = tl.zeros((BP, DH), tl.float32)
    for c in range(0, DE, CH):
        offs_c = c + tl.arange(0, CH)
        base = h*skh + e*ske + offs_c[:, None]*skr + offs_d[None, :]
        Kt = tl.load(K + base)
        Ut = tl.load(U + base)
        Vt = tl.load(V + base)
        m = tl.dot(qb, tl.trans(Kt), out_dtype=tl.float32)
        n = tl.dot(qb, tl.trans(Ut), out_dtype=tl.float32)
        a = (m * tl.sigmoid(m)) * n * r_e[:, None]
        acc += tl.dot(a.to(tl.bfloat16), Vt, out_dtype=tl.float32)
    tl.atomic_add(O + offs_b[:, None]*sob + h*soh + offs_d[None, :], acc, mask=mb[:, None])

def decode_ffn(x, W1, W2, Wr, K, U, V, gamma=None, eps=1e-5, out=None):
    """x [B,1,D]; K/U/V [H,E,de,dh]; Wr [H,dh,E]. Returns [B,1,D] bf16."""
    B = x.shape[0]
    H, E, de, dh = K.shape
    D = H * dh
    q = (x.reshape(B, D) @ W1).view(B, H, dh)
    if out is None:
        out = torch.zeros(B, H, dh, device=x.device, dtype=torch.float32)
    else:
        out.zero_()
    wr = Wr.to(torch.bfloat16)
    _dec[(H*E,)](q, gamma if gamma is not None else wr, wr, K, U, V, out, B,
                 q.stride(0), q.stride(1),
                 K.stride(0), K.stride(1), K.stride(2),
                 out.stride(0), out.stride(1), wr.stride(0),
                 HAS_NORM=gamma is not None, EPS=eps,
                 DH=dh, DE=de, E=E, EP=max(16, triton.next_power_of_2(E)),
                 CH=128, BP=16, num_warps=4)
    return (out.reshape(B, D).to(x.dtype) @ W2).view(B, 1, D)

if __name__ == '__main__':
    torch.manual_seed(0)
    B, H, dh, E, de = 8, 16, 128, 15, 384
    D = H*dh
    dev, dt = 'cuda', torch.bfloat16
    x = torch.randn(B, 1, D, device=dev, dtype=dt)
    W1 = torch.randn(D, D, device=dev, dtype=dt)*0.02
    W2 = torch.randn(D, D, device=dev, dtype=dt)*0.02
    Wr = torch.randn(H, dh, E, device=dev, dtype=dt)*0.1
    K = torch.randn(H, E, de, dh, device=dev, dtype=dt)*0.05
    U = torch.randn(H, E, de, dh, device=dev, dtype=dt)*0.05
    V = torch.randn(H, E, de, dh, device=dev, dtype=dt)*0.05
    g = 1.0 + 0.1*torch.randn(H, dh, device=dev, dtype=dt)
    for gamma in (None, g):
        y = decode_ffn(x, W1, W2, Wr, K, U, V, gamma)
        # fp32 reference, same op order/semantics
        q = (x.reshape(B, D) @ W1).view(B, H, dh).float()
        if gamma is not None:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * gamma.float()
        qb = q.to(dt)
        r = (qb.float().unsqueeze(2) @ Wr.float()).squeeze(2).sigmoid()
        r = r / r.sum(-1, keepdim=True).clamp(min=1e-20)   # [B,H,E]
        o = torch.zeros(B, H, dh, device=dev, dtype=torch.float32)
        for e in range(E):
            m = qb.transpose(0,1).float() @ K[:, e].transpose(-1,-2).float()  # [H,B,de]
            n = qb.transpose(0,1).float() @ U[:, e].transpose(-1,-2).float()
            a = torch.nn.functional.silu(m) * n * r.permute(1,0,2)[:,:,e:e+1]
            o += (a.to(dt).float() @ V[:, e].float()).transpose(0,1)
        ref = (o.reshape(B, D).to(dt) @ W2).view(B, 1, D)
        rel = (y.float()-ref.float()).norm()/ref.float().norm()
        print(f"decode kernel verify (qnorm={gamma is not None}): rel L2 = {rel.item():.2e}")
