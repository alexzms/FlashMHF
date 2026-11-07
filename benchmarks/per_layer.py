#!/usr/bin/env python3
"""Per-layer latency and peak memory over the sequence-length grid of Appendix G:
FlashFFN fused block (W_in -> fused prologue -> fused kernel -> W_out) vs. the SwiGLU FFN
(d_ff=5504) and the naive multi-head FFN without the fused kernel. 1.3B configuration
(H=16, d_h=128, E=15, d_e=384), bs=8, no-grad forward, interleaved rounds. See benchmarks/README.md."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RESULTS_DIR  # puts kernels/triton and kernels/thunderkittens on sys.path
import json, statistics
import torch, triton.testing
import torch.nn.functional as F
import flash_ffn_moe_fast as ext_fast
from fused_prologue import fused_prologue
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mhffnmoe.modeling_mhffnmoe import MHFFNMoEMultiHeadRMSNorm

torch.manual_seed(0)
B,H,dh,de,E,DFF = 8,16,128,384,15,5504
D = H*dh
dev,dt = 'cuda', torch.bfloat16
ROUNDS = 5
qn = MHFFNMoEMultiHeadRMSNorm(H, dh, eps=1e-5).to(dev, dtype=dt)

def mem_of(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()/2**20

rows = {}
hdr = f"{'L':>6} | {'Flash raw':>9} {'x20/24':>7} {'SwiGLU':>7} {'naive':>8} | {'Flash MB':>8} {'SwiGLU MB':>9} {'naive MB':>8}"
print(hdr); print('-'*len(hdr))
for L in [192, 384, 768, 1536, 1920, 2880, 4032, 8064, 16128, 32832]:
    g = torch.Generator(device=dev).manual_seed(L)
    def r(*s, sc=1.0): return torch.randn(*s, generator=g, device=dev, dtype=dt)*sc
    x, W1, W2 = r(B,L,D), r(D,D,sc=0.02), r(D,D,sc=0.02)
    Wr = r(H,dh,E,sc=0.1)
    K,U,V = r(H,E,de,dh,sc=0.05), r(H,E,de,dh,sc=0.05), r(H,E,de,dh,sc=0.05)
    mlp = LlamaMLP(LlamaConfig(hidden_size=D, intermediate_size=DFF, hidden_act='silu', mlp_bias=False)).to(dev, dtype=dt)
    def flash():
        with torch.no_grad():
            q = (x @ W1).view(B,L,H,dh)
            Q, Rt = fused_prologue(q, Wr, qn.weight, eps=1e-5)
            O = torch.empty(B,L,D, device=dev, dtype=dt)
            ext_fast.flashffn_moe_forward(Q,K,U,V,Rt,O)
            return O @ W2
    def swiglu():
        with torch.no_grad(): return mlp(x)
    def naive():
        with torch.no_grad():
            q = qn((x @ W1).view(B,L,H,dh)).transpose(1,2)
            rr = (q.to(torch.float32) @ Wr.to(torch.float32)).sigmoid()
            rr = rr/rr.sum(-1,keepdim=True).clamp(min=1e-20)
            o = torch.zeros(B,H,L,dh, device=dev, dtype=torch.float32)
            for e in range(E):
                a = F.silu(q @ K[:,e].transpose(-1,-2)) * (q @ U[:,e].transpose(-1,-2))
                o += (a @ V[:,e]).float() * rr[:,:,:,e:e+1]
            return o.to(dt).transpose(1,2).reshape(B,L,D) @ W2
    cands = {'flash': flash, 'swiglu': swiglu, 'naive': naive}
    times = {k: [] for k in cands}; mems = {}
    ok = {k: True for k in cands}
    for k, f in cands.items():
        try:
            f(); torch.cuda.synchronize()
            mems[k] = mem_of(f)
        except torch.cuda.OutOfMemoryError:
            ok[k] = False; mems[k] = float('nan'); torch.cuda.empty_cache()
    for rd in range(ROUNDS):
        for k, f in cands.items():
            if not ok[k]: continue
            times[k].append(triton.testing.do_bench(f, warmup=3, rep=10 if k=='naive' else 20))
    med = {k: (statistics.median(v) if v else float('nan')) for k, v in times.items()}
    rows[L] = {'flash_ms': med['flash'], 'flash_scaled': med['flash']*20/24,
               'swiglu_ms': med['swiglu'], 'naive_ms': med['naive'],
               'flash_mb': mems['flash'], 'swiglu_mb': mems['swiglu'], 'naive_mb': mems['naive']}
    print(f"{L:>6} | {med['flash']:9.2f} {med['flash']*20/24:7.2f} {med['swiglu']:7.2f} {med['naive']:8.2f} | "
          f"{mems['flash']:8.0f} {mems['swiglu']:9.0f} {mems['naive']:8.0f}", flush=True)
    del x,W1,W2,Wr,K,U,V,mlp; torch.cuda.empty_cache()
json.dump(rows, open(os.path.join(RESULTS_DIR, 'per_layer.json'),'w'), indent=1)
print("saved per_layer.json")
