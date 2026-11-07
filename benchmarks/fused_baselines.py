#!/usr/bin/env python3
"""Per-layer latency and peak memory: FlashFFN fused block vs. SwiGLU FFN implementations
(eager PyTorch, torch.compile, xFormers fused SwiGLU, Liger fused SwiGLU).
Same scope for every candidate (no-grad forward of the full FFN block), interleaved rounds,
median over rounds. 1.3B FFN configuration, bs=8. Memory columns give the paper's fused-baseline
memory table. See benchmarks/README.md."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RESULTS_DIR  # puts kernels/triton and kernels/thunderkittens on sys.path
import json, statistics
import torch, triton.testing
import flash_ffn_moe_fast as ext_fast
from fused_prologue import fused_prologue
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig
from xformers.ops.swiglu_op import swiglu as xformers_swiglu
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from transformers.models.mhffnmoe.modeling_mhffnmoe import MHFFNMoEMultiHeadRMSNorm

torch.manual_seed(0)
B,H,dh,de,E,DFF = 8,16,128,384,15,5504
D = H*dh
dev,dt = 'cuda', torch.bfloat16
ROUNDS = 5
qn = MHFFNMoEMultiHeadRMSNorm(H, dh, eps=1e-5).to(dev, dtype=dt)

class XFormersSwiGLUMLP(torch.nn.Module):
    """LlamaMLP weights with the forward pass through xFormers' fused SwiGLU op."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return xformers_swiglu(x, self.gate_proj.weight, self.gate_proj.bias,
                               self.up_proj.weight, self.up_proj.bias,
                               self.down_proj.weight, self.down_proj.bias, op=None)

class LigerCfg:
    def __init__(s): s.hidden_size, s.intermediate_size, s.hidden_act = D, DFF, 'silu'

def mem_of(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()/2**20

rows = {}
hdr = f"{'L':>6} | {'Flash':>6} {'x20/24':>7} {'Torch':>6} {'compile':>8} {'xFormers':>8} {'Liger':>6} | {'FlashMB':>7} {'TorchMB':>7} {'xFormMB':>7} {'LigerMB':>7}"
print(hdr); print('-'*len(hdr))
for L in [192, 384, 768, 1536, 1920, 2880, 4032, 8064, 16128, 32832]:
    g = torch.Generator(device=dev).manual_seed(L)
    def r(*s, sc=1.0): return torch.randn(*s, generator=g, device=dev, dtype=dt)*sc
    x, W1, W2 = r(B,L,D), r(D,D,sc=0.02), r(D,D,sc=0.02)
    Wr = r(H,dh,E,sc=0.1)
    K,U,V = r(H,E,de,dh,sc=0.05), r(H,E,de,dh,sc=0.05), r(H,E,de,dh,sc=0.05)
    mlp  = LlamaMLP(LlamaConfig(hidden_size=D, intermediate_size=DFF, hidden_act='silu', mlp_bias=False)).to(dev, dtype=dt)
    mlpc = torch.compile(mlp, dynamic=False)
    fuse = XFormersSwiGLUMLP(D, DFF).to(dev, dtype=dt)
    lig  = LigerSwiGLUMLP(LigerCfg()).to(dev, dtype=dt)
    def flash():
        with torch.no_grad():
            q = (x @ W1).view(B,L,H,dh)
            Q, Rt = fused_prologue(q, Wr, qn.weight, eps=1e-5)
            O = torch.empty(B,L,D, device=dev, dtype=dt)
            ext_fast.flashffn_moe_forward(Q,K,U,V,Rt,O)
            return O @ W2
    def w(m):
        def f():
            with torch.no_grad(): return m(x)
        return f
    cands = {'flash': flash, 'torch': w(mlp), 'compiled': w(mlpc), 'xformers': w(fuse), 'liger': w(lig)}
    for f in cands.values():
        try: f()
        except Exception as e: print("warmup fail:", e)
    for _ in range(3): cands['compiled']()
    torch.cuda.synchronize()
    mems = {k: mem_of(f) for k, f in cands.items()}
    times = {k: [] for k in cands}
    for rd in range(ROUNDS):
        for k, f in cands.items():
            times[k].append(triton.testing.do_bench(f, warmup=3, rep=20))
    med = {k: statistics.median(v) for k, v in times.items()}
    rows[L] = {'ms': med, 'mb': mems}
    print(f"{L:>6} | {med['flash']:6.2f} {med['flash']*20/24:7.2f} {med['torch']:6.2f} {med['compiled']:8.2f} "
          f"{med['xformers']:8.2f} {med['liger']:6.2f} | {mems['flash']:7.0f} {mems['torch']:7.0f} "
          f"{mems['xformers']:7.0f} {mems['liger']:7.0f}", flush=True)
    del x,W1,W2,Wr,K,U,V,mlp,mlpc,fuse,lig; torch.cuda.empty_cache()
json.dump(rows, open(os.path.join(RESULTS_DIR, 'fused_baselines.json'),'w'), indent=1)
print("saved fused_baselines.json")
