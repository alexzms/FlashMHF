#!/usr/bin/env python3
"""Model-level prefill latency and peak memory of the 1.3B configurations: 20-layer FlashFFN
(fused prologue + token-major ThunderKittens kernel patched into every FFN block) vs. the
24-layer SwiGLU baseline. Randomly initialized weights (speed does not depend on the values),
use_cache=False, logits for the last position only. Checks the fused path against the PyTorch
reference first. See benchmarks/README.md."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RESULTS_DIR  # puts kernels/triton and kernels/thunderkittens on sys.path
import json, gc, types
import torch, triton.testing
import torch.nn.functional as F
import flash_ffn_moe_fast as ext_fast
from fused_prologue import fused_prologue
from transformers.models.mhffnmoe.configuration_mhffnmoe import MHFFNMoEConfig
from transformers.models.mhffnmoe.modeling_mhffnmoe import MHFFNMoEForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

dev, dt = 'cuda', torch.bfloat16
VOC = 50432
RES = {}
def save(): json.dump(RES, open(os.path.join(RESULTS_DIR, 'model_prefill.json'),'w'), indent=1)
def free(): gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

common = dict(vocab_size=VOC, hidden_size=2048, num_attention_heads=32, num_key_value_heads=32,
              max_position_embeddings=65536, rms_norm_eps=1e-5, rope_theta=10000.0,
              tie_word_embeddings=False, attention_dropout=0.0, torch_dtype=dt)
fcfg = MHFFNMoEConfig(num_hidden_layers=20, intermediate_size=5760,
    mhffnmoe_num_heads=16, mhffnmoe_num_kuv_heads=16, mhffnmoe_num_experts=15,
    mhffnmoe_skip_linear=False, mhffnmoe_apply_rmsnorm=True, mhffnmoe_use_flash=True,
    mhffnmoe_use_legacy_flash=False, mhffnmoe_apply_dot_scaling=False,
    mhffnmoe_apply_final_scaling=False, mhffnmoe_custom_init=True,
    mhffnmoe_init_wself_identity=False, mlp_bias=False, attention_bias=False, **common)
lcfg = LlamaConfig(num_hidden_layers=24, intermediate_size=5504, mlp_bias=False,
                   attention_bias=False, **common)
fcfg._attn_implementation = 'sdpa'; lcfg._attn_implementation = 'sdpa'

def fast_ffn_forward(self, x):
    B, L, D = x.shape
    H, dh = self.mhffnmoe_num_heads, self.head_dim
    E, de = self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim
    q = self.w_self(x) if self.w_self is not None else x
    Lp = ((L + 191)//192)*192
    if Lp != L: q = F.pad(q, (0, 0, 0, Lp - L))
    if not hasattr(self, '_kuvc'):
        with torch.no_grad():
            mk = lambda w: w.view(self.intermediate_size, H, dh).transpose(0,1)\
                            .view(H, E, de, dh).contiguous()
            self._kuvc = (mk(self.gate_weight), mk(self.up_weight), mk(self.down_weight))
    K, U, V = self._kuvc
    gamma = self.q_norm.weight if self.config.mhffnmoe_apply_rmsnorm else None
    Q, Rt = fused_prologue(q.view(B, Lp, H, dh), self.router, gamma,
                           eps=self.q_norm.variance_epsilon)
    O = torch.empty(B, Lp, D, device=x.device, dtype=x.dtype)
    ext_fast.flashffn_moe_forward(Q, K, U, V, Rt, O)
    o = O[:, :L] if Lp != L else O
    return self.w_self_2(o) if self.w_self_2 is not None else o

print("building models...", flush=True)
with torch.device(dev):
    fm = MHFFNMoEForCausalLM(fcfg).to(dt).eval()
with torch.device(dev):
    lm = LlamaForCausalLM(lcfg).to(dt).eval()
print(f"params: flash={sum(p.numel() for p in fm.parameters())/1e9:.4f}B "
      f"llama={sum(p.numel() for p in lm.parameters())/1e9:.4f}B", flush=True)

def set_mode(mode):  # 'fast' | 'pyref'
    for lyr in fm.model.layers:
        m = lyr.mlp
        if mode == 'fast':
            if not hasattr(m, '_orig_forward'): m._orig_forward = m.forward
            m.forward = types.MethodType(fast_ffn_forward, m)
        else:
            if hasattr(m, '_orig_forward'): m.forward = m._orig_forward
            m.flash = False   # pytorch reference path

# numeric check: fast vs pytorch reference, arbitrary L (tests padding) and 192-multiple
for L in (100, 192):
    ids = torch.randint(0, VOC, (2, L), device=dev)
    set_mode('pyref')
    with torch.no_grad(): ref = fm(input_ids=ids, use_cache=False).logits.float()
    set_mode('fast')
    with torch.no_grad(): out = fm(input_ids=ids, use_cache=False).logits.float()
    print(f"E2E verify L={L}: logits rel L2 = {((out-ref).norm()/ref.norm()).item():.2e}", flush=True)

def bench_prefill(model, bs, L, reps=6):
    ids = torch.randint(0, VOC, (bs, L), device=dev)
    def fn():
        with torch.no_grad(): model(input_ids=ids, use_cache=False, logits_to_keep=1)
    fn(); torch.cuda.synchronize(); free(); fn(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()/2**20
    ms = triton.testing.do_bench(fn, warmup=2, rep=reps)
    del ids; return ms, peak

set_mode('fast')
RES['prefill'] = {}
for name, m, other in (('flash_fast', fm, lm), ('llama', lm, fm)):
    other.to('cpu'); free(); m.to(dev)
    RES['prefill'][name] = {}
    for bs, L in [(8,256),(8,1024),(8,4096),(8,8192),(8,16384),(8,32768),(1,32768)]:
        try:
            ms, peak = bench_prefill(m, bs, L)
            RES['prefill'][name][f"bs{bs}_L{L}"] = {'ms': ms, 'peak_mb': peak}
            print(f"prefill {name} bs={bs} L={L}: {ms:.1f}ms peak={peak:.0f}MB", flush=True)
        except torch.cuda.OutOfMemoryError:
            RES['prefill'][name][f"bs{bs}_L{L}"] = 'OOM'; print(f"prefill {name} bs={bs} L={L}: OOM", flush=True)
        free()
    save()
lm.to(dev)
print("E2E v3 DONE")
