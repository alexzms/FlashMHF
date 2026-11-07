#!/usr/bin/env python3
"""Decode-step estimate under CUDA graphs (no launch overhead), 1.3B configurations.

This is a component-wise estimate, not an end-to-end generation run. It times, separately:
  - the FFN stack for one new token: 20 FlashFFN blocks (PyTorch path and the fused
    decode-step kernel) and 24 SwiGLU blocks, each with distinct per-layer weights;
  - one attention decode step per layer (SDPA over a random KV cache of length C);
  - the LM head.
The model step time is then  FFN stack + n_layers x attention + LM head  (n_layers = 20 for
FlashFFN, 24 for SwiGLU), and tokens/s = bs / step time. Embedding, norms, RoPE and the
K/V projection and cache write of the new token are not included. See benchmarks/README.md."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RESULTS_DIR  # puts kernels/triton and kernels/thunderkittens on sys.path
import json, time
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig
from decode_kernel import decode_ffn

torch.manual_seed(0)
H, dh, E, de = 16, 128, 15, 384
D, DFF = 2048, 5504
dev, dt = 'cuda', torch.bfloat16
RES = {}

def graph_time(fn, static_in, reps=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(static_in)
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(reps): g.replay()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps

class FlashBlockWeights:
    def __init__(s):
        s.W1 = torch.randn(D, D, device=dev, dtype=dt) * 0.02
        s.W2 = torch.randn(D, D, device=dev, dtype=dt) * 0.02
        s.Wr = torch.randn(H, dh, E, device=dev, dtype=dt) * 0.1
        s.g  = torch.ones(H, dh, device=dev, dtype=dt)
        s.K  = torch.randn(H, E*de, dh, device=dev, dtype=dt) * 0.05
        s.U  = torch.randn(H, E*de, dh, device=dev, dtype=dt) * 0.05
        s.V  = torch.randn(H, E*de, dh, device=dev, dtype=dt) * 0.05

def flash_block(w, x):
    # decode layout: heads as bmm batch, batch as GEMM rows (weights read once)
    B, L, _ = x.shape                                          # L == 1
    q = (x @ w.W1).view(B, H, dh)
    qf = q.float()
    q = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(dt) * w.g
    q = q.permute(1, 0, 2)                                     # H,B,dh
    r = (q.float() @ w.Wr.float()).sigmoid()                   # H,B,E
    r = r / r.sum(-1, keepdim=True).clamp(min=1e-20)
    A = F.silu(torch.bmm(q, w.K.transpose(1, 2))) * torch.bmm(q, w.U.transpose(1, 2))
    A = A * r.repeat_interleave(de, dim=-1).to(dt)
    o = torch.bmm(A, w.V)                                      # H,B,dh
    return o.permute(1, 0, 2).reshape(B, L, D) @ w.W2

for bs in (1, 8):
    x = torch.randn(bs, 1, D, device=dev, dtype=dt)
    fw = [FlashBlockWeights() for _ in range(20)]
    lw = [LlamaMLP(LlamaConfig(hidden_size=D, intermediate_size=DFF, hidden_act='silu',
                               mlp_bias=False)).to(dev, dtype=dt) for _ in range(24)]
    with torch.no_grad():
        def flash_stack(xx):
            h = xx
            for w in fw: h = h + flash_block(w, h)
            return h
        def llama_stack(xx):
            h = xx
            for m in lw: h = h + m(h)
            return h
        obufs = [torch.zeros(bs, H, dh, device=dev, dtype=torch.float32) for _ in range(20)]
        def flash_dec_stack(xx):
            h = xx
            for i, w in enumerate(fw):
                h = h + decode_ffn(h, w.W1, w.W2, w.Wr,
                                   w.K.view(H, E, de, dh), w.U.view(H, E, de, dh),
                                   w.V.view(H, E, de, dh), w.g, out=obufs[i])
            return h
        t_f = graph_time(flash_stack, x)
        t_d = graph_time(flash_dec_stack, x)
        t_l = graph_time(llama_stack, x)
    # attention decode step (identical per layer for both models): 24 distinct layers
    aw = [{k: torch.randn(D, D, device=dev, dtype=dt)*0.02 for k in 'qkvo'} for _ in range(24)]
    res_attn = {}
    for C in (4096, 32768):
        kc = torch.randn(24, bs, 32, C, 64, device=dev, dtype=dt)
        vc = torch.randn(24, bs, 32, C, 64, device=dev, dtype=dt)
        with torch.no_grad():
            def attn_stack(xx):
                h = xx
                for i, w in enumerate(aw):
                    q = (h @ w['q']).view(bs, 1, 32, 64).transpose(1, 2)
                    o = F.scaled_dot_product_attention(q, kc[i], vc[i])
                    h = h + (o.transpose(1, 2).reshape(bs, 1, D) @ w['o'])
                return h
            res_attn[C] = graph_time(attn_stack, x) / 24     # per layer
        del kc, vc; torch.cuda.empty_cache()
    Whead = torch.randn(D, 50432, device=dev, dtype=dt) * 0.02
    with torch.no_grad():
        t_head = graph_time(lambda xx: xx.reshape(bs, D) @ Whead, x)
    print(f"\nbs={bs}: FFN stack  flash20={t_f:.3f}ms  flashDEC20={t_d:.3f}ms  llama24={t_l:.3f}ms  (DEC ratio {t_l/t_d:.2f}x)")
    print(f"bs={bs}: attn/layer C=4096: {res_attn[4096]:.4f}ms  C=32768: {res_attn[32768]:.4f}ms  lm_head={t_head:.3f}ms")
    for C in (4096, 32768):
        a = res_attn[C]
        tot_d = t_d + 20*a + t_head
        tot_l = t_l + 24*a + t_head
        print(f"bs={bs} ctx={C}: model decode step flashDEC={tot_d:.3f}ms ({bs/tot_d*1000:.0f} tok/s) "
              f"llama={tot_l:.3f}ms ({bs/tot_l*1000:.0f} tok/s)  -> flashDEC {tot_l/tot_d:.2f}x", flush=True)
    RES[f'bs{bs}'] = {'ffn_flash20_ms': t_f, 'ffn_flashdec20_ms': t_d, 'ffn_llama24_ms': t_l,
                      'attn_per_layer': res_attn, 'lm_head_ms': t_head}
    del fw, lw, aw, Whead; torch.cuda.empty_cache()
json.dump(RES, open(os.path.join(RESULTS_DIR, 'decode_step.json'), 'w'), indent=1)
print("saved decode_step.json")
