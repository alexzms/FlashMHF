# Benchmarks

These scripts produce the efficiency results in the NeurIPS 2026 camera-ready paper. All numbers in the
paper were measured on a single NVIDIA H100 SXM 80GB in bf16, with PyTorch 2.12 and ThunderKittens
commit `6c27e28`. Each script prints a table and writes a JSON file to `results/` (override with
`FLASHFFN_RESULTS_DIR`).

## Setup

1. Build both ThunderKittens modules (`flash_ffn_moe` and the token-major `flash_ffn_moe_fast`):

   ```bash
   export THUNDERKITTENS_ROOT=/path/to/ThunderKittens   # commit 6c27e28
   make -C kernels/thunderkittens
   ```

2. Install the FlashFFN model into Transformers as described in the main [README](../README.md#using-the-model).
   `per_layer.py`, `fused_baselines.py` and `model_prefill.py` import it as `transformers.models.mhffnmoe`.

3. `fused_baselines.py` also needs `liger-kernel` and `xformers`.

Run from the repository root, for example `python benchmarks/model_prefill.py`.

## Scripts

| Script | What it measures | Used for |
|---|---|---|
| `fused_baselines.py` | Per-layer latency and peak memory of the FlashFFN block vs. SwiGLU FFNs in eager PyTorch, `torch.compile`, xFormers and Liger; L = 192 to 32832, bs = 8, 1.3B FFN configuration | Peak-memory comparison against fused SwiGLU implementations (Sec. 5.3) |
| `per_layer.py` | Per-layer latency and peak memory of the FlashFFN block vs. the SwiGLU FFN (d_ff = 5504) and the naive multi-head FFN without the fused kernel; same L grid | Appendix G per-layer results, including the short-sequence speedups |
| `model_prefill.py` | Full-model prefill latency and peak memory, 20-layer FlashFFN vs. 24-layer SwiGLU at 1.3B; bs = 8 with L = 256 to 32768, and bs = 1 with L = 32768 | Model-level prefill throughput (Sec. 5.3) |
| `decode_step.py` | Decode-step estimate under CUDA graphs; bs = 1 and 8, context 4096 and 32768 | Decode throughput (Sec. 5.3) |

## Reading the outputs

- **Scope.** The FlashFFN block always includes W_in, the fused prologue (per-head RMSNorm and router), the
  fused kernel and W_out. The SwiGLU baselines are the full FFN.
- **`x20/24` column** (`fused_baselines.py`, `per_layer.py`): the FlashFFN per-layer time multiplied by 20/24,
  i.e. normalized for the 1.3B models having 20 FlashFFN layers against 24 SwiGLU layers at matched
  parameters and FLOPs. Per-layer comparisons use the raw FlashFFN column.
- **`model_prefill.py`** uses randomly initialized weights, which does not affect speed. Throughput is
  bs x L / latency.
- **`decode_step.py`** is a component-wise estimate, not an end-to-end generation run. It times the FFN
  stack for one new token, one attention decode step per layer (SDPA over a random KV cache), and the LM
  head separately, then reports `FFN stack + n_layers x attention + LM head` (n_layers = 20 for FlashFFN,
  24 for SwiGLU). Embedding, norms, RoPE and the K/V projection and cache write of the new token are not
  included. Because attention is timed once and multiplied by the layer count, part of the difference
  comes from FlashFFN having fewer layers.
