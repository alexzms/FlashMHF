# FlashFFN: Multi-Head Decomposition Enables I/O-Aware Feed-Forward Networks

Official implementation of the NeurIPS 2026 paper *FlashFFN: Multi-Head Decomposition Enables I/O-Aware
Feed-Forward Networks*. An earlier version of the paper is available on arXiv as
[Flash Multi-Head Feed-Forward Network (arXiv:2512.06989)](https://www.arxiv.org/abs/2512.06989).

## Overview

A SwiGLU FFN materializes its `[B, L, d_ff]` intermediate tensor in HBM between the two matmuls.
A FlashAttention-style one-pass fused kernel would keep it on chip, but it has to hold an output
accumulator of width `d_model` in SRAM, which does not fit on current NVIDIA datacenter GPUs
(A100/H100/B200) once `d_model >= 2048` with a sequence tile of 128 or more tokens.

FlashFFN splits the FFN into `H_ffn` heads, so each head's accumulator width `d_h = d_model / H_ffn`
fits in SRAM. To keep the per-head expansion ratio near the SwiGLU optimum of 8/3, every head uses
`E` dense sub-networks combined by a learned router:

```
O = Σ_e R_e · ((SiLU(Q K_eᵀ) ⊙ (Q U_eᵀ)) V_e)
```

Here `Q` is the per-head, RMS-normalized projection of the input and `R_e` are sigmoid-normalized
router weights. The block replaces SwiGLU at matched parameter count and FLOPs and runs as a single
fused kernel that never writes the intermediate to HBM.

## Results

All numbers are from the paper. Models are trained on The Pile (60B tokens at 128M/370M, 100B tokens
at 1.3B), and every comparison is at matched parameters and FLOPs.

**Evaluation perplexity** (lower is better)

| Model | 128M | 370M | 1.3B |
|---|---|---|---|
| Baseline (SwiGLU) | 26.74 | 20.70 | 17.17 |
| MH-FFN (multi-head split only) | 26.31 | 20.74 | – |
| Dense-MoE (sub-networks only) | 26.82 | 21.37 | – |
| **FlashFFN** (`d_h = 128`) | **26.00** | **20.37** | **16.33** |

Average accuracy over six downstream benchmarks (HellaSwag, SIQA, PIQA, OBQA, WinoGrande, RACE)
rises from 39.92 to 40.48 at 370M and from 41.75 to 43.35 at 1.3B.

**Efficiency** (one H100 SXM, bf16, 1.3B configurations)

- Per-layer peak FFN memory at bs = 8 (MB):

  | L | Eager SwiGLU | `torch.compile` | FlashFFN |
  |---|---|---|---|
  | 4032 | 1240 | 901 | 524 |
  | 8064 | 2384 | 1706 | 931 |
  | 16128 | 4666 | 3311 | 1746 |
  | 32832 | 9398 | 6640 | 3434 |

  That is 2.4–2.7× below eager SwiGLU (Liger matches eager) and 1.7–1.9× below `torch.compile`.
- Model-level prefill, 20-layer FlashFFN vs. 24-layer SwiGLU at bs = 8: parity at L = 256 and up to
  1.18× faster at L = 32K. Against a Liger-fused SwiGLU baseline the speedup is 1.02–1.17×.
- Decode: 1.08–1.19× in the CUDA-graph decode-step estimate (bs = 1 and 8, 4K and 32K context).

[`benchmarks/README.md`](benchmarks/README.md) describes exactly what each measurement covers.

## Model configurations

The configs in [`configs/`](configs/) match the paper's configuration table.

| Scale | Model | Config | Layers | `d_model` | `H_ffn` / `d_h` | `E` | Intermediate size |
|---|---|---|---|---|---|---|---|
| 128M | FlashFFN | [`configure-1-tinymhffnmoe.json`](configs/mhffnmoe/configure-1-tinymhffnmoe.json) | 10 | 768 | 6 / 128 | 8 | 2048 |
| 370M | FlashFFN | [`configure-2-mhffnmoe-TriDao-350M.json`](configs/mhffnmoe/configure-2-mhffnmoe-TriDao-350M.json) | 21 | 1024 | 8 / 128 | 7 | 2688 |
| 1.3B | FlashFFN | [`configure-3-mhffnmoe-TriDao-1.3B.json`](configs/mhffnmoe/configure-3-mhffnmoe-TriDao-1.3B.json) | 20 | 2048 | 16 / 128 | 15 | 5760 |
| 128M | SwiGLU | [`configure8-tinyllama.json`](configs/baseline/configure8-tinyllama.json) | 12 | 768 | – | – | 2048 |
| 370M | SwiGLU | [`configure9-TriDao-350M-bf16-pile.json`](configs/baseline/configure9-TriDao-350M-bf16-pile.json) | 24 | 1024 | – | – | 2752 |
| 1.3B | SwiGLU | [`configure10-TriDao-1.3B-bf16-pile.json`](configs/baseline/configure10-TriDao-1.3B-bf16-pile.json) | 24 | 2048 | – | – | 5504 |

The MH-FFN control configs are in [`configs/mhffn/`](configs/mhffn/).

## Using the model

The model code is a Hugging Face Transformers model (`model_type: "mhffnmoe"`), developed in a
Transformers 4.52 source tree. To use it:

1. Copy it into a Transformers 4.52 source checkout and install that checkout:

   ```bash
   cp -r models/flashmhf /path/to/transformers/src/transformers/models/mhffnmoe
   pip install -e /path/to/transformers
   ```

2. Point the model at the Triton kernels:

   ```bash
   export FLASHFFN_TRITON_DIR=/path/to/FlashMHF/kernels/triton
   ```

3. Build a model from one of the configs:

   ```python
   from transformers.models.mhffnmoe.configuration_mhffnmoe import MHFFNMoEConfig
   from transformers.models.mhffnmoe.modeling_mhffnmoe import MHFFNMoEForCausalLM

   config = MHFFNMoEConfig.from_json_file("configs/mhffnmoe/configure-3-mhffnmoe-TriDao-1.3B.json")
   model = MHFFNMoEForCausalLM(config)
   ```

   To use it through the `Auto*` classes, register it first with
   `AutoConfig.register("mhffnmoe", MHFFNMoEConfig)` and
   `AutoModelForCausalLM.register(MHFFNMoEConfig, MHFFNMoEForCausalLM)`.

With `mhffnmoe_use_flash=True` (the default) the FFN uses the fused Triton kernel. If the kernel cannot
be imported, the model prints a warning and falls back to the PyTorch implementation.

## Kernels

- [`kernels/triton/`](kernels/triton/):
  - `flash_mlp_moe.py`: fused multi-head FFN with sub-networks, forward and backward.
  - `flash_mlp.py`: single sub-network (legacy) variant.
  - `fused_prologue.py`: fused per-head RMSNorm, router and layout transform for prefill.
  - `decode_kernel.py`: fused decode-step FFN (norm, router and sub-networks in one launch).
- [`kernels/thunderkittens/`](kernels/thunderkittens/), for H100 (`sm_90a`):
  - `flash_ffn_moe.cu`: forward and backward, head-major output.
  - `flash_ffn_moe_fast.cu`: forward with token-major output. Together with `fused_prologue.py`, this is
    the path used for the efficiency results in the paper.

  Build both with:

  ```bash
  export THUNDERKITTENS_ROOT=/path/to/ThunderKittens   # the paper used commit 6c27e28
  make -C kernels/thunderkittens
  ```

## Repository structure

```
FlashMHF/
├── models/flashmhf/      # Transformers model code (model_type "mhffnmoe")
├── kernels/
│   ├── triton/           # Triton kernels
│   └── thunderkittens/   # ThunderKittens (CUDA) kernels for H100
├── configs/
│   ├── baseline/         # SwiGLU baselines
│   ├── mhffn/            # MH-FFN control
│   └── mhffnmoe/         # FlashFFN
└── benchmarks/           # efficiency benchmarks (see benchmarks/README.md)
```

## Citation

```bibtex
@inproceedings{zhang2026flashffn,
  title     = {{FlashFFN}: Multi-Head Decomposition Enables {I/O}-Aware Feed-Forward Networks},
  author    = {Zhang, Minshen and Hu, Xiang and Li, Jianguo and Wu, Wei and Tu, Kewei},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). The model code in `models/flashmhf/`
is derived from Hugging Face Transformers (Apache-2.0). The kernels are written with
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens) and [Triton](https://github.com/triton-lang/triton),
both MIT-licensed.
