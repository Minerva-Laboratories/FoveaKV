# 12 GPU benchmark: CUDA kernels + end-to-end decoding (Qwen3.5-0.8B, RTX 4050 Laptop GPU)

**TL;DR:** With the improvements from E11 (exact 128-token recent window outside the budget, 128-token block means), FoveaKV-5%
reads ~2.4 bits/elem (2.16 resident) and gives end-to-end perplexity within +0.3% of bf16 at 16K/32K (TurboQuant-MSE 4b: +0.5%,
2b: +13%; KIVI-2b: +19–38%; Quest-5%: +0.5–7%) and the closest-to-bf16 retrieval answers (94% identical vs 88% for TurboQuant-4b/KIVI-4b).
Kernels: all compressed schemes 3–8× faster than bf16 SDPA at 128K, but the gain-shape kernels are compute-bound, so FoveaKV
(0.99 ms/layer at 128K) is slower than 4-bit TurboQuant (0.72 ms) despite reading half the bytes.

## Setup
- Kernels: `gpu/kernels.cu` (split-K flash-decoding; lane-per-token keys, dims-parallel values from shared staging; delta pass);
  every scheme verified against an fp64 dequantize-and-dense reference (≤2e-6 rel err, `gpu/test_kernels.py`).
- Baselines: TurboQuant-MSE (RHT + Lloyd-Max + fp16 norm, no QJL) 2/3/4 bit; KIVI 2/4 bit (group 128, fp16 scale/zero, bf16 residual);
  Quest (bf16 pages of 16, min/max bound, top pages by budget); bf16 dense (PyTorch SDPA). All compressed schemes get the same
  exact bf16 recent window (128–255 tokens).
- End-to-end (`gpu/eval_e2e.py`): bf16 weights + autocast; exact shared chunked prefill; quantized cache built from the prefill
  snapshot; every decode step runs the kernels. PPL: last 256 tokens of 3 docs × {prose (wikitext-2 train), code (transformers
  sources)} × {16K, 32K}. Retrieval: niah / multi-key niah / multi-value niah at 8K/16K/32K, 8 samples each; the question is decoded
  token-by-token through the compressed cache.

## Results
Raw results: `gpu_kernel_bench.json`, `gpu_ppl.json` and `gpu_retrieval.json` in this folder; the summary tables are in
the README.

## Caveats
- 3 docs × 256 tokens per cell: PPL differences < ~0.3% are not significant; retrieval: 8 samples per cell.
- One small model; bf16 is the reference here (the offline study used int8).
- Kernels are compute-bound on this GPU; selection (top-k) + delta pass add ~0.3 ms at 128K. Offloading (PCIe) is still only estimated (E10).
