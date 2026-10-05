# 00b Offline-cache methodology check (Qwen3.5-0.8B, corpus=prose, T=8192, E=512)

**TL;DR:** (1) offline walk-forward attention reproduces the model's o_proj input: max rel err 0.0030 (fp16 cache) — OK. (2) int8 reference: attention-output rel err vs fp16 cache ≤ 0.0219, vs ≥ 0.0707 for naive 4-bit — int8 NOT negligible. (3) captured keys vs NumPy forward: rel diff L3: 0.0055, L23: 0.0063 (capture is AMP bf16; NumPy is fp32).

## Config
- capture: HF transformers on GPU, fp32 weights + torch.autocast(bf16); keys pre-RoPE (post k_norm)
- int8 ref: symmetric per-token/per-head absmax, fp16 scale → 8 + 16/256 = 8.06 bits/elem
- walk-forward: decode query at position p attends to cache[0..p]; RoPE applied in NumPy
- 'naive 4-bit' = per-token/head absmax int4 (scale reference only, not a scheme under test)

| layer | offline(fp16) vs model | offline(int8) vs model | int8 vs fp16 attn out | naive-4b vs fp16 attn out | K int8 rel | V int8 rel |
|---|---|---|---|---|---|---|
| 3 | 0.0029 | 0.0060 | 0.0053 | 0.0971 | 0.0114 | 0.0082 |
| 7 | 0.0030 | 0.0117 | 0.0113 | 0.1562 | 0.0091 | 0.0122 |
| 11 | 0.0030 | 0.0138 | 0.0219 | 0.1873 | 0.0097 | 0.0100 |
| 15 | 0.0030 | 0.0100 | 0.0189 | 0.2135 | 0.0096 | 0.0124 |
| 19 | 0.0029 | 0.0065 | 0.0070 | 0.1293 | 0.0094 | 0.0137 |
| 23 | 0.0028 | 0.0123 | 0.0038 | 0.0707 | 0.0091 | 0.0076 |

## Caveats
- The offline method has no error propagation across layers: K/V of layer L+4 are computed from unquantized upstream activations. Per-layer metrics (E1–E8, E10) are exact under this method; end-to-end perplexity/needle (E9) needs a forward pass with the quantized cache in the loop.
- The model-vs-offline residual reflects bf16 AMP arithmetic inside the model (SDPA in bf16).

runtime 93s
