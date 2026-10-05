# 00 Forward-pass correctness gate — Qwen3.5-0.8B (NumPy)

**TL;DR:** gate PASSED. wikitext-2 PPL 12.576 on 2048 tokens; top-1 agreement with HF transformers fp32 oracle 99.95% over 2047 positions (HF PPL 12.584).
Greedy continuations are coherent (below). Incremental/chunked decode matches full forward (rel diff ~1e-6).
llama.cpp is not installed. Oracle = HF transformers on GPU in pure fp32 (tools/hf_reference_dump.py, ORACLE_FP32=1, memory-capped), used only for this check. An AMP-bf16 oracle run gives 98.05% agreement — all 40 disagreements sit at top-2 logit margins ≤ 0.11 (median overall margin 1.37), i.e. bf16 rounding at near-ties. The single fp32 disagreement has margin 0.003.

## Config
- model: Qwen/Qwen3.5-0.8B (local HF cache), fp32 compute, bf16 weights
- 24 layers: 18 Gated DeltaNet (recurrent loop) + 6 full attention (layers 3, 7, 11, 15, 19, 23)
- full attention: Hq=8, Hkv=2 (GQA group 4), head_dim=256, partial RoPE rot_dim=64 (theta=1e+07), output gate, q/k RMSNorm
- text: wikitext-2-raw test (first 2048 tokens; note: CC-BY-SA, not public domain)

## Greedy continuations
- `'The capital of France is'` → `' Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is'`
- `'def fibonacci(n):\n'` → `'    if n <= 0:\n        return 0\n    elif n == 1:\n        return '`
- `'Once upon a time, in a small village,'` → `' there was a very special kind of animal. It had a very special name, and it was called the "Moth'`

## Numbers
| metric | value |
|---|---|
| NumPy PPL | 12.5761 |
| HF fp32 PPL | 12.5844 |
| top-1 agreement | 99.95% |
| max per-token NLL diff | 3.39e-02 |

runtime 76s
