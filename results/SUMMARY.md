# FoveaKV on real tensors — Qwen3.5-0.8B (hybrid), go/no-go summary

> **Update (phase 2):** improvements (exact recent window outside the budget + 128-token block means, E11) and a CUDA/GPU
> benchmark vs TurboQuant-MSE, KIVI and Quest (results/12_gpu_benchmark.md): FoveaKV-5% is within +0.3% of bf16 perplexity at
> 16K/32K at ~2.4 bits read, better than TurboQuant-MSE 4b. Paper draft: paper/main.pdf. The phase-1 summary below is unchanged.

**TL;DR:** On real Qwen3.5-0.8B activations the dynamic scheme (2.03 resident bits, ~2.13 read bits at k=5%) matches uniform
gain-shape 4-bit: per-layer attention error 0.0610 vs 0.0590, end-to-end PPL within ±0.8% (inside the measurement noise), needle retrieval identical.
3 of 4 criteria pass; the prefetch hit-rate criterion fails as stated (best 0.80–0.85), but the budget it protects (PCIe ≤ 20% of a step) is met
by a wide margin because this hybrid has only 6 attention layers × 2 KV heads. Main caveats: one small model, 8K contexts, small-sample PPL.

## Setup (see README.md, results/00*.md)
- Model: Qwen3.5-0.8B, the only checkpoint on disk (no dense Phase-1 model downloaded: disk 97% full). 6 full-attention layers
  (3,7,11,15,19,23), Hq=8, Hkv=2 (GQA group 4), head_dim 256, partial RoPE (64 dims), output gate. Only attention layers are requantized.
- Forward gate (00): NumPy PPL 12.576 vs HF fp32 12.584 on 2048 wikitext tokens; top-1 agreement 99.95% (1 disagreement at a logit margin of 0.003).
- Offline method (00b): KV cache captured once on GPU (torch AMP) for 5 corpora × 8192 tokens; reference cache = **int8**
  (user decision: compare against a quantized deployment cache). Offline walk-forward attention reproduces the model's o_proj input to 0.3%.
  All calibration uses only prefill tokens; 512 decode queries per layer.

## Go / no-go
| criterion | result | verdict |
|---|---|---|
| DYN within +1% PPL of uniform 4-bit at ≤10% hot (E9) | k=5%: −0.12% prose / +0.74% code; k=10%: −0.20% / +0.24% | **PASS** (but noise ≈ ±0.5–0.8%: GS-4b itself is +0.74%/−0.26% vs the int8 ref) |
| Needle no worse than uniform 4-bit (E9) | GS-4b ✓✓✓, DYN k=2/5/10% ✓✓✓, sign k=5% ✓✓✓ (KIVI-2b ✗✗✓) | **PASS** (easy at 8K; 1 needle) |
| GQA union ≤ ~3× per-head k (E4) | coarse estimate: 2.03× k mean, 3.05× max at k=5% (G=4 caps it at 4×) | **PASS** |
| Prefetch hit ≥ 85% so PCIe ≤ ~20% of a step at 128K (E6/E7/E10) | hit: LRU 4× 0.853, (a)∪(c) prefetch 0.80, spec query alone 0.56; PCIe/step at 128K: 1–2% (measured hits), **12% even with zero reuse** (0.8B, Q4) | hit-rate **FAIL** as stated; PCIe budget **PASS** |

## Key measurements
| experiment | finding |
|---|---|
| E1 Gaussianization | mean+RHT: kurtosis 2.96 (sphere value 2.977), shape MSE 0.02–0.1 dB *better* than the Gaussian table for every head; 0 heads flagged. Without RHT: +1.5–7 dB. |
| E2 Concentration | top-10% holds 0.954 of the mass on average; 4/240 head-cases "flat" (m_opt(10%) < 0.8, worst 0.75). Adaptive k (τ=0.9): 7.5% of tokens on average → 0.931 mass. |
| E3 Selection | m_H/m_opt at k=5%: coarse+RHT 0.989, sign+RHT 0.955, coarse no-RHT 0.976, sign no-RHT 0.915. Bound e^(−2η) never violated but vacuous (median η≈2.8 logits). |
| E5 Pages | at k=5% captured mass 0.88 (page 1) → 0.80 / 0.77 / 0.74 (16/32/64); attn err 0.061 → 0.068 / 0.072 / 0.078. |
| E6 Reuse | step-to-step overlap only 0.56 (synthetic AR(1) assumed far higher); LRU 1×/2×/4×: 0.56 / 0.73 / 0.85. |
| E6b Lazy fetch | misses are low-mass: mass-weighted hits 0.77 / 0.87 / 0.93. Serving misses coarse for one step: err 0.0664 (LRU 2×) vs ideal 0.0606 vs 0.145 without dynamic residuals. |
| E7 Prefetch L→L+4 | (a) previous step 0.59, (b) layer L hot set 0.43, (c) speculative query 0.56 (2× over-fetch 0.78), (a)∪(c) 0.80 at 1.6× transfer. Worst pair 3→7. |
| E8 Per-layer error | attn err: GS-2b 0.234, DYN k=2/5/10% 0.069/0.061/0.059, GS-4b 0.059, KIVI-4b g128 0.044 (4.40 bits). Post- vs pre-RoPE: tie (0.0590 vs 0.0594) → post-RoPE. Gain 6 bits ≈ 8 bits; 4 bits +5%. RHT off: 3× worse. |
| E8b Fair baselines | KIVI's edge is its int8 recent group: without it KIVI-4b 0.074 vs GS-4b 0.059; with the same int8 recent group GS-4b 0.037, DYN k=5% 0.042, KIVI-4b 0.044. Keys favour per-channel quantization; values favour gain-shape. |
| E10 Budget | 128K, Q4 weights: coarse KV 195 MiB (0.8B) vs 387 MiB uniform-4b; PCIe 0.06–0.08 ms vs 3.4 ms step. 9B (extrapolated, config only): 4% of step even with zero reuse. |

## Diagnosis and smallest fixes
1. **Hot-set churn (E6/E7 fail):** the misses are boundary tokens with little mass. Fix: non-blocking *lazy fetch* + LRU 2–4×
   (E6b) instead of demanding ≥85% prediction; quality cost 0.061 → 0.066 attn err at LRU 2×. For this model family PCIe is not the bottleneck anyway.
2. **Recent tokens dominate the remaining error (E8b):** keep the recent window (~64–128 tokens) at int8 instead of 4-bit,
   *outside* the k budget (inside the budget, S/W slightly hurt: E8 S/W table). Cheap: 128 tokens × 8 bits.
3. **Keys:** per-channel structure (channel variance spread 7–15×) is erased by RHT; KIVI-style keys beat gain-shape keys on attention-weighted logit error
   (layer 3: 0.135 vs 0.193; layer 11: 0.173 vs 0.252; diagnostic on prose/code only). Candidate: closed-form per-channel key/query scaling (SmoothQuant-like, from prefill statistics) before the RHT — untested.
4. Selection: coarse beats sign-plane (0.989 vs 0.955 mass); the sign plane is a viable cheaper selector (E9 PPL equal within noise).

## What was not done / caveats
- Phase 1 (dense model) skipped: not on disk and storage is tight. Kimi Linear not attempted.
- Contexts are 8K; all long-context numbers (E10) extrapolate 8K hit rates and concentration. 2B/9B rows use configs only.
- E9 PPL uses 512 decode positions × 2 corpora: differences below ~0.5–0.8% are not resolvable. A longer evaluation would need more CPU time (prefill ≈ 4 min per 8K corpus in NumPy).
- Offline metrics (E1–E8) have no cross-layer error propagation; E9 does.
- Reference text is wikitext-2 (CC-BY-SA, not public domain) + NumPy source code.
