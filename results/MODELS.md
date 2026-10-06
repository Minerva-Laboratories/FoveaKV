# Offline results across models

**TL;DR:** on Qwen3.5 0.8B, 2B and 4B, FoveaKV with 5% hot tokens (2.26 bits) beats TurboQuant-4b and is 21–24% behind
gain-shape 4-bit; selection, GQA union and the advantage over static and SpeCache-style refinement at equal traffic all hold.
On dense Qwen3-0.6B (head_dim 128) the same holds only with per-channel key smoothing: its keys and queries put their
outlier energy in different RoPE pairs. A prefill-statistics predictor tells in advance when smoothing is needed.

All numbers are offline (captured caches, walk-forward over the last 512 positions of 8K-token texts) and measured,
not estimated. Metric: relative error of the attention output vs the reference cache, mean over texts, layers and heads.
Per-model details are in `results/<model>/`, the 0.8B study in `results/`.

## Models and setup

| model | type | attention layers used | KV heads | query heads per KV head | head_dim | RoPE dims | reference |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | hybrid | 6 (all) | 2 | 4 | 256 | 64 | int8 |
| Qwen3.5-2B-Base | hybrid | 6 (all) | 2 | 4 | 256 | 64 | int8 |
| Qwen3.5-4B-Base | hybrid | 4 of 8 (3, 11, 19, 27) | 4 | 4 | 256 | 64 | int8 |
| Qwen3-0.6B | dense | 4 of 28 (0, 9, 18, 27) | 8 | 2 | 128 | 128 | fp16 |

Pipeline check (offline exact attention vs the model's own attention output, E19): max error 1.4% (0.8B), 1.2% (2B),
1.2% (4B), 0.6% (Qwen3-0.6B, fp16 reference). The per-token int8 reference is **not** usable on Qwen3-0.6B: its outlier
key channels (mean |k| up to 195 in one channel against a median of 1.2) push the error to 18% (layer 0).

Captures for 2B, 4B and Qwen3-0.6B used `tools/capture_kv_stream.py` (CPU, one layer at a time). On 0.8B it agrees with
the original GPU capture within bf16 noise (K/V 1.2–1.7%, queries ~1%).

## Main comparison (E11: exact sink + recent window, 128-token block means)

| scheme | bits | 0.8B | 2B | 4B | Qwen3-0.6B | Qwen3-0.6B + smoothing |
|---|---|---|---|---|---|---|
| FoveaKV k=5% | 2.26–2.29 | 0.0205 | 0.0252 | 0.0266 | 0.116 | **0.0178** |
| GS-4b | 4.16–4.19 | 0.0165 | 0.0204 | 0.0219 | 0.050 | 0.0091 |
| TurboQuant-4b (no QJL) | 4.06–4.13 | 0.0246 | 0.0284 | 0.0294 | 0.277 | – |
| KIVI-4b | 4.25 | 0.0439 | 0.0543 | 0.0536 | 0.089 | – |

On Qwen3-0.6B, smoothing also fixes GS-2b (0.507 → 0.054). TurboQuant has neither mean removal nor smoothing and breaks
on Qwen3 for the same reason. The gap between FoveaKV and smoothed GS-4b is larger there (about 2x) than on Qwen3.5 (1.2x).

## Selection and refinement (E3, E4, E16, E17, E18; FoveaKV with shrinkage correction, k = 5%)

| | 0.8B | 2B | 4B | Qwen3-0.6B + smoothing |
|---|---|---|---|---|
| coarse selection captures (m_H / m_opt) | 0.989 | 0.988 | 0.988 | 0.936 (no smoothing) |
| GQA union / k (mean, max) | 1.99, 3.10 | 2.04, 3.20 | 1.96, 3.12 | 1.46, 1.98 (group of 2) |
| oracle selection | 0.0169 | 0.0210 | 0.0226 | 0.0111 |
| FoveaKV (coarse estimate) | 0.0175 | 0.0217 | 0.0233 | 0.0121 |
| two-stage (2k candidates, fine keys) | 0.0168 | 0.0208 | 0.0224 | 0.0107 |
| SpeCache-style, same traffic | 0.0537 | 0.0614 | 0.0501 | 0.0611 |
| SpeCache-style, same token count (8x traffic) | 0.0180 | 0.0210 | 0.0156 | 0.0191 |
| static 4-bit set, same count | 0.0342 | 0.0432 | 0.0437 | 0.0248 |

- The selection gap stays small (3–9% over the oracle) and two-stage selection closes it on every model.
- At equal refinement traffic FoveaKV beats SpeCache-style refinement by 2.2–5x. With 8x the traffic, SpeCache-style
  ties FoveaKV on 0.8B and 2B, is worse on Qwen3-0.6B, and is better on 4B (0.0156 vs 0.0233).
- Per-step selection beats a static set chosen from probe queries by 1.9–2.1x on every model.

## When is key smoothing needed? (E19)

After the rotation the quantization error is spread evenly over dimensions, so the logit error scales with
(Σ_c q_c²)(Σ_c k_c²). Smoothing rescales channels so this becomes (Σ_c √(q_c² k_c²))². The ratio of the two,
computed from prefill statistics, predicts the gain:

| | 0.8B | 2B | 4B | Qwen3-0.6B |
|---|---|---|---|---|
| predicted gain (median over heads) | 1.34x | 1.29x | 1.35x | 4.09x (max 55x) |
| measured 2-bit logit-MSE gain | 1.21x | 1.20x | 1.25x | 3.41x (max 40x) |
| key variance, max / median channel | 7 | 7 | 6 | 42 |
| Spearman(q², k²) over channels | 0.27 | 0.40 | 0.40 | −0.05 |

Predicted and measured gains agree across heads (Spearman 0.97 over the 0.8B, 2B and Qwen3 heads; 0.88 on 4B alone).
On Qwen3-0.6B the key outliers and the query outliers sit in different RoPE pairs (channels c and c+64); see
`e19/plots/e19_channel_profiles.png`. Smoothing costs nothing at run time (the query is scaled once per step; keys
are stored divided by the scale).

## Not done yet
- End-to-end perplexity, retrieval and latency on 2B, 4B and Qwen3-0.6B (GPU is shared; 2B/4B weights do not fit next to
  the other job). Kernels support head_dim 128 (all 21 variants pass), but GQA groups other than 4 and key smoothing
  are not in the GPU path yet.
- Qwen3-0.6B: why the FoveaKV vs GS-4b gap is wider than on Qwen3.5 (attention there is more concentrated, so it is
  not flat attention).
- 4B: why 8x-traffic SpeCache-style refinement beats FoveaKV there.
