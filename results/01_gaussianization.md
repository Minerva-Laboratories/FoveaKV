# E1 Gaussianization (Qwen3.5-0.8B, int8 reference cache, corpora prose+code, 6 attn layers × 2 KV heads)

**TL;DR:** mean removal + RHT brings key-shape kurtosis from median 5.1 (raw, max 8.0) to 2.96 (max 2.98; Gaussian = 3).
Empirical shape-MSE gap vs the Gaussian table after mean+RHT: keys post-RoPE median -0.04 dB (2b) / -0.07 dB (4b), worst -0.02 / -0.03 dB; values -0.05 / -0.08 dB.
0 (corpus, layer, tensor, head) cases exceed the 0.5 dB gap after mean+RHT.

## Config
- z = u·sqrt(d), u = y/||y||; y = x − mu (mu from prefill tokens [0, T−512)) then optionally RHT (seed 0). d = 256.
- Table: 2b nested-coarse MSE 0.1203, 4b Lloyd-Max 0.0095. gap dB = 10·log10(empirical/table) (positive = worse than Gaussian).
- Data: all 8192 tokens per corpus.

## Aggregate (median / max over corpus × layer × head)
| tensor | variant | kurt med | kurt max | gap2b med dB | gap2b max dB | gap4b med dB | gap4b max dB |
|---|---|---|---|---|---|---|---|
| K-post | raw | 5.08 | 8.02 | 1.55 | 2.57 | 3.94 | 7.00 |
| K-post | mean | 4.39 | 6.62 | 1.16 | 2.26 | 2.61 | 5.63 |
| K-post | mean+RHT | 2.96 | 2.98 | -0.04 | -0.02 | -0.07 | -0.03 |
| K-pre | raw | 5.11 | 8.04 | 1.57 | 2.60 | 3.98 | 6.98 |
| K-pre | mean | 4.69 | 6.77 | 1.28 | 2.38 | 3.35 | 5.75 |
| K-pre | mean+RHT | 2.96 | 2.98 | -0.04 | -0.02 | -0.07 | 0.00 |
| V | raw | 5.45 | 14.94 | 1.87 | 4.42 | 4.29 | 10.62 |
| V | mean | 5.42 | 11.50 | 1.60 | 2.96 | 4.45 | 8.69 |
| V | mean+RHT | 2.95 | 2.98 | -0.05 | -0.02 | -0.08 | -0.02 |

## Heads exceeding 0.5 dB after mean+RHT
None.

## Per layer (mean+RHT, keys post-RoPE / values, prose)
| corpus | layer | tensor | head | variant | kurt | mse2 | gap2 dB | mse4 | gap4 dB |
|---|---|---|---|---|---|---|---|---|---|
| prose | 3 | K-post | 0 | mean+RHT | 2.9500 | 0.1189 | -0.0500 | 0.0093 | -0.0769 |
| prose | 3 | K-post | 1 | mean+RHT | 2.9665 | 0.1193 | -0.0354 | 0.0094 | -0.0511 |
| prose | 3 | V | 0 | mean+RHT | 2.9705 | 0.1195 | -0.0273 | 0.0094 | -0.0560 |
| prose | 3 | V | 1 | mean+RHT | 2.9683 | 0.1195 | -0.0286 | 0.0094 | -0.0587 |
| prose | 7 | K-post | 0 | mean+RHT | 2.9499 | 0.1188 | -0.0548 | 0.0093 | -0.0949 |
| prose | 7 | K-post | 1 | mean+RHT | 2.9626 | 0.1192 | -0.0394 | 0.0094 | -0.0574 |
| prose | 7 | V | 0 | mean+RHT | 2.9574 | 0.1190 | -0.0478 | 0.0093 | -0.0899 |
| prose | 7 | V | 1 | mean+RHT | 2.9774 | 0.1196 | -0.0240 | 0.0094 | -0.0432 |
| prose | 11 | K-post | 0 | mean+RHT | 2.9571 | 0.1192 | -0.0405 | 0.0094 | -0.0685 |
| prose | 11 | K-post | 1 | mean+RHT | 2.9534 | 0.1189 | -0.0515 | 0.0093 | -0.0984 |
| prose | 11 | V | 0 | mean+RHT | 2.9471 | 0.1188 | -0.0537 | 0.0093 | -0.0986 |
| prose | 11 | V | 1 | mean+RHT | 2.9500 | 0.1188 | -0.0533 | 0.0093 | -0.0789 |
| prose | 15 | K-post | 0 | mean+RHT | 2.9593 | 0.1193 | -0.0355 | 0.0093 | -0.0883 |
| prose | 15 | K-post | 1 | mean+RHT | 2.9757 | 0.1195 | -0.0289 | 0.0094 | -0.0542 |
| prose | 15 | V | 0 | mean+RHT | 2.9430 | 0.1187 | -0.0578 | 0.0093 | -0.0845 |
| prose | 15 | V | 1 | mean+RHT | 2.8945 | 0.1172 | -0.1131 | 0.0092 | -0.1619 |
| prose | 19 | K-post | 0 | mean+RHT | 2.9537 | 0.1189 | -0.0513 | 0.0093 | -0.0854 |
| prose | 19 | K-post | 1 | mean+RHT | 2.9794 | 0.1198 | -0.0180 | 0.0094 | -0.0510 |
| prose | 19 | V | 0 | mean+RHT | 2.9115 | 0.1179 | -0.0884 | 0.0092 | -0.1363 |
| prose | 19 | V | 1 | mean+RHT | 2.8932 | 0.1173 | -0.1104 | 0.0091 | -0.1663 |
| prose | 23 | K-post | 0 | mean+RHT | 2.9709 | 0.1195 | -0.0274 | 0.0094 | -0.0415 |
| prose | 23 | K-post | 1 | mean+RHT | 2.9713 | 0.1195 | -0.0285 | 0.0094 | -0.0499 |
| prose | 23 | V | 0 | mean+RHT | 2.9188 | 0.1180 | -0.0838 | 0.0092 | -0.1209 |
| prose | 23 | V | 1 | mean+RHT | 2.9353 | 0.1184 | -0.0698 | 0.0092 | -0.1260 |

## Plots
![hist](plots/e01_hist.png) — log-density of z (prose, head 0, layers 3/15) with N(0,1) dashed.

## Caveats
- The reference is the int8 cache (user decision); int8 noise (~1% rel) is far below 4-bit shape error, so it does not affect these gaps.
- Note that a negative gap means *better* than the Gaussian table (lighter tails / sub-Gaussian shape).

runtime 33s
