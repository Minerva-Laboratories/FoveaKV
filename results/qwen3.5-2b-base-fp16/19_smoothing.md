# E19 Why key smoothing matters on some models (pipeline check + characterization)

**TL;DR:** pipeline check (offline exact attention vs the model's own o_proj input), max rel. error: m 0.003.
Median predicted smoothing gain (Σq²·Σk²/(Σ√(q²k²))², from prefill stats): m 1.29x;
measured 2-bit logit-MSE gain: m 1.20x (Spearman predicted vs measured over heads 0.84).

## Config
Reference per model: {'m': 'fp16'}. Corpora ['prose', 'code']; up to 4 attention layers per model; all KV heads; 2-bit (coarse) gain-shape keys with 128-token block means.
Logit error is measured on every 4th decode query over all causal positions. k² = per-channel variance of post-RoPE prefill keys,
q² = per-channel mean square of post-RoPE prefill queries (averaged over the GQA group). runtime 26s.

## (A) Pipeline check
| model | corpus | layer | rel err vs model |
|---|---|---|---|
| m | prose | 3 | 0.0029 |
| m | prose | 11 | 0.0030 |
| m | prose | 15 | 0.0029 |
| m | prose | 23 | 0.0029 |
| m | code | 3 | 0.0030 |
| m | code | 11 | 0.0030 |
| m | code | 15 | 0.0030 |
| m | code | 23 | 0.0029 |

## (B) Summary per model (medians over layer × head × corpus)
| model | n heads | Gpred median | Gpred max | measured gain median | measured max | logit err/spread (no smooth) | logit err/spread (smooth) | k² max/median | q² max/median | Spearman(q², k²) | outlier: max/median mean|k| (median) | (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m | 16 | 1.294 | 1.484 | 1.204 | 1.350 | 0.179 | 0.163 | 7.484 | 21.101 | 0.398 | 5.219 | 9.873 |

## (B) Per head
| model | corpus | layer | head | Gpred | measured gain | err/spread no smooth | err/spread smooth | k² max/med | q² max/med | ρ(q²,k²) | max/med mean|k| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m | prose | 3 | 0 | 1.484 | 1.350 | 0.189 | 0.163 | 17.080 | 44.983 | 0.328 | 9.873 |
| m | prose | 3 | 1 | 1.350 | 1.275 | 0.170 | 0.151 | 7.445 | 22.621 | 0.317 | 7.302 |
| m | prose | 11 | 0 | 1.218 | 1.148 | 0.195 | 0.182 | 6.563 | 19.135 | 0.456 | 3.526 |
| m | prose | 11 | 1 | 1.300 | 1.187 | 0.160 | 0.147 | 11.517 | 24.855 | 0.460 | 4.548 |
| m | prose | 15 | 0 | 1.248 | 1.119 | 0.162 | 0.153 | 10.065 | 18.618 | 0.416 | 5.575 |
| m | prose | 15 | 1 | 1.411 | 1.277 | 0.188 | 0.166 | 4.085 | 21.616 | 0.241 | 4.405 |
| m | prose | 23 | 0 | 1.239 | 1.206 | 0.211 | 0.192 | 3.533 | 19.599 | 0.264 | 4.715 |
| m | prose | 23 | 1 | 1.292 | 1.212 | 0.199 | 0.180 | 4.677 | 18.765 | 0.249 | 5.485 |
| m | code | 3 | 0 | 1.354 | 1.232 | 0.200 | 0.180 | 9.059 | 36.146 | 0.230 | 8.597 |
| m | code | 3 | 1 | 1.324 | 1.295 | 0.166 | 0.146 | 7.524 | 27.430 | 0.389 | 7.540 |
| m | code | 11 | 0 | 1.198 | 1.149 | 0.188 | 0.176 | 7.772 | 22.493 | 0.499 | 3.876 |
| m | code | 11 | 1 | 1.296 | 1.200 | 0.168 | 0.154 | 10.112 | 30.959 | 0.467 | 4.573 |
| m | code | 15 | 0 | 1.231 | 1.145 | 0.167 | 0.156 | 10.419 | 20.587 | 0.516 | 5.816 |
| m | code | 15 | 1 | 1.324 | 1.250 | 0.173 | 0.155 | 5.384 | 16.228 | 0.327 | 4.314 |
| m | code | 23 | 0 | 1.234 | 1.202 | 0.178 | 0.163 | 4.445 | 18.738 | 0.414 | 5.598 |
| m | code | 23 | 1 | 1.232 | 1.186 | 0.179 | 0.165 | 3.976 | 17.490 | 0.407 | 4.953 |

## Plots
![profiles](plots/e19_channel_profiles.png) — per-channel query energy and key variance for the head with the largest predicted gain.
![pred vs meas](plots/e19_gain_pred_vs_meas.png)
