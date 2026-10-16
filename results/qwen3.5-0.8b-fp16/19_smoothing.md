# E19 Why key smoothing matters on some models (pipeline check + characterization)

**TL;DR:** pipeline check (offline exact attention vs the model's own o_proj input), max rel. error: m 0.003.
Median predicted smoothing gain (Σq²·Σk²/(Σ√(q²k²))², from prefill stats): m 1.34x;
measured 2-bit logit-MSE gain: m 1.23x (Spearman predicted vs measured over heads 0.76).

## Config
Reference per model: {'m': 'fp16'}. Corpora ['prose', 'code']; up to 4 attention layers per model; all KV heads; 2-bit (coarse) gain-shape keys with 128-token block means.
Logit error is measured on every 4th decode query over all causal positions. k² = per-channel variance of post-RoPE prefill keys,
q² = per-channel mean square of post-RoPE prefill queries (averaged over the GQA group). runtime 30s.

## (A) Pipeline check
| model | corpus | layer | rel err vs model |
|---|---|---|---|
| m | prose | 3 | 0.0030 |
| m | prose | 11 | 0.0030 |
| m | prose | 15 | 0.0030 |
| m | prose | 23 | 0.0028 |
| m | code | 3 | 0.0029 |
| m | code | 11 | 0.0030 |
| m | code | 15 | 0.0030 |
| m | code | 23 | 0.0029 |

## (B) Summary per model (medians over layer × head × corpus)
| model | n heads | Gpred median | Gpred max | measured gain median | measured max | logit err/spread (no smooth) | logit err/spread (smooth) | k² max/median | q² max/median | Spearman(q², k²) | outlier: max/median mean|k| (median) | (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m | 16 | 1.339 | 1.552 | 1.232 | 1.377 | 0.172 | 0.152 | 7.220 | 21.910 | 0.263 | 4.575 | 11.280 |

## (B) Per head
| model | corpus | layer | head | Gpred | measured gain | err/spread no smooth | err/spread smooth | k² max/med | q² max/med | ρ(q²,k²) | max/med mean|k| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m | prose | 3 | 0 | 1.552 | 1.377 | 0.196 | 0.167 | 14.738 | 41.999 | 0.247 | 11.280 |
| m | prose | 3 | 1 | 1.386 | 1.330 | 0.173 | 0.150 | 6.151 | 24.725 | 0.244 | 6.623 |
| m | prose | 11 | 0 | 1.381 | 1.239 | 0.189 | 0.169 | 7.001 | 20.991 | 0.223 | 2.686 |
| m | prose | 11 | 1 | 1.421 | 1.196 | 0.148 | 0.135 | 12.355 | 19.395 | 0.102 | 3.864 |
| m | prose | 15 | 0 | 1.261 | 1.100 | 0.129 | 0.123 | 11.616 | 16.457 | 0.486 | 4.824 |
| m | prose | 15 | 1 | 1.498 | 1.340 | 0.172 | 0.148 | 4.944 | 28.703 | 0.246 | 4.401 |
| m | prose | 23 | 0 | 1.290 | 1.212 | 0.201 | 0.183 | 3.787 | 16.135 | 0.140 | 4.478 |
| m | prose | 23 | 1 | 1.324 | 1.181 | 0.186 | 0.171 | 5.715 | 21.143 | 0.482 | 3.742 |
| m | code | 3 | 0 | 1.367 | 1.242 | 0.204 | 0.183 | 9.796 | 30.142 | 0.142 | 8.882 |
| m | code | 3 | 1 | 1.346 | 1.276 | 0.167 | 0.147 | 6.429 | 24.290 | 0.265 | 6.894 |
| m | code | 11 | 0 | 1.260 | 1.171 | 0.181 | 0.167 | 8.571 | 21.853 | 0.373 | 3.300 |
| m | code | 11 | 1 | 1.333 | 1.186 | 0.161 | 0.148 | 15.451 | 24.741 | 0.293 | 4.629 |
| m | code | 15 | 0 | 1.289 | 1.189 | 0.162 | 0.148 | 11.740 | 17.202 | 0.463 | 5.153 |
| m | code | 15 | 1 | 1.357 | 1.276 | 0.157 | 0.139 | 4.988 | 15.380 | 0.260 | 3.606 |
| m | code | 23 | 0 | 1.267 | 1.224 | 0.169 | 0.153 | 5.432 | 21.967 | 0.372 | 4.522 |
| m | code | 23 | 1 | 1.326 | 1.239 | 0.172 | 0.155 | 7.440 | 29.542 | 0.412 | 4.903 |

## Plots
![profiles](plots/e19_channel_profiles.png) — per-channel query energy and key variance for the head with the largest predicted gain.
![pred vs meas](plots/e19_gain_pred_vs_meas.png)
