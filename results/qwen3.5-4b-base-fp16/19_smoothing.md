# E19 Why key smoothing matters on some models (pipeline check + characterization)

**TL;DR:** pipeline check (offline exact attention vs the model's own o_proj input), max rel. error: m 0.003.
Median predicted smoothing gain (Σq²·Σk²/(Σ√(q²k²))², from prefill stats): m 1.33x;
measured 2-bit logit-MSE gain: m 1.24x (Spearman predicted vs measured over heads 0.83).

## Config
Reference per model: {'m': 'fp16'}. Corpora ['prose', 'code']; up to 4 attention layers per model; all KV heads; 2-bit (coarse) gain-shape keys with 128-token block means.
Logit error is measured on every 4th decode query over all causal positions. k² = per-channel variance of post-RoPE prefill keys,
q² = per-channel mean square of post-RoPE prefill queries (averaged over the GQA group). runtime 46s.

## (A) Pipeline check
| model | corpus | layer | rel err vs model |
|---|---|---|---|
| m | prose | 3 | 0.0030 |
| m | prose | 11 | 0.0030 |
| m | prose | 19 | 0.0030 |
| m | prose | 27 | 0.0030 |
| m | code | 3 | 0.0031 |
| m | code | 11 | 0.0030 |
| m | code | 19 | 0.0030 |
| m | code | 27 | 0.0030 |

## (B) Summary per model (medians over layer × head × corpus)
| model | n heads | Gpred median | Gpred max | measured gain median | measured max | logit err/spread (no smooth) | logit err/spread (smooth) | k² max/median | q² max/median | Spearman(q², k²) | outlier: max/median mean|k| (median) | (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m | 32 | 1.332 | 1.407 | 1.243 | 1.370 | 0.181 | 0.164 | 6.860 | 25.656 | 0.413 | 6.057 | 10.611 |

## (B) Per head
| model | corpus | layer | head | Gpred | measured gain | err/spread no smooth | err/spread smooth | k² max/med | q² max/med | ρ(q²,k²) | max/med mean|k| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m | prose | 3 | 0 | 1.385 | 1.263 | 0.179 | 0.159 | 17.434 | 47.062 | 0.590 | 9.393 |
| m | prose | 3 | 1 | 1.297 | 1.194 | 0.178 | 0.163 | 11.672 | 19.086 | 0.338 | 8.530 |
| m | prose | 3 | 2 | 1.398 | 1.267 | 0.200 | 0.178 | 11.543 | 65.609 | 0.359 | 7.403 |
| m | prose | 3 | 3 | 1.407 | 1.340 | 0.155 | 0.134 | 3.412 | 20.593 | -0.049 | 4.129 |
| m | prose | 11 | 0 | 1.387 | 1.301 | 0.177 | 0.155 | 14.003 | 37.721 | 0.448 | 6.250 |
| m | prose | 11 | 1 | 1.374 | 1.238 | 0.173 | 0.155 | 14.249 | 42.669 | 0.488 | 5.909 |
| m | prose | 11 | 2 | 1.320 | 1.267 | 0.204 | 0.181 | 6.296 | 18.896 | 0.456 | 2.905 |
| m | prose | 11 | 3 | 1.402 | 1.290 | 0.191 | 0.168 | 12.659 | 50.293 | 0.579 | 5.874 |
| m | prose | 19 | 0 | 1.217 | 1.146 | 0.158 | 0.147 | 8.029 | 21.393 | 0.540 | 3.200 |
| m | prose | 19 | 1 | 1.304 | 1.235 | 0.195 | 0.176 | 4.741 | 18.039 | 0.190 | 3.708 |
| m | prose | 19 | 2 | 1.306 | 1.192 | 0.168 | 0.154 | 10.440 | 22.956 | 0.440 | 4.644 |
| m | prose | 19 | 3 | 1.245 | 1.182 | 0.164 | 0.151 | 4.517 | 16.127 | 0.244 | 6.223 |
| m | prose | 27 | 0 | 1.283 | 1.221 | 0.198 | 0.180 | 4.517 | 36.216 | 0.486 | 8.474 |
| m | prose | 27 | 1 | 1.355 | 1.242 | 0.174 | 0.156 | 4.627 | 38.291 | 0.369 | 7.089 |
| m | prose | 27 | 2 | 1.363 | 1.253 | 0.186 | 0.166 | 4.726 | 21.696 | 0.258 | 3.533 |
| m | prose | 27 | 3 | 1.343 | 1.261 | 0.198 | 0.176 | 3.893 | 27.599 | 0.265 | 6.621 |
| m | code | 3 | 0 | 1.285 | 1.172 | 0.184 | 0.170 | 18.420 | 28.709 | 0.513 | 9.988 |
| m | code | 3 | 1 | 1.254 | 1.180 | 0.198 | 0.182 | 7.985 | 14.818 | 0.183 | 10.611 |
| m | code | 3 | 2 | 1.287 | 1.194 | 0.183 | 0.167 | 6.761 | 37.036 | 0.353 | 6.576 |
| m | code | 3 | 3 | 1.383 | 1.370 | 0.138 | 0.118 | 2.457 | 23.713 | 0.037 | 4.461 |
| m | code | 11 | 0 | 1.385 | 1.271 | 0.168 | 0.149 | 16.878 | 44.763 | 0.429 | 6.178 |
| m | code | 11 | 1 | 1.342 | 1.182 | 0.161 | 0.148 | 16.884 | 44.603 | 0.559 | 6.075 |
| m | code | 11 | 2 | 1.343 | 1.278 | 0.189 | 0.168 | 6.960 | 20.972 | 0.514 | 3.080 |
| m | code | 11 | 3 | 1.379 | 1.247 | 0.184 | 0.165 | 15.649 | 56.899 | 0.591 | 5.839 |
| m | code | 19 | 0 | 1.204 | 1.131 | 0.166 | 0.156 | 10.416 | 18.398 | 0.570 | 3.935 |
| m | code | 19 | 1 | 1.279 | 1.241 | 0.184 | 0.165 | 5.612 | 19.973 | 0.346 | 3.580 |
| m | code | 19 | 2 | 1.253 | 1.174 | 0.175 | 0.161 | 15.429 | 21.346 | 0.477 | 5.209 |
| m | code | 19 | 3 | 1.217 | 1.169 | 0.160 | 0.148 | 4.164 | 15.468 | 0.397 | 6.038 |
| m | code | 27 | 0 | 1.281 | 1.244 | 0.190 | 0.170 | 5.577 | 36.900 | 0.518 | 9.022 |
| m | code | 27 | 1 | 1.372 | 1.267 | 0.179 | 0.159 | 4.260 | 39.897 | 0.148 | 6.904 |
| m | code | 27 | 2 | 1.359 | 1.271 | 0.192 | 0.170 | 5.485 | 22.460 | 0.210 | 3.985 |
| m | code | 27 | 3 | 1.323 | 1.275 | 0.191 | 0.169 | 6.557 | 31.285 | 0.364 | 6.707 |

## Plots
![profiles](plots/e19_channel_profiles.png) — per-channel query energy and key variance for the head with the largest predicted gain.
![pred vs meas](plots/e19_gain_pred_vs_meas.png)
