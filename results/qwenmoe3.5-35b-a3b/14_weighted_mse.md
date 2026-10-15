# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.80 (per corpus×layer range 0.40–1.11),
Spearman ρ over (query, head) 0.95; the purely statistical model (gains, D, no error knowledge) gives ratio
0.78, ρ 0.91. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.5e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1688 | 1.0826 | 1.1989 | 0.9755 | 0.9537 |
| GS-2b (all coarse) | wmse | 1.4277 | 0.9736 | 1.7944 | 0.9021 | 0.8597 |
| GS-2b (all coarse) | model | 0.8264 | 0.5817 | 1.0129 | 0.8821 | 0.8449 |
| GS-4b (all fine) | lin | 1.0155 | 0.9981 | 1.0360 | 0.9973 | 0.9954 |
| GS-4b (all fine) | wmse | 1.0213 | 0.9499 | 1.1151 | 0.9719 | 0.9509 |
| GS-4b (all fine) | model | 1.0871 | 0.9264 | 1.2276 | 0.9090 | 0.8758 |
| DynReq k=5% +A | lin | 1.4405 | 1.1580 | 1.8552 | 0.9719 | 0.9415 |
| DynReq k=5% +A | wmse | 0.8014 | 0.3967 | 1.1139 | 0.9524 | 0.9048 |
| DynReq k=5% +A | model | 0.7805 | 0.2811 | 1.3187 | 0.9121 | 0.8551 |

runtime 1101s
