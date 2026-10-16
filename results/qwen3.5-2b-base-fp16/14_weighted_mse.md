# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.79 (per corpus×layer range 0.40–1.15),
Spearman ρ over (query, head) 0.92; the purely statistical model (gains, D, no error knowledge) gives ratio
0.77, ρ 0.87. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.6e-07 (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1482 | 1.0835 | 1.4448 | 0.9657 | 0.9155 |
| GS-2b (all coarse) | wmse | 1.5965 | 0.9527 | 2.4317 | 0.8900 | 0.8310 |
| GS-2b (all coarse) | model | 0.9082 | 0.5821 | 1.0720 | 0.8816 | 0.8302 |
| GS-4b (all fine) | lin | 1.0127 | 0.9972 | 1.0406 | 0.9969 | 0.9936 |
| GS-4b (all fine) | wmse | 1.0102 | 0.9213 | 1.1529 | 0.9708 | 0.9506 |
| GS-4b (all fine) | model | 1.0859 | 0.9551 | 1.2222 | 0.8959 | 0.8186 |
| DynReq k=5% +A | lin | 1.3322 | 1.0752 | 2.1654 | 0.9778 | 0.9524 |
| DynReq k=5% +A | wmse | 0.7902 | 0.3987 | 1.1480 | 0.9238 | 0.7585 |
| DynReq k=5% +A | model | 0.7734 | 0.2909 | 1.3672 | 0.8662 | 0.7324 |

runtime 787s
