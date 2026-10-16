# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.93 (per corpus×layer range 0.43–1.19),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.97, ρ 0.90. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 5.9e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1568 | 1.0884 | 1.2660 | 0.9677 | 0.9316 |
| GS-2b (all coarse) | wmse | 1.4726 | 1.0019 | 1.8580 | 0.8928 | 0.7052 |
| GS-2b (all coarse) | model | 0.8258 | 0.4977 | 0.9795 | 0.8662 | 0.5429 |
| GS-4b (all fine) | lin | 1.0155 | 0.9891 | 1.0296 | 0.9968 | 0.9914 |
| GS-4b (all fine) | wmse | 1.0309 | 0.9702 | 1.1333 | 0.9599 | 0.9227 |
| GS-4b (all fine) | model | 1.0696 | 0.9647 | 1.1915 | 0.8996 | 0.8450 |
| DynReq k=5% +A | lin | 1.2285 | 1.1001 | 2.1442 | 0.9756 | 0.9400 |
| DynReq k=5% +A | wmse | 0.9333 | 0.4284 | 1.1948 | 0.9394 | 0.7847 |
| DynReq k=5% +A | model | 0.9667 | 0.3016 | 1.2283 | 0.8985 | 0.7862 |

runtime 1072s
