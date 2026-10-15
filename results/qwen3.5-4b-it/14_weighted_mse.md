# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.88 (per corpus×layer range 0.42–1.13),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.83, ρ 0.87. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 5.2e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1480 | 1.0936 | 1.2381 | 0.9689 | 0.9523 |
| GS-2b (all coarse) | wmse | 1.6394 | 1.0287 | 2.0315 | 0.8911 | 0.8454 |
| GS-2b (all coarse) | model | 0.8677 | 0.6200 | 1.1561 | 0.8713 | 0.7838 |
| GS-4b (all fine) | lin | 1.0133 | 0.9703 | 1.0344 | 0.9967 | 0.9948 |
| GS-4b (all fine) | wmse | 1.0174 | 0.9103 | 1.1725 | 0.9666 | 0.9358 |
| GS-4b (all fine) | model | 1.0824 | 0.9496 | 1.2450 | 0.8894 | 0.8065 |
| DynReq k=5% +A | lin | 1.2302 | 1.0129 | 2.1366 | 0.9733 | 0.9162 |
| DynReq k=5% +A | wmse | 0.8847 | 0.4173 | 1.1297 | 0.9424 | 0.8111 |
| DynReq k=5% +A | model | 0.8270 | 0.2994 | 1.5584 | 0.8749 | 0.7486 |

runtime 1228s
