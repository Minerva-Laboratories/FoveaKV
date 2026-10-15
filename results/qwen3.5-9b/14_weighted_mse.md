# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.92 (per corpus×layer range 0.42–1.15),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.96, ρ 0.90. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.9e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1635 | 1.0869 | 1.2823 | 0.9670 | 0.9246 |
| GS-2b (all coarse) | wmse | 1.4737 | 0.9942 | 1.8562 | 0.8928 | 0.7158 |
| GS-2b (all coarse) | model | 0.8271 | 0.5007 | 0.9762 | 0.8663 | 0.5424 |
| GS-4b (all fine) | lin | 1.0138 | 0.9921 | 1.0412 | 0.9968 | 0.9941 |
| GS-4b (all fine) | wmse | 1.0352 | 0.9792 | 1.1246 | 0.9595 | 0.9161 |
| GS-4b (all fine) | model | 1.0902 | 0.9851 | 1.2364 | 0.8948 | 0.8292 |
| DynReq k=5% +A | lin | 1.2330 | 1.0407 | 2.1220 | 0.9763 | 0.9395 |
| DynReq k=5% +A | wmse | 0.9194 | 0.4233 | 1.1472 | 0.9399 | 0.7919 |
| DynReq k=5% +A | model | 0.9568 | 0.3020 | 1.3925 | 0.8959 | 0.7775 |

runtime 1210s
