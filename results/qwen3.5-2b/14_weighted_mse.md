# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.81 (per corpus×layer range 0.39–1.19),
Spearman ρ over (query, head) 0.92; the purely statistical model (gains, D, no error knowledge) gives ratio
0.82, ρ 0.87. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 5.5e-07 (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1492 | 1.0744 | 1.3607 | 0.9663 | 0.9295 |
| GS-2b (all coarse) | wmse | 1.5929 | 1.0625 | 2.3009 | 0.8916 | 0.8032 |
| GS-2b (all coarse) | model | 0.9079 | 0.6406 | 1.0316 | 0.8828 | 0.8246 |
| GS-4b (all fine) | lin | 1.0141 | 1.0045 | 1.0427 | 0.9971 | 0.9944 |
| GS-4b (all fine) | wmse | 1.0372 | 0.9660 | 1.1759 | 0.9730 | 0.9548 |
| GS-4b (all fine) | model | 1.0839 | 0.9633 | 1.2789 | 0.8993 | 0.8350 |
| DynReq k=5% +A | lin | 1.3651 | 1.0815 | 2.1382 | 0.9776 | 0.9521 |
| DynReq k=5% +A | wmse | 0.8110 | 0.3919 | 1.1942 | 0.9240 | 0.7556 |
| DynReq k=5% +A | model | 0.8215 | 0.2964 | 1.6460 | 0.8664 | 0.7252 |

runtime 363s
