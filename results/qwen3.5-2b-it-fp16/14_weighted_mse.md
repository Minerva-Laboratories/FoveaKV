# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.85 (per corpus×layer range 0.39–1.22),
Spearman ρ over (query, head) 0.95; the purely statistical model (gains, D, no error knowledge) gives ratio
0.87, ρ 0.89. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 4.9e-07 (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1452 | 1.1116 | 1.2869 | 0.9620 | 0.8989 |
| GS-2b (all coarse) | wmse | 1.6441 | 0.9820 | 2.1394 | 0.8936 | 0.8348 |
| GS-2b (all coarse) | model | 0.9339 | 0.5873 | 1.1062 | 0.8805 | 0.8354 |
| GS-4b (all fine) | lin | 1.0139 | 0.9990 | 1.0691 | 0.9968 | 0.9928 |
| GS-4b (all fine) | wmse | 1.0304 | 0.9484 | 1.2151 | 0.9696 | 0.9512 |
| GS-4b (all fine) | model | 1.0616 | 0.9492 | 1.2636 | 0.8913 | 0.8219 |
| DynReq k=5% +A | lin | 1.2740 | 1.0970 | 2.3615 | 0.9765 | 0.9436 |
| DynReq k=5% +A | wmse | 0.8461 | 0.3869 | 1.2180 | 0.9472 | 0.7961 |
| DynReq k=5% +A | model | 0.8728 | 0.2792 | 1.4098 | 0.8859 | 0.7539 |

runtime 886s
