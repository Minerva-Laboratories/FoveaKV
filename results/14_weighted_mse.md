# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for FoveaKV k=5% +A, median predicted/actual ratio 0.86 (per corpus×layer range 0.35–1.15),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.88, ρ 0.87. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.6e-07 (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1383 | 1.1098 | 1.2636 | 0.9684 | 0.9374 |
| GS-2b (all coarse) | wmse | 1.6358 | 1.0890 | 2.0086 | 0.8924 | 0.8273 |
| GS-2b (all coarse) | model | 0.9748 | 0.6798 | 1.1355 | 0.8705 | 0.8058 |
| GS-4b (all fine) | lin | 1.0134 | 1.0006 | 1.0451 | 0.9971 | 0.9947 |
| GS-4b (all fine) | wmse | 1.0429 | 0.9791 | 1.1894 | 0.9718 | 0.9595 |
| GS-4b (all fine) | model | 1.1118 | 0.9809 | 1.2493 | 0.8935 | 0.8360 |
| FoveaKV k=5% +A | lin | 1.3379 | 1.0879 | 2.3283 | 0.9752 | 0.9388 |
| FoveaKV k=5% +A | wmse | 0.8603 | 0.3499 | 1.1476 | 0.9357 | 0.7080 |
| FoveaKV k=5% +A | model | 0.8831 | 0.2575 | 1.4323 | 0.8737 | 0.6654 |

runtime 240s
