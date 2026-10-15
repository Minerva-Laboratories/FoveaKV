# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.90 (per corpus×layer range 0.39–1.28),
Spearman ρ over (query, head) 0.95; the purely statistical model (gains, D, no error knowledge) gives ratio
0.87, ρ 0.89. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 4.7e-07 (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1506 | 1.1147 | 1.3219 | 0.9633 | 0.8909 |
| GS-2b (all coarse) | wmse | 1.6477 | 1.0072 | 2.1402 | 0.8942 | 0.8370 |
| GS-2b (all coarse) | model | 0.9304 | 0.5997 | 1.1094 | 0.8802 | 0.8269 |
| GS-4b (all fine) | lin | 1.0145 | 1.0065 | 1.0686 | 0.9965 | 0.9884 |
| GS-4b (all fine) | wmse | 1.0309 | 0.9483 | 1.2399 | 0.9677 | 0.9481 |
| GS-4b (all fine) | model | 1.0640 | 0.9569 | 1.2890 | 0.8926 | 0.8199 |
| DynReq k=5% +A | lin | 1.2997 | 1.1248 | 2.3820 | 0.9767 | 0.9446 |
| DynReq k=5% +A | wmse | 0.9039 | 0.3865 | 1.2757 | 0.9465 | 0.8051 |
| DynReq k=5% +A | model | 0.8662 | 0.2793 | 1.2993 | 0.8872 | 0.7643 |

runtime 852s
