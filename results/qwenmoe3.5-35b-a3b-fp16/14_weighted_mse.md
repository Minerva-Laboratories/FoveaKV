# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.80 (per corpus×layer range 0.40–1.14),
Spearman ρ over (query, head) 0.95; the purely statistical model (gains, D, no error knowledge) gives ratio
0.76, ρ 0.91. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 5.7e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1748 | 1.0750 | 1.1913 | 0.9746 | 0.9521 |
| GS-2b (all coarse) | wmse | 1.3710 | 0.9728 | 1.8707 | 0.9011 | 0.8616 |
| GS-2b (all coarse) | model | 0.8235 | 0.5805 | 0.9942 | 0.8827 | 0.8450 |
| GS-4b (all fine) | lin | 1.0153 | 1.0015 | 1.0403 | 0.9973 | 0.9954 |
| GS-4b (all fine) | wmse | 1.0200 | 0.9379 | 1.1473 | 0.9717 | 0.9563 |
| GS-4b (all fine) | model | 1.0803 | 0.9368 | 1.2125 | 0.9090 | 0.8726 |
| DynReq k=5% +A | lin | 1.4190 | 1.1765 | 1.8772 | 0.9722 | 0.9420 |
| DynReq k=5% +A | wmse | 0.8000 | 0.4023 | 1.1405 | 0.9529 | 0.9075 |
| DynReq k=5% +A | model | 0.7553 | 0.2821 | 1.2868 | 0.9123 | 0.8559 |

runtime 993s
