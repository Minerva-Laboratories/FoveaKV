# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.87 (per corpus×layer range 0.41–1.15),
Spearman ρ over (query, head) 0.95; the purely statistical model (gains, D, no error knowledge) gives ratio
0.83, ρ 0.91. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.6e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1346 | 1.0849 | 1.2148 | 0.9764 | 0.9570 |
| GS-2b (all coarse) | wmse | 1.3743 | 0.9640 | 1.6414 | 0.8924 | 0.8448 |
| GS-2b (all coarse) | model | 0.7945 | 0.5829 | 1.0685 | 0.8738 | 0.8346 |
| GS-4b (all fine) | lin | 1.0116 | 0.9987 | 1.0288 | 0.9974 | 0.9950 |
| GS-4b (all fine) | wmse | 1.0099 | 0.9366 | 1.1335 | 0.9708 | 0.9494 |
| GS-4b (all fine) | model | 1.0848 | 0.9482 | 1.2062 | 0.9074 | 0.8615 |
| DynReq k=5% +A | lin | 1.3574 | 1.1050 | 1.9058 | 0.9728 | 0.9431 |
| DynReq k=5% +A | wmse | 0.8748 | 0.4133 | 1.1475 | 0.9531 | 0.8936 |
| DynReq k=5% +A | model | 0.8262 | 0.3061 | 1.4103 | 0.9102 | 0.8347 |

runtime 1226s
