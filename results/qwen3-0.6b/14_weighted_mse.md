# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.99 (per corpus×layer range 0.01–1.30),
Spearman ρ over (query, head) 0.80; the purely statistical model (gains, D, no error knowledge) gives ratio
1.15, ρ 0.53. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 4.4e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.0696 | 0.0569 | 1.3990 | 0.7076 | -0.1801 |
| GS-2b (all coarse) | wmse | 1.0935 | 0.0415 | 2.8184 | 0.6467 | -0.2727 |
| GS-2b (all coarse) | model | 1.3077 | 0.0495 | 3.4555 | 0.4940 | -0.2903 |
| GS-4b (all fine) | lin | 1.0017 | 0.4185 | 1.0735 | 0.9167 | 0.3913 |
| GS-4b (all fine) | wmse | 0.9990 | 0.4161 | 1.3759 | 0.8559 | 0.3893 |
| GS-4b (all fine) | model | 1.1724 | 0.6129 | 2.4005 | 0.6581 | 0.0923 |
| DynReq k=5% +A | lin | 1.0101 | 0.0866 | 1.5318 | 0.7915 | -0.1796 |
| DynReq k=5% +A | wmse | 0.9873 | 0.0108 | 1.2952 | 0.7985 | -0.2462 |
| DynReq k=5% +A | model | 1.1521 | 0.0224 | 2.8921 | 0.5312 | -0.2442 |

runtime 321s
