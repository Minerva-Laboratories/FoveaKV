# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.91 (per corpus×layer range 0.45–1.09),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.87, ρ 0.88. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 5.6e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1722 | 1.0937 | 1.2688 | 0.9672 | 0.9277 |
| GS-2b (all coarse) | wmse | 1.6156 | 1.1585 | 2.0143 | 0.8907 | 0.8332 |
| GS-2b (all coarse) | model | 0.8784 | 0.7190 | 1.1956 | 0.8742 | 0.7953 |
| GS-4b (all fine) | lin | 1.0164 | 1.0007 | 1.0349 | 0.9969 | 0.9945 |
| GS-4b (all fine) | wmse | 1.0445 | 0.9807 | 1.1467 | 0.9666 | 0.9338 |
| GS-4b (all fine) | model | 1.0926 | 0.9882 | 1.2966 | 0.8915 | 0.8133 |
| DynReq k=5% +A | lin | 1.2696 | 1.0177 | 2.0516 | 0.9744 | 0.9260 |
| DynReq k=5% +A | wmse | 0.9089 | 0.4486 | 1.0913 | 0.9419 | 0.8648 |
| DynReq k=5% +A | model | 0.8654 | 0.3348 | 1.5070 | 0.8794 | 0.7971 |

runtime 581s
