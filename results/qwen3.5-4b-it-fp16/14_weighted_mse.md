# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.91 (per corpus×layer range 0.42–1.15),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.91, ρ 0.88. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 5.9e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1446 | 1.1001 | 1.2397 | 0.9694 | 0.9507 |
| GS-2b (all coarse) | wmse | 1.6336 | 1.0191 | 2.0092 | 0.8936 | 0.8393 |
| GS-2b (all coarse) | model | 0.8585 | 0.6250 | 1.1310 | 0.8735 | 0.7837 |
| GS-4b (all fine) | lin | 1.0157 | 0.9876 | 1.0339 | 0.9968 | 0.9935 |
| GS-4b (all fine) | wmse | 1.0181 | 0.9264 | 1.1920 | 0.9668 | 0.9392 |
| GS-4b (all fine) | model | 1.0850 | 0.9564 | 1.1921 | 0.8931 | 0.7744 |
| DynReq k=5% +A | lin | 1.2590 | 1.0516 | 2.1465 | 0.9732 | 0.9176 |
| DynReq k=5% +A | wmse | 0.9075 | 0.4170 | 1.1477 | 0.9420 | 0.8139 |
| DynReq k=5% +A | model | 0.9147 | 0.3074 | 1.3938 | 0.8811 | 0.7301 |

runtime 1082s
