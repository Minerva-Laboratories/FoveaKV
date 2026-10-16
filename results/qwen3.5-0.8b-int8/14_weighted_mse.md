# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.84 (per corpus×layer range 0.36–1.10),
Spearman ρ over (query, head) 0.94; the purely statistical model (gains, D, no error knowledge) gives ratio
0.86, ρ 0.88. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.1e-07 (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1348 | 1.0980 | 1.2596 | 0.9690 | 0.9320 |
| GS-2b (all coarse) | wmse | 1.6213 | 0.7769 | 2.0536 | 0.8896 | 0.8289 |
| GS-2b (all coarse) | model | 0.9425 | 0.5280 | 1.1184 | 0.8700 | 0.8151 |
| GS-4b (all fine) | lin | 1.0114 | 1.0021 | 1.0335 | 0.9972 | 0.9950 |
| GS-4b (all fine) | wmse | 1.0296 | 0.9058 | 1.1268 | 0.9692 | 0.9562 |
| GS-4b (all fine) | model | 1.1114 | 0.9341 | 1.2821 | 0.8923 | 0.8229 |
| DynReq k=5% +A | lin | 1.3288 | 1.0773 | 2.3290 | 0.9768 | 0.9452 |
| DynReq k=5% +A | wmse | 0.8409 | 0.3577 | 1.0993 | 0.9402 | 0.8237 |
| DynReq k=5% +A | model | 0.8625 | 0.2648 | 1.5823 | 0.8769 | 0.7846 |

runtime 841s
