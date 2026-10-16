# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio 0.92 (per corpus×layer range 0.44–1.06),
Spearman ρ over (query, head) 0.95; the purely statistical model (gains, D, no error knowledge) gives ratio
0.85, ρ 0.88. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to 6.6e-07 (fp64).

## Config
5 corpora × 4 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

| scheme | prediction | median ratio | min (corpus,layer) | max | Spearman ρ mean | ρ min |
|---|---|---|---|---|---|---|
| GS-2b (all coarse) | lin | 1.1653 | 1.0928 | 1.2414 | 0.9664 | 0.9183 |
| GS-2b (all coarse) | wmse | 1.5824 | 0.9932 | 2.0078 | 0.8879 | 0.8319 |
| GS-2b (all coarse) | model | 0.8922 | 0.6064 | 1.2013 | 0.8680 | 0.7915 |
| GS-4b (all fine) | lin | 1.0140 | 0.9821 | 1.0280 | 0.9969 | 0.9912 |
| GS-4b (all fine) | wmse | 1.0458 | 0.9315 | 1.0958 | 0.9643 | 0.9329 |
| GS-4b (all fine) | model | 1.1067 | 0.9229 | 1.3596 | 0.8882 | 0.7946 |
| DynReq k=5% +A | lin | 1.2445 | 0.9945 | 2.0484 | 0.9757 | 0.9404 |
| DynReq k=5% +A | wmse | 0.9239 | 0.4443 | 1.0650 | 0.9480 | 0.8863 |
| DynReq k=5% +A | model | 0.8512 | 0.3253 | 1.6443 | 0.8844 | 0.7635 |

runtime 1298s
