# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0251, GQA sum of squared probabilities 0.0250,
expected-error weight 0.0249, oracle (true p, sum-p2) 0.0231.
At k=2%: 0.0295 / 0.0294 / 0.0293 / oracle 0.0265.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0295 | 0.0329 |
| 2% | sum-p2 | 0.0294 | 0.0328 |
| 2% | sum-p | 0.0295 | 0.0329 |
| 2% | expected-error | 0.0293 | 0.0325 |
| 2% | oracle sum-p2 | 0.0265 | 0.0282 |
| 5% | max-logp | 0.0251 | 0.0255 |
| 5% | sum-p2 | 0.0250 | 0.0253 |
| 5% | sum-p | 0.0250 | 0.0254 |
| 5% | expected-error | 0.0249 | 0.0251 |
| 5% | oracle sum-p2 | 0.0231 | 0.0215 |

runtime 500s
