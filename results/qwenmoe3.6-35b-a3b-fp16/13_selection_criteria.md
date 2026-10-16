# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0214, GQA sum of squared probabilities 0.0214,
expected-error weight 0.0214, oracle (true p, sum-p2) 0.0199.
At k=2%: 0.0254 / 0.0252 / 0.0252 / oracle 0.0228.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0254 | 0.0328 |
| 2% | sum-p2 | 0.0252 | 0.0326 |
| 2% | sum-p | 0.0253 | 0.0327 |
| 2% | expected-error | 0.0252 | 0.0327 |
| 2% | oracle sum-p2 | 0.0228 | 0.0297 |
| 5% | max-logp | 0.0214 | 0.0270 |
| 5% | sum-p2 | 0.0214 | 0.0269 |
| 5% | sum-p | 0.0214 | 0.0270 |
| 5% | expected-error | 0.0214 | 0.0270 |
| 5% | oracle sum-p2 | 0.0199 | 0.0243 |

runtime 757s
