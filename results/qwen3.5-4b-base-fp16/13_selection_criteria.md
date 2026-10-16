# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0255, GQA sum of squared probabilities 0.0254,
expected-error weight 0.0252, oracle (true p, sum-p2) 0.0236.
At k=2%: 0.0298 / 0.0298 / 0.0295 / oracle 0.0269.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0298 | 0.0310 |
| 2% | sum-p2 | 0.0298 | 0.0309 |
| 2% | sum-p | 0.0298 | 0.0311 |
| 2% | expected-error | 0.0295 | 0.0306 |
| 2% | oracle sum-p2 | 0.0269 | 0.0269 |
| 5% | max-logp | 0.0255 | 0.0241 |
| 5% | sum-p2 | 0.0254 | 0.0240 |
| 5% | sum-p | 0.0254 | 0.0241 |
| 5% | expected-error | 0.0252 | 0.0237 |
| 5% | oracle sum-p2 | 0.0236 | 0.0206 |

runtime 485s
