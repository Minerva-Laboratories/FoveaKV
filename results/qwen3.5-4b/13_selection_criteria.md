# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0266, GQA sum of squared probabilities 0.0265,
expected-error weight 0.0264, oracle (true p, sum-p2) 0.0246.
At k=2%: 0.0315 / 0.0314 / 0.0311 / oracle 0.0281.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0315 | 0.0314 |
| 2% | sum-p2 | 0.0314 | 0.0313 |
| 2% | sum-p | 0.0314 | 0.0314 |
| 2% | expected-error | 0.0311 | 0.0310 |
| 2% | oracle sum-p2 | 0.0281 | 0.0271 |
| 5% | max-logp | 0.0266 | 0.0243 |
| 5% | sum-p2 | 0.0265 | 0.0242 |
| 5% | sum-p | 0.0265 | 0.0243 |
| 5% | expected-error | 0.0264 | 0.0240 |
| 5% | oracle sum-p2 | 0.0246 | 0.0208 |

runtime 259s
