# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0261, GQA sum of squared probabilities 0.0261,
expected-error weight 0.0260, oracle (true p, sum-p2) 0.0246.
At k=2%: 0.0301 / 0.0300 / 0.0298 / oracle 0.0275.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0301 | 0.0312 |
| 2% | sum-p2 | 0.0300 | 0.0311 |
| 2% | sum-p | 0.0300 | 0.0312 |
| 2% | expected-error | 0.0298 | 0.0307 |
| 2% | oracle sum-p2 | 0.0275 | 0.0270 |
| 5% | max-logp | 0.0261 | 0.0245 |
| 5% | sum-p2 | 0.0261 | 0.0244 |
| 5% | sum-p | 0.0261 | 0.0245 |
| 5% | expected-error | 0.0260 | 0.0242 |
| 5% | oracle sum-p2 | 0.0246 | 0.0210 |

runtime 450s
