# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0221, GQA sum of squared probabilities 0.0221,
expected-error weight 0.0220, oracle (true p, sum-p2) 0.0203.
At k=2%: 0.0262 / 0.0261 / 0.0261 / oracle 0.0233.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0262 | 0.0248 |
| 2% | sum-p2 | 0.0261 | 0.0247 |
| 2% | sum-p | 0.0262 | 0.0248 |
| 2% | expected-error | 0.0261 | 0.0246 |
| 2% | oracle sum-p2 | 0.0233 | 0.0210 |
| 5% | max-logp | 0.0221 | 0.0193 |
| 5% | sum-p2 | 0.0221 | 0.0192 |
| 5% | sum-p | 0.0221 | 0.0192 |
| 5% | expected-error | 0.0220 | 0.0190 |
| 5% | oracle sum-p2 | 0.0203 | 0.0162 |

runtime 378s
