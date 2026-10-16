# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0192, GQA sum of squared probabilities 0.0191,
expected-error weight 0.0190, oracle (true p, sum-p2) 0.0175.
At k=2%: 0.0229 / 0.0229 / 0.0227 / oracle 0.0202.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0229 | 0.0247 |
| 2% | sum-p2 | 0.0229 | 0.0247 |
| 2% | sum-p | 0.0229 | 0.0248 |
| 2% | expected-error | 0.0227 | 0.0244 |
| 2% | oracle sum-p2 | 0.0202 | 0.0207 |
| 5% | max-logp | 0.0192 | 0.0190 |
| 5% | sum-p2 | 0.0191 | 0.0189 |
| 5% | sum-p | 0.0191 | 0.0189 |
| 5% | expected-error | 0.0190 | 0.0186 |
| 5% | oracle sum-p2 | 0.0175 | 0.0157 |

runtime 363s
