# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0251, GQA sum of squared probabilities 0.0251,
expected-error weight 0.0251, oracle (true p, sum-p2) 0.0230.
At k=2%: 0.0304 / 0.0302 / 0.0302 / oracle 0.0268.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0304 | 0.0237 |
| 2% | sum-p2 | 0.0302 | 0.0236 |
| 2% | sum-p | 0.0303 | 0.0236 |
| 2% | expected-error | 0.0302 | 0.0235 |
| 2% | oracle sum-p2 | 0.0268 | 0.0203 |
| 5% | max-logp | 0.0251 | 0.0184 |
| 5% | sum-p2 | 0.0251 | 0.0183 |
| 5% | sum-p | 0.0251 | 0.0183 |
| 5% | expected-error | 0.0251 | 0.0182 |
| 5% | oracle sum-p2 | 0.0230 | 0.0157 |

runtime 193s
