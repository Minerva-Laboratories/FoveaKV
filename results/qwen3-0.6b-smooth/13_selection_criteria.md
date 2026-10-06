# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0174, GQA sum of squared probabilities 0.0174,
expected-error weight 0.0173, oracle (true p, sum-p2) 0.0142.
At k=2%: 0.0240 / 0.0239 / 0.0237 / oracle 0.0194.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-0 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0240 | 0.0253 |
| 2% | sum-p2 | 0.0239 | 0.0253 |
| 2% | sum-p | 0.0239 | 0.0253 |
| 2% | expected-error | 0.0237 | 0.0260 |
| 2% | oracle sum-p2 | 0.0194 | 0.0221 |
| 5% | max-logp | 0.0174 | 0.0179 |
| 5% | sum-p2 | 0.0174 | 0.0179 |
| 5% | sum-p | 0.0173 | 0.0179 |
| 5% | expected-error | 0.0173 | 0.0184 |
| 5% | oracle sum-p2 | 0.0142 | 0.0158 |

runtime 234s
