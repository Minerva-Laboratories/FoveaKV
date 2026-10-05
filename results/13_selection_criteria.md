# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0205, GQA sum of squared probabilities 0.0204,
expected-error weight 0.0203, oracle (true p, sum-p2) 0.0186.
At k=2%: 0.0247 / 0.0246 / 0.0245 / oracle 0.0216.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0247 | 0.0246 |
| 2% | sum-p2 | 0.0246 | 0.0245 |
| 2% | sum-p | 0.0246 | 0.0245 |
| 2% | expected-error | 0.0245 | 0.0243 |
| 2% | oracle sum-p2 | 0.0216 | 0.0205 |
| 5% | max-logp | 0.0205 | 0.0188 |
| 5% | sum-p2 | 0.0204 | 0.0187 |
| 5% | sum-p | 0.0204 | 0.0187 |
| 5% | expected-error | 0.0203 | 0.0186 |
| 5% | oracle sum-p2 | 0.0186 | 0.0156 |

runtime 131s
