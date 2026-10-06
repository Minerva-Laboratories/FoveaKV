# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.1154, GQA sum of squared probabilities 0.1151,
expected-error weight 0.1087, oracle (true p, sum-p2) 0.4697.
At k=2%: 0.1490 / 0.1489 / 0.1457 / oracle 0.4765.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-0 attn err |
|---|---|---|---|
| 2% | max-logp | 0.1490 | 0.5206 |
| 2% | sum-p2 | 0.1489 | 0.5203 |
| 2% | sum-p | 0.1488 | 0.5201 |
| 2% | expected-error | 0.1457 | 0.5086 |
| 2% | oracle sum-p2 | 0.4765 | 1.8264 |
| 5% | max-logp | 0.1154 | 0.4027 |
| 5% | sum-p2 | 0.1151 | 0.4016 |
| 5% | sum-p | 0.1149 | 0.4007 |
| 5% | expected-error | 0.1087 | 0.3767 |
| 5% | oracle sum-p2 | 0.4697 | 1.8154 |

runtime 242s
