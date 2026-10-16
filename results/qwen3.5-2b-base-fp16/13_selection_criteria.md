# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp 0.0239, GQA sum of squared probabilities 0.0239,
expected-error weight 0.0239, oracle (true p, sum-p2) 0.0219.
At k=2%: 0.0287 / 0.0285 / 0.0285 / oracle 0.0254.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

| k | criterion | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | max-logp | 0.0287 | 0.0230 |
| 2% | sum-p2 | 0.0285 | 0.0229 |
| 2% | sum-p | 0.0286 | 0.0230 |
| 2% | expected-error | 0.0285 | 0.0228 |
| 2% | oracle sum-p2 | 0.0254 | 0.0197 |
| 5% | max-logp | 0.0239 | 0.0179 |
| 5% | sum-p2 | 0.0239 | 0.0178 |
| 5% | sum-p | 0.0239 | 0.0178 |
| 5% | expected-error | 0.0239 | 0.0177 |
| 5% | oracle sum-p2 | 0.0219 | 0.0153 |

runtime 336s
