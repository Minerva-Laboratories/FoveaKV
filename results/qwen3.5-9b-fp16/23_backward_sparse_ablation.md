# E23 Sparse attention from the error model, run backwards (gate- and o_proj-weighted pair costs)

FLOPs = kept (query head, token) pairs / all causal pairs. heads off = share of (query, head) skipped entirely.
resid = relative error of the layer's residual contribution (after gate and o_proj), what the next layer sees.

| arm | setting | FLOPs kept | heads off | attn err | resid err |
|---|---|---|---|---|---|
| backward, no gate/W_o weights | eps=0.01 | 0.4343 | 0.0000 | 0.0014 | 0.0012 |
| pooled top-p (mass only) | eps=0.01 | 0.7747 | 0.0000 | 0.0000 | 0.0000 |
| backward, per-head budget | eps=0.01 | 0.4148 | 0.0000 | 0.0018 | 0.0012 |
| backward, no gate/W_o weights | eps=0.03 | 0.3072 | 0.0000 | 0.0040 | 0.0035 |
| pooled top-p (mass only) | eps=0.03 | 0.6983 | 0.0000 | 0.0001 | 0.0001 |
| backward, per-head budget | eps=0.03 | 0.2904 | 0.0001 | 0.0052 | 0.0040 |
| backward, no gate/W_o weights | eps=0.1 | 0.1741 | 0.0000 | 0.0122 | 0.0112 |
| pooled top-p (mass only) | eps=0.1 | 0.5958 | 0.0000 | 0.0003 | 0.0003 |
| backward, per-head budget | eps=0.1 | 0.1663 | 0.0001 | 0.0163 | 0.0117 |
| backward, no gate/W_o weights | eps=0.3 | 0.0774 | 0.0000 | 0.0348 | 0.0322 |
| pooled top-p (mass only) | eps=0.3 | 0.4870 | 0.0000 | 0.0007 | 0.0009 |
| backward, per-head budget | eps=0.3 | 0.0795 | 0.0006 | 0.0487 | 0.0350 |
| backward, no gate/W_o weights | eps=1 | 0.0186 | 0.0001 | 0.1333 | 0.1087 |
| pooled top-p (mass only) | eps=1 | 0.3568 | 0.0000 | 0.0023 | 0.0028 |
| backward, per-head budget | eps=1 | 0.0238 | 0.0305 | 0.2087 | 0.1400 |
| backward, no gate/W_o weights | eps=3 | 0.0027 | 0.2612 | 0.4651 | 0.5374 |
| pooled top-p (mass only) | eps=3 | 0.2375 | 0.0000 | 0.0065 | 0.0079 |
| backward, per-head budget | eps=3 | 0.0043 | 0.2976 | 0.5485 | 0.5781 |

runtime 1277s
