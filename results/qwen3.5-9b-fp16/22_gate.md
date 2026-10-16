# E22 The output gate in FoveaKV's selection and budget (k = 5% mean per KV head)

## Gate statistics (rms of sigmoid(g) per query and head)
| layer | mean | share < 0.1 | share < 0.25 | std of log rms across heads | corr(gate, |o|) |
|---|---|---|---|---|---|
| L3 | 0.0762 | 0.8192 | 0.9985 | 0.2880 | 0.1564 |
| L11 | 0.3088 | 0.0016 | 0.3083 | 0.2520 | -0.1724 |
| L19 | 0.3107 | 0.0318 | 0.3833 | 0.4008 | -0.2139 |
| L27 | 0.4373 | 0.0146 | 0.1645 | 0.4033 | 0.2100 |

## Errors (resid = after gate and o_proj, what the next layer sees)
| hot-set vote | budget per KV head | attn err | resid err |
|---|---|---|---|
| max log-prob (E17) | uniform k | 0.0234 | 0.0311 |
| max log-prob (E17) | k ~ gate | 0.0235 | 0.0311 |
| max log-prob (E17) | k ~ gate x tail | 0.0236 | 0.0313 |
| max log-prob (E17) | k ~ tail (no gate) | 0.0236 | 0.0314 |
| gate-weighted vote | uniform k | 0.0234 | 0.0311 |
| gate-weighted vote | k ~ gate x tail | 0.0237 | 0.0313 |

runtime 781s
