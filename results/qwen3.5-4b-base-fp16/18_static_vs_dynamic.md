# E18 Static vs per-step mixed precision at equal bits

**TL;DR:** with the same nested code and the same number of 4-bit tokens, a static set chosen from 64 probe queries
(ZipCache/MiKV-style) gives attn err 0.0408 at 5%, the per-step query-dependent set 0.0219
(2%: 0.0517 vs 0.0240; 10%: 0.0357 vs 0.0209).

## Config
Probe = first 64 decode positions (excluded from evaluation); evaluation = remaining positions (stride 2). Shrinkage-corrected
nested code, block means, exact 128-token window, int8 reference, 5 corpora x 4 layers.

| 4-bit tokens | selection | attn err |
|---|---|---|
| 2% | static (probe-chosen) 4-bit set | 0.0517 |
| 2% | per-step (FoveaKV) | 0.0240 |
| 5% | static (probe-chosen) 4-bit set | 0.0408 |
| 5% | per-step (FoveaKV) | 0.0219 |
| 10% | static (probe-chosen) 4-bit set | 0.0357 |
| 10% | per-step (FoveaKV) | 0.0209 |

runtime 319s
