# E18 Static vs per-step mixed precision at equal bits

**TL;DR:** with the same nested code and the same number of 4-bit tokens, a static set chosen from 64 probe queries
(ZipCache/MiKV-style) gives attn err 0.5543 at 5%, the per-step query-dependent set 0.1243
(2%: 0.5664 vs 0.1617; 10%: 0.5461 vs 0.1080).

## Config
Probe = first 64 decode positions (excluded from evaluation); evaluation = remaining positions (stride 2). Shrinkage-corrected
nested code, block means, exact 128-token window, int8 reference, 5 corpora x 4 layers.

| 4-bit tokens | selection | attn err |
|---|---|---|
| 2% | static (probe-chosen) 4-bit set | 0.5664 |
| 2% | per-step (FoveaKV) | 0.1617 |
| 5% | static (probe-chosen) 4-bit set | 0.5543 |
| 5% | per-step (FoveaKV) | 0.1243 |
| 10% | static (probe-chosen) 4-bit set | 0.5461 |
| 10% | per-step (FoveaKV) | 0.1080 |

runtime 159s
