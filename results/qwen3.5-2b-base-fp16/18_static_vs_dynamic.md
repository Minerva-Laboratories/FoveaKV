# E18 Static vs per-step mixed precision at equal bits

**TL;DR:** with the same nested code and the same number of 4-bit tokens, a static set chosen from 64 probe queries
(ZipCache/MiKV-style) gives attn err 0.0404 at 5%, the per-step query-dependent set 0.0203
(2%: 0.0507 vs 0.0224; 10%: 0.0346 vs 0.0193).

## Config
Probe = first 64 decode positions (excluded from evaluation); evaluation = remaining positions (stride 2). Shrinkage-corrected
nested code, block means, exact 128-token window, int8 reference, 5 corpora x 6 layers.

| 4-bit tokens | selection | attn err |
|---|---|---|
| 2% | static (probe-chosen) 4-bit set | 0.0507 |
| 2% | per-step (FoveaKV) | 0.0224 |
| 5% | static (probe-chosen) 4-bit set | 0.0404 |
| 5% | per-step (FoveaKV) | 0.0203 |
| 10% | static (probe-chosen) 4-bit set | 0.0346 |
| 10% | per-step (FoveaKV) | 0.0193 |

runtime 240s
