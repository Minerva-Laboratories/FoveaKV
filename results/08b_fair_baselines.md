# E8b Fair baselines: the KIVI residual group

**TL;DR:** KIVI-4b's E8 advantage comes mostly from its int8 residual group (recent ≤127 tokens): without it KIVI-4b = 0.0736 vs GS-4b 0.0590;
with the same residual group: KIVI-4b 0.0439, GS-4b 0.0371, DYN k=5% 0.0421.
Key/value split: KIVI-style per-channel keys + gain-shape values = 0.0491 — keys favour per-channel quantization, values favour gain-shape.

## Config
Same data/reference as E8 (int8 cache, 5 corpora × 6 layers, 256 decode queries/layer). Residual group = tokens after the last
complete 128-token group at each decode step (mean ≈ 64 tokens), kept at int8 reference precision.

| scheme | attn err (all layers) | attn err (layer 3) |
|---|---|---|
| KIVI-4b g128 (with int8 residual group) | 0.0439 | 0.0149 |
| KIVI-4b g128 (no residual) | 0.0736 | 0.0488 |
| KIVI-2b g128 (with int8 residual group) | 0.2823 | 0.1162 |
| KIVI-2b g128 (no residual) | 0.4437 | 0.2820 |
| GS-4b (no residual) | 0.0590 | 0.0473 |
| GS-4b + int8 residual group | 0.0371 | 0.0154 |
| GS-2b + int8 residual group | 0.1619 | 0.0875 |
| KIVI-4b keys + GS-4b values (no residual) | 0.0491 | 0.0396 |
| DYN k=5% + int8 residual group | 0.0421 | 0.0269 |

runtime 82s
