# E17 FoveaKV vs SpeCache-style refinement (offline, int8 reference, shared exact window)

**TL;DR:** at 5% hot tokens FoveaKV gives 0.0178 while moving 0.10 bits/value/step of refinement data; SpeCache-style
refinement with the same traffic (0.625% of tokens, 16-bit) gives 0.0664, and with the same token count (5%, 8x the
traffic) 0.0224. KIVI-2b alone: 0.1945.

## Config
SpeCache-style = KIVI-2b (group 128, fp16 scale/zero) resident for all tokens; top-k by max-over-group estimated log-prob from the
KIVI keys; selected tokens use the exact reference K/V (16-bit copy in CPU memory). FoveaKV = shrinkage-corrected nested code.
'refinement bits' = bits per value moved per step for hot tokens (FoveaKV: 2·k; SpeCache: 16·k').

| method | k | resident bits | refinement bits/step | attn err | layer-3 attn err |
|---|---|---|---|---|---|
| FoveaKV | 1% | 2.1600 | 0.0200 | 0.0221 | 0.0273 |
| FoveaKV | 2% | 2.1600 | 0.0400 | 0.0197 | 0.0244 |
| FoveaKV | 5% | 2.1600 | 0.1000 | 0.0178 | 0.0210 |
| FoveaKV | 10% | 2.1600 | 0.2000 | 0.0169 | 0.0189 |
| SpeCache-style, equal traffic (k/8) | 1% | 2.2500 | 0.0200 | 0.1116 | 0.0731 |
| SpeCache-style, equal traffic (k/8) | 2% | 2.2500 | 0.0400 | 0.0919 | 0.0644 |
| SpeCache-style, equal traffic (k/8) | 5% | 2.2500 | 0.1000 | 0.0664 | 0.0531 |
| SpeCache-style, equal traffic (k/8) | 10% | 2.2500 | 0.2000 | 0.0488 | 0.0443 |
| SpeCache-style, equal tokens (k) | 1% | 2.2500 | 0.1600 | 0.0542 | 0.0472 |
| SpeCache-style, equal tokens (k) | 2% | 2.2500 | 0.3200 | 0.0384 | 0.0382 |
| SpeCache-style, equal tokens (k) | 5% | 2.2500 | 0.8000 | 0.0224 | 0.0262 |
| SpeCache-style, equal tokens (k) | 10% | 2.2500 | 1.6000 | 0.0133 | 0.0173 |
| KIVI-2b only (no refinement) | 0% | 2.2500 | 0.0000 | 0.1945 | 0.1310 |

runtime 517s
