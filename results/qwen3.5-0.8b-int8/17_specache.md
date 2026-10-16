# E17 FoveaKV vs SpeCache-style refinement (offline, int8 reference, shared exact window)

**TL;DR:** at 5% hot tokens FoveaKV gives 0.0165 while moving 0.10 bits/value/step of refinement data; SpeCache-style
refinement with the same traffic (0.625% of tokens, 16-bit) gives 0.0507, and with the same token count (5%, 8x the
traffic) 0.0164. KIVI-2b alone: 0.1812.

## Config
SpeCache-style = KIVI-2b (group 128, fp16 scale/zero) resident for all tokens; top-k by max-over-group estimated log-prob from the
KIVI keys; selected tokens use the exact reference K/V (16-bit copy in CPU memory). FoveaKV = shrinkage-corrected nested code.
'refinement bits' = bits per value moved per step for hot tokens (FoveaKV: 2·k; SpeCache: 16·k').

| method | k | resident bits | refinement bits/step | attn err | layer-3 attn err |
|---|---|---|---|---|---|
| FoveaKV | 1% | 2.1600 | 0.0200 | 0.0201 | 0.0189 |
| FoveaKV | 2% | 2.1600 | 0.0400 | 0.0181 | 0.0164 |
| FoveaKV | 5% | 2.1600 | 0.1000 | 0.0165 | 0.0136 |
| FoveaKV | 10% | 2.1600 | 0.2000 | 0.0157 | 0.0119 |
| SpeCache-style, equal traffic (k/8) | 1% | 2.2500 | 0.0200 | 0.0860 | 0.0602 |
| SpeCache-style, equal traffic (k/8) | 2% | 2.2500 | 0.0400 | 0.0702 | 0.0525 |
| SpeCache-style, equal traffic (k/8) | 5% | 2.2500 | 0.1000 | 0.0507 | 0.0417 |
| SpeCache-style, equal traffic (k/8) | 10% | 2.2500 | 0.2000 | 0.0374 | 0.0332 |
| SpeCache-style, equal tokens (k) | 1% | 2.2500 | 0.1600 | 0.0415 | 0.0359 |
| SpeCache-style, equal tokens (k) | 2% | 2.2500 | 0.3200 | 0.0294 | 0.0275 |
| SpeCache-style, equal tokens (k) | 5% | 2.2500 | 0.8000 | 0.0164 | 0.0169 |
| SpeCache-style, equal tokens (k) | 10% | 2.2500 | 1.6000 | 0.0092 | 0.0100 |
| KIVI-2b only (no refinement) | 0% | 2.2500 | 0.0000 | 0.1812 | 0.0977 |

runtime 428s
