# E17 FoveaKV vs SpeCache-style refinement (offline, int8 reference, shared exact window)

**TL;DR:** at 5% hot tokens FoveaKV gives 0.0224 while moving 0.10 bits/value/step of refinement data; SpeCache-style
refinement with the same traffic (0.625% of tokens, 16-bit) gives 0.0475, and with the same token count (5%, 8x the
traffic) 0.0146. KIVI-2b alone: 0.2058.

## Config
SpeCache-style = KIVI-2b (group 128, fp16 scale/zero) resident for all tokens; top-k by max-over-group estimated log-prob from the
KIVI keys; selected tokens use the exact reference K/V (16-bit copy in CPU memory). FoveaKV = shrinkage-corrected nested code.
'refinement bits' = bits per value moved per step for hot tokens (FoveaKV: 2·k; SpeCache: 16·k').

| method | k | resident bits | refinement bits/step | attn err | layer-3 attn err |
|---|---|---|---|---|---|
| FoveaKV | 1% | 2.1600 | 0.0200 | 0.0270 | 0.0249 |
| FoveaKV | 2% | 2.1600 | 0.0400 | 0.0245 | 0.0218 |
| FoveaKV | 5% | 2.1600 | 0.1000 | 0.0224 | 0.0182 |
| FoveaKV | 10% | 2.1600 | 0.2000 | 0.0214 | 0.0161 |
| SpeCache-style, equal traffic (k/8) | 1% | 2.2500 | 0.0200 | 0.0875 | 0.0613 |
| SpeCache-style, equal traffic (k/8) | 2% | 2.2500 | 0.0400 | 0.0687 | 0.0532 |
| SpeCache-style, equal traffic (k/8) | 5% | 2.2500 | 0.1000 | 0.0475 | 0.0424 |
| SpeCache-style, equal traffic (k/8) | 10% | 2.2500 | 0.2000 | 0.0340 | 0.0340 |
| SpeCache-style, equal tokens (k) | 1% | 2.2500 | 0.1600 | 0.0381 | 0.0367 |
| SpeCache-style, equal tokens (k) | 2% | 2.2500 | 0.3200 | 0.0264 | 0.0283 |
| SpeCache-style, equal tokens (k) | 5% | 2.2500 | 0.8000 | 0.0146 | 0.0176 |
| SpeCache-style, equal tokens (k) | 10% | 2.2500 | 1.6000 | 0.0082 | 0.0106 |
| KIVI-2b only (no refinement) | 0% | 2.2500 | 0.0000 | 0.2058 | 0.1050 |

runtime 575s
