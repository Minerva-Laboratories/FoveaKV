# E17 FoveaKV vs SpeCache-style refinement (offline, int8 reference, shared exact window)

**TL;DR:** at 5% hot tokens FoveaKV gives 0.1251 while moving 0.10 bits/value/step of refinement data; SpeCache-style
refinement with the same traffic (0.625% of tokens, 16-bit) gives 0.0611, and with the same token count (5%, 8x the
traffic) 0.0191. KIVI-2b alone: 0.1533.

## Config
SpeCache-style = KIVI-2b (group 128, fp16 scale/zero) resident for all tokens; top-k by max-over-group estimated log-prob from the
KIVI keys; selected tokens use the exact reference K/V (16-bit copy in CPU memory). FoveaKV = shrinkage-corrected nested code.
'refinement bits' = bits per value moved per step for hot tokens (FoveaKV: 2·k; SpeCache: 16·k').

| method | k | resident bits | refinement bits/step | attn err | layer-0 attn err |
|---|---|---|---|---|---|
| FoveaKV | 1% | 2.1600 | 0.0200 | 0.2018 | 0.7272 |
| FoveaKV | 2% | 2.1600 | 0.0400 | 0.1625 | 0.5853 |
| FoveaKV | 5% | 2.1600 | 0.1000 | 0.1251 | 0.4503 |
| FoveaKV | 10% | 2.1600 | 0.2000 | 0.1088 | 0.3915 |
| SpeCache-style, equal traffic (k/8) | 1% | 2.2500 | 0.0200 | 0.0908 | 0.1834 |
| SpeCache-style, equal traffic (k/8) | 2% | 2.2500 | 0.0400 | 0.0785 | 0.1640 |
| SpeCache-style, equal traffic (k/8) | 5% | 2.2500 | 0.1000 | 0.0611 | 0.1350 |
| SpeCache-style, equal traffic (k/8) | 10% | 2.2500 | 0.2000 | 0.0469 | 0.1080 |
| SpeCache-style, equal tokens (k) | 1% | 2.2500 | 0.1600 | 0.0516 | 0.1173 |
| SpeCache-style, equal tokens (k) | 2% | 2.2500 | 0.3200 | 0.0369 | 0.0867 |
| SpeCache-style, equal tokens (k) | 5% | 2.2500 | 0.8000 | 0.0191 | 0.0450 |
| SpeCache-style, equal tokens (k) | 10% | 2.2500 | 1.6000 | 0.0098 | 0.0215 |
| KIVI-2b only (no refinement) | 0% | 2.2500 | 0.0000 | 0.1533 | 0.2687 |

runtime 320s
