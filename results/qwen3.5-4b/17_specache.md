# E17 FoveaKV vs SpeCache-style refinement (offline, int8 reference, shared exact window)

**TL;DR:** at 5% hot tokens FoveaKV gives 0.0233 while moving 0.10 bits/value/step of refinement data; SpeCache-style
refinement with the same traffic (0.625% of tokens, 16-bit) gives 0.0501, and with the same token count (5%, 8x the
traffic) 0.0156. KIVI-2b alone: 0.2102.

## Config
SpeCache-style = KIVI-2b (group 128, fp16 scale/zero) resident for all tokens; top-k by max-over-group estimated log-prob from the
KIVI keys; selected tokens use the exact reference K/V (16-bit copy in CPU memory). FoveaKV = shrinkage-corrected nested code.
'refinement bits' = bits per value moved per step for hot tokens (FoveaKV: 2·k; SpeCache: 16·k').

| method | k | resident bits | refinement bits/step | attn err | layer-3 attn err |
|---|---|---|---|---|---|
| FoveaKV | 1% | 2.1600 | 0.0200 | 0.0283 | 0.0252 |
| FoveaKV | 2% | 2.1600 | 0.0400 | 0.0256 | 0.0220 |
| FoveaKV | 5% | 2.1600 | 0.1000 | 0.0233 | 0.0184 |
| FoveaKV | 10% | 2.1600 | 0.2000 | 0.0222 | 0.0162 |
| SpeCache-style, equal traffic (k/8) | 1% | 2.2500 | 0.0200 | 0.0909 | 0.0617 |
| SpeCache-style, equal traffic (k/8) | 2% | 2.2500 | 0.0400 | 0.0719 | 0.0538 |
| SpeCache-style, equal traffic (k/8) | 5% | 2.2500 | 0.1000 | 0.0501 | 0.0430 |
| SpeCache-style, equal traffic (k/8) | 10% | 2.2500 | 0.2000 | 0.0361 | 0.0346 |
| SpeCache-style, equal tokens (k) | 1% | 2.2500 | 0.1600 | 0.0403 | 0.0373 |
| SpeCache-style, equal tokens (k) | 2% | 2.2500 | 0.3200 | 0.0280 | 0.0288 |
| SpeCache-style, equal tokens (k) | 5% | 2.2500 | 0.8000 | 0.0156 | 0.0181 |
| SpeCache-style, equal tokens (k) | 10% | 2.2500 | 1.6000 | 0.0088 | 0.0110 |
| KIVI-2b only (no refinement) | 0% | 2.2500 | 0.0000 | 0.2102 | 0.1042 |

runtime 301s
