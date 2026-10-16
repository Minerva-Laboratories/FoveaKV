# E20 Sharing the hot set across query heads (k = 5%, group size G = 4)

**TL;DR:** one set per KV head (s = 4): FoveaKV 0.0234, SpeCache-style 0.0139; one set per query head
(s = 1): FoveaKV 0.0229, SpeCache-style 0.0097. Attention mass left on the low-bit tail (shared set):
FoveaKV selector 0.101, KIVI selector 0.105. SpeCache-style with FoveaKV's 2-bit tail:
0.0053; FoveaKV with exact hot tokens: 0.0055.

## Config
Same exact window (4 sink + 128 recent tokens) and k for every row; each sub-group of s query heads gets its own top-k set
(max over the sub-group of estimated log-probs). 'distinct hot / k' = tokens fetched per KV head per step, in units of k.

| scheme | heads per set (s) | distinct hot / k | tail mass | attn err |
|---|---|---|---|---|
| FoveaKV | 1 | 2.0144 | 0.0819 | 0.0229 |
| FoveaKV | 2 | 1.4129 | 0.0924 | 0.0232 |
| FoveaKV | 4 | 1.0000 | 0.1008 | 0.0234 |
| SpeCache | 1 | 2.0493 | 0.0854 | 0.0097 |
| SpeCache | 2 | 1.4265 | 0.0962 | 0.0119 |
| SpeCache | 4 | 1.0000 | 0.1047 | 0.0139 |
| Spe+GS tail | 1 | 2.0493 | 0.0854 | 0.0040 |
| Spe+GS tail | 2 | 1.4265 | 0.0962 | 0.0047 |
| Spe+GS tail | 4 | 1.0000 | 0.1047 | 0.0053 |
| Fov+exact | 1 | 2.0144 | 0.0819 | 0.0043 |
| Fov+exact | 2 | 1.4129 | 0.0924 | 0.0049 |
| Fov+exact | 4 | 1.0000 | 0.1008 | 0.0055 |

runtime 986s
