# E20 Sharing the hot set across query heads (k = 5%, group size G = 8)

**TL;DR:** one set per KV head (s = 8): FoveaKV 0.0178, SpeCache-style 0.0226; one set per query head
(s = 1): FoveaKV 0.0171, SpeCache-style 0.0147. Attention mass left on the low-bit tail (shared set):
FoveaKV selector 0.118, KIVI selector 0.123. SpeCache-style with FoveaKV's 2-bit tail:
0.0052; FoveaKV with exact hot tokens: 0.0052.

## Config
Same exact window (4 sink + 128 recent tokens) and k for every row; each sub-group of s query heads gets its own top-k set
(max over the sub-group of estimated log-probs). 'distinct hot / k' = tokens fetched per KV head per step, in units of k.

| scheme | heads per set (s) | distinct hot / k | tail mass | attn err |
|---|---|---|---|---|
| FoveaKV | 1 | 2.8466 | 0.0893 | 0.0171 |
| FoveaKV | 2 | 1.9845 | 0.1008 | 0.0173 |
| FoveaKV | 4 | 1.3922 | 0.1107 | 0.0176 |
| FoveaKV | 8 | 1.0000 | 0.1177 | 0.0178 |
| SpeCache | 1 | 2.9044 | 0.0938 | 0.0147 |
| SpeCache | 2 | 2.0121 | 0.1056 | 0.0176 |
| SpeCache | 4 | 1.4033 | 0.1157 | 0.0203 |
| SpeCache | 8 | 1.0000 | 0.1228 | 0.0226 |
| Spe+GS tail | 1 | 2.9044 | 0.0938 | 0.0036 |
| Spe+GS tail | 2 | 2.0121 | 0.1056 | 0.0042 |
| Spe+GS tail | 4 | 1.4033 | 0.1157 | 0.0048 |
| Spe+GS tail | 8 | 1.0000 | 0.1228 | 0.0052 |
| Fov+exact | 1 | 2.8466 | 0.0893 | 0.0038 |
| Fov+exact | 2 | 1.9845 | 0.1008 | 0.0043 |
| Fov+exact | 4 | 1.3922 | 0.1107 | 0.0048 |
| Fov+exact | 8 | 1.0000 | 0.1177 | 0.0052 |

runtime 1325s
