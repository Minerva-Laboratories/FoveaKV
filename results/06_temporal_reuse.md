# E6 Temporal reuse of the hot set (dynamic part only; sinks+window resident)

**TL;DR:** at k=5%, consecutive decode steps share 0.563 of the dynamic hot set (worst head 0.320);
an on-GPU LRU of residuals of size 1×/2×/4× k gives hit rates 0.559 / 0.726 / 0.853.

## Config
Budget selection (coarse+RHT, S=4, W=64, k counts all hot tokens); 512 consecutive decode steps; per KV head; 5 corpora × 6 layers.
LRU capacity in units of the mean dynamic hot-set size.

| k | step-to-step overlap | LRU 1× | LRU 2× | LRU 4× | worst head overlap |
|---|---|---|---|---|---|
| 2% | 0.4691 | 0.4642 | 0.6091 | 0.7349 | 0.2154 |
| 5% | 0.5627 | 0.5587 | 0.7257 | 0.8528 | 0.3201 |
| 10% | 0.6225 | 0.6191 | 0.8028 | 0.9254 | 0.4097 |

## Per layer (k=5%)
| layer | overlap | LRU 1× | LRU 2× | LRU 4× |
|---|---|---|---|---|
| 3 | 0.4192 | 0.4158 | 0.5859 | 0.7541 |
| 7 | 0.5684 | 0.5645 | 0.7396 | 0.8648 |
| 11 | 0.6356 | 0.6313 | 0.7925 | 0.8913 |
| 15 | 0.6336 | 0.6291 | 0.7778 | 0.8718 |
| 19 | 0.6059 | 0.6017 | 0.7807 | 0.9033 |
| 23 | 0.5134 | 0.5096 | 0.6775 | 0.8315 |

## Caveats
- Queries come from teacher-forced text (the captured corpus), the same as real decoding of that text.
runtime 132s
