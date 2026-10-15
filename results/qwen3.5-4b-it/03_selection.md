# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.988 (overlap 0.825); sign-plane(+RHT) 0.953 (overlap 0.669).
Without RHT: coarse 0.970, sign 0.901.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7881 | 0.9713 | 0.9530 | 0.8863 | 2.8054 | 0.0038 |
| coarse | RHT | 2% | 0.8033 | 0.9800 | 0.9606 | 0.9012 | 2.8054 | 0.0038 |
| coarse | RHT | 5% | 0.8249 | 0.9879 | 0.9697 | 0.9218 | 2.8054 | 0.0038 |
| coarse | RHT | 10% | 0.8444 | 0.9923 | 0.9774 | 0.9387 | 2.8054 | 0.0038 |
| coarse | RHT | 20% | 0.8703 | 0.9957 | 0.9855 | 0.9577 | 2.8054 | 0.0038 |
| coarse | no-RHT | 1% | 0.7161 | 0.9061 | 0.8000 | 0.5140 | 5.7695 | 0.0000 |
| coarse | no-RHT | 2% | 0.7458 | 0.9383 | 0.8613 | 0.6211 | 5.7695 | 0.0000 |
| coarse | no-RHT | 5% | 0.7811 | 0.9704 | 0.9416 | 0.7955 | 5.7695 | 0.0000 |
| coarse | no-RHT | 10% | 0.8072 | 0.9847 | 0.9707 | 0.8845 | 5.7695 | 0.0000 |
| coarse | no-RHT | 20% | 0.8392 | 0.9932 | 0.9821 | 0.9527 | 5.7695 | 0.0000 |
| sign | RHT | 1% | 0.6104 | 0.8960 | 0.8323 | 0.6475 | 5.6647 | 0.0000 |
| sign | RHT | 2% | 0.6332 | 0.9249 | 0.8632 | 0.6961 | 5.6647 | 0.0000 |
| sign | RHT | 5% | 0.6689 | 0.9534 | 0.8929 | 0.7520 | 5.6647 | 0.0000 |
| sign | RHT | 10% | 0.7036 | 0.9696 | 0.9178 | 0.8008 | 5.6647 | 0.0000 |
| sign | RHT | 20% | 0.7514 | 0.9827 | 0.9455 | 0.8581 | 5.6647 | 0.0000 |
| sign | no-RHT | 1% | 0.5194 | 0.7851 | 0.6919 | 0.3755 | 7.6411 | 0.0000 |
| sign | no-RHT | 2% | 0.5579 | 0.8393 | 0.7579 | 0.4604 | 7.6411 | 0.0000 |
| sign | no-RHT | 5% | 0.6105 | 0.9015 | 0.8502 | 0.5983 | 7.6411 | 0.0000 |
| sign | no-RHT | 10% | 0.6532 | 0.9387 | 0.8978 | 0.7034 | 7.6411 | 0.0000 |
| sign | no-RHT | 20% | 0.7077 | 0.9674 | 0.9361 | 0.8121 | 7.6411 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8007 | 0.9767 | 0.9218 |
| 3 | sign | 0.6282 | 0.9158 | 0.7520 |
| 11 | coarse | 0.8385 | 0.9938 | 0.9647 |
| 11 | sign | 0.6907 | 0.9729 | 0.8601 |
| 19 | coarse | 0.8316 | 0.9927 | 0.9647 |
| 19 | sign | 0.6784 | 0.9690 | 0.8696 |
| 27 | coarse | 0.8289 | 0.9886 | 0.9561 |
| 27 | sign | 0.6783 | 0.9560 | 0.8470 |

runtime 978s
