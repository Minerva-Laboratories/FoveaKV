# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.936 (overlap 0.601); sign-plane(+RHT) 0.763 (overlap 0.388).
Without RHT: coarse 0.832, sign 0.621.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.4970 | 0.8525 | 0.3674 | 0.0001 | 5.6483 | 0.0000 |
| coarse | RHT | 2% | 0.5350 | 0.8925 | 0.5231 | 0.0099 | 5.6483 | 0.0000 |
| coarse | RHT | 5% | 0.6013 | 0.9363 | 0.7033 | 0.3095 | 5.6483 | 0.0000 |
| coarse | RHT | 10% | 0.6624 | 0.9630 | 0.8375 | 0.6096 | 5.6483 | 0.0000 |
| coarse | RHT | 20% | 0.7366 | 0.9823 | 0.9261 | 0.7987 | 5.6483 | 0.0000 |
| coarse | no-RHT | 1% | 0.4424 | 0.6599 | 0.2585 | 0.0001 | 8.8529 | 0.0000 |
| coarse | no-RHT | 2% | 0.4796 | 0.7343 | 0.3507 | 0.0073 | 8.8529 | 0.0000 |
| coarse | no-RHT | 5% | 0.5457 | 0.8322 | 0.5191 | 0.0104 | 8.8529 | 0.0000 |
| coarse | no-RHT | 10% | 0.6129 | 0.8979 | 0.6639 | 0.0140 | 8.8529 | 0.0000 |
| coarse | no-RHT | 20% | 0.6963 | 0.9514 | 0.8332 | 0.0242 | 8.8529 | 0.0000 |
| sign | RHT | 1% | 0.2882 | 0.6088 | 0.1740 | 0.0000 | 9.8012 | 0.0000 |
| sign | RHT | 2% | 0.3225 | 0.6648 | 0.2125 | 0.0000 | 9.8012 | 0.0000 |
| sign | RHT | 5% | 0.3877 | 0.7627 | 0.2748 | 0.0000 | 9.8012 | 0.0000 |
| sign | RHT | 10% | 0.4585 | 0.8195 | 0.3603 | 0.0001 | 9.8012 | 0.0000 |
| sign | RHT | 20% | 0.5630 | 0.8693 | 0.5091 | 0.0002 | 9.8012 | 0.0000 |
| sign | no-RHT | 1% | 0.2699 | 0.4596 | 0.0497 | 0.0000 | 11.5402 | 0.0000 |
| sign | no-RHT | 2% | 0.3094 | 0.5255 | 0.0660 | 0.0001 | 11.5402 | 0.0000 |
| sign | no-RHT | 5% | 0.3848 | 0.6214 | 0.0948 | 0.0056 | 11.5402 | 0.0000 |
| sign | no-RHT | 10% | 0.4663 | 0.7000 | 0.1438 | 0.0072 | 11.5402 | 0.0000 |
| sign | no-RHT | 20% | 0.5749 | 0.7802 | 0.2337 | 0.0100 | 11.5402 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 0 | coarse | 0.3922 | 0.7993 | 0.3095 |
| 0 | sign | 0.2026 | 0.4596 | 0.0000 |
| 9 | coarse | 0.6146 | 0.9699 | 0.8459 |
| 9 | sign | 0.3813 | 0.8518 | 0.0775 |
| 18 | coarse | 0.7548 | 0.9855 | 0.9271 |
| 18 | sign | 0.5476 | 0.9444 | 0.7562 |
| 27 | coarse | 0.6436 | 0.9904 | 0.7811 |
| 27 | sign | 0.4192 | 0.7950 | 0.0043 |

runtime 399s
