# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.825); sign-plane(+RHT) 0.957 (overlap 0.669).
Without RHT: coarse 0.980, sign 0.928.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 2457600 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 6 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7865 | 0.9734 | 0.9621 | 0.9066 | 2.7736 | 0.0040 |
| coarse | RHT | 2% | 0.8021 | 0.9816 | 0.9710 | 0.9265 | 2.7736 | 0.0040 |
| coarse | RHT | 5% | 0.8251 | 0.9893 | 0.9796 | 0.9472 | 2.7736 | 0.0040 |
| coarse | RHT | 10% | 0.8454 | 0.9934 | 0.9859 | 0.9614 | 2.7736 | 0.0040 |
| coarse | RHT | 20% | 0.8710 | 0.9966 | 0.9917 | 0.9753 | 2.7736 | 0.0040 |
| coarse | no-RHT | 1% | 0.7406 | 0.9473 | 0.9192 | 0.7665 | 4.2379 | 0.0003 |
| coarse | no-RHT | 2% | 0.7603 | 0.9642 | 0.9462 | 0.8291 | 4.2379 | 0.0003 |
| coarse | no-RHT | 5% | 0.7882 | 0.9804 | 0.9681 | 0.8797 | 4.2379 | 0.0003 |
| coarse | no-RHT | 10% | 0.8124 | 0.9888 | 0.9776 | 0.9163 | 4.2379 | 0.0003 |
| coarse | no-RHT | 20% | 0.8437 | 0.9946 | 0.9869 | 0.9633 | 4.2379 | 0.0003 |
| sign | RHT | 1% | 0.6065 | 0.9012 | 0.8694 | 0.7053 | 5.4939 | 0.0000 |
| sign | RHT | 2% | 0.6307 | 0.9291 | 0.8953 | 0.7546 | 5.4939 | 0.0000 |
| sign | RHT | 5% | 0.6687 | 0.9571 | 0.9239 | 0.8147 | 5.4939 | 0.0000 |
| sign | RHT | 10% | 0.7049 | 0.9732 | 0.9451 | 0.8611 | 5.4939 | 0.0000 |
| sign | RHT | 20% | 0.7520 | 0.9856 | 0.9664 | 0.9066 | 5.4939 | 0.0000 |
| sign | no-RHT | 1% | 0.5421 | 0.8407 | 0.7753 | 0.5053 | 6.6631 | 0.0000 |
| sign | no-RHT | 2% | 0.5724 | 0.8824 | 0.8380 | 0.5984 | 6.6631 | 0.0000 |
| sign | no-RHT | 5% | 0.6187 | 0.9283 | 0.8870 | 0.7161 | 6.6631 | 0.0000 |
| sign | no-RHT | 10% | 0.6601 | 0.9552 | 0.9181 | 0.7979 | 6.6631 | 0.0000 |
| sign | no-RHT | 20% | 0.7138 | 0.9759 | 0.9490 | 0.8613 | 6.6631 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8021 | 0.9834 | 0.9472 |
| 3 | sign | 0.6293 | 0.9367 | 0.8147 |
| 7 | coarse | 0.8242 | 0.9938 | 0.9723 |
| 7 | sign | 0.6671 | 0.9739 | 0.8925 |
| 11 | coarse | 0.8309 | 0.9922 | 0.9703 |
| 11 | sign | 0.6805 | 0.9676 | 0.8811 |
| 15 | coarse | 0.8269 | 0.9905 | 0.9674 |
| 15 | sign | 0.6719 | 0.9607 | 0.8728 |
| 19 | coarse | 0.8366 | 0.9872 | 0.9636 |
| 19 | sign | 0.6877 | 0.9493 | 0.8682 |
| 23 | coarse | 0.8299 | 0.9885 | 0.9631 |
| 23 | sign | 0.6757 | 0.9546 | 0.8601 |

runtime 693s
