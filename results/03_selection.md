# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.821); sign-plane(+RHT) 0.955 (overlap 0.662).
Without RHT: coarse 0.976, sign 0.915.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 2457600 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 6 layers × 8 heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7764 | 0.9724 | 0.9614 | 0.9035 | 2.8057 | 0.0037 |
| coarse | RHT | 2% | 0.7939 | 0.9806 | 0.9690 | 0.9243 | 2.8057 | 0.0037 |
| coarse | RHT | 5% | 0.8207 | 0.9886 | 0.9779 | 0.9450 | 2.8057 | 0.0037 |
| coarse | RHT | 10% | 0.8426 | 0.9930 | 0.9844 | 0.9599 | 2.8057 | 0.0037 |
| coarse | RHT | 20% | 0.8700 | 0.9964 | 0.9906 | 0.9740 | 2.8057 | 0.0037 |
| coarse | no-RHT | 1% | 0.7258 | 0.9390 | 0.9031 | 0.7688 | 4.9080 | 0.0002 |
| coarse | no-RHT | 2% | 0.7488 | 0.9567 | 0.9280 | 0.8051 | 4.9080 | 0.0002 |
| coarse | no-RHT | 5% | 0.7815 | 0.9760 | 0.9519 | 0.8516 | 4.9080 | 0.0002 |
| coarse | no-RHT | 10% | 0.8085 | 0.9869 | 0.9699 | 0.8920 | 4.9080 | 0.0002 |
| coarse | no-RHT | 20% | 0.8426 | 0.9941 | 0.9869 | 0.9438 | 4.9080 | 0.0002 |
| sign | RHT | 1% | 0.5907 | 0.8992 | 0.8607 | 0.6897 | 5.6267 | 0.0000 |
| sign | RHT | 2% | 0.6182 | 0.9265 | 0.8894 | 0.7509 | 5.6267 | 0.0000 |
| sign | RHT | 5% | 0.6623 | 0.9551 | 0.9187 | 0.8133 | 5.6267 | 0.0000 |
| sign | RHT | 10% | 0.7012 | 0.9717 | 0.9408 | 0.8589 | 5.6267 | 0.0000 |
| sign | RHT | 20% | 0.7510 | 0.9849 | 0.9630 | 0.9051 | 5.6267 | 0.0000 |
| sign | no-RHT | 1% | 0.5136 | 0.8147 | 0.6525 | 0.2708 | 7.3177 | 0.0000 |
| sign | no-RHT | 2% | 0.5482 | 0.8622 | 0.7449 | 0.4174 | 7.3177 | 0.0000 |
| sign | no-RHT | 5% | 0.5994 | 0.9153 | 0.8561 | 0.6343 | 7.3177 | 0.0000 |
| sign | no-RHT | 10% | 0.6453 | 0.9471 | 0.9113 | 0.7792 | 7.3177 | 0.0000 |
| sign | no-RHT | 20% | 0.7050 | 0.9719 | 0.9479 | 0.8449 | 7.3177 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7855 | 0.9799 | 0.9450 |
| 3 | sign | 0.6009 | 0.9245 | 0.8133 |
| 7 | coarse | 0.8145 | 0.9923 | 0.9551 |
| 7 | sign | 0.6516 | 0.9686 | 0.8371 |
| 11 | coarse | 0.8292 | 0.9916 | 0.9580 |
| 11 | sign | 0.6768 | 0.9645 | 0.8430 |
| 15 | coarse | 0.8326 | 0.9900 | 0.9578 |
| 15 | sign | 0.6851 | 0.9601 | 0.8495 |
| 19 | coarse | 0.8401 | 0.9894 | 0.9611 |
| 19 | sign | 0.6967 | 0.9583 | 0.8628 |
| 23 | coarse | 0.8223 | 0.9884 | 0.9634 |
| 23 | sign | 0.6630 | 0.9547 | 0.8650 |

runtime 296s
