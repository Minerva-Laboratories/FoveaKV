# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.823); sign-plane(+RHT) 0.957 (overlap 0.667).
Without RHT: coarse 0.978, sign 0.921.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 2457600 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 6 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7779 | 0.9722 | 0.9593 | 0.9007 | 2.8022 | 0.0037 |
| coarse | RHT | 2% | 0.7961 | 0.9810 | 0.9689 | 0.9238 | 2.8022 | 0.0037 |
| coarse | RHT | 5% | 0.8233 | 0.9892 | 0.9779 | 0.9453 | 2.8022 | 0.0037 |
| coarse | RHT | 10% | 0.8453 | 0.9936 | 0.9845 | 0.9605 | 2.8022 | 0.0037 |
| coarse | RHT | 20% | 0.8717 | 0.9967 | 0.9906 | 0.9737 | 2.8022 | 0.0037 |
| coarse | no-RHT | 1% | 0.7306 | 0.9416 | 0.9037 | 0.7683 | 4.6658 | 0.0003 |
| coarse | no-RHT | 2% | 0.7538 | 0.9593 | 0.9364 | 0.8073 | 4.6658 | 0.0003 |
| coarse | no-RHT | 5% | 0.7862 | 0.9782 | 0.9575 | 0.8527 | 4.6658 | 0.0003 |
| coarse | no-RHT | 10% | 0.8125 | 0.9884 | 0.9757 | 0.8920 | 4.6658 | 0.0003 |
| coarse | no-RHT | 20% | 0.8447 | 0.9948 | 0.9872 | 0.9668 | 4.6658 | 0.0003 |
| sign | RHT | 1% | 0.5938 | 0.8985 | 0.8544 | 0.6889 | 5.6632 | 0.0000 |
| sign | RHT | 2% | 0.6222 | 0.9275 | 0.8889 | 0.7513 | 5.6632 | 0.0000 |
| sign | RHT | 5% | 0.6672 | 0.9572 | 0.9186 | 0.8134 | 5.6632 | 0.0000 |
| sign | RHT | 10% | 0.7061 | 0.9737 | 0.9407 | 0.8594 | 5.6632 | 0.0000 |
| sign | RHT | 20% | 0.7538 | 0.9861 | 0.9630 | 0.9054 | 5.6632 | 0.0000 |
| sign | no-RHT | 1% | 0.5219 | 0.8191 | 0.6547 | 0.2767 | 7.3057 | 0.0000 |
| sign | no-RHT | 2% | 0.5574 | 0.8676 | 0.7448 | 0.4153 | 7.3057 | 0.0000 |
| sign | no-RHT | 5% | 0.6084 | 0.9208 | 0.8561 | 0.6353 | 7.3057 | 0.0000 |
| sign | no-RHT | 10% | 0.6530 | 0.9517 | 0.9153 | 0.7823 | 7.3057 | 0.0000 |
| sign | no-RHT | 20% | 0.7092 | 0.9748 | 0.9481 | 0.8627 | 7.3057 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7881 | 0.9800 | 0.9453 |
| 3 | sign | 0.6064 | 0.9247 | 0.8134 |
| 7 | coarse | 0.8199 | 0.9929 | 0.9629 |
| 7 | sign | 0.6617 | 0.9710 | 0.8673 |
| 11 | coarse | 0.8316 | 0.9923 | 0.9686 |
| 11 | sign | 0.6808 | 0.9669 | 0.8781 |
| 15 | coarse | 0.8357 | 0.9914 | 0.9669 |
| 15 | sign | 0.6895 | 0.9644 | 0.8732 |
| 19 | coarse | 0.8391 | 0.9901 | 0.9684 |
| 19 | sign | 0.6952 | 0.9605 | 0.8822 |
| 23 | coarse | 0.8255 | 0.9887 | 0.9632 |
| 23 | sign | 0.6696 | 0.9558 | 0.8635 |

runtime 723s
