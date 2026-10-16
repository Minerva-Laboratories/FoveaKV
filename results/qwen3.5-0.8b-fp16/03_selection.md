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
| coarse | RHT | 1% | 0.7780 | 0.9722 | 0.9592 | 0.9004 | 2.8089 | 0.0037 |
| coarse | RHT | 2% | 0.7961 | 0.9810 | 0.9691 | 0.9255 | 2.8089 | 0.0037 |
| coarse | RHT | 5% | 0.8234 | 0.9892 | 0.9780 | 0.9455 | 2.8089 | 0.0037 |
| coarse | RHT | 10% | 0.8454 | 0.9936 | 0.9845 | 0.9601 | 2.8089 | 0.0037 |
| coarse | RHT | 20% | 0.8717 | 0.9967 | 0.9906 | 0.9740 | 2.8089 | 0.0037 |
| coarse | no-RHT | 1% | 0.7303 | 0.9412 | 0.9064 | 0.7707 | 4.5797 | 0.0004 |
| coarse | no-RHT | 2% | 0.7537 | 0.9592 | 0.9358 | 0.8081 | 4.5797 | 0.0004 |
| coarse | no-RHT | 5% | 0.7860 | 0.9782 | 0.9568 | 0.8520 | 4.5797 | 0.0004 |
| coarse | no-RHT | 10% | 0.8124 | 0.9885 | 0.9753 | 0.8894 | 4.5797 | 0.0004 |
| coarse | no-RHT | 20% | 0.8446 | 0.9948 | 0.9871 | 0.9666 | 4.5797 | 0.0004 |
| sign | RHT | 1% | 0.5938 | 0.8986 | 0.8531 | 0.6903 | 5.6753 | 0.0000 |
| sign | RHT | 2% | 0.6223 | 0.9275 | 0.8896 | 0.7509 | 5.6753 | 0.0000 |
| sign | RHT | 5% | 0.6673 | 0.9573 | 0.9187 | 0.8139 | 5.6753 | 0.0000 |
| sign | RHT | 10% | 0.7062 | 0.9737 | 0.9409 | 0.8583 | 5.6753 | 0.0000 |
| sign | RHT | 20% | 0.7539 | 0.9861 | 0.9630 | 0.9058 | 5.6753 | 0.0000 |
| sign | no-RHT | 1% | 0.5220 | 0.8193 | 0.6570 | 0.2770 | 7.3853 | 0.0000 |
| sign | no-RHT | 2% | 0.5574 | 0.8675 | 0.7516 | 0.4276 | 7.3853 | 0.0000 |
| sign | no-RHT | 5% | 0.6083 | 0.9207 | 0.8638 | 0.6550 | 7.3853 | 0.0000 |
| sign | no-RHT | 10% | 0.6528 | 0.9518 | 0.9154 | 0.7839 | 7.3853 | 0.0000 |
| sign | no-RHT | 20% | 0.7090 | 0.9750 | 0.9481 | 0.8593 | 7.3853 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7882 | 0.9800 | 0.9455 |
| 3 | sign | 0.6068 | 0.9250 | 0.8139 |
| 7 | coarse | 0.8199 | 0.9929 | 0.9633 |
| 7 | sign | 0.6615 | 0.9711 | 0.8681 |
| 11 | coarse | 0.8318 | 0.9923 | 0.9683 |
| 11 | sign | 0.6808 | 0.9669 | 0.8792 |
| 15 | coarse | 0.8359 | 0.9914 | 0.9669 |
| 15 | sign | 0.6897 | 0.9643 | 0.8737 |
| 19 | coarse | 0.8391 | 0.9901 | 0.9685 |
| 19 | sign | 0.6953 | 0.9605 | 0.8828 |
| 23 | coarse | 0.8256 | 0.9887 | 0.9634 |
| 23 | sign | 0.6699 | 0.9558 | 0.8663 |

runtime 727s
