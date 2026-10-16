# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.990 (overlap 0.826); sign-plane(+RHT) 0.959 (overlap 0.672).
Without RHT: coarse 0.980, sign 0.925.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 2457600 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 6 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7827 | 0.9731 | 0.9618 | 0.9076 | 2.7942 | 0.0039 |
| coarse | RHT | 2% | 0.8000 | 0.9817 | 0.9704 | 0.9274 | 2.7942 | 0.0039 |
| coarse | RHT | 5% | 0.8256 | 0.9897 | 0.9791 | 0.9479 | 2.7942 | 0.0039 |
| coarse | RHT | 10% | 0.8472 | 0.9939 | 0.9854 | 0.9621 | 2.7942 | 0.0039 |
| coarse | RHT | 20% | 0.8737 | 0.9969 | 0.9913 | 0.9754 | 2.7942 | 0.0039 |
| coarse | no-RHT | 1% | 0.7306 | 0.9447 | 0.9069 | 0.7613 | 4.2894 | 0.0004 |
| coarse | no-RHT | 2% | 0.7522 | 0.9627 | 0.9428 | 0.8389 | 4.2894 | 0.0004 |
| coarse | no-RHT | 5% | 0.7835 | 0.9799 | 0.9660 | 0.8879 | 4.2894 | 0.0004 |
| coarse | no-RHT | 10% | 0.8095 | 0.9889 | 0.9770 | 0.9335 | 4.2894 | 0.0004 |
| coarse | no-RHT | 20% | 0.8427 | 0.9948 | 0.9864 | 0.9625 | 4.2894 | 0.0004 |
| sign | RHT | 1% | 0.6034 | 0.9017 | 0.8652 | 0.6910 | 5.6515 | 0.0000 |
| sign | RHT | 2% | 0.6304 | 0.9304 | 0.8923 | 0.7521 | 5.6515 | 0.0000 |
| sign | RHT | 5% | 0.6719 | 0.9591 | 0.9215 | 0.8155 | 5.6515 | 0.0000 |
| sign | RHT | 10% | 0.7094 | 0.9750 | 0.9433 | 0.8608 | 5.6515 | 0.0000 |
| sign | RHT | 20% | 0.7572 | 0.9869 | 0.9651 | 0.9076 | 5.6515 | 0.0000 |
| sign | no-RHT | 1% | 0.5297 | 0.8253 | 0.6629 | 0.3234 | 6.7097 | 0.0000 |
| sign | no-RHT | 2% | 0.5642 | 0.8731 | 0.7576 | 0.4679 | 6.7097 | 0.0000 |
| sign | no-RHT | 5% | 0.6135 | 0.9246 | 0.8729 | 0.6776 | 6.7097 | 0.0000 |
| sign | no-RHT | 10% | 0.6549 | 0.9536 | 0.9150 | 0.7701 | 6.7097 | 0.0000 |
| sign | no-RHT | 20% | 0.7092 | 0.9754 | 0.9469 | 0.8525 | 6.7097 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7963 | 0.9823 | 0.9479 |
| 3 | sign | 0.6215 | 0.9333 | 0.8155 |
| 7 | coarse | 0.8228 | 0.9928 | 0.9679 |
| 7 | sign | 0.6660 | 0.9703 | 0.8736 |
| 11 | coarse | 0.8271 | 0.9921 | 0.9701 |
| 11 | sign | 0.6765 | 0.9670 | 0.8799 |
| 15 | coarse | 0.8325 | 0.9911 | 0.9667 |
| 15 | sign | 0.6848 | 0.9631 | 0.8673 |
| 19 | coarse | 0.8427 | 0.9904 | 0.9702 |
| 19 | sign | 0.7015 | 0.9619 | 0.8810 |
| 23 | coarse | 0.8319 | 0.9896 | 0.9709 |
| 23 | sign | 0.6809 | 0.9590 | 0.8855 |

runtime 721s
