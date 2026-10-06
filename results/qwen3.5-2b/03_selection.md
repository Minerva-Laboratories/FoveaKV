# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.988 (overlap 0.823); sign-plane(+RHT) 0.955 (overlap 0.665).
Without RHT: coarse 0.979, sign 0.924.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 2457600 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 6 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7834 | 0.9726 | 0.9622 | 0.9069 | 2.7387 | 0.0044 |
| coarse | RHT | 2% | 0.7990 | 0.9807 | 0.9711 | 0.9275 | 2.7387 | 0.0044 |
| coarse | RHT | 5% | 0.8227 | 0.9885 | 0.9797 | 0.9478 | 2.7387 | 0.0044 |
| coarse | RHT | 10% | 0.8438 | 0.9929 | 0.9859 | 0.9613 | 2.7387 | 0.0044 |
| coarse | RHT | 20% | 0.8705 | 0.9963 | 0.9917 | 0.9754 | 2.7387 | 0.0044 |
| coarse | no-RHT | 1% | 0.7360 | 0.9450 | 0.9179 | 0.7785 | 4.4252 | 0.0002 |
| coarse | no-RHT | 2% | 0.7553 | 0.9619 | 0.9360 | 0.8215 | 4.4252 | 0.0002 |
| coarse | no-RHT | 5% | 0.7841 | 0.9787 | 0.9575 | 0.8660 | 4.4252 | 0.0002 |
| coarse | no-RHT | 10% | 0.8097 | 0.9877 | 0.9719 | 0.8954 | 4.4252 | 0.0002 |
| coarse | no-RHT | 20% | 0.8430 | 0.9940 | 0.9852 | 0.9325 | 4.4252 | 0.0002 |
| sign | RHT | 1% | 0.6019 | 0.8994 | 0.8692 | 0.7013 | 5.4173 | 0.0000 |
| sign | RHT | 2% | 0.6262 | 0.9265 | 0.8950 | 0.7573 | 5.4173 | 0.0000 |
| sign | RHT | 5% | 0.6649 | 0.9546 | 0.9239 | 0.8163 | 5.4173 | 0.0000 |
| sign | RHT | 10% | 0.7021 | 0.9712 | 0.9452 | 0.8626 | 5.4173 | 0.0000 |
| sign | RHT | 20% | 0.7510 | 0.9845 | 0.9664 | 0.9074 | 5.4173 | 0.0000 |
| sign | no-RHT | 1% | 0.5366 | 0.8376 | 0.7741 | 0.5055 | 6.5761 | 0.0000 |
| sign | no-RHT | 2% | 0.5659 | 0.8781 | 0.8268 | 0.6038 | 6.5761 | 0.0000 |
| sign | no-RHT | 5% | 0.6129 | 0.9239 | 0.8875 | 0.7192 | 6.5761 | 0.0000 |
| sign | no-RHT | 10% | 0.6561 | 0.9517 | 0.9182 | 0.7967 | 6.5761 | 0.0000 |
| sign | no-RHT | 20% | 0.7127 | 0.9739 | 0.9491 | 0.8610 | 6.5761 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7992 | 0.9827 | 0.9478 |
| 3 | sign | 0.6247 | 0.9343 | 0.8163 |
| 7 | coarse | 0.8194 | 0.9929 | 0.9584 |
| 7 | sign | 0.6601 | 0.9713 | 0.8490 |
| 11 | coarse | 0.8284 | 0.9914 | 0.9645 |
| 11 | sign | 0.6762 | 0.9652 | 0.8647 |
| 15 | coarse | 0.8239 | 0.9893 | 0.9621 |
| 15 | sign | 0.6663 | 0.9562 | 0.8551 |
| 19 | coarse | 0.8374 | 0.9865 | 0.9631 |
| 19 | sign | 0.6895 | 0.9470 | 0.8673 |
| 23 | coarse | 0.8281 | 0.9881 | 0.9626 |
| 23 | sign | 0.6728 | 0.9535 | 0.8602 |

runtime 396s
