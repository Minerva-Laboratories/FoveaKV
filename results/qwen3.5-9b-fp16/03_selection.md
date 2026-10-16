# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.829); sign-plane(+RHT) 0.957 (overlap 0.676).
Without RHT: coarse 0.970, sign 0.913.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7945 | 0.9731 | 0.9563 | 0.8879 | 2.8548 | 0.0033 |
| coarse | RHT | 2% | 0.8098 | 0.9817 | 0.9625 | 0.9048 | 2.8548 | 0.0033 |
| coarse | RHT | 5% | 0.8286 | 0.9891 | 0.9706 | 0.9238 | 2.8548 | 0.0033 |
| coarse | RHT | 10% | 0.8454 | 0.9930 | 0.9775 | 0.9398 | 2.8548 | 0.0033 |
| coarse | RHT | 20% | 0.8697 | 0.9961 | 0.9853 | 0.9574 | 2.8548 | 0.0033 |
| coarse | no-RHT | 1% | 0.7199 | 0.9184 | 0.7846 | 0.4160 | 5.4500 | 0.0000 |
| coarse | no-RHT | 2% | 0.7451 | 0.9436 | 0.8518 | 0.5166 | 5.4500 | 0.0000 |
| coarse | no-RHT | 5% | 0.7781 | 0.9701 | 0.9247 | 0.7176 | 5.4500 | 0.0000 |
| coarse | no-RHT | 10% | 0.8036 | 0.9845 | 0.9707 | 0.8897 | 5.4500 | 0.0000 |
| coarse | no-RHT | 20% | 0.8360 | 0.9932 | 0.9819 | 0.9410 | 5.4500 | 0.0000 |
| sign | RHT | 1% | 0.6197 | 0.8996 | 0.8491 | 0.6571 | 5.6176 | 0.0000 |
| sign | RHT | 2% | 0.6440 | 0.9295 | 0.8697 | 0.6995 | 5.6176 | 0.0000 |
| sign | RHT | 5% | 0.6762 | 0.9574 | 0.8968 | 0.7559 | 5.6176 | 0.0000 |
| sign | RHT | 10% | 0.7064 | 0.9724 | 0.9199 | 0.8029 | 5.6176 | 0.0000 |
| sign | RHT | 20% | 0.7507 | 0.9842 | 0.9461 | 0.8555 | 5.6176 | 0.0000 |
| sign | no-RHT | 1% | 0.5270 | 0.8051 | 0.6607 | 0.3232 | 7.3004 | 0.0000 |
| sign | no-RHT | 2% | 0.5610 | 0.8570 | 0.7464 | 0.4269 | 7.3004 | 0.0000 |
| sign | no-RHT | 5% | 0.6096 | 0.9131 | 0.8466 | 0.5608 | 7.3004 | 0.0000 |
| sign | no-RHT | 10% | 0.6498 | 0.9460 | 0.8983 | 0.7046 | 7.3004 | 0.0000 |
| sign | no-RHT | 20% | 0.7036 | 0.9715 | 0.9356 | 0.8416 | 7.3004 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8094 | 0.9799 | 0.9238 |
| 3 | sign | 0.6411 | 0.9252 | 0.7559 |
| 11 | coarse | 0.8365 | 0.9937 | 0.9664 |
| 11 | sign | 0.6892 | 0.9725 | 0.8703 |
| 19 | coarse | 0.8413 | 0.9933 | 0.9664 |
| 19 | sign | 0.7002 | 0.9721 | 0.8739 |
| 27 | coarse | 0.8270 | 0.9897 | 0.9646 |
| 27 | sign | 0.6742 | 0.9597 | 0.8720 |

runtime 921s
