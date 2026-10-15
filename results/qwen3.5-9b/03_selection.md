# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.828); sign-plane(+RHT) 0.957 (overlap 0.676).
Without RHT: coarse 0.970, sign 0.913.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7944 | 0.9731 | 0.9563 | 0.8881 | 2.8420 | 0.0034 |
| coarse | RHT | 2% | 0.8097 | 0.9817 | 0.9624 | 0.9058 | 2.8420 | 0.0034 |
| coarse | RHT | 5% | 0.8285 | 0.9891 | 0.9705 | 0.9231 | 2.8420 | 0.0034 |
| coarse | RHT | 10% | 0.8453 | 0.9930 | 0.9775 | 0.9393 | 2.8420 | 0.0034 |
| coarse | RHT | 20% | 0.8696 | 0.9961 | 0.9854 | 0.9575 | 2.8420 | 0.0034 |
| coarse | no-RHT | 1% | 0.7201 | 0.9183 | 0.7846 | 0.4144 | 5.4415 | 0.0000 |
| coarse | no-RHT | 2% | 0.7454 | 0.9433 | 0.8524 | 0.5149 | 5.4415 | 0.0000 |
| coarse | no-RHT | 5% | 0.7784 | 0.9701 | 0.9267 | 0.7289 | 5.4415 | 0.0000 |
| coarse | no-RHT | 10% | 0.8038 | 0.9845 | 0.9708 | 0.8909 | 5.4415 | 0.0000 |
| coarse | no-RHT | 20% | 0.8361 | 0.9932 | 0.9820 | 0.9414 | 5.4415 | 0.0000 |
| sign | RHT | 1% | 0.6198 | 0.8997 | 0.8484 | 0.6556 | 5.6062 | 0.0000 |
| sign | RHT | 2% | 0.6441 | 0.9296 | 0.8691 | 0.6979 | 5.6062 | 0.0000 |
| sign | RHT | 5% | 0.6762 | 0.9574 | 0.8968 | 0.7526 | 5.6062 | 0.0000 |
| sign | RHT | 10% | 0.7063 | 0.9724 | 0.9198 | 0.8031 | 5.6062 | 0.0000 |
| sign | RHT | 20% | 0.7507 | 0.9842 | 0.9461 | 0.8552 | 5.6062 | 0.0000 |
| sign | no-RHT | 1% | 0.5275 | 0.8048 | 0.6628 | 0.3289 | 7.2949 | 0.0000 |
| sign | no-RHT | 2% | 0.5615 | 0.8566 | 0.7486 | 0.4261 | 7.2949 | 0.0000 |
| sign | no-RHT | 5% | 0.6101 | 0.9130 | 0.8488 | 0.5734 | 7.2949 | 0.0000 |
| sign | no-RHT | 10% | 0.6502 | 0.9459 | 0.8986 | 0.7111 | 7.2949 | 0.0000 |
| sign | no-RHT | 20% | 0.7039 | 0.9714 | 0.9358 | 0.8416 | 7.2949 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8095 | 0.9799 | 0.9231 |
| 3 | sign | 0.6412 | 0.9252 | 0.7526 |
| 11 | coarse | 0.8364 | 0.9937 | 0.9663 |
| 11 | sign | 0.6893 | 0.9725 | 0.8708 |
| 19 | coarse | 0.8412 | 0.9933 | 0.9663 |
| 19 | sign | 0.7003 | 0.9722 | 0.8744 |
| 27 | coarse | 0.8268 | 0.9897 | 0.9643 |
| 27 | sign | 0.6741 | 0.9596 | 0.8712 |

runtime 976s
