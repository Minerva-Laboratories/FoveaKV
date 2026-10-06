# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 2.04× k on average (max 3.20×); with the coarse estimate 2.08× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.958 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.2584 | 3.6341 | 1.0601 | 0.9366 | 0.7463 |
| true | 2% | 2.1746 | 3.4146 | 1.0451 | 0.9537 | 0.7966 |
| true | 5% | 2.0403 | 3.2049 | 1.0294 | 0.9708 | 0.8729 |
| true | 10% | 1.9113 | 2.8523 | 1.0194 | 0.9812 | 0.9072 |
| true | 20% | 1.7352 | 2.3863 | 1.0107 | 0.9898 | 0.9404 |
| coarse | 1% | 2.3125 | 3.6750 | 1.0418 | 0.9092 | 0.6783 |
| coarse | 2% | 2.2226 | 3.3720 | 1.0327 | 0.9338 | 0.7502 |
| coarse | 5% | 2.0811 | 3.1927 | 1.0226 | 0.9585 | 0.8265 |
| coarse | 10% | 1.9451 | 2.8376 | 1.0157 | 0.9732 | 0.8714 |
| coarse | 20% | 1.7608 | 2.4018 | 1.0091 | 0.9854 | 0.9169 |
| sign | 1% | 2.4555 | 3.5949 | 0.9954 | 0.8419 | 0.5277 |
| sign | 2% | 2.3489 | 3.4012 | 0.9997 | 0.8818 | 0.6059 |
| sign | 5% | 2.1850 | 3.1658 | 1.0034 | 0.9239 | 0.7037 |
| sign | 10% | 2.0271 | 2.8718 | 1.0046 | 0.9499 | 0.7743 |
| sign | 20% | 1.8212 | 2.4368 | 1.0040 | 0.9719 | 0.8489 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.2060 | 3.0648 | 2.2497 | 0.9375 |
| 7 | 1.9456 | 2.9022 | 2.0026 | 0.9799 |
| 11 | 1.8401 | 2.7286 | 1.8922 | 0.9738 |
| 15 | 1.9373 | 2.7293 | 1.9854 | 0.9632 |
| 19 | 1.9669 | 2.9853 | 1.9916 | 0.9462 |
| 23 | 2.3456 | 3.2049 | 2.3650 | 0.9502 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 396s
