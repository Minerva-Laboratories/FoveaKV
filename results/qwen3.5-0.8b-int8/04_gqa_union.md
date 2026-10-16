# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.95× k on average (max 3.08×); with the coarse estimate 2.00× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.966 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.1604 | 3.5897 | 1.0534 | 0.9492 | 0.7715 |
| true | 2% | 2.0782 | 3.4172 | 1.0389 | 0.9636 | 0.8218 |
| true | 5% | 1.9499 | 3.0765 | 1.0243 | 0.9778 | 0.8780 |
| true | 10% | 1.8303 | 2.7969 | 1.0157 | 0.9860 | 0.9108 |
| true | 20% | 1.6753 | 2.4080 | 1.0086 | 0.9925 | 0.9421 |
| coarse | 1% | 2.2229 | 3.5897 | 1.0362 | 0.9215 | 0.6849 |
| coarse | 2% | 2.1346 | 3.3742 | 1.0276 | 0.9441 | 0.7522 |
| coarse | 5% | 1.9952 | 3.0324 | 1.0184 | 0.9664 | 0.8258 |
| coarse | 10% | 1.8667 | 2.7867 | 1.0126 | 0.9789 | 0.8732 |
| coarse | 20% | 1.7027 | 2.4010 | 1.0073 | 0.9887 | 0.9171 |
| sign | 1% | 2.3783 | 3.6049 | 0.9917 | 0.8544 | 0.5255 |
| sign | 2% | 2.2701 | 3.4534 | 0.9969 | 0.8932 | 0.5980 |
| sign | 5% | 2.1048 | 3.1517 | 1.0014 | 0.9344 | 0.7006 |
| sign | 10% | 1.9563 | 2.8381 | 1.0031 | 0.9582 | 0.7743 |
| sign | 20% | 1.7699 | 2.4029 | 1.0031 | 0.9770 | 0.8476 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1741 | 2.9870 | 2.2297 | 0.9358 |
| 7 | 1.8446 | 2.8533 | 1.9039 | 0.9823 |
| 11 | 1.7880 | 2.7751 | 1.8445 | 0.9777 |
| 15 | 1.7655 | 2.6927 | 1.8133 | 0.9763 |
| 19 | 1.8063 | 2.6585 | 1.8408 | 0.9682 |
| 23 | 2.3209 | 3.0765 | 2.3391 | 0.9579 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 723s
