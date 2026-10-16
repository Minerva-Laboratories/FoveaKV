# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.93× k on average (max 3.07×); with the coarse estimate 1.97× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.967 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.1398 | 3.6667 | 1.0537 | 0.9487 | 0.7577 |
| true | 2% | 2.0579 | 3.3899 | 1.0389 | 0.9636 | 0.8155 |
| true | 5% | 1.9327 | 3.0699 | 1.0243 | 0.9779 | 0.8724 |
| true | 10% | 1.8139 | 2.8161 | 1.0156 | 0.9861 | 0.9065 |
| true | 20% | 1.6560 | 2.3999 | 1.0085 | 0.9927 | 0.9393 |
| coarse | 1% | 2.1972 | 3.6500 | 1.0368 | 0.9218 | 0.6874 |
| coarse | 2% | 2.1091 | 3.4465 | 1.0279 | 0.9449 | 0.7498 |
| coarse | 5% | 1.9744 | 3.1321 | 1.0185 | 0.9670 | 0.8238 |
| coarse | 10% | 1.8476 | 2.8251 | 1.0125 | 0.9794 | 0.8711 |
| coarse | 20% | 1.6827 | 2.4206 | 1.0072 | 0.9891 | 0.9156 |
| sign | 1% | 2.3435 | 3.6750 | 0.9941 | 0.8564 | 0.5162 |
| sign | 2% | 2.2385 | 3.4214 | 0.9985 | 0.8963 | 0.6044 |
| sign | 5% | 2.0794 | 3.1917 | 1.0022 | 0.9365 | 0.6999 |
| sign | 10% | 1.9321 | 2.8886 | 1.0033 | 0.9597 | 0.7701 |
| sign | 20% | 1.7472 | 2.4433 | 1.0030 | 0.9781 | 0.8450 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1978 | 3.0699 | 2.2431 | 0.9377 |
| 7 | 1.8358 | 2.7531 | 1.8922 | 0.9812 |
| 11 | 1.7906 | 2.6778 | 1.8396 | 0.9775 |
| 15 | 1.7923 | 2.7225 | 1.8499 | 0.9745 |
| 19 | 1.7663 | 2.8900 | 1.7883 | 0.9699 |
| 23 | 2.2133 | 2.9854 | 2.2334 | 0.9612 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 721s
