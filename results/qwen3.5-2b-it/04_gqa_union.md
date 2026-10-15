# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.93× k on average (max 3.07×); with the coarse estimate 1.97× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.967 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.1398 | 3.6790 | 1.0537 | 0.9487 | 0.7571 |
| true | 2% | 2.0579 | 3.4025 | 1.0389 | 0.9636 | 0.8158 |
| true | 5% | 1.9328 | 3.0674 | 1.0243 | 0.9779 | 0.8719 |
| true | 10% | 1.8140 | 2.8148 | 1.0156 | 0.9861 | 0.9066 |
| true | 20% | 1.6560 | 2.4025 | 1.0085 | 0.9927 | 0.9392 |
| coarse | 1% | 2.1973 | 3.6750 | 1.0368 | 0.9218 | 0.6842 |
| coarse | 2% | 2.1094 | 3.3962 | 1.0279 | 0.9448 | 0.7505 |
| coarse | 5% | 1.9745 | 3.1451 | 1.0185 | 0.9670 | 0.8227 |
| coarse | 10% | 1.8476 | 2.8264 | 1.0125 | 0.9794 | 0.8709 |
| coarse | 20% | 1.6826 | 2.4245 | 1.0072 | 0.9891 | 0.9160 |
| sign | 1% | 2.3434 | 3.6296 | 0.9941 | 0.8564 | 0.5178 |
| sign | 2% | 2.2386 | 3.4591 | 0.9984 | 0.8963 | 0.6013 |
| sign | 5% | 2.0794 | 3.1891 | 1.0022 | 0.9365 | 0.6954 |
| sign | 10% | 1.9321 | 2.8769 | 1.0033 | 0.9597 | 0.7693 |
| sign | 20% | 1.7472 | 2.4563 | 1.0030 | 0.9781 | 0.8434 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1981 | 3.0674 | 2.2435 | 0.9376 |
| 7 | 1.8359 | 2.7653 | 1.8925 | 0.9812 |
| 11 | 1.7907 | 2.6846 | 1.8398 | 0.9775 |
| 15 | 1.7925 | 2.7225 | 1.8499 | 0.9745 |
| 19 | 1.7663 | 2.8900 | 1.7887 | 0.9699 |
| 23 | 2.2133 | 2.9805 | 2.2323 | 0.9612 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 731s
