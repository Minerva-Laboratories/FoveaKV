# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.92× k on average (max 3.27×); with the coarse estimate 1.96× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.962 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.0796 | 3.6667 | 1.0559 | 0.9393 | 0.6381 |
| true | 2% | 2.0101 | 3.5590 | 1.0410 | 0.9567 | 0.7171 |
| true | 5% | 1.9181 | 3.2670 | 1.0274 | 0.9726 | 0.7934 |
| true | 10% | 1.8200 | 2.9163 | 1.0190 | 0.9818 | 0.8440 |
| true | 20% | 1.6689 | 2.4753 | 1.0113 | 0.9895 | 0.8928 |
| coarse | 1% | 2.1404 | 3.7531 | 1.0396 | 0.9134 | 0.5707 |
| coarse | 2% | 2.0658 | 3.5528 | 1.0302 | 0.9387 | 0.6449 |
| coarse | 5% | 1.9634 | 3.2444 | 1.0213 | 0.9618 | 0.7309 |
| coarse | 10% | 1.8562 | 2.9106 | 1.0155 | 0.9746 | 0.7910 |
| coarse | 20% | 1.6971 | 2.4856 | 1.0097 | 0.9853 | 0.8563 |
| sign | 1% | 2.3020 | 3.7317 | 0.9972 | 0.8497 | 0.4448 |
| sign | 2% | 2.2075 | 3.5570 | 1.0010 | 0.8921 | 0.5106 |
| sign | 5% | 2.0748 | 3.2475 | 1.0043 | 0.9323 | 0.5996 |
| sign | 10% | 1.9453 | 2.9446 | 1.0054 | 0.9547 | 0.6759 |
| sign | 20% | 1.7644 | 2.4965 | 1.0048 | 0.9733 | 0.7663 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1739 | 3.2670 | 2.2160 | 0.9209 |
| 11 | 1.7557 | 2.7000 | 1.8103 | 0.9827 |
| 19 | 1.7864 | 2.9389 | 1.8312 | 0.9788 |
| 27 | 1.9565 | 3.1516 | 1.9961 | 0.9646 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 976s
