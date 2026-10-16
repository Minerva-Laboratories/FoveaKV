# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.92× k on average (max 3.27×); with the coarse estimate 1.96× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.962 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.0794 | 3.6543 | 1.0559 | 0.9394 | 0.6380 |
| true | 2% | 2.0100 | 3.5714 | 1.0410 | 0.9567 | 0.7178 |
| true | 5% | 1.9181 | 3.2670 | 1.0274 | 0.9726 | 0.7934 |
| true | 10% | 1.8199 | 2.9187 | 1.0190 | 0.9818 | 0.8445 |
| true | 20% | 1.6689 | 2.4747 | 1.0113 | 0.9895 | 0.8927 |
| coarse | 1% | 2.1404 | 3.7407 | 1.0396 | 0.9135 | 0.5703 |
| coarse | 2% | 2.0656 | 3.5590 | 1.0302 | 0.9387 | 0.6493 |
| coarse | 5% | 1.9632 | 3.2419 | 1.0213 | 0.9618 | 0.7314 |
| coarse | 10% | 1.8562 | 2.9131 | 1.0155 | 0.9746 | 0.7922 |
| coarse | 20% | 1.6970 | 2.4830 | 1.0097 | 0.9853 | 0.8562 |
| sign | 1% | 2.3021 | 3.7284 | 0.9971 | 0.8495 | 0.4454 |
| sign | 2% | 2.2073 | 3.5443 | 1.0010 | 0.8920 | 0.5106 |
| sign | 5% | 2.0750 | 3.2574 | 1.0044 | 0.9323 | 0.6011 |
| sign | 10% | 1.9454 | 2.9421 | 1.0054 | 0.9547 | 0.6763 |
| sign | 20% | 1.7644 | 2.5029 | 1.0048 | 0.9733 | 0.7659 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1739 | 3.2670 | 2.2159 | 0.9209 |
| 11 | 1.7556 | 2.6976 | 1.8105 | 0.9828 |
| 19 | 1.7863 | 2.9291 | 1.8308 | 0.9788 |
| 27 | 1.9564 | 3.1491 | 1.9955 | 0.9647 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 921s
