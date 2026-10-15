# E4 GQA union (group size G = 8 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 8 per-head true top-k sets is 2.73× k on average (max 5.07×); with the coarse estimate 2.81× k.
Criterion "union ≤ ~3× per-head k": PASS (G=8 bounds the union at 8×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.948 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 3.1192 | 6.3537 | 1.0859 | 0.9141 | 0.5667 |
| true | 2% | 2.9676 | 5.7914 | 1.0640 | 0.9371 | 0.6503 |
| true | 5% | 2.7311 | 5.0709 | 1.0429 | 0.9590 | 0.7399 |
| true | 10% | 2.4892 | 4.2509 | 1.0299 | 0.9716 | 0.8030 |
| true | 20% | 2.1569 | 3.2735 | 1.0179 | 0.9825 | 0.8652 |
| coarse | 1% | 3.2328 | 6.3659 | 1.0726 | 0.8890 | 0.5061 |
| coarse | 2% | 3.0665 | 5.7730 | 1.0552 | 0.9190 | 0.5824 |
| coarse | 5% | 2.8099 | 5.1589 | 1.0381 | 0.9478 | 0.6785 |
| coarse | 10% | 2.5502 | 4.2668 | 1.0273 | 0.9640 | 0.7515 |
| coarse | 20% | 2.1994 | 3.2931 | 1.0168 | 0.9779 | 0.8279 |
| sign | 1% | 3.5548 | 6.5244 | 1.0404 | 0.8298 | 0.3951 |
| sign | 2% | 3.3408 | 5.9695 | 1.0328 | 0.8740 | 0.4666 |
| sign | 5% | 3.0215 | 5.2078 | 1.0255 | 0.9181 | 0.5652 |
| sign | 10% | 2.7092 | 4.3856 | 1.0202 | 0.9434 | 0.6430 |
| sign | 20% | 2.3044 | 3.3534 | 1.0138 | 0.9649 | 0.7372 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 3.6240 | 5.0709 | 3.7001 | 0.8953 |
| 15 | 2.3884 | 3.8484 | 2.4756 | 0.9735 |
| 27 | 2.0497 | 3.4813 | 2.1500 | 0.9821 |
| 39 | 2.8625 | 4.2302 | 2.9140 | 0.9402 |

## Caveats
- With G=8 the union can never exceed 8×. Larger groups (G=8–16) are not tested here.
runtime 903s
