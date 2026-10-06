# E4 GQA union (group size G = 2 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 2 per-head true top-k sets is 1.46× k on average (max 1.98×); with the coarse estimate 1.49× k.
Criterion "union ≤ ~3× per-head k": PASS (G=2 bounds the union at 2×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.913 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 1.5281 | 1.9878 | 1.0151 | 0.9865 | 0.8466 |
| true | 2% | 1.5084 | 1.9809 | 1.0128 | 0.9891 | 0.8729 |
| true | 5% | 1.4638 | 1.9799 | 1.0094 | 0.9924 | 0.9086 |
| true | 10% | 1.4174 | 1.9301 | 1.0068 | 0.9948 | 0.9341 |
| true | 20% | 1.3469 | 1.7709 | 1.0042 | 0.9971 | 0.9588 |
| coarse | 1% | 1.5705 | 2.0000 | 0.8795 | 0.8180 | 0.0000 |
| coarse | 2% | 1.5441 | 1.9756 | 0.9186 | 0.8585 | 0.0000 |
| coarse | 5% | 1.4933 | 1.9049 | 0.9566 | 0.9129 | 0.0002 |
| coarse | 10% | 1.4396 | 1.8430 | 0.9778 | 0.9496 | 0.0986 |
| coarse | 20% | 1.3642 | 1.7414 | 0.9914 | 0.9755 | 0.7602 |
| sign | 1% | 1.6281 | 2.0000 | 0.6586 | 0.5982 | 0.0000 |
| sign | 2% | 1.5925 | 2.1132 | 0.7143 | 0.6542 | 0.0000 |
| sign | 5% | 1.5341 | 2.2408 | 0.8051 | 0.7502 | 0.0000 |
| sign | 10% | 1.4696 | 1.9577 | 0.8493 | 0.8123 | 0.0001 |
| sign | 20% | 1.3778 | 1.8102 | 0.8922 | 0.8640 | 0.0001 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 0 | 1.4927 | 1.9799 | 1.5454 | 0.7293 |
| 9 | 1.5042 | 1.9098 | 1.5482 | 0.9586 |
| 18 | 1.4433 | 1.8977 | 1.4671 | 0.9728 |
| 27 | 1.4150 | 1.7261 | 1.4124 | 0.9911 |

## Caveats
- With G=2 the union can never exceed 2×. Larger groups (G=8–16) are not tested here.
runtime 399s
