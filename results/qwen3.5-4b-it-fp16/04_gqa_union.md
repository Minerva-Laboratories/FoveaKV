# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.92× k on average (max 3.18×); with the coarse estimate 1.97× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.958 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.0871 | 3.6296 | 1.0585 | 0.9378 | 0.6433 |
| true | 2% | 2.0222 | 3.5094 | 1.0438 | 0.9544 | 0.7160 |
| true | 5% | 1.9192 | 3.1814 | 1.0295 | 0.9704 | 0.7981 |
| true | 10% | 1.8116 | 2.8010 | 1.0204 | 0.9801 | 0.8519 |
| true | 20% | 1.6595 | 2.3938 | 1.0121 | 0.9886 | 0.9007 |
| coarse | 1% | 2.1511 | 3.6296 | 1.0406 | 0.9106 | 0.5700 |
| coarse | 2% | 2.0816 | 3.4845 | 1.0318 | 0.9350 | 0.6493 |
| coarse | 5% | 1.9695 | 3.1751 | 1.0228 | 0.9584 | 0.7369 |
| coarse | 10% | 1.8526 | 2.8595 | 1.0166 | 0.9722 | 0.7997 |
| coarse | 20% | 1.6906 | 2.4145 | 1.0103 | 0.9839 | 0.8639 |
| sign | 1% | 2.3236 | 3.7215 | 0.9965 | 0.8474 | 0.4331 |
| sign | 2% | 2.2343 | 3.5466 | 1.0005 | 0.8869 | 0.5061 |
| sign | 5% | 2.0919 | 3.2289 | 1.0044 | 0.9269 | 0.6041 |
| sign | 10% | 1.9511 | 2.8706 | 1.0057 | 0.9506 | 0.6849 |
| sign | 20% | 1.7635 | 2.4476 | 1.0052 | 0.9709 | 0.7743 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.2064 | 3.1814 | 2.2525 | 0.9111 |
| 11 | 1.7118 | 2.6727 | 1.7735 | 0.9835 |
| 19 | 1.8275 | 2.9756 | 1.8811 | 0.9771 |
| 27 | 1.9309 | 3.0698 | 1.9709 | 0.9620 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 918s
