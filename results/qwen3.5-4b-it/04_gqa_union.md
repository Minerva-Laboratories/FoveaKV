# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.92× k on average (max 3.18×); with the coarse estimate 1.97× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.958 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.0870 | 3.6420 | 1.0585 | 0.9378 | 0.6411 |
| true | 2% | 2.0223 | 3.4969 | 1.0438 | 0.9544 | 0.7152 |
| true | 5% | 1.9192 | 3.1839 | 1.0295 | 0.9704 | 0.7979 |
| true | 10% | 1.8117 | 2.7997 | 1.0204 | 0.9801 | 0.8520 |
| true | 20% | 1.6595 | 2.3925 | 1.0120 | 0.9886 | 0.9005 |
| coarse | 1% | 2.1507 | 3.6173 | 1.0405 | 0.9106 | 0.5721 |
| coarse | 2% | 2.0812 | 3.5031 | 1.0318 | 0.9350 | 0.6473 |
| coarse | 5% | 1.9695 | 3.1617 | 1.0228 | 0.9584 | 0.7368 |
| coarse | 10% | 1.8527 | 2.8602 | 1.0166 | 0.9722 | 0.7993 |
| coarse | 20% | 1.6906 | 2.4081 | 1.0103 | 0.9839 | 0.8644 |
| sign | 1% | 2.3240 | 3.7037 | 0.9966 | 0.8475 | 0.4350 |
| sign | 2% | 2.2344 | 3.5404 | 1.0004 | 0.8869 | 0.5081 |
| sign | 5% | 2.0917 | 3.2363 | 1.0044 | 0.9270 | 0.6065 |
| sign | 10% | 1.9510 | 2.8794 | 1.0057 | 0.9506 | 0.6856 |
| sign | 20% | 1.7635 | 2.4425 | 1.0052 | 0.9709 | 0.7737 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.2065 | 3.1839 | 2.2527 | 0.9110 |
| 11 | 1.7119 | 2.6649 | 1.7735 | 0.9835 |
| 19 | 1.8276 | 2.9804 | 1.8812 | 0.9771 |
| 27 | 1.9309 | 3.0648 | 1.9705 | 0.9620 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 978s
