# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.94× k on average (max 3.12×); with the coarse estimate 1.99× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.959 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.1299 | 3.6543 | 1.0595 | 0.9341 | 0.6414 |
| true | 2% | 2.0489 | 3.4534 | 1.0433 | 0.9532 | 0.7202 |
| true | 5% | 1.9381 | 3.1222 | 1.0279 | 0.9710 | 0.8073 |
| true | 10% | 1.8327 | 2.8449 | 1.0188 | 0.9809 | 0.8571 |
| true | 20% | 1.6848 | 2.3980 | 1.0109 | 0.9891 | 0.9041 |
| coarse | 1% | 2.1915 | 3.6375 | 1.0417 | 0.9066 | 0.5637 |
| coarse | 2% | 2.1074 | 3.4403 | 1.0317 | 0.9338 | 0.6496 |
| coarse | 5% | 1.9880 | 3.1827 | 1.0217 | 0.9593 | 0.7445 |
| coarse | 10% | 1.8730 | 2.8361 | 1.0153 | 0.9734 | 0.8081 |
| coarse | 20% | 1.7143 | 2.4059 | 1.0094 | 0.9849 | 0.8708 |
| sign | 1% | 2.3566 | 3.6914 | 0.9966 | 0.8414 | 0.4269 |
| sign | 2% | 2.2580 | 3.5000 | 1.0003 | 0.8844 | 0.5124 |
| sign | 5% | 2.1113 | 3.1956 | 1.0039 | 0.9274 | 0.6169 |
| sign | 10% | 1.9713 | 2.8995 | 1.0051 | 0.9522 | 0.6958 |
| sign | 20% | 1.7851 | 2.4620 | 1.0046 | 0.9724 | 0.7853 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1602 | 3.1222 | 2.2075 | 0.9184 |
| 11 | 1.7547 | 2.8220 | 1.8157 | 0.9834 |
| 19 | 1.8626 | 3.0415 | 1.9161 | 0.9769 |
| 27 | 1.9749 | 3.1220 | 2.0127 | 0.9587 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 1022s
