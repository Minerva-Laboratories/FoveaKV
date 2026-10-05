# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.99× k on average (max 3.10×); with the coarse estimate 2.03× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.963 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.2069 | 3.7342 | 1.0551 | 0.9459 | 0.7692 |
| true | 2% | 2.1235 | 3.4684 | 1.0412 | 0.9603 | 0.8158 |
| true | 5% | 1.9887 | 3.0987 | 1.0266 | 0.9751 | 0.8763 |
| true | 10% | 1.8622 | 2.7982 | 1.0174 | 0.9842 | 0.9114 |
| true | 20% | 1.6933 | 2.4074 | 1.0096 | 0.9915 | 0.9421 |
| coarse | 1% | 2.2678 | 3.6835 | 1.0374 | 0.9189 | 0.6873 |
| coarse | 2% | 2.1789 | 3.4258 | 1.0293 | 0.9406 | 0.7526 |
| coarse | 5% | 2.0338 | 3.0532 | 1.0202 | 0.9631 | 0.8266 |
| coarse | 10% | 1.8989 | 2.7982 | 1.0140 | 0.9765 | 0.8721 |
| coarse | 20% | 1.7211 | 2.3940 | 1.0081 | 0.9874 | 0.9168 |
| sign | 1% | 2.4212 | 3.6543 | 0.9922 | 0.8529 | 0.5262 |
| sign | 2% | 2.3138 | 3.4658 | 0.9973 | 0.8895 | 0.6032 |
| sign | 5% | 2.1454 | 3.1460 | 1.0020 | 0.9296 | 0.7003 |
| sign | 10% | 1.9893 | 2.8306 | 1.0037 | 0.9541 | 0.7764 |
| sign | 20% | 1.7884 | 2.4072 | 1.0036 | 0.9745 | 0.8488 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.2000 | 2.9826 | 2.2555 | 0.9347 |
| 7 | 1.8818 | 2.8582 | 1.9431 | 0.9804 |
| 11 | 1.8085 | 2.7702 | 1.8649 | 0.9750 |
| 15 | 1.8000 | 2.6878 | 1.8483 | 0.9710 |
| 19 | 1.8612 | 2.6931 | 1.8937 | 0.9621 |
| 23 | 2.3804 | 3.0987 | 2.3974 | 0.9555 |

## Caveats
- With only 2 KV heads (G=4) the union can never exceed 4×; this model is a mild GQA case. Larger groups (G=8–16) are not tested here.
runtime 296s
