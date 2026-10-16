# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 2.01× k on average (max 3.20×); with the coarse estimate 2.05× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.962 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.2102 | 3.6341 | 1.0574 | 0.9410 | 0.7477 |
| true | 2% | 2.1287 | 3.4268 | 1.0422 | 0.9578 | 0.8149 |
| true | 5% | 2.0064 | 3.2049 | 1.0270 | 0.9737 | 0.8737 |
| true | 10% | 1.8884 | 2.8449 | 1.0178 | 0.9831 | 0.9065 |
| true | 20% | 1.7233 | 2.3856 | 1.0099 | 0.9908 | 0.9400 |
| coarse | 1% | 2.2643 | 3.6125 | 1.0400 | 0.9141 | 0.6791 |
| coarse | 2% | 2.1773 | 3.3677 | 1.0306 | 0.9388 | 0.7500 |
| coarse | 5% | 2.0468 | 3.1927 | 1.0207 | 0.9622 | 0.8262 |
| coarse | 10% | 1.9212 | 2.8364 | 1.0143 | 0.9757 | 0.8708 |
| coarse | 20% | 1.7485 | 2.4232 | 1.0083 | 0.9867 | 0.9168 |
| sign | 1% | 2.4090 | 3.6076 | 0.9954 | 0.8473 | 0.5278 |
| sign | 2% | 2.3059 | 3.4198 | 0.9993 | 0.8881 | 0.6109 |
| sign | 5% | 2.1502 | 3.1606 | 1.0027 | 0.9295 | 0.7037 |
| sign | 10% | 2.0021 | 2.8795 | 1.0039 | 0.9540 | 0.7738 |
| sign | 20% | 1.8076 | 2.4524 | 1.0035 | 0.9742 | 0.8487 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1693 | 3.0622 | 2.2149 | 0.9408 |
| 7 | 1.9128 | 2.8998 | 1.9687 | 0.9824 |
| 11 | 1.8243 | 2.7366 | 1.8735 | 0.9765 |
| 15 | 1.9039 | 2.7188 | 1.9515 | 0.9682 |
| 19 | 1.9154 | 2.9829 | 1.9405 | 0.9527 |
| 23 | 2.3128 | 3.2049 | 2.3316 | 0.9528 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 693s
