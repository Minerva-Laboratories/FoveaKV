# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.95× k on average (max 3.07×); with the coarse estimate 1.99× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.966 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.1603 | 3.5897 | 1.0535 | 0.9492 | 0.7681 |
| true | 2% | 2.0781 | 3.4294 | 1.0389 | 0.9636 | 0.8219 |
| true | 5% | 1.9498 | 3.0740 | 1.0243 | 0.9778 | 0.8781 |
| true | 10% | 1.8303 | 2.8020 | 1.0157 | 0.9860 | 0.9108 |
| true | 20% | 1.6752 | 2.4132 | 1.0086 | 0.9925 | 0.9420 |
| coarse | 1% | 2.2227 | 3.5641 | 1.0361 | 0.9216 | 0.6864 |
| coarse | 2% | 2.1340 | 3.4258 | 1.0276 | 0.9441 | 0.7519 |
| coarse | 5% | 1.9948 | 3.0285 | 1.0184 | 0.9664 | 0.8259 |
| coarse | 10% | 1.8665 | 2.8046 | 1.0126 | 0.9789 | 0.8732 |
| coarse | 20% | 1.7024 | 2.3914 | 1.0073 | 0.9887 | 0.9175 |
| sign | 1% | 2.3783 | 3.5949 | 0.9919 | 0.8546 | 0.5255 |
| sign | 2% | 2.2697 | 3.4258 | 0.9970 | 0.8934 | 0.6037 |
| sign | 5% | 2.1041 | 3.1468 | 1.0015 | 0.9345 | 0.7022 |
| sign | 10% | 1.9557 | 2.8356 | 1.0031 | 0.9583 | 0.7755 |
| sign | 20% | 1.7695 | 2.4134 | 1.0031 | 0.9770 | 0.8484 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1741 | 2.9925 | 2.2286 | 0.9358 |
| 7 | 1.8446 | 2.8606 | 1.9039 | 0.9824 |
| 11 | 1.7879 | 2.7628 | 1.8441 | 0.9778 |
| 15 | 1.7654 | 2.6902 | 1.8130 | 0.9763 |
| 19 | 1.8062 | 2.6537 | 1.8406 | 0.9683 |
| 23 | 2.3208 | 3.0740 | 2.3387 | 0.9579 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 727s
