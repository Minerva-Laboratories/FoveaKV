# E4 GQA union (group size G = 8 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 8 per-head true top-k sets is 2.73× k on average (max 5.06×); with the coarse estimate 2.81× k.
Criterion "union ≤ ~3× per-head k": PASS (G=8 bounds the union at 8×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.948 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 3.1194 | 6.3293 | 1.0859 | 0.9142 | 0.5665 |
| true | 2% | 2.9674 | 5.7914 | 1.0640 | 0.9371 | 0.6500 |
| true | 5% | 2.7311 | 5.0636 | 1.0429 | 0.9590 | 0.7388 |
| true | 10% | 2.4892 | 4.2558 | 1.0299 | 0.9716 | 0.8034 |
| true | 20% | 2.1569 | 3.2735 | 1.0179 | 0.9825 | 0.8653 |
| coarse | 1% | 3.2318 | 6.3415 | 1.0725 | 0.8889 | 0.5051 |
| coarse | 2% | 3.0657 | 5.7791 | 1.0552 | 0.9190 | 0.5839 |
| coarse | 5% | 2.8102 | 5.1100 | 1.0381 | 0.9478 | 0.6792 |
| coarse | 10% | 2.5502 | 4.3060 | 1.0273 | 0.9640 | 0.7514 |
| coarse | 20% | 2.1993 | 3.2962 | 1.0168 | 0.9779 | 0.8274 |
| sign | 1% | 3.5536 | 6.5488 | 1.0404 | 0.8302 | 0.3955 |
| sign | 2% | 3.3413 | 5.9755 | 1.0328 | 0.8740 | 0.4689 |
| sign | 5% | 3.0218 | 5.1932 | 1.0255 | 0.9180 | 0.5600 |
| sign | 10% | 2.7094 | 4.3672 | 1.0202 | 0.9433 | 0.6424 |
| sign | 20% | 2.3043 | 3.3527 | 1.0138 | 0.9649 | 0.7359 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 3.6241 | 5.0636 | 3.7008 | 0.8952 |
| 15 | 2.3881 | 3.8509 | 2.4772 | 0.9735 |
| 27 | 2.0495 | 3.4713 | 2.1493 | 0.9821 |
| 39 | 2.8625 | 4.2277 | 2.9137 | 0.9402 |

## Caveats
- With G=8 the union can never exceed 8×. Larger groups (G=8–16) are not tested here.
runtime 906s
