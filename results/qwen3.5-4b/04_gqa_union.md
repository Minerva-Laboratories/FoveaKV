# E4 GQA union (group size G = 4 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 4 per-head true top-k sets is 1.96× k on average (max 3.12×); with the coarse estimate 2.01× k.
Criterion "union ≤ ~3× per-head k": PASS (G=4 bounds the union at 4×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.956 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 2.1680 | 3.6667 | 1.0615 | 0.9304 | 0.6430 |
| true | 2% | 2.0861 | 3.4522 | 1.0458 | 0.9496 | 0.7211 |
| true | 5% | 1.9640 | 3.1247 | 1.0299 | 0.9684 | 0.8071 |
| true | 10% | 1.8480 | 2.8413 | 1.0202 | 0.9793 | 0.8566 |
| true | 20% | 1.6922 | 2.3999 | 1.0116 | 0.9883 | 0.9041 |
| coarse | 1% | 2.2306 | 3.6173 | 1.0431 | 0.9027 | 0.5686 |
| coarse | 2% | 2.1444 | 3.4286 | 1.0336 | 0.9295 | 0.6488 |
| coarse | 5% | 2.0138 | 3.1751 | 1.0233 | 0.9560 | 0.7434 |
| coarse | 10% | 1.8886 | 2.8460 | 1.0165 | 0.9713 | 0.8079 |
| coarse | 20% | 1.7224 | 2.4010 | 1.0100 | 0.9838 | 0.8708 |
| sign | 1% | 2.3919 | 3.6790 | 0.9967 | 0.8373 | 0.4308 |
| sign | 2% | 2.2923 | 3.4845 | 1.0009 | 0.8794 | 0.5148 |
| sign | 5% | 2.1375 | 3.1870 | 1.0047 | 0.9229 | 0.6170 |
| sign | 10% | 1.9889 | 2.8787 | 1.0058 | 0.9490 | 0.6954 |
| sign | 20% | 1.7950 | 2.4485 | 1.0051 | 0.9706 | 0.7852 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 2.1713 | 3.1247 | 2.2204 | 0.9166 |
| 11 | 1.7763 | 2.8244 | 1.8382 | 0.9818 |
| 19 | 1.8956 | 3.0415 | 1.9486 | 0.9729 |
| 27 | 2.0128 | 3.1244 | 2.0480 | 0.9529 |

## Caveats
- With G=4 the union can never exceed 4×. Larger groups (G=8–16) are not tested here.
runtime 427s
