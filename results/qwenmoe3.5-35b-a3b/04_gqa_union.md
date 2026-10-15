# E4 GQA union (group size G = 8 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 8 per-head true top-k sets is 2.69× k on average (max 4.93×); with the coarse estimate 2.77× k.
Criterion "union ≤ ~3× per-head k": PASS (G=8 bounds the union at 8×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.947 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 3.0867 | 6.0976 | 1.0894 | 0.9135 | 0.5413 |
| true | 2% | 2.9305 | 5.7239 | 1.0667 | 0.9364 | 0.6282 |
| true | 5% | 2.6852 | 4.9312 | 1.0449 | 0.9584 | 0.7288 |
| true | 10% | 2.4415 | 4.1636 | 1.0313 | 0.9712 | 0.7966 |
| true | 20% | 2.1188 | 3.2491 | 1.0188 | 0.9823 | 0.8621 |
| coarse | 1% | 3.1967 | 6.2683 | 1.0761 | 0.8882 | 0.4849 |
| coarse | 2% | 3.0283 | 5.7362 | 1.0579 | 0.9185 | 0.5639 |
| coarse | 5% | 2.7669 | 4.9483 | 1.0402 | 0.9472 | 0.6659 |
| coarse | 10% | 2.5062 | 4.1641 | 1.0288 | 0.9636 | 0.7413 |
| coarse | 20% | 2.1633 | 3.2784 | 1.0177 | 0.9777 | 0.8205 |
| sign | 1% | 3.5241 | 6.3333 | 1.0441 | 0.8296 | 0.3841 |
| sign | 2% | 3.3043 | 5.8323 | 1.0357 | 0.8734 | 0.4534 |
| sign | 5% | 2.9774 | 5.1525 | 1.0276 | 0.9178 | 0.5502 |
| sign | 10% | 2.6661 | 4.3411 | 1.0218 | 0.9432 | 0.6336 |
| sign | 20% | 2.2714 | 3.3605 | 1.0148 | 0.9647 | 0.7289 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 3.6496 | 4.9312 | 3.7362 | 0.8904 |
| 15 | 2.3471 | 3.7311 | 2.4367 | 0.9733 |
| 27 | 1.9411 | 3.1881 | 2.0435 | 0.9818 |
| 39 | 2.8029 | 4.1830 | 2.8513 | 0.9433 |

## Caveats
- With G=8 the union can never exceed 8×. Larger groups (G=8–16) are not tested here.
runtime 902s
