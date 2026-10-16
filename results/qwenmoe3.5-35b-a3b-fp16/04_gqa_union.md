# E4 GQA union (group size G = 8 query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the 8 per-head true top-k sets is 2.68× k on average (max 4.91×); with the coarse estimate 2.77× k.
Criterion "union ≤ ~3× per-head k": PASS (G=8 bounds the union at 8×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures 0.947 of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

| scores | k | union/k mean | union/k max | m_H/m_opt union | m_H/m_opt budget | budget worst p5 |
|---|---|---|---|---|---|---|
| true | 1% | 3.0866 | 6.1098 | 1.0894 | 0.9135 | 0.5398 |
| true | 2% | 2.9305 | 5.7362 | 1.0667 | 0.9364 | 0.6277 |
| true | 5% | 2.6849 | 4.9115 | 1.0449 | 0.9584 | 0.7293 |
| true | 10% | 2.4414 | 4.1648 | 1.0313 | 0.9712 | 0.7961 |
| true | 20% | 2.1187 | 3.2466 | 1.0188 | 0.9823 | 0.8619 |
| coarse | 1% | 3.1965 | 6.3293 | 1.0761 | 0.8883 | 0.4802 |
| coarse | 2% | 3.0279 | 5.7301 | 1.0579 | 0.9185 | 0.5640 |
| coarse | 5% | 2.7663 | 4.9457 | 1.0402 | 0.9472 | 0.6657 |
| coarse | 10% | 2.5060 | 4.1731 | 1.0288 | 0.9636 | 0.7419 |
| coarse | 20% | 2.1631 | 3.2765 | 1.0177 | 0.9777 | 0.8202 |
| sign | 1% | 3.5245 | 6.3077 | 1.0443 | 0.8298 | 0.3816 |
| sign | 2% | 3.3040 | 5.8645 | 1.0357 | 0.8736 | 0.4541 |
| sign | 5% | 2.9768 | 5.1292 | 1.0276 | 0.9178 | 0.5484 |
| sign | 10% | 2.6660 | 4.3463 | 1.0217 | 0.9432 | 0.6335 |
| sign | 20% | 2.2712 | 3.3643 | 1.0148 | 0.9647 | 0.7293 |

## Per layer at k=5%
| layer | true union/k | true max | coarse union/k | coarse budget m_H/m_opt |
|---|---|---|---|---|
| 3 | 3.6493 | 4.9115 | 3.7348 | 0.8904 |
| 15 | 2.3468 | 3.7262 | 2.4367 | 0.9733 |
| 27 | 1.9409 | 3.1769 | 2.0427 | 0.9818 |
| 39 | 2.8028 | 4.1704 | 2.8509 | 0.9433 |

## Caveats
- With G=8 the union can never exceed 8×. Larger groups (G=8–16) are not tested here.
runtime 860s
