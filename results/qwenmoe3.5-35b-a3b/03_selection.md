# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.826); sign-plane(+RHT) 0.955 (overlap 0.671).
Without RHT: coarse 0.960, sign 0.904.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7887 | 0.9721 | 0.9616 | 0.8886 | 2.7367 | 0.0046 |
| coarse | RHT | 2% | 0.8052 | 0.9808 | 0.9674 | 0.9071 | 2.7367 | 0.0046 |
| coarse | RHT | 5% | 0.8265 | 0.9885 | 0.9748 | 0.9302 | 2.7367 | 0.0046 |
| coarse | RHT | 10% | 0.8441 | 0.9926 | 0.9808 | 0.9468 | 2.7367 | 0.0046 |
| coarse | RHT | 20% | 0.8685 | 0.9958 | 0.9874 | 0.9649 | 2.7367 | 0.0046 |
| coarse | no-RHT | 1% | 0.7008 | 0.9072 | 0.8583 | 0.5955 | 5.6041 | 0.0000 |
| coarse | no-RHT | 2% | 0.7268 | 0.9300 | 0.8822 | 0.6748 | 5.6041 | 0.0000 |
| coarse | no-RHT | 5% | 0.7623 | 0.9604 | 0.9235 | 0.7655 | 5.6041 | 0.0000 |
| coarse | no-RHT | 10% | 0.7905 | 0.9801 | 0.9603 | 0.8696 | 5.6041 | 0.0000 |
| coarse | no-RHT | 20% | 0.8257 | 0.9917 | 0.9836 | 0.9531 | 5.6041 | 0.0000 |
| sign | RHT | 1% | 0.6109 | 0.8982 | 0.8698 | 0.6714 | 5.4006 | 0.0000 |
| sign | RHT | 2% | 0.6365 | 0.9272 | 0.8885 | 0.7175 | 5.4006 | 0.0000 |
| sign | RHT | 5% | 0.6715 | 0.9554 | 0.9124 | 0.7766 | 5.4006 | 0.0000 |
| sign | RHT | 10% | 0.7038 | 0.9709 | 0.9320 | 0.8244 | 5.4006 | 0.0000 |
| sign | RHT | 20% | 0.7489 | 0.9831 | 0.9538 | 0.8774 | 5.4006 | 0.0000 |
| sign | no-RHT | 1% | 0.5030 | 0.7979 | 0.7460 | 0.4139 | 7.5835 | 0.0000 |
| sign | no-RHT | 2% | 0.5370 | 0.8471 | 0.7889 | 0.5173 | 7.5835 | 0.0000 |
| sign | no-RHT | 5% | 0.5855 | 0.9037 | 0.8485 | 0.6507 | 7.5835 | 0.0000 |
| sign | no-RHT | 10% | 0.6279 | 0.9389 | 0.8968 | 0.7419 | 7.5835 | 0.0000 |
| sign | no-RHT | 20% | 0.6844 | 0.9668 | 0.9418 | 0.8406 | 7.5835 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7939 | 0.9802 | 0.9302 |
| 3 | sign | 0.6160 | 0.9284 | 0.7766 |
| 15 | coarse | 0.8317 | 0.9926 | 0.9612 |
| 15 | sign | 0.6779 | 0.9687 | 0.8531 |
| 27 | coarse | 0.8477 | 0.9934 | 0.9645 |
| 27 | sign | 0.7095 | 0.9720 | 0.8580 |
| 39 | coarse | 0.8325 | 0.9879 | 0.9669 |
| 39 | sign | 0.6826 | 0.9526 | 0.8700 |

runtime 902s
