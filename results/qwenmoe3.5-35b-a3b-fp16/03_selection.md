# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.826); sign-plane(+RHT) 0.955 (overlap 0.672).
Without RHT: coarse 0.960, sign 0.903.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7887 | 0.9721 | 0.9615 | 0.8900 | 2.7455 | 0.0045 |
| coarse | RHT | 2% | 0.8053 | 0.9808 | 0.9673 | 0.9060 | 2.7455 | 0.0045 |
| coarse | RHT | 5% | 0.8265 | 0.9885 | 0.9747 | 0.9299 | 2.7455 | 0.0045 |
| coarse | RHT | 10% | 0.8441 | 0.9926 | 0.9807 | 0.9460 | 2.7455 | 0.0045 |
| coarse | RHT | 20% | 0.8684 | 0.9957 | 0.9874 | 0.9649 | 2.7455 | 0.0045 |
| coarse | no-RHT | 1% | 0.7004 | 0.9071 | 0.8573 | 0.5924 | 5.6511 | 0.0000 |
| coarse | no-RHT | 2% | 0.7265 | 0.9303 | 0.8822 | 0.6792 | 5.6511 | 0.0000 |
| coarse | no-RHT | 5% | 0.7620 | 0.9605 | 0.9234 | 0.7652 | 5.6511 | 0.0000 |
| coarse | no-RHT | 10% | 0.7903 | 0.9799 | 0.9605 | 0.8662 | 5.6511 | 0.0000 |
| coarse | no-RHT | 20% | 0.8255 | 0.9917 | 0.9836 | 0.9546 | 5.6511 | 0.0000 |
| sign | RHT | 1% | 0.6112 | 0.8983 | 0.8702 | 0.6715 | 5.4014 | 0.0000 |
| sign | RHT | 2% | 0.6367 | 0.9272 | 0.8888 | 0.7151 | 5.4014 | 0.0000 |
| sign | RHT | 5% | 0.6717 | 0.9555 | 0.9125 | 0.7761 | 5.4014 | 0.0000 |
| sign | RHT | 10% | 0.7038 | 0.9709 | 0.9320 | 0.8239 | 5.4014 | 0.0000 |
| sign | RHT | 20% | 0.7489 | 0.9831 | 0.9538 | 0.8776 | 5.4014 | 0.0000 |
| sign | no-RHT | 1% | 0.5023 | 0.7970 | 0.7457 | 0.4149 | 7.6111 | 0.0000 |
| sign | no-RHT | 2% | 0.5364 | 0.8462 | 0.7873 | 0.5202 | 7.6111 | 0.0000 |
| sign | no-RHT | 5% | 0.5851 | 0.9034 | 0.8481 | 0.6522 | 7.6111 | 0.0000 |
| sign | no-RHT | 10% | 0.6276 | 0.9387 | 0.8971 | 0.7445 | 7.6111 | 0.0000 |
| sign | no-RHT | 20% | 0.6841 | 0.9667 | 0.9419 | 0.8373 | 7.6111 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7940 | 0.9802 | 0.9299 |
| 3 | sign | 0.6162 | 0.9285 | 0.7761 |
| 15 | coarse | 0.8317 | 0.9926 | 0.9611 |
| 15 | sign | 0.6781 | 0.9687 | 0.8521 |
| 27 | coarse | 0.8478 | 0.9934 | 0.9642 |
| 27 | sign | 0.7096 | 0.9720 | 0.8574 |
| 39 | coarse | 0.8324 | 0.9879 | 0.9672 |
| 39 | sign | 0.6828 | 0.9526 | 0.8719 |

runtime 860s
