# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.827); sign-plane(+RHT) 0.955 (overlap 0.670).
Without RHT: coarse 0.975, sign 0.914.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7906 | 0.9717 | 0.9551 | 0.8866 | 2.8163 | 0.0037 |
| coarse | RHT | 2% | 0.8061 | 0.9806 | 0.9623 | 0.9044 | 2.8163 | 0.0037 |
| coarse | RHT | 5% | 0.8268 | 0.9886 | 0.9717 | 0.9265 | 2.8163 | 0.0037 |
| coarse | RHT | 10% | 0.8458 | 0.9929 | 0.9792 | 0.9437 | 2.8163 | 0.0037 |
| coarse | RHT | 20% | 0.8707 | 0.9962 | 0.9868 | 0.9614 | 2.8163 | 0.0037 |
| coarse | no-RHT | 1% | 0.7300 | 0.9157 | 0.8380 | 0.5585 | 5.4091 | 0.0000 |
| coarse | no-RHT | 2% | 0.7568 | 0.9457 | 0.8947 | 0.6871 | 5.4091 | 0.0000 |
| coarse | no-RHT | 5% | 0.7882 | 0.9747 | 0.9481 | 0.8269 | 5.4091 | 0.0000 |
| coarse | no-RHT | 10% | 0.8127 | 0.9871 | 0.9699 | 0.8784 | 5.4091 | 0.0000 |
| coarse | no-RHT | 20% | 0.8434 | 0.9941 | 0.9842 | 0.9442 | 5.4091 | 0.0000 |
| sign | RHT | 1% | 0.6127 | 0.8968 | 0.8448 | 0.6469 | 5.5934 | 0.0000 |
| sign | RHT | 2% | 0.6360 | 0.9259 | 0.8690 | 0.6960 | 5.5934 | 0.0000 |
| sign | RHT | 5% | 0.6705 | 0.9550 | 0.8994 | 0.7589 | 5.5934 | 0.0000 |
| sign | RHT | 10% | 0.7046 | 0.9713 | 0.9240 | 0.8092 | 5.5934 | 0.0000 |
| sign | RHT | 20% | 0.7511 | 0.9841 | 0.9503 | 0.8668 | 5.5934 | 0.0000 |
| sign | no-RHT | 1% | 0.5340 | 0.8028 | 0.7032 | 0.3989 | 7.2244 | 0.0000 |
| sign | no-RHT | 2% | 0.5695 | 0.8545 | 0.7727 | 0.5117 | 7.2244 | 0.0000 |
| sign | no-RHT | 5% | 0.6200 | 0.9136 | 0.8648 | 0.6777 | 7.2244 | 0.0000 |
| sign | no-RHT | 10% | 0.6629 | 0.9479 | 0.9060 | 0.7581 | 7.2244 | 0.0000 |
| sign | no-RHT | 20% | 0.7168 | 0.9731 | 0.9424 | 0.8230 | 7.2244 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8073 | 0.9787 | 0.9265 |
| 3 | sign | 0.6368 | 0.9212 | 0.7589 |
| 11 | coarse | 0.8341 | 0.9938 | 0.9682 |
| 11 | sign | 0.6834 | 0.9730 | 0.8753 |
| 19 | coarse | 0.8327 | 0.9929 | 0.9712 |
| 19 | sign | 0.6803 | 0.9696 | 0.8851 |
| 27 | coarse | 0.8330 | 0.9892 | 0.9673 |
| 27 | sign | 0.6813 | 0.9560 | 0.8766 |

runtime 1022s
