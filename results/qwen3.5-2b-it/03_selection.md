# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.990 (overlap 0.826); sign-plane(+RHT) 0.959 (overlap 0.672).
Without RHT: coarse 0.980, sign 0.924.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 2457600 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 6 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7827 | 0.9731 | 0.9617 | 0.9072 | 2.8021 | 0.0039 |
| coarse | RHT | 2% | 0.8000 | 0.9817 | 0.9703 | 0.9273 | 2.8021 | 0.0039 |
| coarse | RHT | 5% | 0.8256 | 0.9897 | 0.9790 | 0.9476 | 2.8021 | 0.0039 |
| coarse | RHT | 10% | 0.8471 | 0.9939 | 0.9854 | 0.9620 | 2.8021 | 0.0039 |
| coarse | RHT | 20% | 0.8737 | 0.9969 | 0.9913 | 0.9755 | 2.8021 | 0.0039 |
| coarse | no-RHT | 1% | 0.7306 | 0.9439 | 0.9059 | 0.7641 | 4.3863 | 0.0003 |
| coarse | no-RHT | 2% | 0.7523 | 0.9620 | 0.9424 | 0.8386 | 4.3863 | 0.0003 |
| coarse | no-RHT | 5% | 0.7837 | 0.9797 | 0.9657 | 0.8861 | 4.3863 | 0.0003 |
| coarse | no-RHT | 10% | 0.8097 | 0.9888 | 0.9771 | 0.9246 | 4.3863 | 0.0003 |
| coarse | no-RHT | 20% | 0.8428 | 0.9948 | 0.9865 | 0.9628 | 4.3863 | 0.0003 |
| sign | RHT | 1% | 0.6035 | 0.9017 | 0.8658 | 0.6915 | 5.6573 | 0.0000 |
| sign | RHT | 2% | 0.6305 | 0.9304 | 0.8915 | 0.7473 | 5.6573 | 0.0000 |
| sign | RHT | 5% | 0.6718 | 0.9591 | 0.9215 | 0.8139 | 5.6573 | 0.0000 |
| sign | RHT | 10% | 0.7094 | 0.9750 | 0.9434 | 0.8614 | 5.6573 | 0.0000 |
| sign | RHT | 20% | 0.7572 | 0.9868 | 0.9651 | 0.9073 | 5.6573 | 0.0000 |
| sign | no-RHT | 1% | 0.5297 | 0.8256 | 0.6649 | 0.3271 | 6.7210 | 0.0000 |
| sign | no-RHT | 2% | 0.5644 | 0.8731 | 0.7577 | 0.4740 | 6.7210 | 0.0000 |
| sign | no-RHT | 5% | 0.6138 | 0.9241 | 0.8711 | 0.6731 | 6.7210 | 0.0000 |
| sign | no-RHT | 10% | 0.6551 | 0.9530 | 0.9152 | 0.7543 | 6.7210 | 0.0000 |
| sign | no-RHT | 20% | 0.7094 | 0.9748 | 0.9471 | 0.8363 | 6.7210 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7963 | 0.9822 | 0.9476 |
| 3 | sign | 0.6212 | 0.9332 | 0.8139 |
| 7 | coarse | 0.8228 | 0.9928 | 0.9676 |
| 7 | sign | 0.6657 | 0.9702 | 0.8750 |
| 11 | coarse | 0.8271 | 0.9921 | 0.9702 |
| 11 | sign | 0.6766 | 0.9671 | 0.8805 |
| 15 | coarse | 0.8326 | 0.9911 | 0.9668 |
| 15 | sign | 0.6850 | 0.9631 | 0.8682 |
| 19 | coarse | 0.8428 | 0.9904 | 0.9698 |
| 19 | sign | 0.7015 | 0.9619 | 0.8832 |
| 23 | coarse | 0.8318 | 0.9896 | 0.9708 |
| 23 | sign | 0.6811 | 0.9591 | 0.8858 |

runtime 731s
