# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.988 (overlap 0.825); sign-plane(+RHT) 0.953 (overlap 0.668).
Without RHT: coarse 0.973, sign 0.910.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7880 | 0.9712 | 0.9550 | 0.8884 | 2.8019 | 0.0038 |
| coarse | RHT | 2% | 0.8033 | 0.9798 | 0.9624 | 0.9064 | 2.8019 | 0.0038 |
| coarse | RHT | 5% | 0.8251 | 0.9880 | 0.9717 | 0.9269 | 2.8019 | 0.0038 |
| coarse | RHT | 10% | 0.8450 | 0.9925 | 0.9792 | 0.9440 | 2.8019 | 0.0038 |
| coarse | RHT | 20% | 0.8706 | 0.9959 | 0.9868 | 0.9609 | 2.8019 | 0.0038 |
| coarse | no-RHT | 1% | 0.7256 | 0.9128 | 0.8366 | 0.5557 | 5.3648 | 0.0000 |
| coarse | no-RHT | 2% | 0.7520 | 0.9429 | 0.8951 | 0.6925 | 5.3648 | 0.0000 |
| coarse | no-RHT | 5% | 0.7849 | 0.9729 | 0.9489 | 0.8259 | 5.3648 | 0.0000 |
| coarse | no-RHT | 10% | 0.8110 | 0.9861 | 0.9711 | 0.8793 | 5.3648 | 0.0000 |
| coarse | no-RHT | 20% | 0.8432 | 0.9937 | 0.9841 | 0.9413 | 5.3648 | 0.0000 |
| sign | RHT | 1% | 0.6089 | 0.8953 | 0.8453 | 0.6511 | 5.5829 | 0.0000 |
| sign | RHT | 2% | 0.6318 | 0.9237 | 0.8690 | 0.6988 | 5.5829 | 0.0000 |
| sign | RHT | 5% | 0.6679 | 0.9528 | 0.8995 | 0.7602 | 5.5829 | 0.0000 |
| sign | RHT | 10% | 0.7034 | 0.9697 | 0.9241 | 0.8079 | 5.5829 | 0.0000 |
| sign | RHT | 20% | 0.7511 | 0.9832 | 0.9504 | 0.8671 | 5.5829 | 0.0000 |
| sign | no-RHT | 1% | 0.5302 | 0.8018 | 0.7065 | 0.4127 | 7.1069 | 0.0000 |
| sign | no-RHT | 2% | 0.5646 | 0.8515 | 0.7725 | 0.5127 | 7.1069 | 0.0000 |
| sign | no-RHT | 5% | 0.6161 | 0.9101 | 0.8667 | 0.6765 | 7.1069 | 0.0000 |
| sign | no-RHT | 10% | 0.6608 | 0.9449 | 0.9055 | 0.7593 | 7.1069 | 0.0000 |
| sign | no-RHT | 20% | 0.7168 | 0.9712 | 0.9420 | 0.8168 | 7.1069 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8053 | 0.9782 | 0.9269 |
| 3 | sign | 0.6339 | 0.9199 | 0.7602 |
| 11 | coarse | 0.8319 | 0.9933 | 0.9578 |
| 11 | sign | 0.6793 | 0.9712 | 0.8405 |
| 19 | coarse | 0.8299 | 0.9919 | 0.9619 |
| 19 | sign | 0.6756 | 0.9662 | 0.8565 |
| 27 | coarse | 0.8332 | 0.9884 | 0.9553 |
| 27 | sign | 0.6826 | 0.9538 | 0.8421 |

runtime 427s
