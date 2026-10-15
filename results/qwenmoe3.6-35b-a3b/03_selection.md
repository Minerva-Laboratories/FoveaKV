# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.824); sign-plane(+RHT) 0.956 (overlap 0.667).
Without RHT: coarse 0.962, sign 0.902.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7884 | 0.9729 | 0.9624 | 0.8933 | 2.7551 | 0.0044 |
| coarse | RHT | 2% | 0.8039 | 0.9813 | 0.9684 | 0.9110 | 2.7551 | 0.0044 |
| coarse | RHT | 5% | 0.8239 | 0.9888 | 0.9760 | 0.9342 | 2.7551 | 0.0044 |
| coarse | RHT | 10% | 0.8411 | 0.9928 | 0.9820 | 0.9505 | 2.7551 | 0.0044 |
| coarse | RHT | 20% | 0.8656 | 0.9959 | 0.9881 | 0.9672 | 2.7551 | 0.0044 |
| coarse | no-RHT | 1% | 0.7009 | 0.9008 | 0.8168 | 0.4923 | 5.6819 | 0.0000 |
| coarse | no-RHT | 2% | 0.7271 | 0.9275 | 0.8546 | 0.6150 | 5.6819 | 0.0000 |
| coarse | no-RHT | 5% | 0.7626 | 0.9620 | 0.9161 | 0.7309 | 5.6819 | 0.0000 |
| coarse | no-RHT | 10% | 0.7899 | 0.9810 | 0.9603 | 0.8334 | 5.6819 | 0.0000 |
| coarse | no-RHT | 20% | 0.8243 | 0.9918 | 0.9838 | 0.9313 | 5.6819 | 0.0000 |
| sign | RHT | 1% | 0.6099 | 0.9000 | 0.8626 | 0.6779 | 5.3554 | 0.0000 |
| sign | RHT | 2% | 0.6344 | 0.9287 | 0.8909 | 0.7205 | 5.3554 | 0.0000 |
| sign | RHT | 5% | 0.6673 | 0.9562 | 0.9148 | 0.7800 | 5.3554 | 0.0000 |
| sign | RHT | 10% | 0.6983 | 0.9714 | 0.9343 | 0.8289 | 5.3554 | 0.0000 |
| sign | RHT | 20% | 0.7437 | 0.9834 | 0.9557 | 0.8830 | 5.3554 | 0.0000 |
| sign | no-RHT | 1% | 0.5057 | 0.7948 | 0.7049 | 0.3893 | 7.5519 | 0.0000 |
| sign | no-RHT | 2% | 0.5388 | 0.8441 | 0.7485 | 0.4747 | 7.5519 | 0.0000 |
| sign | no-RHT | 5% | 0.5864 | 0.9016 | 0.8198 | 0.6095 | 7.5519 | 0.0000 |
| sign | no-RHT | 10% | 0.6272 | 0.9379 | 0.8794 | 0.7196 | 7.5519 | 0.0000 |
| sign | no-RHT | 20% | 0.6827 | 0.9666 | 0.9360 | 0.8210 | 7.5519 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7962 | 0.9812 | 0.9342 |
| 3 | sign | 0.6195 | 0.9314 | 0.7800 |
| 15 | coarse | 0.8298 | 0.9928 | 0.9596 |
| 15 | sign | 0.6763 | 0.9696 | 0.8466 |
| 27 | coarse | 0.8407 | 0.9938 | 0.9643 |
| 27 | sign | 0.6979 | 0.9732 | 0.8609 |
| 39 | coarse | 0.8288 | 0.9874 | 0.9654 |
| 39 | sign | 0.6754 | 0.9507 | 0.8727 |

runtime 903s
