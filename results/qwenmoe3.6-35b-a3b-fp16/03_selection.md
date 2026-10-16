# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.989 (overlap 0.824); sign-plane(+RHT) 0.956 (overlap 0.667).
Without RHT: coarse 0.960, sign 0.899.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7884 | 0.9729 | 0.9626 | 0.8948 | 2.7566 | 0.0044 |
| coarse | RHT | 2% | 0.8040 | 0.9813 | 0.9685 | 0.9112 | 2.7566 | 0.0044 |
| coarse | RHT | 5% | 0.8239 | 0.9888 | 0.9760 | 0.9339 | 2.7566 | 0.0044 |
| coarse | RHT | 10% | 0.8411 | 0.9928 | 0.9820 | 0.9498 | 2.7566 | 0.0044 |
| coarse | RHT | 20% | 0.8656 | 0.9959 | 0.9881 | 0.9669 | 2.7566 | 0.0044 |
| coarse | no-RHT | 1% | 0.7007 | 0.8993 | 0.8167 | 0.4935 | 5.7114 | 0.0000 |
| coarse | no-RHT | 2% | 0.7268 | 0.9254 | 0.8554 | 0.6156 | 5.7114 | 0.0000 |
| coarse | no-RHT | 5% | 0.7624 | 0.9596 | 0.9160 | 0.7182 | 5.7114 | 0.0000 |
| coarse | no-RHT | 10% | 0.7897 | 0.9799 | 0.9595 | 0.7895 | 5.7114 | 0.0000 |
| coarse | no-RHT | 20% | 0.8241 | 0.9916 | 0.9846 | 0.9243 | 5.7114 | 0.0000 |
| sign | RHT | 1% | 0.6100 | 0.9002 | 0.8638 | 0.6762 | 5.3593 | 0.0000 |
| sign | RHT | 2% | 0.6344 | 0.9287 | 0.8905 | 0.7214 | 5.3593 | 0.0000 |
| sign | RHT | 5% | 0.6672 | 0.9562 | 0.9147 | 0.7803 | 5.3593 | 0.0000 |
| sign | RHT | 10% | 0.6982 | 0.9714 | 0.9343 | 0.8303 | 5.3593 | 0.0000 |
| sign | RHT | 20% | 0.7437 | 0.9834 | 0.9556 | 0.8825 | 5.3593 | 0.0000 |
| sign | no-RHT | 1% | 0.5051 | 0.7934 | 0.7057 | 0.3892 | 7.5778 | 0.0000 |
| sign | no-RHT | 2% | 0.5382 | 0.8419 | 0.7491 | 0.4784 | 7.5778 | 0.0000 |
| sign | no-RHT | 5% | 0.5859 | 0.8986 | 0.8200 | 0.6143 | 7.5778 | 0.0000 |
| sign | no-RHT | 10% | 0.6268 | 0.9349 | 0.8800 | 0.7033 | 7.5778 | 0.0000 |
| sign | no-RHT | 20% | 0.6824 | 0.9647 | 0.9355 | 0.7813 | 7.5778 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.7962 | 0.9812 | 0.9339 |
| 3 | sign | 0.6192 | 0.9313 | 0.7803 |
| 15 | coarse | 0.8297 | 0.9928 | 0.9599 |
| 15 | sign | 0.6762 | 0.9696 | 0.8486 |
| 27 | coarse | 0.8408 | 0.9938 | 0.9644 |
| 27 | sign | 0.6980 | 0.9733 | 0.8592 |
| 39 | coarse | 0.8288 | 0.9874 | 0.9652 |
| 39 | sign | 0.6753 | 0.9506 | 0.8721 |

runtime 906s
