# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = 0.988 (overlap 0.825); sign-plane(+RHT) 0.953 (overlap 0.669).
Without RHT: coarse 0.970, sign 0.901.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): 0 violations / 3276800 (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over 5 corpora × 4 layers × all query heads × 512 queries)
| estimator | rotation | k | overlap | m_H/m_opt mean | worst layer-mean | worst p5 | median η | median e^-2η |
|---|---|---|---|---|---|---|---|---|
| coarse | RHT | 1% | 0.7880 | 0.9713 | 0.9526 | 0.8859 | 2.8141 | 0.0037 |
| coarse | RHT | 2% | 0.8032 | 0.9800 | 0.9606 | 0.9007 | 2.8141 | 0.0037 |
| coarse | RHT | 5% | 0.8249 | 0.9879 | 0.9697 | 0.9215 | 2.8141 | 0.0037 |
| coarse | RHT | 10% | 0.8444 | 0.9923 | 0.9774 | 0.9386 | 2.8141 | 0.0037 |
| coarse | RHT | 20% | 0.8702 | 0.9957 | 0.9855 | 0.9578 | 2.8141 | 0.0037 |
| coarse | no-RHT | 1% | 0.7159 | 0.9061 | 0.7936 | 0.5008 | 5.8195 | 0.0000 |
| coarse | no-RHT | 2% | 0.7457 | 0.9381 | 0.8572 | 0.6081 | 5.8195 | 0.0000 |
| coarse | no-RHT | 5% | 0.7810 | 0.9697 | 0.9392 | 0.7917 | 5.8195 | 0.0000 |
| coarse | no-RHT | 10% | 0.8071 | 0.9843 | 0.9704 | 0.8646 | 5.8195 | 0.0000 |
| coarse | no-RHT | 20% | 0.8391 | 0.9930 | 0.9821 | 0.9525 | 5.8195 | 0.0000 |
| sign | RHT | 1% | 0.6104 | 0.8958 | 0.8316 | 0.6488 | 5.6784 | 0.0000 |
| sign | RHT | 2% | 0.6332 | 0.9248 | 0.8630 | 0.6927 | 5.6784 | 0.0000 |
| sign | RHT | 5% | 0.6689 | 0.9534 | 0.8928 | 0.7507 | 5.6784 | 0.0000 |
| sign | RHT | 10% | 0.7036 | 0.9696 | 0.9177 | 0.7992 | 5.6784 | 0.0000 |
| sign | RHT | 20% | 0.7513 | 0.9827 | 0.9455 | 0.8578 | 5.6784 | 0.0000 |
| sign | no-RHT | 1% | 0.5191 | 0.7853 | 0.6808 | 0.3716 | 7.6697 | 0.0000 |
| sign | no-RHT | 2% | 0.5575 | 0.8389 | 0.7483 | 0.4514 | 7.6697 | 0.0000 |
| sign | no-RHT | 5% | 0.6101 | 0.9009 | 0.8453 | 0.5884 | 7.6697 | 0.0000 |
| sign | no-RHT | 10% | 0.6529 | 0.9382 | 0.8977 | 0.6973 | 7.6697 | 0.0000 |
| sign | no-RHT | 20% | 0.7074 | 0.9669 | 0.9361 | 0.8095 | 7.6697 | 0.0000 |

## Per layer at k=5% (+RHT)
| layer | estimator | overlap | m_H/m_opt | worst p5 |
|---|---|---|---|---|
| 3 | coarse | 0.8009 | 0.9767 | 0.9215 |
| 3 | sign | 0.6283 | 0.9158 | 0.7507 |
| 11 | coarse | 0.8384 | 0.9938 | 0.9651 |
| 11 | sign | 0.6907 | 0.9729 | 0.8614 |
| 19 | coarse | 0.8315 | 0.9926 | 0.9645 |
| 19 | sign | 0.6783 | 0.9689 | 0.8670 |
| 27 | coarse | 0.8289 | 0.9886 | 0.9558 |
| 27 | sign | 0.6782 | 0.9561 | 0.8472 |

runtime 918s
