# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.962 of the attention mass (min head-avg 0.770); top-1% holds 0.861.
8 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 3.4% of tokens and captures 0.709 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 0 | 1% | 0.7909 | 0.3542 | 0.9996 |
| 0 | 2% | 0.8421 | 0.4640 | 0.9998 |
| 0 | 5% | 0.9050 | 0.6329 | 1.0000 |
| 0 | 10% | 0.9444 | 0.7702 | 1.0000 |
| 0 | 20% | 0.9753 | 0.8932 | 1.0000 |
| 9 | 1% | 0.8599 | 0.4412 | 0.9887 |
| 9 | 2% | 0.8951 | 0.5440 | 0.9930 |
| 9 | 5% | 0.9369 | 0.6997 | 0.9970 |
| 9 | 10% | 0.9631 | 0.8164 | 0.9987 |
| 9 | 20% | 0.9830 | 0.9143 | 0.9996 |
| 18 | 1% | 0.8320 | 0.4561 | 0.9825 |
| 18 | 2% | 0.8719 | 0.5569 | 0.9884 |
| 18 | 5% | 0.9209 | 0.7075 | 0.9941 |
| 18 | 10% | 0.9526 | 0.8192 | 0.9969 |
| 18 | 20% | 0.9773 | 0.9082 | 0.9988 |
| 27 | 1% | 0.9623 | 0.8425 | 0.9993 |
| 27 | 2% | 0.9715 | 0.8713 | 0.9995 |
| 27 | 5% | 0.9826 | 0.9082 | 0.9998 |
| 27 | 10% | 0.9898 | 0.9362 | 0.9999 |
| 27 | 20% | 0.9953 | 0.9636 | 1.0000 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 0 | 6 | 0.781 |
| needle10 | 0 | 11 | 0.781 |
| needle50 | 0 | 6 | 0.781 |
| needle50 | 0 | 11 | 0.781 |
| needle90 | 0 | 6 | 0.781 |
| needle90 | 0 | 11 | 0.781 |
| prose | 0 | 6 | 0.770 |
| prose | 0 | 11 | 0.789 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 1.3742 | 17.0499 | 0.6454 | 0.0000 |
| 0.9000 | 3.3602 | 30.1500 | 0.7092 | 0.0000 |
| 0.9500 | 6.4080 | 43.4887 | 0.7500 | 0.0000 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 399s
