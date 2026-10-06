# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.948 of the attention mass (min head-avg 0.735); top-1% holds 0.776.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 8.2% of tokens and captures 0.932 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.7032 | 0.4013 | 0.8439 |
| 3 | 2% | 0.7712 | 0.4880 | 0.8833 |
| 3 | 5% | 0.8542 | 0.6224 | 0.9357 |
| 3 | 10% | 0.9094 | 0.7347 | 0.9662 |
| 3 | 20% | 0.9544 | 0.8470 | 0.9862 |
| 7 | 1% | 0.8727 | 0.5275 | 0.9667 |
| 7 | 2% | 0.9165 | 0.6346 | 0.9812 |
| 7 | 5% | 0.9570 | 0.7749 | 0.9917 |
| 7 | 10% | 0.9772 | 0.8690 | 0.9961 |
| 7 | 20% | 0.9902 | 0.9418 | 0.9985 |
| 11 | 1% | 0.8363 | 0.6193 | 0.9150 |
| 11 | 2% | 0.8938 | 0.7229 | 0.9489 |
| 11 | 5% | 0.9470 | 0.8436 | 0.9800 |
| 11 | 10% | 0.9728 | 0.9118 | 0.9916 |
| 11 | 20% | 0.9888 | 0.9592 | 0.9978 |
| 15 | 1% | 0.8107 | 0.6838 | 0.9078 |
| 15 | 2% | 0.8708 | 0.7593 | 0.9563 |
| 15 | 5% | 0.9306 | 0.8495 | 0.9850 |
| 15 | 10% | 0.9621 | 0.9092 | 0.9928 |
| 15 | 20% | 0.9834 | 0.9565 | 0.9981 |
| 19 | 1% | 0.6985 | 0.5236 | 0.7786 |
| 19 | 2% | 0.7827 | 0.6010 | 0.8619 |
| 19 | 5% | 0.8779 | 0.7190 | 0.9388 |
| 19 | 10% | 0.9329 | 0.8147 | 0.9728 |
| 19 | 20% | 0.9711 | 0.9042 | 0.9910 |
| 23 | 1% | 0.7330 | 0.5735 | 0.9905 |
| 23 | 2% | 0.8029 | 0.6700 | 0.9930 |
| 23 | 5% | 0.8850 | 0.7890 | 0.9959 |
| 23 | 10% | 0.9346 | 0.8694 | 0.9976 |
| 23 | 20% | 0.9705 | 0.9366 | 0.9988 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 4 | 0.764 |
| needle50 | 3 | 4 | 0.764 |
| needle90 | 3 | 4 | 0.766 |
| prose | 3 | 4 | 0.735 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.5960 | 16.5620 | 0.8516 | 0.7421 |
| 0.9000 | 8.1965 | 30.7143 | 0.9321 | 0.8813 |
| 0.9500 | 14.6579 | 45.2280 | 0.9687 | 0.9481 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 396s
