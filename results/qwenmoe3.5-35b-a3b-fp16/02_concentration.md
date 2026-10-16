# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.943 of the attention mass (min head-avg 0.677); top-1% holds 0.779.
14 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 9.0% of tokens and captures 0.930 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6661 | 0.3038 | 0.9402 |
| 3 | 2% | 0.7303 | 0.3977 | 0.9529 |
| 3 | 5% | 0.8138 | 0.5474 | 0.9724 |
| 3 | 10% | 0.8745 | 0.6767 | 0.9853 |
| 3 | 20% | 0.9294 | 0.8032 | 0.9938 |
| 15 | 1% | 0.8542 | 0.5786 | 0.9503 |
| 15 | 2% | 0.9088 | 0.6864 | 0.9747 |
| 15 | 5% | 0.9564 | 0.8159 | 0.9939 |
| 15 | 10% | 0.9780 | 0.8959 | 0.9980 |
| 15 | 20% | 0.9910 | 0.9544 | 0.9994 |
| 27 | 1% | 0.8606 | 0.5886 | 0.9481 |
| 27 | 2% | 0.9174 | 0.6949 | 0.9726 |
| 27 | 5% | 0.9635 | 0.8205 | 0.9916 |
| 27 | 10% | 0.9823 | 0.8980 | 0.9964 |
| 27 | 20% | 0.9927 | 0.9544 | 0.9991 |
| 39 | 1% | 0.7347 | 0.5784 | 0.9998 |
| 39 | 2% | 0.8088 | 0.6622 | 0.9999 |
| 39 | 5% | 0.8902 | 0.7729 | 1.0000 |
| 39 | 10% | 0.9377 | 0.8527 | 1.0000 |
| 39 | 20% | 0.9717 | 0.9226 | 1.0000 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| code | 3 | 14 | 0.792 |
| needle10 | 3 | 1 | 0.780 |
| needle10 | 3 | 14 | 0.729 |
| needle10 | 3 | 15 | 0.714 |
| needle50 | 3 | 1 | 0.779 |
| needle50 | 3 | 14 | 0.728 |
| needle50 | 3 | 15 | 0.715 |
| needle90 | 3 | 1 | 0.780 |
| needle90 | 3 | 14 | 0.733 |
| needle90 | 3 | 15 | 0.729 |
| prose | 3 | 1 | 0.751 |
| prose | 3 | 10 | 0.799 |
| prose | 3 | 14 | 0.682 |
| prose | 3 | 15 | 0.677 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 4.1336 | 24.5240 | 0.8468 | 0.7404 |
| 0.9000 | 9.0498 | 40.9471 | 0.9299 | 0.8795 |
| 0.9500 | 15.6300 | 55.9045 | 0.9676 | 0.9456 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 860s
