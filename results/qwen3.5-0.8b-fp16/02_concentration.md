# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.957 of the attention mass (min head-avg 0.749); top-1% holds 0.801.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.0% of tokens and captures 0.930 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6899 | 0.4443 | 0.8594 |
| 3 | 2% | 0.7599 | 0.5237 | 0.9036 |
| 3 | 5% | 0.8470 | 0.6464 | 0.9457 |
| 3 | 10% | 0.9051 | 0.7492 | 0.9753 |
| 3 | 20% | 0.9522 | 0.8526 | 0.9909 |
| 7 | 1% | 0.8723 | 0.5775 | 0.9718 |
| 7 | 2% | 0.9172 | 0.6934 | 0.9816 |
| 7 | 5% | 0.9594 | 0.8318 | 0.9903 |
| 7 | 10% | 0.9796 | 0.9123 | 0.9949 |
| 7 | 20% | 0.9918 | 0.9646 | 0.9979 |
| 11 | 1% | 0.8520 | 0.6599 | 0.9331 |
| 11 | 2% | 0.9096 | 0.7690 | 0.9601 |
| 11 | 5% | 0.9597 | 0.8848 | 0.9840 |
| 11 | 10% | 0.9812 | 0.9432 | 0.9939 |
| 11 | 20% | 0.9930 | 0.9779 | 0.9981 |
| 15 | 1% | 0.8388 | 0.6490 | 0.9192 |
| 15 | 2% | 0.8979 | 0.7331 | 0.9671 |
| 15 | 5% | 0.9495 | 0.8365 | 0.9917 |
| 15 | 10% | 0.9734 | 0.9007 | 0.9963 |
| 15 | 20% | 0.9885 | 0.9497 | 0.9992 |
| 19 | 1% | 0.7843 | 0.6233 | 0.8766 |
| 19 | 2% | 0.8538 | 0.7068 | 0.9143 |
| 19 | 5% | 0.9237 | 0.8177 | 0.9642 |
| 19 | 10% | 0.9600 | 0.8918 | 0.9853 |
| 19 | 20% | 0.9834 | 0.9442 | 0.9960 |
| 23 | 1% | 0.7663 | 0.5701 | 0.9960 |
| 23 | 2% | 0.8301 | 0.6561 | 0.9971 |
| 23 | 5% | 0.9026 | 0.7723 | 0.9983 |
| 23 | 10% | 0.9453 | 0.8553 | 0.9990 |
| 23 | 20% | 0.9755 | 0.9262 | 0.9995 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 4 | 0.771 |
| needle50 | 3 | 4 | 0.771 |
| needle90 | 3 | 4 | 0.772 |
| prose | 3 | 4 | 0.749 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.0166 | 16.5961 | 0.8470 | 0.7021 |
| 0.9000 | 6.9753 | 30.5889 | 0.9299 | 0.8575 |
| 0.9500 | 12.5901 | 45.0575 | 0.9677 | 0.9347 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 727s
