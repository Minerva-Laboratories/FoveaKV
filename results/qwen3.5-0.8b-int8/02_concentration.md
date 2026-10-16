# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.957 of the attention mass (min head-avg 0.749); top-1% holds 0.801.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.0% of tokens and captures 0.930 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6898 | 0.4442 | 0.8596 |
| 3 | 2% | 0.7599 | 0.5236 | 0.9037 |
| 3 | 5% | 0.8471 | 0.6464 | 0.9458 |
| 3 | 10% | 0.9051 | 0.7492 | 0.9754 |
| 3 | 20% | 0.9522 | 0.8526 | 0.9909 |
| 7 | 1% | 0.8723 | 0.5774 | 0.9719 |
| 7 | 2% | 0.9172 | 0.6933 | 0.9816 |
| 7 | 5% | 0.9594 | 0.8318 | 0.9903 |
| 7 | 10% | 0.9796 | 0.9122 | 0.9949 |
| 7 | 20% | 0.9918 | 0.9645 | 0.9979 |
| 11 | 1% | 0.8518 | 0.6596 | 0.9331 |
| 11 | 2% | 0.9095 | 0.7688 | 0.9601 |
| 11 | 5% | 0.9597 | 0.8847 | 0.9839 |
| 11 | 10% | 0.9812 | 0.9431 | 0.9939 |
| 11 | 20% | 0.9930 | 0.9779 | 0.9981 |
| 15 | 1% | 0.8389 | 0.6493 | 0.9192 |
| 15 | 2% | 0.8980 | 0.7334 | 0.9671 |
| 15 | 5% | 0.9496 | 0.8367 | 0.9917 |
| 15 | 10% | 0.9735 | 0.9007 | 0.9963 |
| 15 | 20% | 0.9885 | 0.9497 | 0.9992 |
| 19 | 1% | 0.7844 | 0.6234 | 0.8764 |
| 19 | 2% | 0.8538 | 0.7069 | 0.9142 |
| 19 | 5% | 0.9237 | 0.8177 | 0.9643 |
| 19 | 10% | 0.9601 | 0.8919 | 0.9853 |
| 19 | 20% | 0.9834 | 0.9443 | 0.9960 |
| 23 | 1% | 0.7663 | 0.5702 | 0.9960 |
| 23 | 2% | 0.8301 | 0.6562 | 0.9971 |
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
| 0.8000 | 3.0218 | 16.5575 | 0.8475 | 0.7116 |
| 0.9000 | 6.9852 | 30.6474 | 0.9301 | 0.8632 |
| 0.9500 | 12.6050 | 45.1519 | 0.9678 | 0.9365 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 723s
