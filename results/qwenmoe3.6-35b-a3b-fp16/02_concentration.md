# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.945 of the attention mass (min head-avg 0.689); top-1% holds 0.785.
13 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 8.8% of tokens and captures 0.930 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6767 | 0.3146 | 0.9308 |
| 3 | 2% | 0.7406 | 0.4110 | 0.9497 |
| 3 | 5% | 0.8227 | 0.5615 | 0.9739 |
| 3 | 10% | 0.8813 | 0.6890 | 0.9859 |
| 3 | 20% | 0.9338 | 0.8178 | 0.9940 |
| 15 | 1% | 0.8602 | 0.5719 | 0.9545 |
| 15 | 2% | 0.9131 | 0.6796 | 0.9773 |
| 15 | 5% | 0.9588 | 0.8103 | 0.9946 |
| 15 | 10% | 0.9793 | 0.8924 | 0.9982 |
| 15 | 20% | 0.9915 | 0.9530 | 0.9994 |
| 27 | 1% | 0.8728 | 0.6251 | 0.9483 |
| 27 | 2% | 0.9245 | 0.7223 | 0.9734 |
| 27 | 5% | 0.9662 | 0.8358 | 0.9920 |
| 27 | 10% | 0.9834 | 0.9060 | 0.9968 |
| 27 | 20% | 0.9933 | 0.9578 | 0.9989 |
| 39 | 1% | 0.7315 | 0.5698 | 0.9998 |
| 39 | 2% | 0.8054 | 0.6537 | 0.9999 |
| 39 | 5% | 0.8878 | 0.7653 | 0.9999 |
| 39 | 10% | 0.9361 | 0.8471 | 1.0000 |
| 39 | 20% | 0.9709 | 0.9197 | 1.0000 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| code | 3 | 14 | 0.793 |
| needle10 | 3 | 1 | 0.794 |
| needle10 | 3 | 14 | 0.750 |
| needle10 | 3 | 15 | 0.729 |
| needle50 | 3 | 1 | 0.794 |
| needle50 | 3 | 14 | 0.750 |
| needle50 | 3 | 15 | 0.734 |
| needle90 | 3 | 1 | 0.796 |
| needle90 | 3 | 14 | 0.754 |
| needle90 | 3 | 15 | 0.744 |
| prose | 3 | 1 | 0.764 |
| prose | 3 | 14 | 0.710 |
| prose | 3 | 15 | 0.689 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.9497 | 23.4872 | 0.8467 | 0.7301 |
| 0.9000 | 8.7975 | 39.8481 | 0.9299 | 0.8739 |
| 0.9500 | 15.3885 | 54.9864 | 0.9676 | 0.9431 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 906s
