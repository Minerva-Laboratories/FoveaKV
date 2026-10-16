# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.952 of the attention mass (min head-avg 0.735); top-1% holds 0.786.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.7% of tokens and captures 0.931 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.7110 | 0.4015 | 0.8707 |
| 3 | 2% | 0.7790 | 0.4882 | 0.9108 |
| 3 | 5% | 0.8605 | 0.6227 | 0.9558 |
| 3 | 10% | 0.9138 | 0.7350 | 0.9800 |
| 3 | 20% | 0.9567 | 0.8471 | 0.9921 |
| 7 | 1% | 0.8800 | 0.6246 | 0.9810 |
| 7 | 2% | 0.9236 | 0.7377 | 0.9884 |
| 7 | 5% | 0.9626 | 0.8605 | 0.9944 |
| 7 | 10% | 0.9809 | 0.9268 | 0.9973 |
| 7 | 20% | 0.9920 | 0.9701 | 0.9990 |
| 11 | 1% | 0.8452 | 0.7240 | 0.9211 |
| 11 | 2% | 0.9019 | 0.8165 | 0.9499 |
| 11 | 5% | 0.9524 | 0.9046 | 0.9799 |
| 11 | 10% | 0.9759 | 0.9471 | 0.9916 |
| 11 | 20% | 0.9901 | 0.9761 | 0.9978 |
| 15 | 1% | 0.8269 | 0.6867 | 0.9076 |
| 15 | 2% | 0.8842 | 0.7594 | 0.9562 |
| 15 | 5% | 0.9390 | 0.8495 | 0.9850 |
| 15 | 10% | 0.9668 | 0.9093 | 0.9928 |
| 15 | 20% | 0.9853 | 0.9565 | 0.9981 |
| 19 | 1% | 0.7130 | 0.5839 | 0.8837 |
| 19 | 2% | 0.7944 | 0.6640 | 0.9200 |
| 19 | 5% | 0.8848 | 0.7801 | 0.9539 |
| 19 | 10% | 0.9365 | 0.8649 | 0.9734 |
| 19 | 20% | 0.9726 | 0.9313 | 0.9910 |
| 23 | 1% | 0.7416 | 0.5738 | 0.9905 |
| 23 | 2% | 0.8109 | 0.6703 | 0.9930 |
| 23 | 5% | 0.8906 | 0.7950 | 0.9959 |
| 23 | 10% | 0.9380 | 0.8760 | 0.9976 |
| 23 | 20% | 0.9719 | 0.9402 | 0.9988 |

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
| 0.8000 | 3.3320 | 16.5900 | 0.8496 | 0.7417 |
| 0.9000 | 7.6603 | 30.5552 | 0.9313 | 0.8822 |
| 0.9500 | 13.8089 | 45.0912 | 0.9684 | 0.9473 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 693s
