# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.954 of the attention mass (min head-avg 0.750); top-1% holds 0.793.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.5% of tokens and captures 0.931 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6908 | 0.4447 | 0.8418 |
| 3 | 2% | 0.7590 | 0.5241 | 0.8758 |
| 3 | 5% | 0.8449 | 0.6468 | 0.9189 |
| 3 | 10% | 0.9030 | 0.7495 | 0.9561 |
| 3 | 20% | 0.9508 | 0.8528 | 0.9814 |
| 7 | 1% | 0.8665 | 0.5052 | 0.9578 |
| 7 | 2% | 0.9110 | 0.6125 | 0.9707 |
| 7 | 5% | 0.9540 | 0.7569 | 0.9858 |
| 7 | 10% | 0.9758 | 0.8566 | 0.9936 |
| 7 | 20% | 0.9899 | 0.9357 | 0.9977 |
| 11 | 1% | 0.8433 | 0.5779 | 0.9331 |
| 11 | 2% | 0.9010 | 0.6846 | 0.9601 |
| 11 | 5% | 0.9533 | 0.8154 | 0.9839 |
| 11 | 10% | 0.9772 | 0.8976 | 0.9939 |
| 11 | 20% | 0.9913 | 0.9573 | 0.9981 |
| 15 | 1% | 0.8222 | 0.6498 | 0.9191 |
| 15 | 2% | 0.8833 | 0.7338 | 0.9669 |
| 15 | 5% | 0.9398 | 0.8370 | 0.9917 |
| 15 | 10% | 0.9678 | 0.9009 | 0.9962 |
| 15 | 20% | 0.9862 | 0.9499 | 0.9992 |
| 19 | 1% | 0.7710 | 0.5596 | 0.8495 |
| 19 | 2% | 0.8423 | 0.6368 | 0.9124 |
| 19 | 5% | 0.9164 | 0.7516 | 0.9643 |
| 19 | 10% | 0.9561 | 0.8420 | 0.9853 |
| 19 | 20% | 0.9818 | 0.9219 | 0.9961 |
| 23 | 1% | 0.7616 | 0.5702 | 0.9960 |
| 23 | 2% | 0.8241 | 0.6562 | 0.9971 |
| 23 | 5% | 0.8971 | 0.7723 | 0.9983 |
| 23 | 10% | 0.9415 | 0.8553 | 0.9990 |
| 23 | 20% | 0.9737 | 0.9262 | 0.9995 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 4 | 0.771 |
| needle50 | 3 | 4 | 0.771 |
| needle90 | 3 | 4 | 0.771 |
| prose | 3 | 4 | 0.750 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.2585 | 16.4737 | 0.8498 | 0.7126 |
| 0.9000 | 7.4871 | 30.4872 | 0.9310 | 0.8635 |
| 0.9500 | 13.4183 | 45.0445 | 0.9681 | 0.9376 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 296s
