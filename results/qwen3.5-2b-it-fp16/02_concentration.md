# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.958 of the attention mass (min head-avg 0.743); top-1% holds 0.800.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.0% of tokens and captures 0.930 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.7055 | 0.4083 | 0.8701 |
| 3 | 2% | 0.7732 | 0.4968 | 0.9117 |
| 3 | 5% | 0.8558 | 0.6317 | 0.9588 |
| 3 | 10% | 0.9105 | 0.7429 | 0.9824 |
| 3 | 20% | 0.9548 | 0.8525 | 0.9933 |
| 7 | 1% | 0.8660 | 0.5923 | 0.9673 |
| 7 | 2% | 0.9149 | 0.7107 | 0.9800 |
| 7 | 5% | 0.9598 | 0.8471 | 0.9904 |
| 7 | 10% | 0.9808 | 0.9239 | 0.9953 |
| 7 | 20% | 0.9927 | 0.9720 | 0.9983 |
| 11 | 1% | 0.8475 | 0.6890 | 0.9270 |
| 11 | 2% | 0.9048 | 0.7913 | 0.9550 |
| 11 | 5% | 0.9557 | 0.8962 | 0.9821 |
| 11 | 10% | 0.9786 | 0.9471 | 0.9936 |
| 11 | 20% | 0.9917 | 0.9779 | 0.9988 |
| 15 | 1% | 0.8397 | 0.6494 | 0.9302 |
| 15 | 2% | 0.8974 | 0.7342 | 0.9689 |
| 15 | 5% | 0.9489 | 0.8361 | 0.9923 |
| 15 | 10% | 0.9727 | 0.8977 | 0.9968 |
| 15 | 20% | 0.9880 | 0.9482 | 0.9997 |
| 19 | 1% | 0.7777 | 0.6075 | 0.8761 |
| 19 | 2% | 0.8475 | 0.6952 | 0.9170 |
| 19 | 5% | 0.9188 | 0.7975 | 0.9662 |
| 19 | 10% | 0.9567 | 0.8677 | 0.9861 |
| 19 | 20% | 0.9816 | 0.9323 | 0.9959 |
| 23 | 1% | 0.7644 | 0.6060 | 0.9945 |
| 23 | 2% | 0.8296 | 0.6993 | 0.9961 |
| 23 | 5% | 0.9034 | 0.8149 | 0.9978 |
| 23 | 10% | 0.9464 | 0.8838 | 0.9988 |
| 23 | 20% | 0.9763 | 0.9402 | 0.9995 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 4 | 0.760 |
| needle50 | 3 | 4 | 0.759 |
| needle90 | 3 | 4 | 0.761 |
| prose | 3 | 4 | 0.743 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.0199 | 16.3782 | 0.8481 | 0.7336 |
| 0.9000 | 6.9889 | 30.6310 | 0.9304 | 0.8738 |
| 0.9500 | 12.6418 | 45.1841 | 0.9679 | 0.9414 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 721s
