# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.958 of the attention mass (min head-avg 0.743); top-1% holds 0.800.
4 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.0% of tokens and captures 0.930 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.7056 | 0.4083 | 0.8701 |
| 3 | 2% | 0.7732 | 0.4968 | 0.9117 |
| 3 | 5% | 0.8559 | 0.6317 | 0.9588 |
| 3 | 10% | 0.9105 | 0.7429 | 0.9824 |
| 3 | 20% | 0.9548 | 0.8525 | 0.9933 |
| 7 | 1% | 0.8660 | 0.5924 | 0.9673 |
| 7 | 2% | 0.9148 | 0.7108 | 0.9800 |
| 7 | 5% | 0.9598 | 0.8472 | 0.9904 |
| 7 | 10% | 0.9808 | 0.9240 | 0.9953 |
| 7 | 20% | 0.9927 | 0.9720 | 0.9983 |
| 11 | 1% | 0.8474 | 0.6888 | 0.9270 |
| 11 | 2% | 0.9048 | 0.7913 | 0.9550 |
| 11 | 5% | 0.9557 | 0.8962 | 0.9821 |
| 11 | 10% | 0.9786 | 0.9472 | 0.9936 |
| 11 | 20% | 0.9917 | 0.9779 | 0.9988 |
| 15 | 1% | 0.8397 | 0.6491 | 0.9302 |
| 15 | 2% | 0.8974 | 0.7339 | 0.9689 |
| 15 | 5% | 0.9489 | 0.8361 | 0.9923 |
| 15 | 10% | 0.9727 | 0.8977 | 0.9968 |
| 15 | 20% | 0.9880 | 0.9482 | 0.9997 |
| 19 | 1% | 0.7778 | 0.6077 | 0.8761 |
| 19 | 2% | 0.8476 | 0.6954 | 0.9170 |
| 19 | 5% | 0.9188 | 0.7975 | 0.9662 |
| 19 | 10% | 0.9567 | 0.8677 | 0.9861 |
| 19 | 20% | 0.9816 | 0.9323 | 0.9959 |
| 23 | 1% | 0.7644 | 0.6063 | 0.9945 |
| 23 | 2% | 0.8296 | 0.6996 | 0.9961 |
| 23 | 5% | 0.9035 | 0.8151 | 0.9978 |
| 23 | 10% | 0.9464 | 0.8838 | 0.9988 |
| 23 | 20% | 0.9764 | 0.9403 | 0.9995 |

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
| 0.8000 | 3.0182 | 16.4708 | 0.8480 | 0.7301 |
| 0.9000 | 6.9872 | 30.6350 | 0.9304 | 0.8713 |
| 0.9500 | 12.6410 | 45.2157 | 0.9679 | 0.9405 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 731s
