# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.945 of the attention mass (min head-avg 0.618); top-1% holds 0.783.
14 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 8.3% of tokens and captures 0.927 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6236 | 0.2773 | 0.8737 |
| 3 | 2% | 0.7014 | 0.3575 | 0.9123 |
| 3 | 5% | 0.8018 | 0.4931 | 0.9491 |
| 3 | 10% | 0.8726 | 0.6180 | 0.9753 |
| 3 | 20% | 0.9334 | 0.7560 | 0.9908 |
| 11 | 1% | 0.8744 | 0.6675 | 0.9622 |
| 11 | 2% | 0.9259 | 0.7695 | 0.9807 |
| 11 | 5% | 0.9674 | 0.8758 | 0.9936 |
| 11 | 10% | 0.9845 | 0.9338 | 0.9976 |
| 11 | 20% | 0.9940 | 0.9726 | 0.9992 |
| 19 | 1% | 0.8638 | 0.6462 | 0.9504 |
| 19 | 2% | 0.9122 | 0.7175 | 0.9751 |
| 19 | 5% | 0.9552 | 0.8117 | 0.9936 |
| 19 | 10% | 0.9758 | 0.8783 | 0.9978 |
| 19 | 20% | 0.9892 | 0.9357 | 0.9996 |
| 27 | 1% | 0.7700 | 0.5499 | 0.9077 |
| 27 | 2% | 0.8380 | 0.6447 | 0.9379 |
| 27 | 5% | 0.9085 | 0.7586 | 0.9687 |
| 27 | 10% | 0.9479 | 0.8379 | 0.9855 |
| 27 | 20% | 0.9756 | 0.9110 | 0.9946 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 4 | 0.791 |
| needle10 | 3 | 7 | 0.786 |
| needle10 | 3 | 14 | 0.640 |
| needle50 | 3 | 4 | 0.791 |
| needle50 | 3 | 7 | 0.786 |
| needle50 | 3 | 14 | 0.640 |
| needle90 | 3 | 4 | 0.793 |
| needle90 | 3 | 7 | 0.787 |
| needle90 | 3 | 14 | 0.642 |
| prose | 3 | 4 | 0.763 |
| prose | 3 | 5 | 0.781 |
| prose | 3 | 7 | 0.761 |
| prose | 3 | 12 | 0.775 |
| prose | 3 | 14 | 0.618 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.8285 | 24.8510 | 0.8404 | 0.7265 |
| 0.9000 | 8.3394 | 41.3309 | 0.9272 | 0.8713 |
| 0.9500 | 14.4374 | 56.3917 | 0.9665 | 0.9414 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 978s
