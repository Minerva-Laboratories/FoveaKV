# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.945 of the attention mass (min head-avg 0.618); top-1% holds 0.783.
14 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 8.4% of tokens and captures 0.927 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6235 | 0.2772 | 0.8736 |
| 3 | 2% | 0.7014 | 0.3574 | 0.9122 |
| 3 | 5% | 0.8018 | 0.4930 | 0.9491 |
| 3 | 10% | 0.8726 | 0.6179 | 0.9753 |
| 3 | 20% | 0.9333 | 0.7560 | 0.9908 |
| 11 | 1% | 0.8743 | 0.6676 | 0.9622 |
| 11 | 2% | 0.9259 | 0.7697 | 0.9807 |
| 11 | 5% | 0.9674 | 0.8759 | 0.9936 |
| 11 | 10% | 0.9845 | 0.9338 | 0.9976 |
| 11 | 20% | 0.9940 | 0.9726 | 0.9992 |
| 19 | 1% | 0.8634 | 0.6448 | 0.9505 |
| 19 | 2% | 0.9120 | 0.7163 | 0.9751 |
| 19 | 5% | 0.9550 | 0.8109 | 0.9936 |
| 19 | 10% | 0.9757 | 0.8778 | 0.9978 |
| 19 | 20% | 0.9892 | 0.9355 | 0.9996 |
| 27 | 1% | 0.7701 | 0.5501 | 0.9078 |
| 27 | 2% | 0.8381 | 0.6448 | 0.9380 |
| 27 | 5% | 0.9086 | 0.7582 | 0.9688 |
| 27 | 10% | 0.9479 | 0.8376 | 0.9855 |
| 27 | 20% | 0.9756 | 0.9109 | 0.9946 |

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
| 0.8000 | 3.8367 | 24.8401 | 0.8406 | 0.7279 |
| 0.9000 | 8.3552 | 41.3991 | 0.9273 | 0.8729 |
| 0.9500 | 14.4600 | 56.4062 | 0.9666 | 0.9421 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 918s
