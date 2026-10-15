# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.948 of the attention mass (min head-avg 0.603); top-1% holds 0.792.
10 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 8.0% of tokens and captures 0.928 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6439 | 0.2648 | 0.8779 |
| 3 | 2% | 0.7206 | 0.3431 | 0.9183 |
| 3 | 5% | 0.8159 | 0.4771 | 0.9553 |
| 3 | 10% | 0.8817 | 0.6027 | 0.9750 |
| 3 | 20% | 0.9379 | 0.7443 | 0.9897 |
| 11 | 1% | 0.8734 | 0.6533 | 0.9613 |
| 11 | 2% | 0.9245 | 0.7600 | 0.9797 |
| 11 | 5% | 0.9659 | 0.8767 | 0.9939 |
| 11 | 10% | 0.9834 | 0.9359 | 0.9978 |
| 11 | 20% | 0.9934 | 0.9732 | 0.9993 |
| 19 | 1% | 0.8606 | 0.6091 | 0.9543 |
| 19 | 2% | 0.9123 | 0.6899 | 0.9784 |
| 19 | 5% | 0.9560 | 0.7927 | 0.9943 |
| 19 | 10% | 0.9763 | 0.8650 | 0.9977 |
| 19 | 20% | 0.9894 | 0.9277 | 0.9991 |
| 27 | 1% | 0.7889 | 0.5947 | 0.9071 |
| 27 | 2% | 0.8507 | 0.6880 | 0.9381 |
| 27 | 5% | 0.9155 | 0.7873 | 0.9724 |
| 27 | 10% | 0.9518 | 0.8593 | 0.9874 |
| 27 | 20% | 0.9774 | 0.9241 | 0.9954 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 7 | 0.798 |
| needle10 | 3 | 14 | 0.668 |
| needle50 | 3 | 7 | 0.798 |
| needle50 | 3 | 14 | 0.668 |
| needle90 | 3 | 7 | 0.800 |
| needle90 | 3 | 14 | 0.671 |
| prose | 3 | 4 | 0.757 |
| prose | 3 | 7 | 0.750 |
| prose | 3 | 12 | 0.787 |
| prose | 3 | 14 | 0.603 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.5905 | 25.2525 | 0.8428 | 0.7075 |
| 0.9000 | 7.9973 | 42.1056 | 0.9283 | 0.8616 |
| 0.9500 | 14.0810 | 57.4155 | 0.9670 | 0.9370 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 976s
