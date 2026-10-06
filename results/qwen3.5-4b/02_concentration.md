# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.946 of the attention mass (min head-avg 0.642); top-1% holds 0.776.
12 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 8.3% of tokens and captures 0.929 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6327 | 0.3001 | 0.8568 |
| 3 | 2% | 0.7106 | 0.3827 | 0.9005 |
| 3 | 5% | 0.8094 | 0.5190 | 0.9429 |
| 3 | 10% | 0.8783 | 0.6417 | 0.9670 |
| 3 | 20% | 0.9369 | 0.7744 | 0.9848 |
| 11 | 1% | 0.8704 | 0.5831 | 0.9563 |
| 11 | 2% | 0.9204 | 0.6880 | 0.9778 |
| 11 | 5% | 0.9626 | 0.8164 | 0.9922 |
| 11 | 10% | 0.9813 | 0.8963 | 0.9969 |
| 11 | 20% | 0.9924 | 0.9548 | 0.9990 |
| 19 | 1% | 0.8461 | 0.6289 | 0.9377 |
| 19 | 2% | 0.9003 | 0.7166 | 0.9681 |
| 19 | 5% | 0.9499 | 0.8240 | 0.9889 |
| 19 | 10% | 0.9739 | 0.8944 | 0.9956 |
| 19 | 20% | 0.9890 | 0.9500 | 0.9990 |
| 27 | 1% | 0.7529 | 0.4736 | 0.8486 |
| 27 | 2% | 0.8278 | 0.5528 | 0.9040 |
| 27 | 5% | 0.9065 | 0.6761 | 0.9573 |
| 27 | 10% | 0.9492 | 0.7798 | 0.9800 |
| 27 | 20% | 0.9777 | 0.8809 | 0.9925 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| code | 3 | 14 | 0.716 |
| code | 27 | 7 | 0.780 |
| needle10 | 3 | 4 | 0.800 |
| needle10 | 3 | 7 | 0.799 |
| needle10 | 3 | 14 | 0.665 |
| needle50 | 3 | 7 | 0.799 |
| needle50 | 3 | 14 | 0.666 |
| needle90 | 3 | 14 | 0.668 |
| prose | 3 | 4 | 0.773 |
| prose | 3 | 5 | 0.796 |
| prose | 3 | 7 | 0.776 |
| prose | 3 | 14 | 0.642 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.7668 | 22.7139 | 0.8436 | 0.7225 |
| 0.9000 | 8.3171 | 39.0603 | 0.9290 | 0.8716 |
| 0.9500 | 14.6017 | 54.4983 | 0.9674 | 0.9426 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 427s
