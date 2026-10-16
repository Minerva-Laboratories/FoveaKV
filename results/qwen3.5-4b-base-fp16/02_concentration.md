# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average 0.949 of the attention mass (min head-avg 0.642); top-1% holds 0.784.
9 (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average 7.9% of tokens and captures 0.928 true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
| layer | k | mean | min | max |
|---|---|---|---|---|
| 3 | 1% | 0.6369 | 0.3002 | 0.8857 |
| 3 | 2% | 0.7155 | 0.3827 | 0.9220 |
| 3 | 5% | 0.8139 | 0.5191 | 0.9555 |
| 3 | 10% | 0.8816 | 0.6417 | 0.9742 |
| 3 | 20% | 0.9387 | 0.7744 | 0.9898 |
| 11 | 1% | 0.8745 | 0.6701 | 0.9562 |
| 11 | 2% | 0.9250 | 0.7681 | 0.9778 |
| 11 | 5% | 0.9659 | 0.8776 | 0.9922 |
| 11 | 10% | 0.9833 | 0.9351 | 0.9969 |
| 11 | 20% | 0.9933 | 0.9724 | 0.9989 |
| 19 | 1% | 0.8601 | 0.6840 | 0.9625 |
| 19 | 2% | 0.9118 | 0.7585 | 0.9783 |
| 19 | 5% | 0.9568 | 0.8496 | 0.9897 |
| 19 | 10% | 0.9777 | 0.9084 | 0.9956 |
| 19 | 20% | 0.9905 | 0.9548 | 0.9990 |
| 27 | 1% | 0.7653 | 0.5497 | 0.9270 |
| 27 | 2% | 0.8380 | 0.6348 | 0.9520 |
| 27 | 5% | 0.9127 | 0.7452 | 0.9754 |
| 27 | 10% | 0.9525 | 0.8271 | 0.9872 |
| 27 | 20% | 0.9790 | 0.9048 | 0.9948 |

## Flat heads (m_opt(10%) < 0.8)
| corpus | layer | q-head | m_opt(10%) |
|---|---|---|---|
| needle10 | 3 | 7 | 0.799 |
| needle10 | 3 | 14 | 0.666 |
| needle50 | 3 | 7 | 0.799 |
| needle50 | 3 | 14 | 0.665 |
| needle90 | 3 | 14 | 0.668 |
| prose | 3 | 4 | 0.773 |
| prose | 3 | 5 | 0.796 |
| prose | 3 | 7 | 0.776 |
| prose | 3 | 14 | 0.642 |

## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
| τ | mean k % | max (p90 per layer) k % | true mass mean | true mass worst p5 |
|---|---|---|---|---|
| 0.8000 | 3.5514 | 22.7610 | 0.8420 | 0.7259 |
| 0.9000 | 7.9011 | 39.2129 | 0.9284 | 0.8722 |
| 0.9500 | 13.9652 | 54.5529 | 0.9673 | 0.9422 |

## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime 1022s
