# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle 0.5336, coarse estimate 0.1251, simulated true-logits + predicted Gaussian noise
0.5277 (independent per head and step) and 0.5325 (one perturbation per key, shared by heads and steps)
— persistent key noise does NOT explain the gap
(measured/predicted noise std 0.99). The estimate misses 39.4% of the oracle tokens but only 24.8% of their mass;
35% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 0.2254,
c=1 0.2692, c=2 0.3065. Matching the oracle at 5% needs k≈6.5%: 0.1180, 8%: 0.1134, 10%: 0.1088.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

| k | selector | attn err | layer-0 attn err |
|---|---|---|---|
| 2% | oracle | 0.5438 | 2.0683 |
| 2% | est | 0.1625 | 0.5853 |
| 2% | sim (true + predicted noise) | 0.5388 | 2.0355 |
| 2% | sim (persistent key noise) | 0.5432 | 2.0513 |
| 2% | est, MMSE logits (no scaling) | 0.1618 | 0.5821 |
| 2% | est, unbiased logits x1/(1-Dc) | 0.1635 | 0.5896 |
| 2% | est from fine (4-bit) keys | 0.5257 | 1.9972 |
| 2% | est + 0.5·sigma² | 0.2972 | 1.1246 |
| 2% | est + 1.0·sigma² | 0.3511 | 1.3399 |
| 2% | est + 2.0·sigma² | 0.3936 | 1.5067 |
| 2% | two-stage (2k fine keys, k fine values) | 0.1328 | 0.4777 |
| 5% | oracle | 0.5336 | 2.0535 |
| 5% | est | 0.1251 | 0.4503 |
| 5% | sim (true + predicted noise) | 0.5277 | 2.0193 |
| 5% | sim (persistent key noise) | 0.5325 | 2.0366 |
| 5% | est, MMSE logits (no scaling) | 0.1250 | 0.4496 |
| 5% | est, unbiased logits x1/(1-Dc) | 0.1257 | 0.4528 |
| 5% | est from fine (4-bit) keys | 0.5128 | 1.9709 |
| 5% | est + 0.5·sigma² | 0.2254 | 0.8519 |
| 5% | est + 1.0·sigma² | 0.2692 | 1.0273 |
| 5% | est + 2.0·sigma² | 0.3065 | 1.1748 |
| 5% | est at k=6.5% | 0.1180 | 0.4249 |
| 5% | est at k=7.5% | 0.1147 | 0.4128 |
| 5% | est at k=8% | 0.1134 | 0.4084 |
| 5% | est at k=10% | 0.1088 | 0.3915 |
| 5% | two-stage (2k fine keys, k fine values) | 0.1089 | 0.3915 |

## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| 0.394 | 0.2479 | 0.35 | 0.73 |

runtime 764s
