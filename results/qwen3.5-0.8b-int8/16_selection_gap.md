# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle 0.0159, coarse estimate 0.0165, simulated true-logits + predicted Gaussian noise
0.0161 (independent per head and step) and 0.0161 (one perturbation per key, shared by heads and steps)
— persistent key noise does NOT explain the gap
(measured/predicted noise std 1.03). The estimate misses 16.6% of the oracle tokens but only 3.9% of their mass;
56% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 0.0165,
c=1 0.0165, c=2 0.0165. Matching the oracle at 5% needs k≈6.5%: 0.0161, 8%: 0.0159, 10%: 0.0157.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

| k | selector | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | oracle | 0.0171 | 0.0150 |
| 2% | est | 0.0181 | 0.0164 |
| 2% | sim (true + predicted noise) | 0.0174 | 0.0156 |
| 2% | sim (persistent key noise) | 0.0175 | 0.0157 |
| 2% | est, MMSE logits (no scaling) | 0.0182 | 0.0164 |
| 2% | est, unbiased logits x1/(1-Dc) | 0.0181 | 0.0164 |
| 2% | est from fine (4-bit) keys | 0.0171 | 0.0150 |
| 2% | est + 0.5·sigma² | 0.0181 | 0.0164 |
| 2% | est + 1.0·sigma² | 0.0181 | 0.0163 |
| 2% | est + 2.0·sigma² | 0.0181 | 0.0163 |
| 2% | two-stage (2k fine keys, k fine values) | 0.0170 | 0.0147 |
| 5% | oracle | 0.0159 | 0.0126 |
| 5% | est | 0.0165 | 0.0136 |
| 5% | sim (true + predicted noise) | 0.0161 | 0.0130 |
| 5% | sim (persistent key noise) | 0.0161 | 0.0130 |
| 5% | est, MMSE logits (no scaling) | 0.0165 | 0.0136 |
| 5% | est, unbiased logits x1/(1-Dc) | 0.0165 | 0.0136 |
| 5% | est from fine (4-bit) keys | 0.0159 | 0.0126 |
| 5% | est + 0.5·sigma² | 0.0165 | 0.0136 |
| 5% | est + 1.0·sigma² | 0.0165 | 0.0135 |
| 5% | est + 2.0·sigma² | 0.0165 | 0.0135 |
| 5% | est at k=6.5% | 0.0161 | 0.0129 |
| 5% | est at k=7.5% | 0.0160 | 0.0125 |
| 5% | est at k=8% | 0.0159 | 0.0124 |
| 5% | est at k=10% | 0.0157 | 0.0119 |
| 5% | two-stage (2k fine keys, k fine values) | 0.0158 | 0.0122 |

## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| 0.166 | 0.0388 | 0.56 | 0.93 |

runtime 1101s
