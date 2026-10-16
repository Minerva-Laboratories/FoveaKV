# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle 0.0201, coarse estimate 0.0207, simulated true-logits + predicted Gaussian noise
0.0202 (independent per head and step) and 0.0203 (one perturbation per key, shared by heads and steps)
— persistent key noise does NOT explain the gap
(measured/predicted noise std 1.02). The estimate misses 16.8% of the oracle tokens but only 3.9% of their mass;
55% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 0.0207,
c=1 0.0207, c=2 0.0207. Matching the oracle at 5% needs k≈6.5%: 0.0203, 8%: 0.0200, 10%: 0.0198.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

| k | selector | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | oracle | 0.0216 | 0.0149 |
| 2% | est | 0.0228 | 0.0157 |
| 2% | sim (true + predicted noise) | 0.0220 | 0.0154 |
| 2% | sim (persistent key noise) | 0.0220 | 0.0155 |
| 2% | est, MMSE logits (no scaling) | 0.0228 | 0.0158 |
| 2% | est, unbiased logits x1/(1-Dc) | 0.0228 | 0.0157 |
| 2% | est from fine (4-bit) keys | 0.0216 | 0.0149 |
| 2% | est + 0.5·sigma² | 0.0228 | 0.0157 |
| 2% | est + 1.0·sigma² | 0.0228 | 0.0157 |
| 2% | est + 2.0·sigma² | 0.0228 | 0.0157 |
| 2% | two-stage (2k fine keys, k fine values) | 0.0214 | 0.0144 |
| 5% | oracle | 0.0201 | 0.0129 |
| 5% | est | 0.0207 | 0.0132 |
| 5% | sim (true + predicted noise) | 0.0202 | 0.0131 |
| 5% | sim (persistent key noise) | 0.0203 | 0.0132 |
| 5% | est, MMSE logits (no scaling) | 0.0207 | 0.0132 |
| 5% | est, unbiased logits x1/(1-Dc) | 0.0207 | 0.0132 |
| 5% | est from fine (4-bit) keys | 0.0201 | 0.0129 |
| 5% | est + 0.5·sigma² | 0.0207 | 0.0132 |
| 5% | est + 1.0·sigma² | 0.0207 | 0.0132 |
| 5% | est + 2.0·sigma² | 0.0207 | 0.0132 |
| 5% | est at k=6.5% | 0.0203 | 0.0126 |
| 5% | est at k=7.5% | 0.0201 | 0.0124 |
| 5% | est at k=8% | 0.0200 | 0.0122 |
| 5% | est at k=10% | 0.0198 | 0.0118 |
| 5% | two-stage (2k fine keys, k fine values) | 0.0199 | 0.0122 |

## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| 0.168 | 0.0395 | 0.55 | 0.94 |

runtime 1082s
