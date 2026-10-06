# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle 0.0111, coarse estimate 0.0121, simulated true-logits + predicted Gaussian noise
0.0118 (independent per head and step) and 0.0118 (one perturbation per key, shared by heads and steps)
— persistent key noise explains the gap
(measured/predicted noise std 0.77). The estimate misses 17.8% of the oracle tokens but only 6.6% of their mass;
53% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 0.0120,
c=1 0.0120, c=2 0.0121. Matching the oracle at 5% needs k≈6.5%: 0.0113, 8%: 0.0108, 10%: 0.0103.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

| k | selector | attn err | layer-0 attn err |
|---|---|---|---|
| 2% | oracle | 0.0136 | 0.0161 |
| 2% | est | 0.0150 | 0.0163 |
| 2% | sim (true + predicted noise) | 0.0147 | 0.0178 |
| 2% | sim (persistent key noise) | 0.0148 | 0.0180 |
| 2% | est, MMSE logits (no scaling) | 0.0151 | 0.0163 |
| 2% | est, unbiased logits x1/(1-Dc) | 0.0150 | 0.0163 |
| 2% | est from fine (4-bit) keys | 0.0136 | 0.0160 |
| 2% | est + 0.5·sigma² | 0.0149 | 0.0161 |
| 2% | est + 1.0·sigma² | 0.0149 | 0.0162 |
| 2% | est + 2.0·sigma² | 0.0151 | 0.0171 |
| 2% | two-stage (2k fine keys, k fine values) | 0.0134 | 0.0147 |
| 5% | oracle | 0.0111 | 0.0119 |
| 5% | est | 0.0121 | 0.0123 |
| 5% | sim (true + predicted noise) | 0.0118 | 0.0132 |
| 5% | sim (persistent key noise) | 0.0118 | 0.0133 |
| 5% | est, MMSE logits (no scaling) | 0.0121 | 0.0123 |
| 5% | est, unbiased logits x1/(1-Dc) | 0.0120 | 0.0123 |
| 5% | est from fine (4-bit) keys | 0.0110 | 0.0118 |
| 5% | est + 0.5·sigma² | 0.0120 | 0.0122 |
| 5% | est + 1.0·sigma² | 0.0120 | 0.0121 |
| 5% | est + 2.0·sigma² | 0.0121 | 0.0125 |
| 5% | est at k=6.5% | 0.0113 | 0.0112 |
| 5% | est at k=7.5% | 0.0109 | 0.0107 |
| 5% | est at k=8% | 0.0108 | 0.0104 |
| 5% | est at k=10% | 0.0103 | 0.0096 |
| 5% | two-stage (2k fine keys, k fine values) | 0.0107 | 0.0108 |

## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| 0.178 | 0.0664 | 0.53 | 0.92 |

runtime 767s
