# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle 0.0184, coarse estimate 0.0187, simulated true-logits + predicted Gaussian noise
0.0185 (independent per head and step) and 0.0186 (one perturbation per key, shared by heads and steps)
— persistent key noise does NOT explain the gap
(measured/predicted noise std 1.02). The estimate misses 17.2% of the oracle tokens but only 4.0% of their mass;
55% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 0.0187,
c=1 0.0187, c=2 0.0187. Matching the oracle at 5% needs k≈6.5%: 0.0184, 8%: 0.0181, 10%: 0.0179.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

| k | selector | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | oracle | 0.0200 | 0.0232 |
| 2% | est | 0.0206 | 0.0240 |
| 2% | sim (true + predicted noise) | 0.0203 | 0.0238 |
| 2% | sim (persistent key noise) | 0.0205 | 0.0239 |
| 2% | est, MMSE logits (no scaling) | 0.0206 | 0.0241 |
| 2% | est, unbiased logits x1/(1-Dc) | 0.0206 | 0.0240 |
| 2% | est from fine (4-bit) keys | 0.0200 | 0.0232 |
| 2% | est + 0.5·sigma² | 0.0206 | 0.0240 |
| 2% | est + 1.0·sigma² | 0.0206 | 0.0240 |
| 2% | est + 2.0·sigma² | 0.0206 | 0.0239 |
| 2% | two-stage (2k fine keys, k fine values) | 0.0195 | 0.0224 |
| 5% | oracle | 0.0184 | 0.0205 |
| 5% | est | 0.0187 | 0.0209 |
| 5% | sim (true + predicted noise) | 0.0185 | 0.0208 |
| 5% | sim (persistent key noise) | 0.0186 | 0.0209 |
| 5% | est, MMSE logits (no scaling) | 0.0187 | 0.0209 |
| 5% | est, unbiased logits x1/(1-Dc) | 0.0187 | 0.0209 |
| 5% | est from fine (4-bit) keys | 0.0184 | 0.0204 |
| 5% | est + 0.5·sigma² | 0.0187 | 0.0208 |
| 5% | est + 1.0·sigma² | 0.0187 | 0.0208 |
| 5% | est + 2.0·sigma² | 0.0187 | 0.0208 |
| 5% | est at k=6.5% | 0.0184 | 0.0201 |
| 5% | est at k=7.5% | 0.0182 | 0.0197 |
| 5% | est at k=8% | 0.0181 | 0.0195 |
| 5% | est at k=10% | 0.0179 | 0.0189 |
| 5% | two-stage (2k fine keys, k fine values) | 0.0181 | 0.0195 |

## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| 0.172 | 0.0404 | 0.55 | 0.93 |

runtime 2474s
