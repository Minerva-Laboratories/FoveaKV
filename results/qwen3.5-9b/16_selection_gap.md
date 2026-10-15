# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle 0.0229, coarse estimate 0.0234, simulated true-logits + predicted Gaussian noise
0.0230 (independent per head and step) and 0.0231 (one perturbation per key, shared by heads and steps)
— persistent key noise does NOT explain the gap
(measured/predicted noise std 1.02). The estimate misses 16.6% of the oracle tokens but only 3.6% of their mass;
56% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 0.0234,
c=1 0.0234, c=2 0.0234. Matching the oracle at 5% needs k≈6.5%: 0.0231, 8%: 0.0228, 10%: 0.0226.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

| k | selector | attn err | layer-3 attn err |
|---|---|---|---|
| 2% | oracle | 0.0246 | 0.0205 |
| 2% | est | 0.0254 | 0.0220 |
| 2% | sim (true + predicted noise) | 0.0250 | 0.0212 |
| 2% | sim (persistent key noise) | 0.0251 | 0.0213 |
| 2% | est, MMSE logits (no scaling) | 0.0255 | 0.0221 |
| 2% | est, unbiased logits x1/(1-Dc) | 0.0253 | 0.0220 |
| 2% | est from fine (4-bit) keys | 0.0246 | 0.0205 |
| 2% | est + 0.5·sigma² | 0.0253 | 0.0220 |
| 2% | est + 1.0·sigma² | 0.0253 | 0.0220 |
| 2% | est + 2.0·sigma² | 0.0253 | 0.0219 |
| 2% | two-stage (2k fine keys, k fine values) | 0.0242 | 0.0201 |
| 5% | oracle | 0.0229 | 0.0175 |
| 5% | est | 0.0234 | 0.0186 |
| 5% | sim (true + predicted noise) | 0.0230 | 0.0179 |
| 5% | sim (persistent key noise) | 0.0231 | 0.0179 |
| 5% | est, MMSE logits (no scaling) | 0.0235 | 0.0187 |
| 5% | est, unbiased logits x1/(1-Dc) | 0.0234 | 0.0186 |
| 5% | est from fine (4-bit) keys | 0.0229 | 0.0175 |
| 5% | est + 0.5·sigma² | 0.0234 | 0.0186 |
| 5% | est + 1.0·sigma² | 0.0234 | 0.0186 |
| 5% | est + 2.0·sigma² | 0.0234 | 0.0186 |
| 5% | est at k=6.5% | 0.0231 | 0.0178 |
| 5% | est at k=7.5% | 0.0229 | 0.0174 |
| 5% | est at k=8% | 0.0228 | 0.0172 |
| 5% | est at k=10% | 0.0226 | 0.0166 |
| 5% | two-stage (2k fine keys, k fine values) | 0.0227 | 0.0171 |

## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| 0.166 | 0.0361 | 0.56 | 0.94 |

runtime 1535s
