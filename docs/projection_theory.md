# Sequence-fitted low-rank cache projections: theory (working notes, 2026-10-16)

Setting: one attention head (GQA: one KV head and its query heads), queries q, keys k, values v, scale s = d^-1/2,
p_t = softmax_t(s q.k_t), o = sum_t p_t v_t. The cache stores a low-rank code of each key / value, as MLA does, but
the projection is fitted to the current sequence (online, or on the whole prompt) instead of being fixed in training.
Softmax attention and the per-token cache are unchanged. Earlier results: docs/online_projection_spec.md in OpenDecider
(online subspace iteration beats a fixed basis by 10-30% NLL; a non-causal oracle halves it again). Experiment: E29.

## 1. The error of a compressed cache, to first order

Replace k_t by k_t + e_t and v_t by v_t + u_t. With ds_t = s q.e_t,

    do = sum_t p_t ds_t (v_t - o) + sum_t p_t u_t + O(ds^2)                                         (1)

(softmax Jacobian: d p_t = p_t (ds_t - sum_s p_s ds_s); the mean of ds under p cancels, so any error that is constant
across keys costs nothing: a key offset common to all tokens is free, which is why keys are centered first).

**Key truncation is a query perturbation.** For a rank-r map k_hat = D E (k - mu) + mu, e_t = -(I - DE)(k_t - mu) and
q.e_t = -((I - B) q).(k_t - mu) with B = (DE)^T. So (1) reads

    do_K = - J(q) (I - B) q,      J(q) = s sum_t p_t (v_t - o)(k_t - mu)^T = d o / d q                  (2)

The key error is the attention output's response to moving the query by its component outside the code's subspace.
J is the query Jacobian of attention: large where keys that co-vary with values under the current attention exist.

**Value truncation is exact, not first order.** v_hat = D E (v - mu_v) + mu_v gives o_hat = DE (o - mu_v) + mu_v, so

    do_V = -(I - DE)(o - mu_v)                                                                     (3)

The value code only has to preserve the subspace of the attention OUTPUTS, not of the values: the best value basis is
the PCA of o (in the gate x o_proj metric), not of v. Values with large variance that attention averages away do not
need dimensions.

## 2. Objectives and their optima

All are two-sided Eckart-Young problems min_{rank r A} ||X (I - A) Y||_F with solution A = Y-side projector built from
the top-r left singular vectors U of (X Y) (proof: write C = X A Y, the best rank-r C is [X Y]_r = U U^T X Y, so
A = X^-1 U U^T X when X is invertible).

| objective | minimizes | optimum (encoder E, decoder D) |
|---|---|---|
| key PCA (Oja target) | E ||(I - DE)(k - mu)||^2 | U = top-r eigvecs of S_k; E = U^T, D = U |
| score-preserving (QW) | E (q^T (I - DE)(k - mu))^2 = ||M^1/2 (I - DE) S_k^1/2||^2 | U from M^1/2 S_k^1/2; E = U^T M^1/2, D = M^-1/2 U |
| output, keys (J) | E_q ||J(q)(I - B) q||^2 ~ ||G^1/2 (I - B) M^1/2||^2, G = E J^T J | U from G^1/2 M^1/2; B = G^-1/2 U U^T G^1/2, DE = B^T |
| output, values (O) | E_q ||W (I - DE)(o - mu_v)||^2 | U from W (O - mu_v); E = U^T W, D = W^-1 U |

M = E[q q^T] (position-free part), S_k = key covariance, W = gate x o_proj column norms. The J objective decouples G and
M (exact form: sum_q ||J_q (I - B) q||^2, a quadratic in B; alternating least squares if the decoupled form falls short).

**Two forms of G.** G = E_q J_q^T J_q = s^2 E_q sum_{t,s} p_t p_s (v_t - o).(v_s - o) k_t k_s^T.
- Diagonal (t = s only): G_diag = s^2 E_q sum_t p_t^2 ||v_t - o||^2 k_t k_t^T. This is the right metric when the
  per-key errors are independent across keys (quantization noise): each key is weighted by p_t^2 ||v_t - o||^2, the
  E23 error-model weight. Its total scales like 1 / n_eff, n_eff = 1 / sum p^2.
- Full: keeps the cross terms. Truncation errors are NOT independent across keys (every key loses the same directions),
  so they add coherently; the full G is the right metric for them.

## 3. Two kinds of error, and why only one of them averages out

From (1), a key error contributes sum_t p_t ds_t (v_t - o).
- **Independent per-key errors** (FoveaKV codes after a random rotation; dithered quantization): E||do||^2 =
  sum_t p_t^2 ||v_t - o||^2 E ds_t^2, which shrinks with n_eff. Long contexts with diffuse attention average them out.
- **Systematic errors** (truncation): ds_t = -s ((I - B) q).(k_t - mu) is a deterministic function of the key, so the
  output error is a covariance under attention, Cov_p(ds, v), that does not shrink with n_eff unless the dropped
  directions are uncorrelated with the values under p. This is exactly what J measures.

Prediction: at long context and diffuse attention, bits get cheaper relative to rank. **E29 does not support the
asymmetry**: the absolute output error falls with n_eff for both kinds (2-bit full-dimension code -21% from the lowest
to the highest n_eff quartile, rank-24 truncation -27%). The dropped directions' score contributions are mostly
uncorrelated with the values under attention, so truncation errors average too; the coherent part (full G vs diagonal
G) is small (sec. 6). Treat kappa below as about equal for both kinds.

## 4. The equal-bytes optimum

Budget B bits per token-head, latent rank r, bits per coordinate b = B / r. With a rotated gain-shape code the
quantization noise is isotropic inside the kept subspace with relative MSE delta(b) (Lloyd-Max: 0.118 at 2 bits, 0.0095
at 4). With lambda_i the eigenvalues in the relevant metric (sec. 2) and S(r) = sum_{i <= r} lambda_i,

    D(r) = T(r) + kappa delta(B/r) S(r),     T(r) = sum_{i > r} lambda_i                              (4)

kappa < 1 is the averaging factor of independent errors relative to systematic ones (sec. 3; about sum p^2 ||v - o||^2
/ ||sum p (v - o)||^2-like, measured in E29). With delta(b) ~ c 2^(-2b), dD/dr = 0 gives

    lambda_r (1 - kappa delta) = 2 ln2 kappa b delta(b) S(r) / r                                       (5)

Keep adding dimensions while the next eigenvalue exceeds 2 ln2 kappa b delta(b) times the mean kept one. Without
averaging (kappa = 1): about 5% at 4 bits and 33% at 2 bits. Averaging (kappa << 1) moves the optimum to higher rank,
i.e. towards plain FoveaKV at full dimension. Low rank wins at equal bytes only if the metric spectrum falls faster
than (5) allows, which E29 measures directly with the ladder 192 x 2b = 96 x 4b = 48 x 8b = 24 x 16b.
The basis optimum does not depend on b (isotropic noise in the kept subspace), so basis and bits decouple: only r couples.

## 5. Online estimation with linear-attention machinery

All the statistics are decayed sums of outer products, i.e. linear-attention states, and the bases are their top
singular vectors.

| statistic | state | write per token / query |
|---|---|---|
| S_k (key PCA) | sum gamma^(T-t) (k_t - mu)(k_t - mu)^T | key outer product |
| M (query metric) | sum gamma^(T-t) q_t q_t^T | query outer product |
| G_diag | sum_t w_t (k_t - mu)(k_t - mu)^T, w_t = sum_q p_qt^2 ||v_t - o_q||^2 | key outer product, strength w_t |
| output PCA | sum gamma^(T-t) (o_t - mu_v)(o_t - mu_v)^T | output outer product (o is computed anyway) |

- Oja's subspace rule is a delta rule on a projector: with P = E^T E and loss ||(I - P) k||^2,
  P <- P + eta [(I - P) k k^T P + P k k^T (I - P)], the symmetric counterpart of DeltaNet's S <- S + beta (v - S k) k^T.
  So GDN's gates carry over: a data-dependent step beta_t ~ surprise = ||(I - P) k_t||^2 / ||k_t||^2 (moves at topic
  shifts, freezes on familiar text, so old codes stay valid) and a forget gate alpha_t.
- In prefill, the states at every chunk boundary are a decayed prefix scan (chunkwise-parallel as in GLA / Mamba-2); each
  chunk's basis is a few power iterations warm-started from the previous chunk's, plus an r x d QR.
- A rotation-free update (Procrustes-aligned, or GHA's ordered basis) keeps old codes readable under the newest basis
  without per-chunk bases or re-maps.
- The cheapest legal non-causal fit: the whole prompt's statistics at the end of prefill (the cache is computed exactly
  during prefill anyway); online updates only during generation.
- G_diag needs w_t, the attention a key receives weighted by value deviation: the column sums of p^2 (v - o)^2 terms,
  computable inside a flash-attention pass as an extra accumulator per key (E23 already estimates it from the cheap pass).
  The full G needs per-query J_q (d_v x d), too costly online; E29 shows whether the diagonal form is enough.

## 6. Results (E29, CPU, Qwen3.5-9B prefill capture: prose + code, layers 11 / 27, 4 KV heads; results/qwen3.5-9b-pf-S1024/29_projection_objectives.md)

Prompt = first 7,680 tokens, evaluation = the next 512 queries (decode-like), sinks + 128-token window exact. Relative
attention-output error, mean over 16 (corpus, layer, head); residual error (after gate and o_proj) in the file.

**A basis fitted to the sequence beats a fixed one by about 2-3x** (static = fitted on the other corpus, the MLA-like
case): keys r = 64 out of 192: static 0.193, prompt key-PCA 0.113, best objective 0.081; values r = 64 out of 256:
static 0.273, prompt value-PCA 0.134, output-PCA 0.090. This is a far larger lever than per-token routing among fixed
projections (mixture of latents: 10-20%).

**Objectives.** Keys: key-PCA 0.113 -> QW 0.085 -> J diag 0.082 -> J full 0.081 at r = 64 (r = 96: 0.078 / 0.056 /
0.054 / 0.053). Score-aware objectives cut 28-32%; most of it is the query metric M, the Jacobian adds ~5%, and the
diagonal G (computable online from per-key weights) is as good as the full one. Values: output-PCA beats value-PCA by
33-37% at every rank (r = 128: 0.052 vs 0.083), as eq. (3) predicts; the gate x o_proj metric adds nothing measurable.
An oracle fitted on the evaluation queries' own outputs is still 1.5-1.8x better (0.028 at r = 128): the outputs of
later queries are only partly predicted by the prompt's.

**First-order formula (2).** The predicted key-truncation error J(q)(I - B)q points the right way (median cosine with
the measured error 0.98 key-PCA / 0.99 J basis) and overestimates its size by 1.3-1.5x (softmax saturation).

**Equal bytes against FoveaKV's gain-shape codes** (static codes, no dynamic tier):

| budget per head-token | full dimension | half rank, 4-bit | quarter rank, 8-bit | eighth rank, bf16 |
|---|---|---|---|---|
| keys 384 bits (2 bits/dim) | 0.097 (2-bit) | **0.068** (J basis) | 0.100 | 0.148 |
| keys 768 bits (4 bits/dim) | **0.018** (4-bit) | 0.053 (8-bit) | 0.100 (bf16) | |
| values 512 bits | 0.070 (2-bit) | **0.054** (output basis) | 0.090 | 0.127 |
| values 1024 bits | **0.015** (4-bit) | 0.052 (8-bit) | 0.090 (bf16) | |

At 2 bits per dimension, half the rank at 4 bits beats the full dimension at 2 bits by 22-30%; at 4 bits per dimension
the full dimension wins 3x. This is eq. (5): at 2 bits the threshold is ~33% of the mean kept eigenvalue, so the lower
half of the spectrum is not worth its bits; at 4 bits (~5%) it is. Only the 2-bit tier stands to gain from a sequence
basis; the 4-bit tier should stay full-dimensional.

## 7. Without a basis per chunk (E30, E31)

E30: true online learning (causal chunks): fast block-Oja (eta = 8) beats the batch whitened fit by ~40% (keys 0.052 vs
0.085, values 0.033 vs 0.059 at 256-token chunks), but only if every chunk keeps the basis it was written with; its
limit is plain per-chunk PCA (0.034 / 0.021). The gain is locality. Read through the newest basis only: 0.19-0.26.

Bound: if all codes are read through one current rank-r decoder, every reconstruction lies in one r-dim subspace, so
no update rule beats the best fixed rank-r subspace. And the intermediate bases cannot be rebuilt from (initial basis,
current basis, latents): what moves the basis is the chunk's residual (I - U U^T) y, which the latents discard.
They can be rebuilt from a log of the innovations: E31 "swap" = each chunk replaces its m weakest slots by its top-m
residual directions; the log (m directions + m slot indices per chunk) replays every chunk's basis from the initial
one, and its end is the current basis.

E31 (r = 64 keys / 128 values): keys at 256-token chunks: initial basis 0.085, swap4 / 8 / 16 0.078 / 0.072 / 0.064
(+6 / 12 / 24 B per token on 128 B of codes), growing dictionary (one current dictionary, greedy atom choice) 0.075,
coordinate selection in a fixed full basis 0.084 (no gain), per-chunk bases 0.033 (+96 B). At 1024-token chunks:
swap16 0.075 (+6 B), per-chunk bases 0.057 (+24 B). Values: at most -12% (swap16 0.046 vs 0.052). Against a larger
global rank at the same bytes (r = 76 for +24 B, about 0.074) swap16 keeps ~13% (keys); the local subspaces differ from
each other in many directions, so a few innovations per chunk recover only a third of the per-chunk-basis gain.
