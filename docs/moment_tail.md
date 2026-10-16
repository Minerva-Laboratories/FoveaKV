# Query-dependent tail summaries (Minerva in-house attention acceleration, working notes, 2026-10-15)

Goal: an attention approximation that is query dependent at every step (what is read exactly, and how the rest is
estimated, is decided per query), that keeps the tail instead of dropping it, and that is judged on attention error
per byte moved (decode) and per FLOP (prefill). Lessons carried from FoveaKV (E11-E20): the error is roughly the
attention mass left on the approximated tail times the error of the tail code; per-query refinement beats any static
approximation fixed once the cache is written; selection by an estimated score is the bottleneck more than the code.

## Block moment summaries

For a block of B keys/values (mean mu, covariance Sigma, value mean mu_v, cross covariance C_vk) and scale s, under a
Gaussian model of the keys the block's softmax contribution is, for the current query q:

    log sum_i e^{s q.k_i}  ~  log B + s q.mu + (s^2 / 2) q^T Sigma q          (cumulant expansion, exact if Gaussian)
    sum_i e^{s q.k_i} v_i / sum_i e^{s q.k_i}  ~  mu_v + s C_vk q              (mean of v under the tilted Gaussian)

Mean pooling keeps only s q.mu, so by Jensen it underestimates every block's mass. The tilted value mean is the
key-space local-linear correction, but it needs C_vk q, not an inverse covariance: the exponential tilt replaces the
regression solve, so there is no ill-conditioned d x d inverse per query. Sigma and C_vk are kept on the top-r key
directions (plus an isotropic remainder for the mass), and the log mass is clipped at the per-channel bound
log B + s (q+ . max + q- . min).

Where the covariances come from (all evaluated per query; none is a frozen approximation of the output):
1. computed from each block's keys (E21; sums of k, k k^T, v k^T, i.e. a chunkwise linear-attention state);
2. learned: a small compressor maps a block to (mu, U, W) trained to minimize the attention error over real
   queries, on captured caches offline (fits the model's own attention; no LLM teacher). This is the fix for
   non-Gaussian blocks. Prior art to check: learned block compression in DeepSeek NSA;
3. running linear-attention statistics with a feature map, for the far tier only (exp-kernel feature maps are poor
   for the peaked heads we measured).

## First result (E21 smoke test, Qwen3.5-9B layer 11, prose, 8K context, fp16 reference)

Mass estimate per block, mass-weighted |log error|: mean pooling 2.88 nats (all underestimates), second moments
isotropic 1.60, rank 2 1.06, rank 8 0.99 (unbiased). Attention error at a fixed fraction of blocks read exactly:

| refined blocks | drop tail (bound select) | mean pooling | moments r=2 | moments r=8, oracle selection |
|---|---|---|---|---|
| 3% | 0.142 | 0.101 | 0.088 | 0.091 |
| 13% | 0.064 | 0.050 | 0.045 | 0.024 |
| 25% | 0.029 | 0.040 | 0.036 | 0.007 |

Reading: summaries beat dropping when few blocks are refined, lose to it when many are (the summary's own error
exceeds the small mass left), and the selector leaves a 4-5x gap to the oracle. Per byte, block summaries are far
behind FoveaKV's per-token 2-bit tail on this model (0.023 at a similar byte budget). Full runs: results/<model>-fp16/
21_moment_tail.md.

## Open
- Better selector (the oracle gap): score blocks with per-token coarse keys (FoveaKV's nested code) instead of
  summaries; error proxy from the coarse scores.
- Summaries as the FAR tier of a hierarchy over FoveaKV (contexts where even 2 bits per token is too much).
- Learned compressor (item 2 above).
- Prior-art check before naming anything: NSA / DSA / CSA (DeepSeek), MRA, H-Transformer-1D, Fast Multipole
  Attention, Quest, MagicPIG, SpeCache.
