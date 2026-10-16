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

## Overnight 2026-10-15/16 (all numbers: Qwen3.5-9B, 5 corpora x 4 layers, fp16 reference, decode at 8K, error of the
## layer's residual contribution after gate and o_proj unless noted)

**E21 full** (attention-output error, pre-gate). 9B: moments beat dropping at <= 13% refined blocks, lose at 25%
(drop 0.061, moments r=8 0.049 at 25% but with more bytes). Qwen3.5-35B-A3B: moments beat dropping at every budget
(13%: 0.091 drop vs 0.049 r=2; 25%: 0.058 vs 0.032). Mass error per block: mean pooling -2.6 / -2.2 nats (9B / MoE),
second moments rank 8 about 1.0 / 0.7 nats.

**E22 gate.** Gate-weighted votes and per-head budgets change nothing (0.0311 in every arm): the gate never closes a
head (head rms 0.04-0.7, 2-3x spread inside a layer); it is channel selective (a quarter of the channels carry 51-88% of
the gate energy; layer 3 has 70% of its channels below 0.05). Untested: gate-weighted value coding per channel.

**E23 backward sparsity.** Dropping (head, token) pairs by p_i ||w (v_i - o)|| against an error budget: 2-3x better than
top-k and 1.4-2x better than top-p at equal pairs kept; coarse 2-bit estimates of p and o almost as good as exact.
Ablation: the gain comes from the value-deviation term (gate weighting adds ~10%, pooling the budget over heads little).
Kernel granularity (layers 11, 27): per-query marks of 5-25% become 18-59% as a union over 16 consecutive queries and
32-79% over 64; choosing the dropped tokens per 16-query tile gives about 2x the compute of per-query marks at the same
error (31% for 0.010, 17% for 0.028), the same curve as using the whole union.

**E24 Barnes-Hut nodes** (rank-2 second-cumulant nodes, opened by estimated mass, compute incl. node evaluations):

| nodes | 20% compute | 30% compute |
|---|---|---|
| positional 64-token blocks | 0.090 | 0.062 |
| k-means on keys | 0.030 | 0.017 |
| gain strata x spherical shapes | 0.044 | 0.024 |
| k-means in the query metric E[qq^T]^(1/2) (Li & Gong 2008) | 0.028 | 0.016 |
| same, Sigma scaled by 0.5 (crossover kernel, Ito et al. 2020) | **0.026** | **0.013** |
| same, oracle opening | 0.018 | 0.009 |

Opening rules that do not beat estimated mass (smoke test, layer 11): the Barnes-Hut radius bound (much worse), mass x
value deviation, mass x (deviation + spread), a regression of the true node error on cheap features (plain or
error-weighted, cross-fitted). The remaining oracle gap needs information the summaries do not carry.
Comparison at equal total compute: top-k needs a scoring pass (INT4 ~ 1/8 of dense): top-k 20% + pass ~ 32% -> 0.016;
E23 backward (coarse est.) 17% + pass ~ 30% -> 0.010; tree 30% -> 0.013. At 8K the tree is between the two; its case
is long context (node evaluations ~ T/64 x 4 vs a T/8 scoring pass, and a two-level tree evaluates fewer nodes).
Prior art to cite for clustered keys: ClusterKV, PQCache, RetrievalAttention (selection only; unselected clusters dropped).

**E25 two-level tree + precision ladder** (query-metric nodes, Sigma x 0.5; parents = k-means of node centroids, ~11):
flat (all nodes evaluated) 20.5% -> 0.025, 29.9% -> 0.013; flat + ladder (heaviest half of opened tokens exact, rest
via the 2-bit code at 1/4 cost) 17.0% -> 0.032, 22.9% -> 0.017 (~1.2x better than flat at equal compute); tree with 6
of 11 parents opened 18.5% -> 0.027 (66% of nodes evaluated); tree Pp=6 + ladder 14.9% -> 0.033, 20.8% -> 0.019 (~1.3x
better than flat). Opening only 2-4 parents saturates (mass lost in parent summaries). At 8K this is level with E23
backward pruning (~30% with its INT4 scoring pass -> 0.010) while selection grows sub-linearly with context.
Same on Qwen3.5-35B-A3B (GQA 8) and post-trained 4B: tree Pp=6 + ladder at ~21% compute gives 0.033 / 0.023 against
0.041 / 0.030 for the flat tree at ~20.5% (1.2-1.3x).

**E26 prefill (causal, decisions per 16-query tile x GQA group; 9B layers 11/27, prose/code, 24 tiles each).** Capture
with queries at every position after 512 (data/qwen3.5-9b-pf). Chunks of S as parents, query-metric clusters inside,
each chunk's last 128 tokens as positional nodes read exactly inside a query's recent window. Result: the tree LOSES in
prefill. S=1024 at ~30% compute: tile top-k on exact logits (tail dropped, scoring pass not counted) 0.055; top-k +
summaries of untouched nodes 0.051 (31%); tree, all chunks opened, 0.078 (35%). Errors are 3-4x the decode ones at
equal compute: 16 queries share each decision, and mid-sequence queries (2-6K) attend more diffusely (per-tile errors
0.03-0.11; a tile at a chunk start 0.29). Summaries add little on top of a tile top-k. Prefill needs a different design
(finer query tiles / reordering queries by similarity, or the precision ladder instead of dropping).
