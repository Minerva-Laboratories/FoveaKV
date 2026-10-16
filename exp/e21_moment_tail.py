"""E21 Query-dependent tail summaries: keep the attention tail as per-block moments instead of dropping it.

Decode attention = exact tier (4 sinks, 128-token recent window, any token outside a full summary block) + R refined
blocks read exactly + every other block of B keys replaced by a summary evaluated FOR EACH QUERY. Per KV head and block
(keys k, values v, scale s):
  mean      mass B e^{s q.mu}, value mu_v                                   (mean pooling; Jensen: underestimates mass)
  mom-r     log mass ~ log B + s q.mu + s^2/2 q^T Sigma q   (Gaussian cumulants; Sigma = top-r PCA + isotropic residual)
            value    ~ mu_v + s Cov(v, k) q                 (mean of v under the tilted Gaussian, Cov(v, k) on the
                                                             top-r key subspace)
            log mass clipped at log B + s max_i q.k_i <= log B + s (q+ . max + q- . min) (per-channel key bounds)
  drop      no summary (block top-k, Quest-style bound for selection)
Which blocks are read exactly is chosen per query (shared by the GQA group, max over its query heads, as in E17):
  by estimated mass ('mass'), by an error proxy mass * sqrt(s^2 q^T Sigma q) * std(v) ('err'), or by the true
  summary error ('oracle', upper bound of what a selector can do). 'adaptive': per-query budget, refine in proxy
  order until the remaining estimated error is below tau * estimated total mass.
  +o1: the largest-norm key of each block is kept exact and left out of the block's moments.
Bytes per decode step = exact tokens (K and V) + all summaries, relative to reading the whole fp16 cache.
FOVEAKV_E21_B (default 64), FOVEAKV_E21_STRIDE (query subsampling, default 2)."""
from common import *
from foveakv.attention import scores, softmax, attend

t0 = time.time()
B = int(os.environ.get("FOVEAKV_E21_B", "64"))
STRIDE = int(os.environ.get("FOVEAKV_E21_STRIDE", "2"))
RHO = (1 / 32, 1 / 16, 1 / 8, 1 / 4)
RS = (0, 2, 8)
TAUS = (0.1, 0.03, 0.01, 0.003)
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]


def block_stats(K, V, b0, nb, r, o1):
    """Per (KV head, block) moments. K, V (T, Hkv, d). Returns dict of arrays with leading (Hkv, nb)."""
    kb = K[b0:b0 + nb * B].reshape(nb, B, *K.shape[1:]).transpose(2, 0, 1, 3).astype(np.float64)    # (H,nb,B,d)
    vb = V[b0:b0 + nb * B].reshape(nb, B, *V.shape[1:]).transpose(2, 0, 1, 3).astype(np.float64)
    keep = np.ones(kb.shape[:3], bool)
    if o1:
        out = np.linalg.norm(kb, axis=-1).argmax(-1)                                                   # (H,nb)
        np.put_along_axis(keep, out[..., None], False, -1)
    n = keep.sum(-1)[..., None]                                                                        # (H,nb,1)
    w = keep[..., None]
    mu = (kb * w).sum(2) / n
    muv = (vb * w).sum(2) / n
    X, Y = (kb - mu[:, :, None]) * w, (vb - muv[:, :, None]) * w
    _, S, Vt = np.linalg.svd(X, full_matrices=False)                                                   # (H,nb,B), (H,nb,B,d)
    lam = S ** 2 / n                                                                                   # eigenvalues of Sigma
    d = K.shape[-1]
    U = Vt[:, :, :r].transpose(0, 1, 3, 2)                                                             # (H,nb,d,r)
    res = (lam.sum(-1) - lam[..., :r].sum(-1)) / (d - r)                                               # isotropic rest
    W = np.einsum("hnbv,hnbd,hndr->hnvr", Y, X, U) / n[..., None]                                       # Cov(v,k) U
    kmask = np.where(w, kb, np.nan)
    return dict(mu=mu, muv=muv, U=U, lam=lam[..., :r], res=res, W=W, logn=np.log(n[..., 0]),
                mx=np.nanmax(kmask, 2), mn=np.nanmin(kmask, 2), sv=np.sqrt((Y ** 2).sum((-1, -2)) / n[..., 0]),
                out=None if not o1 else out)


def summaries(st, qh, s, kind):
    """qh (H,G,E,d). Returns log mass (H,G,E,nb), tilt coords z (H,G,E,nb,r), quad term (H,G,E,nb)."""
    a = s * np.einsum("hged,hnd->hgen", qh, st["mu"])
    if kind == "mean":
        return st["logn"][:, None, None] + a, None, None
    z = np.einsum("hged,hndr->hgenr", qh, st["U"])
    q2 = (qh ** 2).sum(-1)[..., None]                                                                  # (H,G,E,1)
    quad = s * s * ((st["lam"][:, None, None] * z ** 2).sum(-1) + st["res"][:, None, None] * (q2 - (z ** 2).sum(-1)))
    ub = s * (np.einsum("hged,hnd->hgen", np.maximum(qh, 0), st["mx"]) + np.einsum("hged,hnd->hgen", np.minimum(qh, 0), st["mn"]))
    return np.minimum(st["logn"][:, None, None] + a + quad / 2, st["logn"][:, None, None] + ub), z, quad


def combine(sr, Vh, exact_tok, logm, sum_mask, st, z, s):
    """Softmax over exact tokens plus summary blocks. sr (H,G,E,T) logits, Vh (H,T,dv), exact_tok (H,G|1,E,T),
    logm (H,G,E,nb), sum_mask (H,G|1,E,nb) = blocks served by their summary."""
    se = np.where(exact_tok, sr, -np.inf)
    lm = np.where(sum_mask, logm, -np.inf)
    M = np.maximum(se.max(-1, keepdims=True), lm.max(-1, keepdims=True))
    pe, pm = np.exp(se - M), np.exp(lm - M)
    num = np.matmul(pe, Vh[:, None]) + np.einsum("hgen,hnv->hgev", pm, st["muv"])
    if z is not None:
        num = num + s * np.einsum("hgenr,hnvr->hgev", pm[..., None] * z, st["W"])
    o = num / (pe.sum(-1) + pm.sum(-1))[..., None]
    H, G, E, dv = o.shape
    return o.transpose(2, 0, 1, 3).reshape(E, H * G, dv)


def pick(score, R):
    """score (H,E,nb) -> boolean (H,E,nb) with the R largest per row (R may be an (H,E) array for adaptive)."""
    order = np.argsort(-score, -1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(score.shape[-1])[None, None], -1)
    return rank < np.asarray(R)[..., None]


res, diag = [], []
for C in CORPORA:
    for Li, L in ((Li, Lc) for Li in LAYERS for Lc in head_chunks(kvdata.load_layer(C, Li))):
        sl = slice(0, None, STRIDE)
        q, qpos = L["q"][sl].astype(np.float64), L["qpos"][sl]
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)
        T, H, d = K.shape
        E, Hq, _ = q.shape
        G, s = Hq // H, d ** -0.5
        qh = q.reshape(E, H, G, d).transpose(1, 2, 0, 3)
        sr = scores(q, K, qpos)
        o_ref = attend(softmax(sr), V)
        err = lambda o: float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
        Vh = V.transpose(1, 0, 2)
        b0 = B                                                         # block 0 (with the sinks) stays exact
        nb = (int(qpos.min()) - 127 - b0) // B                         # full blocks before every query's window
        tok_blk = np.full(T, -1); tok_blk[b0:b0 + nb * B] = np.arange(nb).repeat(B)
        causal = np.arange(T)[None, :] <= qpos[:, None]                # (E,T)
        base = (causal & (tok_blk[None] < 0))[None, None]              # exact tier outside the summary region
        ctx = (qpos + 1).mean()                                        # tokens a full read moves
        srb = sr[..., b0:b0 + nb * B].reshape(H, G, E, nb, B)          # true per-block log mass (diagnostics, oracle)
        mt = srb.max(-1)
        true_lm = mt + np.log(np.exp(srb - mt[..., None]).sum(-1))
        wt = np.exp(true_lm - true_lm.max(-1, keepdims=True))
        dlm = lambda lm: (float((wt * np.abs(lm - true_lm)).sum(-1).mean() / wt.sum(-1).mean()),
                         float((wt * (lm - true_lm)).sum(-1).mean() / wt.sum(-1).mean()))
        for r, o1 in [(None, False)] + [(r, False) for r in RS] + [(8, True)]:
            st = block_stats(K, V, b0, nb, r or 0, o1)
            ex_extra = np.zeros((H, 1, E, T), bool)
            if o1:
                pos = b0 + np.arange(nb)[None] * B + st["out"]         # (H,nb) positions of the per-block outliers
                for h in range(H):
                    ex_extra[h, 0][:, pos[h]] = True
                ex_extra &= causal[None, None]
            if r is None:
                lm_mean, _, _ = summaries(st, qh, s, "mean")
                ub = s * (np.einsum("hged,hnd->hgen", np.maximum(qh, 0), st["mx"]) + np.einsum("hged,hnd->hgen", np.minimum(qh, 0), st["mn"]))
                names = [("drop", "bound", None, ub, None), ("mean", "mass", lm_mean, lm_mean, None)]
                diag.append((C, Li, "mean", dlm(lm_mean)))
            else:
                lm, z, quad = summaries(st, qh, s, "mom")
                tag = f"mom{r}" + ("+o1" if o1 else "")
                if not o1:
                    diag.append((C, Li, tag, dlm(lm)))
                proxy = lm + 0.5 * np.log(quad + 1e-12) + np.log(st["sv"] + 1e-12)[:, None, None]
                names = [(tag, "mass", lm, lm, z), (tag, "err", lm, proxy, z), (tag + " no v-tilt", "mass", lm, lm, None)]
                if r == 8 and not o1:                                  # oracle: true error of each block's summary
                    top = true_lm.max(-1, keepdims=True)
                    num_t = np.einsum("hgenb,hnbv->hgenv", np.exp(srb - top[..., None]),
                                      V[b0:b0 + nb * B].reshape(nb, B, H, -1).transpose(2, 0, 1, 3))
                    val = st["muv"][:, None, None] + s * np.einsum("hgenr,hnvr->hgenv", z, st["W"])
                    gap = np.linalg.norm(num_t - val * np.exp(lm - top)[..., None], axis=-1)
                    del num_t, val
                    names.append((tag, "oracle", lm, np.log(gap + 1e-300), z))
            for name, sel, lmx, score, z_ in names:
                sc = score.max(1)                                      # (H,E,nb): one set per KV head
                plans = [(f"rho={rho:g}", np.full((H, E), max(1, round(rho * nb)))) for rho in RHO]
                if sel == "err":                                       # per-query budget from the error proxy
                    se = np.where(base | ex_extra, sr, -np.inf)
                    tot = np.logaddexp(se.max(-1) + np.log(np.exp(se - se.max(-1, keepdims=True)).sum(-1)),
                                       np.log(np.exp(lmx - lmx.max(-1, keepdims=True)).sum(-1)) + lmx.max(-1)).max(1)
                    m = np.maximum(sc.max(-1), tot)
                    cum = np.cumsum(np.sort(np.exp(sc - m[..., None]), -1), -1)
                    thr = np.exp(tot - m) * st["sv"].mean(-1)[:, None]
                    for tau in TAUS:
                        plans.append((f"tau={tau:g}", nb - (cum <= tau * thr[..., None]).sum(-1)))
                for plan, R in plans:
                    ref = pick(sc, R)                                  # (H,E,nb) refined blocks
                    ref_tok = np.concatenate([np.zeros((H, E, b0), bool), ref.repeat(B, -1),
                                              np.zeros((H, E, T - b0 - nb * B), bool)], -1)[:, None]
                    exact_tok = (base | ex_extra | ref_tok) & causal[None, None]
                    if lmx is None:                                    # drop: no summaries, per-channel bounds only
                        o = combine(sr, Vh, exact_tok, np.zeros((H, G, E, nb)), np.zeros((H, 1, E, nb), bool), st, None, s)
                        sval = 2 * d
                    else:
                        o = combine(sr, Vh, exact_tok, lmx, ~ref[:, None], st, z_, s)
                        rr = int(name[3]) if name.startswith("mom") else 0
                        sval = 2 * d if name == "mean" else 4 * d + rr * d * (1 if z_ is None else 2) + rr + 1   # mu, mu_v (+ U, W, lam, res, bounds)
                    nbytes = (exact_tok[:, 0].sum(-1).mean() * 2 * d + nb * sval) / (ctx * 2 * d)
                    res.append((C, Li, name, sel, plan, float(nbytes), float(ref.mean()), err(o)))
        print(C, Li, L["h0"], f"{time.time()-t0:.0f}s", flush=True)

R_ = np.array(res, dtype=object)
np.save(os.path.join(RES, "e21_raw.npy"), R_)
rows = []
for key in dict.fromkeys(map(tuple, R_[:, 2:5])):
    a = R_[(R_[:, 2] == key[0]) & (R_[:, 3] == key[1]) & (R_[:, 4] == key[2])]
    rows.append((*key, float(a[:, 5].astype(float).mean()), float(a[:, 6].astype(float).mean()), float(a[:, 7].astype(float).mean())))
D = np.array(diag, dtype=object)
drows = [(k, float(np.mean([x[0] for x in D[D[:, 2] == k][:, 3]])), float(np.mean([x[1] for x in D[D[:, 2] == k][:, 3]])))
         for k in dict.fromkeys(D[:, 2])]
write("21_moment_tail.md", f"""# E21 Query-dependent moment summaries for the attention tail (block {B})

Exact tier: 4 sinks + first block, 128-token window, tokens outside full blocks. Every other block is either read
exactly (refined, chosen per query and shared by the GQA group) or served by its summary evaluated for that query.
'bytes' = K/V values moved per decode step relative to a full fp16 read (summaries included); 'refined' = fraction of
blocks read exactly; err = relative error of the attention output vs exact attention.

## Mass estimate (mass-weighted mean |log mass error| of the block summaries, nats)
""" + md_table(["summary", "|d log mass|", "signed (+ = over)"], drows) + """
## All runs
""" + md_table(["summary", "selector", "budget", "bytes", "refined", "attn err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
