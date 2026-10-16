"""E26 Prefill (compute) version of the E25 tree: causal, incremental, decisions shared by a 16-query tile.

Needs a capture with queries at most positions (tools/capture_kv_stream.py --eval 7680; tag qwen3.5-9b-pf).
The sequence is cut into chunks of S tokens. A completed chunk gets a tree: the chunk itself is the parent node and
its keys are clustered (k-means in the query metric E[q q^T]^(1/2)) into S/64 child nodes, each with the second-cumulant
summary of E24/E25 (rank 2, sigma-scaled). A query tile (16 consecutive positions, one tensor-core row block) in chunk c
attends exactly to the first 64 tokens and to its own chunk (causal); for each earlier chunk it uses the chunk summary,
or, for the Pp heaviest chunks (estimated mass, max over the tile's queries and the GQA group), the child summaries,
opening the heaviest children (exact keys) until the token budget rho of the past tokens. All decisions are per tile,
so the kernel can gather the opened keys once per tile.
Compute = sum over queries of (exact tokens + opened tokens + node evaluations x (4 + 2r) / 2) / sum of (t + 1) (dense
causal). Errors on the evaluated tiles: attention output and residual contribution (gate, o_proj).
FOVEAKV_E26_S (chunk, default 1024), FOVEAKV_E25_SIGMA (default 0.5), FOVEAKV_E26_TILES (evaluated tiles, default 24)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.load import model_weights

t0 = time.time()
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
B, RK, TQ, WIN = 64, 2, 16, 128
S_ = int(os.environ.get("FOVEAKV_E26_S", "1024"))
SIG = float(os.environ.get("FOVEAKV_E25_SIGMA", "0.5"))
NT = int(os.environ.get("FOVEAKV_E26_TILES", "24"))
RHO = (0.02, 0.05, 0.1, 0.2)
PPS = (1, 2, 4, 1000)                         # 1000 = open every past chunk (flat over children)
rng = np.random.default_rng(0)
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]


def kmeans(X, k, iters=12):
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((X ** 2).sum(-1)[:, None] - 2 * X @ C.T + (C ** 2).sum(-1)[None]).argmin(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab


def stats(k, v):
    """One node: dict of its summary pieces (k (n,d), v (n,d))."""
    mu, muv = k.mean(0), v.mean(0)
    X, Y = k - mu, v - muv
    _, Sv, Vt = np.linalg.svd(X, full_matrices=False)
    lam = Sv ** 2 / len(k)
    r = min(RK, len(lam))
    U = np.zeros((k.shape[1], RK)); U[:, :r] = Vt[:r].T
    lr = np.zeros(RK); lr[:r] = lam[:r]
    return dict(mu=mu, muv=muv, U=U, lam=lr, res=(lam.sum() - lam[:r].sum()) / (k.shape[1] - r),
                W=Y.T @ (X @ U) / len(k), logn=np.log(len(k)), mx=k.max(0), mn=k.min(0))


def summarize(nodes, qh, s):
    """nodes: list of stats dicts (one KV head); qh (G,E,d) -> log mass (G,E,n), value (G,E,n,d)."""
    st = {k_: np.stack([x[k_] for x in nodes]) for k_ in nodes[0]}
    qn2 = (qh ** 2).sum(-1)[..., None]
    a = s * qh @ st["mu"].T
    z = np.einsum("ged,ndr->genr", qh, st["U"])
    quad = s * s * ((st["lam"] * z ** 2).sum(-1) + st["res"] * (qn2 - (z ** 2).sum(-1)))
    ub = s * (np.maximum(qh, 0) @ st["mx"].T + np.minimum(qh, 0) @ st["mn"].T)
    lm = np.minimum(st["logn"] + a + SIG * quad / 2, st["logn"] + ub)
    val = st["muv"] + SIG * s * np.einsum("genr,ndr->gend", z, st["W"])
    return lm, val


res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float32)
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)
        T, H, d = K.shape
        Q, QP = L["q"].astype(np.float64), L["qpos"]
        Hq = Q.shape[1]
        G, s = Hq // H, d ** -0.5
        GT = sigmoid(L["gate"]).reshape(len(QP), Hq, d) if "gate" in L else np.ones((len(QP), Hq, d))
        nch = T // S_
        # query metric per KV head from every 7th query
        QM = []
        for h in range(H):
            qm = Q[::7].reshape(-1, H, G, d)[:, h].reshape(-1, d)
            w_, U_ = np.linalg.eigh(qm.T @ qm / len(qm))
            QM.append(U_ @ np.diag(np.sqrt(np.maximum(w_, 0))) @ U_.T)
        # trees of the completed chunks (chunk 0 without its first 64 tokens, which stay exact)
        # the last WIN tokens of each chunk are two positional nodes (read exactly when inside a query's recent window)
        trees = {}
        for c in range(nch):
            lo, hi = max(c * S_, B), (c + 1) * S_
            for h in range(H):
                k, v = K[lo:hi - WIN, h], V[lo:hi - WIN, h]
                nn = max(1, (len(k) + B - 1) // B)
                lab = kmeans(k @ QM[h], nn)
                kids = [(np.nonzero(lab == j)[0] + lo, stats(k[lab == j], v[lab == j])) for j in range(nn) if (lab == j).any()]
                kids += [(np.arange(a, a + B), stats(K[a:a + B, h], V[a:a + B, h])) for a in range(hi - WIN, hi, B)]
                trees[(c, h)] = (stats(K[lo:hi, h], V[lo:hi, h]), kids)
        # evaluated tiles: spread over the positions that have at least one completed chunk behind them
        starts = QP[0] + np.arange(0, len(QP) - TQ + 1, TQ)
        starts = starts[starts >= S_]
        pick = starts[np.linspace(0, len(starts) - 1, NT).astype(int)]
        acc = {}
        for t0_ in pick:
            rows = np.arange(t0_ - QP[0], t0_ - QP[0] + TQ)
            q, qpos, gt = Q[rows], QP[rows], GT[rows]
            c_own = int(qpos[0] // S_)
            sr = scores(q, K, qpos)
            o_ref = attend(softmax(sr), V)
            r_ref = (o_ref * gt).reshape(TQ, -1) @ Wo.T
            qh = q.reshape(TQ, H, G, d).transpose(1, 2, 0, 3)                  # (H,G,TQ,d)
            exact_base = np.zeros(T, bool); exact_base[:B] = True; exact_base[c_own * S_:] = True
            past = c_own * S_ - B
            dense = (qpos + 1).sum()
            # baseline: tile-shared top-k on exact logits (oracle scores; the tail is dropped), same exact tier
            for HYB, rho in [(hy, r_) for hy in (False, True) for r_ in RHO]:
                outs, cost = [], 0.0
                for h in range(H):
                    ex = exact_base.copy(); ex[max(0, int(qpos[0]) - WIN):] = True
                    sc_ = np.where(ex[None, None], -np.inf, sr[h]).max((0, 1))
                    kk = int(rho * past)
                    if kk:
                        ex[np.argsort(-sc_)[:kk]] = True
                    se = np.where(ex[None, None] & (np.arange(T)[None, None] <= qpos[None, :, None]), sr[h], -np.inf)
                    if HYB:                                                     # + summaries of untouched nodes
                        kids = [kd for c in range(c_own) for kd in trees[(c, h)][1] if not ex[kd[0]].any()]
                        if kids:
                            klm, kval = summarize([kd[1] for kd in kids], qh[h], s)
                            M = np.maximum(se.max(-1), klm.max(-1))[..., None]
                            pe, pm = np.exp(se - M), np.exp(klm - M)
                            outs.append((pe @ V[:, h] + np.einsum("gtn,gtnd->gtd", pm, kval)) / (pe.sum(-1) + pm.sum(-1))[..., None])
                            cost += G * (ex[None] & (np.arange(T)[None] <= qpos[:, None])).sum() + G * TQ * len(kids) * (4 + 2 * RK) / 2
                            continue
                    pe = np.exp(se - se.max(-1, keepdims=True))
                    outs.append(pe @ V[:, h] / pe.sum(-1)[..., None])
                    cost += G * (ex[None] & (np.arange(T)[None] <= qpos[:, None])).sum()
                o = np.stack(outs).transpose(2, 0, 1, 3).reshape(TQ, Hq, d)
                r = (o * gt).reshape(TQ, -1) @ Wo.T
                a_ = acc.setdefault(("topk+sum" if HYB else "topk", rho), [0.0, 0.0, [], []])
                a_[0] += cost; a_[1] += dense * Hq
                a_[2].append((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
                a_[3].append((np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean())
            for pp in PPS:
                for rho in RHO:
                    outs, cost = [], 0.0
                    for h in range(H):
                        par = [summarize([trees[(c, h)][0]], qh[h], s) for c in range(c_own)]
                        plm = np.stack([p[0][..., 0] for p in par], -1)        # (G,TQ,c_own)
                        pval = np.stack([p[1][..., 0, :] for p in par], -2)    # (G,TQ,c_own,d)
                        prev = c_own - 1                                       # holds the recent window: always opened
                        rest = [c for c in np.argsort(-plm.max((0, 1))) if c != prev][:max(0, min(pp, c_own) - 1)]
                        popen = np.array([prev] + rest)
                        terms_l, terms_v = [], []
                        ex = exact_base.copy()
                        kid_lm, kid_val, kid_idx = [], [], []
                        for c in popen:
                            klm, kval = summarize([kd[1] for kd in trees[(c, h)][1]], qh[h], s)
                            kid_lm.append(klm); kid_val.append(kval); kid_idx += [kd[0] for kd in trees[(c, h)][1]]
                        nkid = len(kid_idx)
                        if nkid:
                            klm, kval = np.concatenate(kid_lm, -1), np.concatenate(kid_val, -2)
                            inwin = np.array([kid_idx[j].max() > qpos[0] - WIN for j in range(nkid)])   # recent window
                            order = np.argsort(-np.where(inwin, -np.inf, klm.max((0, 1))))
                            order = order[~inwin[order]]
                            sz = np.array([len(kid_idx[j]) for j in order])
                            nopen = int((np.cumsum(sz) <= rho * past).sum())
                            opened = np.concatenate([np.nonzero(inwin)[0], order[:nopen]]).astype(int)
                            for j in opened:
                                ex[kid_idx[j]] = True
                            keep = np.ones(nkid, bool); keep[opened] = False
                            terms_l.append(klm[..., keep]); terms_v.append(kval[..., keep, :])
                        pkeep = np.ones(c_own, bool); pkeep[popen] = False
                        terms_l.append(plm[..., pkeep]); terms_v.append(pval[..., pkeep, :])
                        se = np.where(ex[None, None] & (np.arange(T)[None, None] <= qpos[None, :, None]), sr[h], -np.inf)
                        lmA, valA = np.concatenate(terms_l, -1), np.concatenate(terms_v, -2)
                        M = np.maximum(se.max(-1), lmA.max(-1) if lmA.shape[-1] else -np.inf)[..., None]
                        pe, pm = np.exp(se - M), np.exp(lmA - M)
                        num = pe @ V[:, h] + np.einsum("gtn,gtnd->gtd", pm, valA)
                        outs.append(num / (pe.sum(-1) + pm.sum(-1))[..., None])
                        exact_cnt = (ex[None] & (np.arange(T)[None] <= qpos[:, None])).sum()
                        cost += G * (exact_cnt + TQ * (c_own + nkid) * (4 + 2 * RK) / 2)
                    o = np.stack(outs).transpose(2, 0, 1, 3).reshape(TQ, Hq, d)
                    r = (o * gt).reshape(TQ, -1) @ Wo.T
                    ae = (np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean()
                    re_ = (np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean()
                    if os.environ.get("FOVEAKV_E26_DIAG") and pp == 1000 and rho == 0.2:
                        print("DIAG", int(t0_), c_own, f"{re_:.4f}", f"{ae:.4f}", flush=True)
                    a_ = acc.setdefault((pp, rho), [0.0, 0.0, [], []])
                    a_[0] += cost; a_[1] += dense * Hq; a_[2].append(ae); a_[3].append(re_)
        for (pp, rho), a_ in acc.items():
            res.append((C, Li, "tile top-k (exact scores, tail dropped)" if pp == "topk" else "tile top-k + summaries of untouched nodes" if pp == "topk+sum" else ("flat" if pp == 1000 else f"Pp={pp}"), f"rho={rho:g}", a_[0] / a_[1], float(np.mean(a_[2])), float(np.mean(a_[3]))))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e26_raw.npy"), R)
rows = []
for key in dict.fromkeys(map(tuple, R[:, 2:4])):
    a = R[(R[:, 2] == key[0]) & (R[:, 3] == key[1])]
    rows.append((*key, *[float(a[:, j].astype(float).mean()) for j in (4, 5, 6)]))
write("26_prefill_tree.md", f"""# E26 Causal prefill tree (chunks of {S_} as parents, query-metric clusters of ~64 as children; sigma {SIG})

Decisions shared by {TQ}-query tiles and the GQA group. compute = (exact + opened + node evaluations) / dense causal
attention, over the evaluated tiles ({NT} per layer and corpus, spread over positions >= {S_}). Pp = past chunks opened.

""" + md_table(["parents opened", "budget", "compute", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
