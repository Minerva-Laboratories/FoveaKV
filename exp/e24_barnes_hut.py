"""E24 Barnes-Hut for exp-dot Nadaraya-Watson (one level: nodes + leaves), with FoveaKV's gain / shape split.

Node of N keys, centroid mu: Z(q) = e^{s q.mu} sum_i e^{s q.(k_i - mu)} ~ N e^{s q.mu + s^2/2 q^T Sigma q} (second cumulant,
top-r Sigma + isotropic rest, clipped at the node's per-channel bound); value ~ mu_v + s Cov(v,k) q. The expansion
parameter is the node's logit spread for the query, delta = s |q| r (r = max |k_i - mu|): the mass is within e^{+-delta}.
Opening a node = reading its tokens exactly. Per query (shared by the GQA group), nodes are opened in order of a score
until a token budget is spent:
  mass    estimated node mass
  err     estimated mass x (e^{delta} - 1), delta from the node radius (Barnes-Hut opening criterion as an error bound)
  oracle  true error of the node summary
Node constructions (same number of nodes as 64-token positional blocks):
  pos        consecutive 64-token blocks (E21)
  kmeans     k-means on the keys
  gain-shape keys k = g u: the top 1% gains are leaves (always exact, counted in the budget); the rest are split into
             4 gain quartiles, each clustered by spherical k-means on u, with nodes per stratum ~ size x mean gain
             (high-gain keys get tighter nodes, since delta grows with g)
Compute (decode, per query head) = exact tokens + opened tokens + node evaluations ((4 + 2r) d / 2d token-equivalents
each), relative to dense attention. Errors: attention output and residual contribution (after gate and o_proj)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.load import model_weights

t0 = time.time()
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
B, RK = 64, 2
STRIDE = int(os.environ.get("FOVEAKV_E24_STRIDE", "4"))
RHO = (0.02, 0.05, 0.1, 0.2)
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]
rng = np.random.default_rng(0)
NODES = os.environ.get("FOVEAKV_E24_NODES", "pos,kmeans,gain-shape").split(",")
TAG = os.environ.get("FOVEAKV_E24_TAG", "")


def kmeans(X, k, iters=12, sphere=False):
    if sphere:
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = (X @ C.T).argmax(-1) if sphere else ((X ** 2).sum(-1)[:, None] - 2 * X @ C.T + (C ** 2).sum(-1)[None]).argmin(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].mean(0)
                C[j] = c / np.linalg.norm(c) if sphere else c
    return lab


def build_labels(Kh, nb, how):
    """Kh (n, d) keys of one KV head in the summary region -> labels (n,) in [0, nb), -1 = leaf (always exact)."""
    n = len(Kh)
    if how == "pos":
        return np.minimum(np.arange(n) // B, nb - 1)
    if how == "kmeans":
        return kmeans(Kh, nb)
    if how == "kmeans-q":                    # query metric: only directions the queries use matter (Li & Gong 2008)
        return kmeans(Kh @ QM, nb)
    g = np.linalg.norm(Kh, axis=-1)
    lab = np.full(n, -1)
    rest = g < np.quantile(g, 0.99)
    q4 = np.quantile(g[rest], [0.25, 0.5, 0.75])
    strata = np.digitize(g, q4)
    w = np.array([(rest & (strata == j)).sum() * g[rest & (strata == j)].mean() for j in range(4)])
    cnt = np.maximum(1, np.round(w / w.sum() * nb).astype(int))
    cnt[-1] += nb - cnt.sum()
    off = 0
    for j in range(4):
        m = rest & (strata == j)
        lab[m] = off + kmeans(Kh[m], cnt[j], sphere=True)
        off += cnt[j]
    return lab


def node_stats(Kh, Vh, lab, nb, d):
    out = dict(mu=np.zeros((nb, d)), muv=np.zeros((nb, d)), U=np.zeros((nb, d, RK)), lam=np.zeros((nb, RK)),
               res=np.zeros(nb), W=np.zeros((nb, d, RK)), logn=np.full(nb, -np.inf), mx=np.zeros((nb, d)),
               mn=np.zeros((nb, d)), rad=np.zeros(nb))
    for j in range(nb):
        m = lab == j
        if m.sum() == 0:
            continue
        k, v = Kh[m], Vh[m]
        n = len(k)
        mu, muv = k.mean(0), v.mean(0)
        X, Y = k - mu, v - muv
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        lam = S ** 2 / n
        r = min(RK, len(lam))
        U = Vt[:r].T
        out["mu"][j], out["muv"][j], out["logn"][j] = mu, muv, np.log(n)
        out["U"][j, :, :r], out["lam"][j, :r] = U, lam[:r]
        out["res"][j] = (lam.sum() - lam[:r].sum()) / (d - r)
        out["W"][j, :, :r] = Y.T @ (X @ U) / n
        out["mx"][j], out["mn"][j] = k.max(0), k.min(0)
        out["rad"][j] = np.linalg.norm(X, axis=-1).max()
    return out


res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float32)
        sl = slice(0, None, STRIDE)
        q, qpos = L["q"][sl].astype(np.float64), L["qpos"][sl]
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)
        T, H, d = K.shape
        E, Hq, _ = q.shape
        G, s = Hq // H, d ** -0.5
        gt = sigmoid(L["gate"][sl]).reshape(E, Hq, d) if "gate" in L else np.ones((E, Hq, d))
        qh = q.reshape(E, H, G, d).transpose(1, 2, 0, 3)
        sr = scores(q, K, qpos)
        o_ref = attend(softmax(sr), V)
        r_ref = (o_ref * gt).reshape(E, -1) @ Wo.T
        def errs(o):
            r = (o * gt).reshape(E, -1) @ Wo.T
            return (float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean()),
                    float((np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean()))
        b0 = B
        nb = (int(qpos.min()) - 127 - b0) // B
        reg = slice(b0, b0 + nb * B)
        causal = np.arange(T)[None, :] <= qpos[:, None]
        inreg = np.zeros(T, bool); inreg[reg] = True
        base = (causal & ~inreg[None])[None, None]                          # (1,1,E,T)
        Vh = V.transpose(1, 0, 2)
        qn = np.linalg.norm(qh, axis=-1)                                     # (H,G,E)
        cache = {}
        for how in NODES:
            base_how, sig = how.split(" s=")[0], float(how.split(" s=")[1]) if " s=" in how else 1.0
            if base_how not in cache:
                labs_h = []
                for h in range(H):
                    global QM
                    qm = L["q"][2::4].reshape(-1, H, G, d)[:, h].reshape(-1, d).astype(np.float64)   # held-out queries
                    w_, U_ = np.linalg.eigh(qm.T @ qm / len(qm))
                    QM = U_ @ np.diag(np.sqrt(np.maximum(w_, 0))) @ U_.T                # E[q q^T]^(1/2)
                    labs_h.append(build_labels(K[reg, h], nb, base_how))
                cache[base_how] = np.stack(labs_h)
            labs = cache[base_how]
            sts = [node_stats(K[reg, h], V[reg, h], labs[h], nb, d) for h in range(H)]
            st = {key: np.stack([x[key] for x in sts]) for key in sts[0]}
            leaf = np.zeros((H, 1, E, T), bool)
            leaf[:, 0, :, reg] = (labs < 0)[:, None, :]
            a = s * np.einsum("hged,hnd->hgen", qh, st["mu"])
            z = np.einsum("hged,hndr->hgenr", qh, st["U"])
            quad = s * s * ((st["lam"][:, None, None] * z ** 2).sum(-1) + st["res"][:, None, None, :] * (qn[..., None] ** 2 - (z ** 2).sum(-1)))
            ub = s * (np.einsum("hged,hnd->hgen", np.maximum(qh, 0), st["mx"]) + np.einsum("hged,hnd->hgen", np.minimum(qh, 0), st["mn"]))
            ln = st["logn"][:, None, None]
            lm = np.minimum(ln + a + sig * quad / 2, ln + ub)               # sig: crossover-kernel scale of Sigma (Ito et al. 2020)                          # (H,G,E,nb), -inf for empty nodes
            delta = s * qn[..., None] * st["rad"][:, None, None]
            # true node mass and summary error (oracle)
            tok_node = np.full((H, T), -1); tok_node[:, reg] = labs
            top = sr.max(-1, keepdims=True)
            ex = np.exp(sr - top)
            num_t = np.zeros((H, G, E, nb, d))
            for h in range(H):
                for j in range(nb):
                    idx = np.nonzero(tok_node[h] == j)[0]
                    if len(idx):
                        num_t[h, :, :, j] = ex[h][..., idx] @ V[idx, h]
            val = st["muv"][:, None, None] + sig * s * np.einsum("hgenr,hndr->hgend", z, st["W"])
            est = np.exp(lm - top)[..., None] * val
            gap = np.linalg.norm(num_t - np.nan_to_num(est), axis=-1)
            sizes = np.stack([np.bincount(labs[h][labs[h] >= 0], minlength=nb) for h in range(H)])  # (H,nb)
            scorers = {"mass": lm, "err": lm + np.log(np.expm1(np.minimum(delta, 50)) + 1e-300), "oracle": np.log(gap + 1e-300)}
            ntail = inreg.sum()
            for sel, scr in scorers.items():
                sc = np.where(np.isfinite(lm), scr, -np.inf).max(1)              # (H,E,nb)
                order = np.argsort(-sc, -1)
                csz = np.cumsum(np.take_along_axis(np.broadcast_to(sizes[:, None], sc.shape), order, -1), -1)
                for rho in RHO:
                    nopen = (csz <= rho * ntail).sum(-1, keepdims=True)
                    ref = np.zeros_like(sc, bool)
                    np.put_along_axis(ref, order, np.arange(nb)[None, None] < nopen, -1)
                    ref_tok = np.zeros((H, 1, E, T), bool)
                    for h in range(H):
                        ref_tok[h, 0][:, reg] = np.take_along_axis(ref[h], np.maximum(labs[h], 0)[None].repeat(E, 0), -1) & (labs[h] >= 0)[None]
                    exact = (base | leaf | ref_tok) & causal[None, None]
                    summ = (~ref)[:, None] & np.isfinite(lm)
                    se = np.where(exact, sr, -np.inf)
                    lmx = np.where(summ, lm, -np.inf)
                    M = np.maximum(se.max(-1, keepdims=True), lmx.max(-1, keepdims=True))
                    pe, pm = np.exp(se - M), np.exp(lmx - M)
                    numr = np.matmul(pe, Vh[:, None]) + np.einsum("hgen,hgend->hged", pm, val)
                    o = (numr / (pe.sum(-1) + pm.sum(-1))[..., None]).transpose(2, 0, 1, 3).reshape(E, Hq, d)
                    ae, re_ = errs(o)
                    flops = (exact[:, 0].sum(-1).mean() + nb * (4 + 2 * RK) / 2) / (qpos + 1).mean()
                    res.append((C, Li, how, sel, f"rho={rho:g}", float(flops), ae, re_))
            print(C, Li, how, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, f"e24_raw{TAG}.npy"), R)
rows = []
for key in dict.fromkeys(map(tuple, R[:, 2:5])):
    a = R[(R[:, 2] == key[0]) & (R[:, 3] == key[1]) & (R[:, 4] == key[2])]
    rows.append((*key, *[float(a[:, j].astype(float).mean()) for j in (5, 6, 7)]))
write(f"24_barnes_hut{TAG}.md", f"""# E24 One-level Barnes-Hut for exp-dot attention: node construction x opening rule (rank-{RK} second-cumulant nodes)

compute = (exact + opened tokens + node evaluations) / dense, decode, per query head. rho = token budget for opened
nodes as a fraction of the summarized region.

""" + md_table(["nodes", "opening", "budget", "compute", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
