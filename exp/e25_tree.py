"""E25 Two-level Barnes-Hut tree over keys, query dependent, with a precision ladder for the opened leaves.

Per KV head, over the summarized region (as E21/E24): nodes = k-means of the keys in the query metric E[q q^T]^(1/2)
(held-out queries; Li & Gong 2008: only the directions the queries use matter), ~64 keys each; parents = k-means of the
node centroids (~sqrt(#nodes)). Every node and parent carries the second-cumulant summary (rank-2 Sigma + isotropic rest,
scaled by sigma as in the crossover kernel of Ito et al. 2020; clipped at the per-channel bound) and the tilted value
mean mu_v + sigma s Cov(v,k) q.

Per query (shared by the GQA group, max over its query heads): evaluate all parents; open the Pp heaviest; evaluate their
children; open children by estimated mass until the token budget rho; the rest is served by child summaries (inside
opened parents) or parent summaries. 'flat' = every node evaluated (E24). 'ladder': the heaviest half of the opened
tokens is read exactly, the other half through FoveaKV's 2-bit nested code (coarse keys and values, block-centred,
unbiased rescale), counted at 1/4 of an exact token (INT4-class tensor-core throughput).
Compute = exact tokens + opened tokens (x1 or x1/4) + node evaluations ((4 + 2r) d / 2d token-equivalents each),
relative to dense attention, decode, per query head. Errors: attention output and residual contribution (gate, o_proj).
FOVEAKV_E25_SIGMA (default 0.5)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.scheme import gain_shape_cache_v2, block_means
from foveakv.quant import NestedCodebook
from foveakv.load import model_weights

t0 = time.time()
ST = model_weights()
Dc = NestedCodebook().mse_coarse
sigmoid = lambda x: 1 / (1 + np.exp(-x))
B, RK = 64, 2
SIG = float(os.environ.get("FOVEAKV_E25_SIGMA", "0.5"))
STRIDE = int(os.environ.get("FOVEAKV_E24_STRIDE", "4"))
RHO = (0.02, 0.05, 0.1, 0.2)
PPS = (2, 4, 6)
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]
rng = np.random.default_rng(0)


def kmeans(X, k, iters=12):
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((X ** 2).sum(-1)[:, None] - 2 * X @ C.T + (C ** 2).sum(-1)[None]).argmin(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab


def stats(Kh, Vh, lab, n, d):
    out = dict(mu=np.zeros((n, d)), muv=np.zeros((n, d)), U=np.zeros((n, d, RK)), lam=np.zeros((n, RK)), res=np.zeros(n),
               W=np.zeros((n, d, RK)), logn=np.full(n, -np.inf), mx=np.zeros((n, d)), mn=np.zeros((n, d)))
    for j in range(n):
        m = lab == j
        if not m.any():
            continue
        k, v = Kh[m], Vh[m]
        mu, muv = k.mean(0), v.mean(0)
        X, Y = k - mu, v - muv
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        lam = S ** 2 / len(k)
        r = min(RK, len(lam))
        U = Vt[:r].T
        out["mu"][j], out["muv"][j], out["logn"][j] = mu, muv, np.log(len(k))
        out["U"][j, :, :r], out["lam"][j, :r] = U, lam[:r]
        out["res"][j] = (lam.sum() - lam[:r].sum()) / (d - r)
        out["W"][j, :, :r] = Y.T @ (X @ U) / len(k)
        out["mx"][j], out["mn"][j] = k.max(0), k.min(0)
    return out


def summarize(st, qh, s, qn):
    """-> log mass (H,G,E,n), tilted value (H,G,E,n,d)."""
    a = s * np.einsum("hged,hnd->hgen", qh, st["mu"])
    z = np.einsum("hged,hndr->hgenr", qh, st["U"])
    quad = s * s * ((st["lam"][:, None, None] * z ** 2).sum(-1) + st["res"][:, None, None] * (qn[..., None] ** 2 - (z ** 2).sum(-1)))
    ub = s * (np.einsum("hged,hnd->hgen", np.maximum(qh, 0), st["mx"]) + np.einsum("hged,hnd->hgen", np.minimum(qh, 0), st["mn"]))
    ln = st["logn"][:, None, None]
    lm = np.minimum(ln + a + SIG * quad / 2, ln + ub)
    val = st["muv"][:, None, None] + SIG * s * np.einsum("hgenr,hndr->hgend", z, st["W"])
    return lm, val


def lse_terms(terms):
    """terms: list of (logits (H,G,E,N), values (H,G,E,N,d) or (H,1,N,d) for token values via matmul flag)."""
    M = np.max([t[0].max(-1) for t in terms], axis=0)[..., None]
    num, den = 0, 0
    for lg, val, tok in terms:
        p = np.exp(lg - M)
        den = den + p.sum(-1)
        num = num + (np.matmul(p, val) if tok else np.einsum("hgen,hgend->hged", p, val))
    return num / den[..., None]


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
        qn = np.linalg.norm(qh, axis=-1)
        sr = scores(q, K, qpos)
        o_ref = attend(softmax(sr), V)
        r_ref = (o_ref * gt).reshape(E, -1) @ Wo.T
        def errs(o):
            o = o.transpose(2, 0, 1, 3).reshape(E, Hq, d)
            r = (o * gt).reshape(E, -1) @ Wo.T
            return (float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean()),
                    float((np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean()))
        P_ = prefill_len(L)
        gs = gain_shape_cache_v2(L, P_, block=128, smooth=SMOOTH)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = (Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc)).astype(np.float64)
        Vc = (Mv + (gs["Vc"] - Mv) / (1 - Dc)).astype(np.float64)
        sc = scores(q, Kc, qpos)
        b0 = B
        nb = (int(qpos.min()) - 127 - b0) // B
        reg = slice(b0, b0 + nb * B)
        n = nb * B
        npar = int(np.ceil(np.sqrt(nb)))
        causal = np.arange(T)[None, :] <= qpos[:, None]
        inreg = np.zeros(T, bool); inreg[reg] = True
        base = causal & ~inreg[None]                                          # (E,T)
        labs, plabs = [], []
        for h in range(H):
            qm = L["q"][2::4].reshape(-1, H, G, d)[:, h].reshape(-1, d).astype(np.float64)
            w_, U_ = np.linalg.eigh(qm.T @ qm / len(qm))
            QM = U_ @ np.diag(np.sqrt(np.maximum(w_, 0))) @ U_.T
            Xq = K[reg, h] @ QM
            lab = kmeans(Xq, nb)
            cent = np.stack([Xq[lab == j].mean(0) if (lab == j).any() else np.full(d, 1e6) for j in range(nb)])
            pl = kmeans(cent, npar)                                           # parent of each node
            labs.append(lab); plabs.append(pl)
        labs, plabs = np.stack(labs), np.stack(plabs)                          # (H,n), (H,nb)
        stc = [stats(K[reg, h], V[reg, h], labs[h], nb, d) for h in range(H)]
        stp = [stats(K[reg, h], V[reg, h], plabs[h][labs[h]], npar, d) for h in range(H)]
        stc = {k_: np.stack([x[k_] for x in stc]) for k_ in stc[0]}
        stp = {k_: np.stack([x[k_] for x in stp]) for k_ in stp[0]}
        lmc, valc = summarize(stc, qh, s, qn)
        lmp, valp = summarize(stp, qh, s, qn)
        sizes = np.stack([np.bincount(labs[h], minlength=nb) for h in range(H)])            # (H,nb)
        scc, scp = lmc.max(1), lmp.max(1)                                       # (H,E,nb), (H,E,npar)
        Vh, Vch = V.transpose(1, 0, 2)[:, None], Vc.transpose(1, 0, 2)[:, None]
        node_cost = (4 + 2 * RK) / 2
        ctx = (qpos + 1).mean()
        for mode in ["flat"] + [f"tree Pp={pp}" for pp in PPS if pp < npar]:
            if mode == "flat":
                popen = np.ones((H, E, npar), bool)
            else:
                pp = int(mode.split("=")[1])
                popen = np.zeros((H, E, npar), bool)
                np.put_along_axis(popen, np.argsort(-scp, -1)[..., :pp], True, -1)
            celig = np.take_along_axis(popen, np.broadcast_to(plabs[:, None], (H, E, nb)), -1)   # child evaluated
            neval = (npar if mode != "flat" else 0) + celig.sum(-1).mean()
            order = np.argsort(-np.where(celig, scc, -np.inf), -1)
            csz = np.cumsum(np.take_along_axis(np.broadcast_to(sizes[:, None], (H, E, nb)), order, -1), -1)
            for rho in RHO:
                for ladder in (False, True):
                    nopen = np.minimum((csz <= rho * n).sum(-1, keepdims=True), celig.sum(-1, keepdims=True))
                    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(nb)[None, None], -1)
                    copen = rank < nopen                                        # (H,E,nb)
                    if ladder:                                                  # heaviest half of the tokens exact
                        nex = (csz <= rho * n / 2).sum(-1, keepdims=True)
                        cexact = copen & (rank < nex)
                    else:
                        cexact = copen
                    ccoarse = copen & ~cexact
                    tok = lambda m: np.stack([np.concatenate([np.zeros((E, b0), bool), m[h][:, labs[h]], np.zeros((E, T - b0 - n), bool)], -1)
                                              for h in range(H)])[:, None] & causal[None, None]
                    ex_tok = tok(cexact) | base[None, None]
                    co_tok = tok(ccoarse)
                    csum = (celig & ~copen)[:, None]                            # child summaries in use
                    psum = (~popen)[:, None]                                    # parent summaries in use
                    terms = [(np.where(ex_tok, sr, -np.inf), Vh, True), (np.where(co_tok, sc, -np.inf), Vch, True),
                             (np.where(csum, lmc, -np.inf), valc, False), (np.where(psum, lmp, -np.inf), valp, False)]
                    ae, re_ = errs(lse_terms(terms))
                    flops = (ex_tok[:, 0].sum(-1).mean() + 0.25 * co_tok[:, 0].sum(-1).mean() + neval * node_cost) / ctx
                    res.append((C, Li, mode + (" +ladder" if ladder else ""), f"rho={rho:g}", float(flops), float(neval / nb), ae, re_))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
TAG = os.environ.get("FOVEAKV_E25_TAG", "")
np.save(os.path.join(RES, f"e25_raw{TAG}.npy"), R)
rows = []
for key in dict.fromkeys(map(tuple, R[:, 2:4])):
    a = R[(R[:, 2] == key[0]) & (R[:, 3] == key[1])]
    rows.append((*key, *[float(a[:, j].astype(float).mean()) for j in (4, 5, 6, 7)]))
write(f"25_tree{TAG}.md", f"""# E25 Two-level query-metric tree over keys + precision ladder (sigma = {SIG}, rank-{RK} nodes)

compute = (exact + opened tokens [coarse at 1/4] + node evaluations) / dense, decode, per query head; 'nodes evaluated'
= share of leaf-level nodes whose summary was computed (flat = 1). rho = opened-token budget, fraction of the region.

""" + md_table(["mode", "budget", "compute", "nodes evaluated", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
