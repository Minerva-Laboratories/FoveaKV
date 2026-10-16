"""E29 Objectives for a sequence-fitted (online / prompt-fitted) low-rank cache projection, and the rank x bits trade
against FoveaKV's codes at equal bytes (theory: docs/projection_theory.md).

Setting (Qwen3.5-9B prefill capture, layers 11 / 27, prose / code, 4 KV heads x 4 query heads): the prompt is positions
[0, 7680); the evaluation queries are the last 512 positions (decode-like), attending causally to everything before
them. Only the 192 position-free key dims are projected (the 64 RoPE dims stay exact); values are projected in all 256
dims. Sinks (first 4) and a 128-token window per query stay exact in every arm. Bases are fitted per KV head from the
PROMPT only (legal at inference), except "static" (fitted on the other corpus, the MLA-like fixed projection) and
"oracle" (fitted on the evaluation queries themselves; upper bound).

Key objectives (cache c = E k~, score q~^T D c, so the key error is (I - DE) k~ and acts as the query perturbation
(I - B) q~ with B = (DE)^T):
  kpca   top-r PCA of centered prompt keys                 (Oja / subspace iteration target)
  qw     min E (q^T (I-DE) k)^2: SVD of M^1/2 S_k^1/2        (score-preserving, spec section 2)
  jdiag  min E ||J_q (I-B) q||^2, J_q = d o / d q = s sum_t p_t (v_t - o) k_t^T, with G = E J^T J taken diagonal in t
         (sum_t p_t^2 ||v_t - o||^2 k_t k_t^T: independent per-key errors)
  jfull  same with the full G = E_q J_q^T J_q (coherent errors)
Value objectives (exact, not first order: o_hat = D E o, so the value error is (I - DE) o):
  vpca   top-r PCA of prompt values
  opca   top-r PCA of prompt attention OUTPUTS o_q
  ow     outputs in the gate x o_proj column-norm metric (oblique: E = U^T W, D = W^-1 U)
Equal bytes: latent coded with FoveaKV's gain-shape nested codes (random rotation, 2-bit coarse / 4-bit fine), 8-bit
per-token absmax, or bf16, at 384 / 768 bits per key and 512 / 1024 bits per value (head), vs full dimension.
Also: measured truncation error vs the first-order prediction J_q (I - B) q, and error vs n_eff = 1 / sum p^2 for
truncation and for quantization (does each average out?)."""
from common import *
from foveakv.load import model_weights
from foveakv.quant import NestedCodebook, GainShape

t0 = time.time()
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
CB = NestedCodebook(4, 2)
P0, NEV, WIN, NS, ROT = 7680, 512, 128, 4, 64
NQS = int(os.environ.get("FOVEAKV_E29_NQ", "96"))          # prompt queries sampled for the J statistics
SMOKE = bool(os.environ.get("FOVEAKV_E29_SMOKE"))
HEADS = [0] if SMOKE else [0, 1, 2, 3]
RNG = np.random.default_rng(0)


def rand_orth(d, seed):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))[0]


def psd_pow(A, p, eps=1e-6):
    w, U = np.linalg.eigh((A + A.T) / 2)
    w = np.maximum(w, eps * max(w.max(), 1e-30))
    return (U * w ** p) @ U.T


def topk_left(X, r):
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :r]


def attn(q, K, V, qpos, s):
    """q (n, d) at positions qpos, causal over K, V (T, d): p (n, T), o (n, dv)."""
    sc = s * q @ K.T
    sc[np.arange(K.shape[0])[None] > qpos[:, None]] = -np.inf
    sc -= sc.max(1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(1, keepdims=True)
    return p, p @ V


def fit_gs(X, Xcal, bits, seed):
    """Gain-shape code of rows of X (rotation fitted with Xcal): reconstruction."""
    d = X.shape[1]
    R = rand_orth(d, seed)
    gs = GainShape(R, CB, 8, center=True).fit(Xcal)
    idx, gq, _, _ = gs.encode(X)
    return gs.decode(idx, gq, fine=(bits == 4))


def code(X, Xcal, bits, seed=0):
    if bits >= 16:                                                    # bf16: round the fp32 mantissa to 8 bits
        u = X.astype(np.float32).view(np.uint32)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return u.view(np.float32).astype(np.float64)
    if bits == 8:
        mu = Xcal.mean(0)
        Y = X - mu
        sc = np.abs(Y).max(1, keepdims=True) / 127 + 1e-30
        return np.round(Y / sc) * sc + mu
    return fit_gs(X, Xcal, bits, seed)


res, pred_rows, neff_rows = [], [], []
corpora = ["prose"] if SMOKE else ["prose", "code"]
layers = [11] if SMOKE else [11, 27]
for C_ in corpora:
    other = "code" if C_ == "prose" else "prose"
    for Li in layers:
        L = kvdata.load_layer(C_, Li)
        Lo = kvdata.load_layer(other, Li)
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float64)
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)      # post-RoPE keys (RoPE part exact)
        T, H, d = K.shape
        Q, QP = L["q"].astype(np.float64), L["qpos"]
        Hq = Q.shape[1]
        G, s = Hq // H, d ** -0.5
        GT = sigmoid(L["gate"].astype(np.float64)).reshape(len(QP), Hq, d)
        ncol = np.linalg.norm(Wo, axis=0).reshape(Hq, d)
        ev = np.where(QP >= P0)[0][:NEV]
        pr = np.where(QP < P0)[0]
        prs = np.sort(RNG.choice(pr, NQS, replace=False))
        qpos_e = QP[ev]
        for h in HEADS:
            Kh, Vh = K[:, h], V[:, h]
            Kn = Kh[:, ROT:]                                                   # position-free key dims (192)
            qh = Q[:, h * G:(h + 1) * G]                                       # (E, G, d)
            # ---- prompt statistics (per KV head, pooled over its G query heads)
            Kp, Vp = Kn[:P0], Vh[:P0]
            muk = Kp.mean(0)
            Sk = np.cov(Kp.T, bias=True)
            qn_pr = qh[pr][:, :, ROT:].reshape(-1, d - ROT)
            M = qn_pr.T @ qn_pr / len(qn_pr)
            Gd = np.zeros((d - ROT, d - ROT)); Gf = np.zeros_like(Gd)
            O_pr, Ow_pr = [], []
            Kc = Kp - muk
            Kc32, Vp32 = Kc.astype(np.float32), Vp.astype(np.float32)
            for g in range(G):
                pp, oo = attn(qh[prs, g], Kh[:P0], Vp, QP[prs], s)
                O_pr.append(oo)
                Ow_pr.append((GT[prs, h * G + g] * ncol[h * G + g]) ** 2)
                P2 = pp ** 2                                                     # sum_n p^2 ||v_t - o_n||^2
                wt = P2.sum(0) * (Vp ** 2).sum(1) - 2 * (P2 * (oo @ Vp.T)).sum(0) + P2.T @ (oo ** 2).sum(1)
                Gd += (Kc * wt[:, None]).T @ Kc
                for i in range(len(prs)):
                    tm = int(QP[prs[i]]) + 1
                    pi = pp[i, :tm].astype(np.float32)
                    J = (Vp32[:tm].T * pi) @ Kc32[:tm] - np.outer(oo[i], pi @ Kc32[:tm])   # (dv, 192)
                    Gf += (J.T @ J).astype(np.float64)
            # output statistics from ALL prompt queries of the last 2048 prompt positions (96 samples overfit r = 128)
            O_pr, Ow_pr, O_rec = [], [], []
            pr2 = pr[QP[pr] >= P0 - 2048]
            for g in range(G):
                _, oo = attn(qh[pr2, g].astype(np.float32), Kh[:P0].astype(np.float32), Vp32, QP[pr2], s)
                O_pr.append(oo); O_rec.append(oo[-512:])
                Ow_pr.append((GT[pr2, h * G + g] * ncol[h * G + g]) ** 2)
            O_pr, Ow_pr, O_rec = np.concatenate(O_pr), np.concatenate(Ow_pr), np.concatenate(O_rec)
            Vmu = Vp.mean(0)
            # ---- static (other corpus) and oracle (evaluation queries) statistics
            Ko = Lo["k"][:P0, h, ROT:].astype(np.float64)
            Vo = Lo["v"][:P0, h].astype(np.float64)
            # exact evaluation reference
            ref = [attn(qh[ev, g], Kh, Vh, qpos_e, s) for g in range(G)]
            O_ev = np.concatenate([o for _, o in ref])

            def kbasis(kind, r):
                """(E (r, 192), D (192, r)) for keys."""
                if kind == "kpca":
                    U = topk_left((Kp - muk).T, r); return U.T, U
                if kind == "static":
                    U = topk_left((Ko - Ko.mean(0)).T, r); return U.T, U
                if kind == "qw":
                    Mh, Mi = psd_pow(M, .5), psd_pow(M, -.5)
                    U = topk_left(Mh @ psd_pow(Sk, .5), r)
                    return U.T @ Mh, Mi @ U                                    # DE = M^-1/2 U U^T M^1/2
                Gm = Gd if kind == "jdiag" else Gf
                Gh, Gi = psd_pow(Gm, .5), psd_pow(Gm, -.5)
                U = topk_left(Gh @ psd_pow(M, .5), r)
                return U.T @ Gi, Gh @ U                                        # DE = B^T, B = G^-1/2 U U^T G^1/2

            def vbasis(kind, r):
                if kind == "vpca":
                    U = topk_left((Vp - Vmu).T, r); return U.T, U
                if kind == "vstatic":
                    U = topk_left((Vo - Vo.mean(0)).T, r); return U.T, U
                if kind == "orecent":
                    U = topk_left((O_rec - Vmu).T, r); return U.T, U
                if kind == "opca":
                    U = topk_left((O_pr - Vmu).T, r); return U.T, U
                if kind == "oracle":
                    U = topk_left((O_ev - Vmu).T, r); return U.T, U
                if kind == "ow":
                    w = np.sqrt(Ow_pr.mean(0)) + 1e-6
                    U = topk_left(((O_pr - Vmu) * w).T, r)
                    return U.T * w[None], U / w[:, None]

            t_ = np.arange(T)[None]
            EX = ((t_ > qpos_e[:, None] - WIN) & (t_ <= qpos_e[:, None])) | (t_ < NS)
            FUT = t_ > qpos_e[:, None]
            Kh32, Vh32 = Kh.astype(np.float32), Vh.astype(np.float32)
            SC = [s * qh[ev, g].astype(np.float32) @ Kh32.T for g in range(G)]
            PR = [r_[0].astype(np.float32) for r_ in ref]
            GTe = [GT[ev, h * G + g] for g in range(G)]
            Wh = [Wo[:, (h * G + g) * d:(h * G + g + 1) * d].astype(np.float32) for g in range(G)]
            R0 = [(ref[g][1] * GTe[g]).astype(np.float32) @ Wh[g].T for g in range(G)]

            def evaluate(Khat_n=None, Vhat=None):
                """Approximate keys (position-free part replaced) / values outside sinks + window."""
                ea, er, outs = [], [], []
                dK = None if Khat_n is None else (Khat_n - Kn).astype(np.float32)
                dV = None if Vhat is None else (Vhat - Vh).astype(np.float32)
                for g in range(G):
                    if dK is None:
                        p = PR[g]
                    else:
                        sa = SC[g] + s * qh[ev, g][:, ROT:].astype(np.float32) @ dK.T
                        sc = np.where(EX, SC[g], sa)
                        sc[FUT] = -np.inf
                        sc -= sc.max(1, keepdims=True)
                        p = np.exp(sc); p /= p.sum(1, keepdims=True)
                    o = p @ Vh32
                    if dV is not None:
                        o = o + (p * ~EX) @ dV
                    o0 = ref[g][1]
                    r1 = (o * GTe[g]).astype(np.float32) @ Wh[g].T
                    ea.append(np.linalg.norm(o - o0, axis=1) / np.linalg.norm(o0, axis=1))
                    er.append(np.linalg.norm(r1 - R0[g], axis=1) / np.linalg.norm(R0[g], axis=1))
                    outs.append(o)
                return np.concatenate(ea), np.concatenate(er), outs

            def add(arm, r, bits, kv, ea, er):
                res.append((C_, Li, h, arm, r, bits, kv, float(ea.mean()), float(er.mean())))

            Kn_c = Kn - muk
            # ---- keys: objectives at bf16 latent
            for r in (32, 64, 96):
                for kind in ("kpca", "static", "qw", "jdiag", "jfull"):
                    E_, D_ = kbasis(kind, r)
                    mu_s = Ko.mean(0) if kind == "static" else muk
                    Kh_n = (Kn - mu_s) @ E_.T @ D_.T + mu_s
                    ea, er, outs = evaluate(Khat_n=Kh_n)
                    add(kind, r, 16, "k", ea, er)
                    if kind in ("kpca", "jfull") and r == 64:                 # first-order prediction check
                        B = (D_ @ E_).T
                        for g in range(G):
                            p0, o0 = ref[g]
                            qn = qh[ev, g][:, ROT:]
                            dq = qn - qn @ B.T                                 # (I - B) q
                            dsc = np.where(EX, 0.0, -s * (dq @ Kn_c.T)) * p0    # p_t ds_t (exact keys: 0)
                            Jdq = dsc @ Vh - o0 * dsc.sum(1, keepdims=True)
                            meas = outs[g] - o0
                            cos = (Jdq * meas).sum(1) / (np.linalg.norm(Jdq, axis=1) * np.linalg.norm(meas, axis=1) + 1e-30)
                            ratio = np.linalg.norm(Jdq, axis=1) / (np.linalg.norm(meas, axis=1) + 1e-30)
                            pred_rows.append((C_, Li, h, kind, float(np.median(cos)), float(np.median(ratio))))
            # ---- values: objectives at bf16 latent
            for r in (32, 64, 128):
                for kind in ("vpca", "vstatic", "opca", "orecent", "ow", "oracle"):
                    E_, D_ = vbasis(kind, r)
                    mu_s = Vo.mean(0) if kind == "vstatic" else Vmu
                    Vhat = (Vh - mu_s) @ E_.T @ D_.T + mu_s
                    ea, er, _ = evaluate(Vhat=Vhat)
                    add(kind, r, 16, "v", ea, er)
            # ---- equal bytes: rank x bits (keys 192 dims, values 256 dims)
            for budget, ladder in ((384, ((192, 2), (96, 4), (48, 8), (24, 16))), (768, ((192, 4), (96, 8), (48, 16)))):
                for r, b in ladder:
                    for kind in (("full",) if r == 192 else ("kpca", "qw", "jfull")):
                        if kind == "full":
                            Kh_n = code(Kn, Kp, b, seed=h)
                        else:
                            E_, D_ = kbasis(kind, r)
                            Z = Kn_c @ E_.T
                            Kh_n = code(Z, Z[:P0], b, seed=h) @ D_.T + muk
                        ea, er, outs = evaluate(Khat_n=Kh_n)
                        add(f"eqbytes{budget}:{kind}", r, b, "k", ea, er)
                        if budget == 384 and kind in ("full", "jfull") and (r, b) in ((192, 2), (96, 4), (24, 16)):
                            pe = ref[0][0]
                            ne = 1 / (pe ** 2).sum(1)
                            qs = np.quantile(ne, [0, .25, .5, .75, 1])
                            e0 = ea[:NEV]
                            ab = np.linalg.norm(outs[0] - ref[0][1], axis=1) / np.sqrt((Vh[:P0] ** 2).sum(1).mean())
                            for a_ in range(4):
                                m_ = (ne >= qs[a_]) & (ne <= qs[a_ + 1])
                                neff_rows.append((C_, Li, h, f"{kind} r{r} {b}b", a_, float(np.median(ne[m_])),
                                                  float(e0[m_].mean()), float(ab[m_].mean())))
            for budget, ladder in ((512, ((256, 2), (128, 4), (64, 8), (32, 16))), (1024, ((256, 4), (128, 8), (64, 16)))):
                for r, b in ladder:
                    for kind in (("full",) if r == 256 else ("vpca", "opca")):
                        if kind == "full":
                            Vhat = code(Vh, Vp, b, seed=10 + h)
                        else:
                            E_, D_ = vbasis(kind, r)
                            Z = (Vh - Vmu) @ E_.T
                            Vhat = code(Z, Z[:P0], b, seed=10 + h) @ D_.T + Vmu
                        ea, er, _ = evaluate(Vhat=Vhat)
                        add(f"eqbytes{budget}:{kind}", r, b, "v", ea, er)
            print(C_, Li, h, f"{time.time() - t0:.0f}s", flush=True)
            np.save(os.path.join(RES, "e29_raw.npy"), np.array(res, dtype=object))

R_ = np.array(res, dtype=object)


def table(kv, arms_filter):
    rows = []
    keys = [(a, r, b) for a, r, b in dict.fromkeys((x[3], x[4], x[5]) for x in res if x[6] == kv and arms_filter(x[3]))]
    for a, r, b in keys:
        sel = [x for x in res if x[3] == a and x[4] == r and x[5] == b and x[6] == kv]
        rows.append((a, r, b, float(np.mean([x[7] for x in sel])), float(np.mean([x[8] for x in sel]))))
    return md_table(["arm", "r", "bits", "attn err", "resid err"], rows)


pt = md_table(["corpus", "layer", "head", "basis", "median cos(pred, meas)", "median |pred|/|meas|"], pred_rows)
nt = {}
for c, l, h, arm, a_, ne, e, ab in neff_rows:
    nt.setdefault((arm, a_), []).append((ne, e, ab))
nrows = [(arm, a_ + 1, *[float(np.mean([x[j] for x in v])) for j in range(3)]) for (arm, a_), v in nt.items()]
write("29_projection_objectives.md", f"""# E29 Sequence-fitted low-rank cache projections: objectives and rank x bits at equal bytes

9B prefill capture, {corpora} x layers {layers} x KV heads {HEADS}; prompt [0, {P0}), evaluation = {NEV} queries after it;
sinks {NS} + window {WIN} exact. Errors: relative attention-output error (pre-gate) and residual error (gate, o_proj),
mean over evaluation queries, then over (corpus, layer, head). Keys: 192 position-free dims projected (RoPE 64 exact).

## Key objectives (bf16 latent)
{table("k", lambda a: not a.startswith("eqbytes"))}
## Value objectives (bf16 latent)
{table("v", lambda a: not a.startswith("eqbytes"))}
## Equal bytes: keys (bits per head-token for the 192 dims)
{table("k", lambda a: a.startswith("eqbytes"))}
## Equal bytes: values (bits per head-token for 256 dims)
{table("v", lambda a: a.startswith("eqbytes"))}
## First-order check: J_q (I - B) q vs measured output error (key truncation, r = 64)
{pt}
## Error vs n_eff = 1 / sum p^2 (quartiles; keys, 384-bit arms; first query head of each group)
{md_table(["arm", "n_eff quartile", "median n_eff", "attn err (rel. to |o|)", "|do| / rms|v|"], nrows)}
runtime {time.time() - t0:.0f}s
""")
