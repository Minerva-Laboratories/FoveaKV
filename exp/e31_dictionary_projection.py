"""E31 Local cache bases WITHOUT storing a basis per chunk (follow-up of E30, same capture and evaluation).

Bound (docs/projection_theory.md sec. 7): if every code is read through one current rank-r decoder, all reconstructions
lie in one r-dim subspace, so no update rule can beat the best fixed rank-r subspace of the whole sequence. E30's local
gain needs more than r dims in the stored object plus a per-chunk choice inside it. Arms (r = 64 keys / 128 values;
keys in the query-whitened metric, values against the chunk's partial attention outputs):
  global     the initial (prompt) basis, top r                                    (stores: initial basis)
  local      exact PCA of each chunk, own basis per chunk                         (stores: one basis per chunk; E30)
  reencode   local basis, but at every chunk all old codes are re-encoded into the newest basis (stores: current only)
  coordsel   initial FULL rotated basis, each chunk keeps its r best coordinates  (stores: initial basis + r indices / chunk)
  dict<R>    growing dictionary: initial basis + atoms appended online from chunk residuals (top-m residual eigvecs,
             frozen once added, at most R atoms); each chunk picks r atoms greedily (stores: current dictionary +
             r indices / chunk)
  static     fixed basis from the other corpus."""
from common import *

t0 = time.time()
P0, NEV, WIN, NS, ROT = 7680, 512, 128, 4, 64
RK, RV, MNEW = 64, 128, 16
SMOKE = bool(os.environ.get("FOVEAKV_E31_SMOKE"))
HEADS = [0] if SMOKE else [0, 1, 2, 3]
CHUNKS = (256, 1024)


def psd_pow(A, p, eps=1e-6):
    w, U = np.linalg.eigh((A + A.T) / 2)
    w = np.maximum(w, eps * max(w.max(), 1e-30))
    return (U * w ** p) @ U.T


def eigtop(S, r):
    w, U = np.linalg.eigh((S + S.T) / 2)
    return U[:, ::-1][:, :r]


def greedy(A, S, r):
    """Pick r columns of A (unit norm) greedily maximizing the captured energy tr(P S P) of their span.
    Returns (indices, Q orthonormal basis of their span)."""
    Ap, SAp = A.copy(), S @ A
    chosen, Qs = [], []
    for _ in range(r):
        den = (Ap * Ap).sum(0)
        sc = (Ap * SAp).sum(0) / np.maximum(den, 1e-12)
        sc[den < 1e-6] = -np.inf
        sc[chosen] = -np.inf
        j = int(np.argmax(sc))
        q = Ap[:, j] / np.sqrt(den[j])
        coef = q @ Ap
        SAp -= np.outer(S @ q, coef)
        Ap -= np.outer(q, coef)
        chosen.append(j); Qs.append(q)
    return chosen, np.stack(Qs, 1)


def attn(q, K, V, qpos, s):
    sc = s * q @ K.T
    sc[np.arange(K.shape[0])[None] > qpos[:, None]] = -np.inf
    sc -= sc.max(1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(1, keepdims=True)
    return p, p @ V


res = []
corpora = ["prose"] if SMOKE else ["prose", "code"]
layers = [11] if SMOKE else [11, 27]
for C_ in corpora:
    other = "code" if C_ == "prose" else "prose"
    for Li in layers:
        L = kvdata.load_layer(C_, Li)
        Lo = kvdata.load_layer(other, Li)
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)
        T, H, d = K.shape
        Q, QP = L["q"].astype(np.float64), L["qpos"]
        Hq = Q.shape[1]
        G, s = Hq // H, d ** -0.5
        ev = np.where(QP >= P0)[0][:NEV]
        qpos_e = QP[ev]
        for h in HEADS:
            Kh, Vh = K[:, h], V[:, h]
            Kn = Kh[:, ROT:]
            dk = d - ROT
            qh = Q[:, h * G:(h + 1) * G]
            K32, V32 = Kh.astype(np.float32), Vh.astype(np.float32)
            t_ = np.arange(T)[None]
            EX = ((t_ > qpos_e[:, None] - WIN) & (t_ <= qpos_e[:, None])) | (t_ < NS)
            FUT = t_ > qpos_e[:, None]
            ref = [attn(qh[ev, g], Kh, Vh, qpos_e, s) for g in range(G)]
            SC = [s * qh[ev, g].astype(np.float32) @ K32.T for g in range(G)]
            PR = [r_[0].astype(np.float32) for r_ in ref]

            def evaluate(Khat_n=None, Vhat=None):
                ea = []
                dK = None if Khat_n is None else (Khat_n - Kn).astype(np.float32)
                dV = None if Vhat is None else (Vhat - Vh).astype(np.float32)
                for g in range(G):
                    if dK is None:
                        p = PR[g]
                    else:
                        sc = np.where(EX, SC[g], SC[g] + s * qh[ev, g][:, ROT:].astype(np.float32) @ dK.T)
                        sc[FUT] = -np.inf
                        sc -= sc.max(1, keepdims=True)
                        p = np.exp(sc); p /= p.sum(1, keepdims=True)
                    o = p @ V32
                    if dV is not None:
                        o = o + (p * ~EX) @ dV
                    o0 = ref[g][1]
                    ea.append(np.linalg.norm(o - o0, axis=1) / np.linalg.norm(o0, axis=1))
                return float(np.concatenate(ea).mean())

            # keys: whitened coordinates y = (k - mu) M^1/2 (prompt queries); values: x = v - mu_v
            muk = Kn[:P0].mean(0)
            qn = qh[QP < P0][:, :, ROT:].reshape(-1, dk)
            Mq = qn.T @ qn / len(qn)
            Mh, Mi = psd_pow(Mq, .5), psd_pow(Mq, -.5)
            Yk = (Kn - muk) @ Mh
            muv = Vh[:P0].mean(0)
            Xv = Vh - muv
            # prompt targets for the initial bases: whitened key covariance; output covariance of the last 2048 prompt queries
            Sk0 = Yk[:P0].T @ Yk[:P0] / P0
            pr2 = np.where((QP < P0) & (QP >= P0 - 2048))[0]
            O2 = np.concatenate([attn(qh[pr2, g].astype(np.float32), K32[:P0], V32[:P0], QP[pr2], s)[1] for g in range(G)])
            Sv0 = (O2 - muv).T @ (O2 - muv) / len(O2)
            B0k, B0v = eigtop(Sk0, dk), eigtop(Sv0, d)                  # full rotated initial bases
            # static references
            Ko = Lo["k"][:P0, h, ROT:].astype(np.float64); mo = Ko.mean(0)
            U = eigtop(np.cov(Ko.T, bias=True), RK)
            res.append((C_, Li, h, "k", 0, "static (other corpus)", evaluate(Khat_n=(Kn - mo) @ U @ U.T + mo), 0))
            Vo = Lo["v"][:P0, h].astype(np.float64); mov = Vo.mean(0)
            U = eigtop(np.cov(Vo.T, bias=True), RV)
            res.append((C_, Li, h, "v", 0, "static (other corpus)", evaluate(Vhat=(Vh - mov) @ U @ U.T + mov), 0))
            U = B0k[:, :RK]
            res.append((C_, Li, h, "k", 0, "global (initial basis)", evaluate(Khat_n=Yk @ U @ U.T @ Mi + muk), 0))
            U = B0v[:, :RV]
            res.append((C_, Li, h, "v", 0, "global (initial basis)", evaluate(Vhat=Xv @ U @ U.T + muv), 0))
            for CH in CHUNKS:
                nch = T // CH
                # per-chunk targets: keys = whitened chunk covariance; values = covariance of the chunk's partial outputs
                Sk_b, Sv_b = [], []
                for b in range(nch):
                    sl = slice(b * CH, (b + 1) * CH)
                    Sk_b.append(Yk[sl].T @ Yk[sl] / CH)
                    m = np.where((QP >= b * CH) & (QP < (b + 1) * CH))[0]
                    if len(m):
                        parts = []
                        for g in range(G):
                            p, _ = attn(qh[m, g].astype(np.float32), K32[:(b + 1) * CH], V32[:(b + 1) * CH], QP[m], s)
                            parts.append(p[:, sl] @ Xv[sl])
                        Pm = np.concatenate(parts)
                        Sv_b.append(Pm.T @ Pm / len(Pm))
                    else:
                        Sv_b.append(Xv[sl].T @ Xv[sl] / CH)
                for kv, X, S_b, B0, r, dim in (("k", Yk, Sk_b, B0k, RK, dk), ("v", Xv, Sv_b, B0v, RV, d)):
                    back = (lambda Y: Y @ Mi + muk) if kv == "k" else (lambda Y: Y + muv)
                    put = (lambda Y: dict(Khat_n=back(Y))) if kv == "k" else (lambda Y: dict(Vhat=back(Y)))
                    # local (own basis per chunk) and reencode (current basis only)
                    loc = np.empty_like(X); rec = np.empty_like(X)
                    for b in range(nch):
                        sl = slice(b * CH, (b + 1) * CH)
                        U = eigtop(S_b[b], r)
                        loc[sl] = X[sl] @ U @ U.T
                        rec[:b * CH] = rec[:b * CH] @ U @ U.T
                        rec[sl] = loc[sl]
                    res.append((C_, Li, h, kv, CH, "local (basis per chunk)", evaluate(**put(loc)), r * dim * 2 / CH))
                    res.append((C_, Li, h, kv, CH, "reencode (current basis only)", evaluate(**put(rec)), 0))
                    # coordinate selection in the initial full basis
                    cs = np.empty_like(X)
                    for b in range(nch):
                        sl = slice(b * CH, (b + 1) * CH)
                        _, Qb = greedy(B0, S_b[b], r)
                        cs[sl] = X[sl] @ Qb @ Qb.T
                    res.append((C_, Li, h, kv, CH, "coordsel (initial full basis)", evaluate(**put(cs)), r / CH))
                    # replayable column swaps: U_b = U_{b-1} with its m weakest columns (for this chunk) replaced by the
                    # top-m residual directions; the log (m new directions + m slot indices per chunk) rebuilds every
                    # chunk's basis from the initial one, and its end is the current basis
                    for msw in (4, 8, 16):
                        U = B0[:, :r].copy()
                        sw = np.empty_like(X)
                        for b in range(nch):
                            sl = slice(b * CH, (b + 1) * CH)
                            S = S_b[b]
                            en = np.einsum("dr,de,er->r", U, S, U)
                            drop = np.argsort(en)[:msw]
                            keep = np.setdiff1d(np.arange(r), drop)
                            Uk = U[:, keep]
                            Pr = np.eye(dim) - Uk @ Uk.T
                            U[:, drop] = eigtop(Pr @ S @ Pr, msw)
                            sw[sl] = X[sl] @ U @ U.T
                        res.append((C_, Li, h, kv, CH, f"swap{msw} (initial basis + swap log)", evaluate(**put(sw)),
                                    msw * dim * 2 / CH))
                    # growing dictionary of frozen atoms
                    for mult in (2, 4):
                        Rmax = mult * dim
                        A = B0.copy()
                        dc = np.empty_like(X)
                        for b in range(nch):
                            sl = slice(b * CH, (b + 1) * CH)
                            S = S_b[b]
                            _, Qb = greedy(A, S, r)
                            if A.shape[1] < Rmax:
                                Pr = np.eye(dim) - Qb @ Qb.T
                                cand = eigtop(Pr @ S @ Pr, MNEW)
                                Ac = np.concatenate([A, cand], 1)
                                idx, Qb = greedy(Ac, S, r)
                                sel_new = [j for j in idx if j >= A.shape[1]]
                                new = sel_new[: Rmax - A.shape[1]]
                                A = np.concatenate([A, Ac[:, new]], 1)
                                if len(new) < len(sel_new):              # no room for all: re-pick from the dictionary
                                    Qb = greedy(A, S, r)[1]
                            dc[sl] = X[sl] @ Qb @ Qb.T
                        res.append((C_, Li, h, kv, CH, f"dict{mult}d (current dictionary)", evaluate(**put(dc)),
                                    A.shape[1] * dim * 2 / T))
            print(C_, Li, h, f"{time.time() - t0:.0f}s", flush=True)
            np.save(os.path.join(RES, "e31_raw.npy"), np.array(res, dtype=object))

rows = []
for kv in ("k", "v"):
    for CH, arm in dict.fromkeys((x[4], x[5]) for x in res if x[3] == kv):
        sel = [x for x in res if x[3] == kv and x[4] == CH and x[5] == arm]
        rows.append((kv, CH or "-", arm, float(np.mean([x[6] for x in sel])), float(np.mean([x[7] for x in sel]))))
write("31_dictionary_projection.md", f"""# E31 Local bases without per-chunk storage: keys r = {RK} / 192 (query-whitened), values r = {RV} / 256

Relative attention-output error (pre-gate), mean over {corpora} x layers {layers} x heads {HEADS}; evaluation as E29/E30.
"extra B/token" = storage beyond the r codes per token and head (per-chunk bases, indices, or the dictionary amortized
over this {T}-token sequence; the dictionary is a constant, so its share falls with context). Codes: {RK * 2} B (keys),
{RV * 2} B (values) per token and head in bf16.

""" + md_table(["", "chunk", "arm", "attn err", "extra B/token"], rows) + f"""
runtime {time.time() - t0:.0f}s
""")
