"""E30 Online (causal, streaming) learning of the cache projection, against the batch fits of E29.

Same capture and evaluation as E29 (9B prefill capture, prompt [0, 7680), 512 evaluation queries, sinks + window
exact). The whole sequence is written chunk by chunk (C = 256 tokens); each chunk's keys / values are coded with the
basis learned from everything before and including that chunk, and kept with that basis (read with their own basis,
"own"), or read through the newest basis only ("latest": the drift cost of a single basis without re-mapping).
Keys: the score-preserving objective (whitened by the query metric M, QW), values: the output objective (PCA of
attention outputs). All statistics are decayed states, gamma per chunk.

Learners (basis U, r columns, in the objective's metric):
  ideal    exact top-r eigenvectors of the decayed state at every chunk (what the online rules track)
  power    one subspace-iteration step per chunk on the decayed state, warm-started: U <- qr(T U)
  oja      block Oja on the current chunk only: U <- qr(U + eta C_chunk U / tr C_chunk)
  oja_s    same, eta scaled by the chunk's surprise (residual energy fraction / its running mean)
References: batch (E29: one fit on the whole prompt), static (fitted on the other corpus)."""
from common import *
from foveakv.load import model_weights

t0 = time.time()
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
P0, NEV, WIN, NS, ROT = 7680, 512, 128, 4, 64
CH = int(os.environ.get("FOVEAKV_E30_CHUNK", "256"))
RK, RV = 64, 128
SMOKE = bool(os.environ.get("FOVEAKV_E30_SMOKE"))
HEADS = [0] if SMOKE else [0, 1, 2, 3]
GAMMAS = (0.97, 1.0)
ETAS = (0.5, 2.0, 8.0)


def psd_pow(A, p, eps=1e-6):
    w, U = np.linalg.eigh((A + A.T) / 2)
    w = np.maximum(w, eps * max(w.max(), 1e-30))
    return (U * w ** p) @ U.T


def top(T_, r):
    w, U = np.linalg.eigh((T_ + T_.T) / 2)
    return U[:, ::-1][:, :r]


def orth(X):
    return np.linalg.qr(X)[0]


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
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float64)
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)
        T, H, d = K.shape
        Q, QP = L["q"].astype(np.float64), L["qpos"]
        Hq = Q.shape[1]
        G, s = Hq // H, d ** -0.5
        GT = sigmoid(L["gate"].astype(np.float64)).reshape(len(QP), Hq, d)
        ev = np.where(QP >= P0)[0][:NEV]
        qpos_e = QP[ev]
        nch = T // CH
        for h in HEADS:
            Kh, Vh = K[:, h], V[:, h]
            Kn = Kh[:, ROT:]
            qh = Q[:, h * G:(h + 1) * G]
            K32, V32 = Kh.astype(np.float32), Vh.astype(np.float32)
            # exact outputs of every captured query (causal), for the value statistics
            O_all = np.zeros((len(QP), G, d))
            for a in range(0, len(QP), 1024):
                sl = slice(a, a + 1024)
                for g in range(G):
                    O_all[sl, g] = attn(qh[sl, g].astype(np.float32), K32, V32, QP[sl], s)[1]
            # per-chunk statistics
            ck = [Kn[b * CH:(b + 1) * CH] for b in range(nch)]
            cv = [Vh[b * CH:(b + 1) * CH] for b in range(nch)]
            cq, co = [], []
            for b in range(nch):
                m = (QP >= b * CH) & (QP < (b + 1) * CH)
                cq.append(qh[m][:, :, ROT:].reshape(-1, d - ROT))
                co.append(O_all[m].reshape(-1, d))

            # exact evaluation reference and cached scores
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

            def run_keys(learner, gamma, eta=None):
                """Stream the keys: returns reconstructions with own-chunk bases and with the latest basis."""
                Sk = np.zeros((d - ROT, d - ROT)); Mq = np.zeros_like(Sk); n_k = n_q = 0.0; mu = np.zeros(d - ROT)
                U = None; own = np.empty_like(Kn); codes = []; surp = []
                for b in range(nch):
                    x = ck[b]
                    n_k = gamma * n_k + len(x)
                    mu = mu + (x.sum(0) - len(x) * mu) / n_k
                    xc = x - mu
                    Sk = gamma * Sk + xc.T @ xc
                    if len(cq[b]):
                        Mq = gamma * Mq + cq[b].T @ cq[b]; n_q = gamma * n_q + len(cq[b])
                    Mh = psd_pow(Mq / n_q, .5) if n_q > 0 else np.eye(d - ROT)
                    Mi = psd_pow(Mq / n_q, -.5) if n_q > 0 else np.eye(d - ROT)
                    Tm = Mh @ (Sk / n_k) @ Mh                                    # QW target: top eigvecs of M^1/2 S M^1/2
                    if learner == "local":                                       # PCA of this chunk alone (metric)
                        U = top(Mh @ (xc.T @ xc / len(xc)) @ Mh, RK)
                    elif learner == "ideal" or U is None:
                        U = top(Tm, RK)
                    elif learner == "power":
                        U = orth(Tm @ U)
                    else:
                        Cc = Mh @ (xc.T @ xc / len(xc)) @ Mh
                        res_e = 1 - np.trace(U.T @ Cc @ U) / np.trace(Cc)
                        surp.append(res_e)
                        e = eta * (res_e / np.mean(surp) if learner == "oja_s" else 1.0)
                        U = orth(U + e * Cc @ U / np.trace(Cc) * (d - ROT))
                    E_, D_ = U.T @ Mh, Mi @ U
                    own[b * CH:(b + 1) * CH] = xc @ E_.T @ D_.T + mu
                    codes.append((xc @ E_.T, mu))
                E_, D_ = U.T @ Mh, Mi @ U
                latest = np.concatenate([c @ D_.T + m for c, m in codes])
                return own, latest

            def run_values(learner, gamma, eta=None):
                So = np.zeros((d, d)); n_o = n_v = 0.0; mu = np.zeros(d)
                U = None; own = np.empty_like(Vh); codes = []; surp = []
                for b in range(nch):
                    x = cv[b]
                    n_v = gamma * n_v + len(x)
                    mu = mu + (x.sum(0) - len(x) * mu) / n_v
                    oc = co[b] - mu if len(co[b]) else (x - mu)                  # first chunks (no queries): values
                    So = gamma * So + oc.T @ oc; n_o = gamma * n_o + len(oc)
                    Tm = So / n_o
                    if learner == "local":
                        U = top(oc.T @ oc / len(oc), RV)
                    elif learner == "ideal" or U is None:
                        U = top(Tm, RV)
                    elif learner == "power":
                        U = orth(Tm @ U)
                    else:
                        Cc = oc.T @ oc / len(oc)
                        res_e = 1 - np.trace(U.T @ Cc @ U) / np.trace(Cc)
                        surp.append(res_e)
                        e = eta * (res_e / np.mean(surp) if learner == "oja_s" else 1.0)
                        U = orth(U + e * Cc @ U / np.trace(Cc) * d)
                    own[b * CH:(b + 1) * CH] = (x - mu) @ U @ U.T + mu
                    codes.append(((x - mu) @ U, mu))
                latest = np.concatenate([c @ U.T + m for c, m in codes])
                return own, latest

            def add(kv, arm, err):
                res.append((C_, Li, h, kv, arm, err))

            # references: batch on the prompt (E29 QW / output PCA) and static
            Kp = Kn[:P0]; muk = Kp.mean(0)
            qn = qh[QP < P0][:, :, ROT:].reshape(-1, d - ROT)
            M = qn.T @ qn / len(qn)
            Mh, Mi = psd_pow(M, .5), psd_pow(M, -.5)
            U = top(Mh @ np.cov(Kp.T, bias=True) @ Mh, RK)
            add("k", "batch prompt (E29 qw)", evaluate(Khat_n=(Kn - muk) @ (U.T @ Mh).T @ (Mi @ U).T + muk))
            Ko = Lo["k"][:P0, h, ROT:].astype(np.float64); mo = Ko.mean(0)
            U = top(np.cov(Ko.T, bias=True), RK)
            add("k", "static (other corpus, pca)", evaluate(Khat_n=(Kn - mo) @ U @ U.T + mo))
            Vmu = Vh[:P0].mean(0)
            Op = O_all[(QP < P0) & (QP >= P0 - 2048)].reshape(-1, d)
            U = top(np.cov(Op.T, bias=True), RV)
            add("v", "batch prompt (E29 opca)", evaluate(Vhat=(Vh - Vmu) @ U @ U.T + Vmu))
            Vo = Lo["v"][:P0, h].astype(np.float64); mo = Vo.mean(0)
            U = top(np.cov(Vo.T, bias=True), RV)
            add("v", "static (other corpus, pca)", evaluate(Vhat=(Vh - mo) @ U @ U.T + mo))
            for gamma in GAMMAS:
                arms = [("ideal", None), ("power", None)] + ([("local", None)] if gamma == GAMMAS[0] else []) + [(l, e) for l in ("oja", "oja_s") for e in ETAS]
                for learner, eta in arms:
                    nm = f"{learner}{'' if eta is None else f' eta={eta:g}'} gamma={gamma:g}"
                    own, lat = run_keys(learner, gamma, eta)
                    add("k", nm + " own", evaluate(Khat_n=own))
                    add("k", nm + " latest", evaluate(Khat_n=lat))
                    own, lat = run_values(learner, gamma, eta)
                    add("v", nm + " own", evaluate(Vhat=own))
                    add("v", nm + " latest", evaluate(Vhat=lat))
            print(C_, Li, h, f"{time.time() - t0:.0f}s", flush=True)
            np.save(os.path.join(RES, f"e30_raw_C{CH}.npy"), np.array(res, dtype=object))

rows = []
for kv in ("k", "v"):
    for arm in dict.fromkeys(x[4] for x in res if x[3] == kv):
        rows.append((kv, arm, float(np.mean([x[5] for x in res if x[3] == kv and x[4] == arm]))))
write(f"30_online_projection_C{CH}.md", f"""# E30 Online learning of the cache projection (causal, chunks of {CH}), keys r = {RK} / 192, values r = {RV} / 256

Relative attention-output error (pre-gate), mean over {corpora} x layers {layers} x heads {HEADS}; evaluation as E29.
"own" = each chunk read with the basis it was written with; "latest" = all codes read through the newest basis.
Per-chunk bases cost r x d bf16 per chunk and KV head: keys {RK * (192) * 2 / CH:.0f} B/token on top of {RK * 2} B of codes,
values {RV * 256 * 2 / CH:.0f} B/token on top of {RV * 2} B ("local" = exact PCA of each chunk alone, the limit of fast Oja).

""" + md_table(["", "arm", "attn err"], rows) + f"""
runtime {time.time() - t0:.0f}s
""")
