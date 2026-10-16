"""E22 The output gate (Qwen3.5 gated attention: out = attn * sigmoid(g), g from the current token) in the error model.
FoveaKV's metrics already include the gate and o_proj ('resid err', E8/E11), but its selection and budget ignore it.
The gate is known before attention runs (same projection as the query), so it is free, query-dependent side
information: a head whose gate is near zero contributes nothing, so its errors do not matter.

Measured here (FoveaKV k = 5%, shared hot set per KV head, nested 2-bit tail / 4-bit hot, exact sinks + window):
  1. gate statistics: per (query, head) rms of sigmoid(g); share of heads that are nearly closed
  2. selection vote: hot set by max over the group of estimated log-prob (E17) vs weighted by the head's gate
     (log-prob + log gate rms: closed heads do not vote)
  3. budget: same total tokens per layer and query, k per KV head proportional to the group's gate rms
     (optionally x the group's estimated tail mass, the FoveaKV error model)
Metric: residual-contribution error, ||W_o (gate * (o - o_ref))|| / ||W_o (gate * o_ref)|| per query (all heads of the
layer together, as the next layer sees it), plus the plain attention-output error."""
from common import *
from foveakv.attention import scores, softmax, attend, logprob
from foveakv.select import topk_rows, k_for
from foveakv.scheme import gain_shape_cache_v2, block_means
from foveakv.quant import NestedCodebook
from foveakv.load import model_weights

t0 = time.time()
CB = NestedCodebook()
Dc, Df = CB.mse_coarse, CB.mse_fine
ST = model_weights()
KF = 0.05
sigmoid = lambda x: 1 / (1 + np.exp(-x))
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]


def tiered(sc, sf, sr, Vc, Vf, Vr, hot, win):
    w = win[None, None]
    h = hot[:, None] & ~w
    p = softmax(np.where(w, sr, np.where(h, sf, sc)))
    pw, ph = p * w, p * h
    return attend(p - pw - ph, Vc) + attend(ph, Vf) + attend(pw, Vr)


res, gst = [], []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        if "gate" not in L:
            raise SystemExit("no gate captured for this model")
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float32)
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos = L["q"][sl], L["qpos"][sl]
        E, Hq, d = q.shape
        T, Hkv, _ = L["k"].shape
        G = Hq // Hkv
        gt = sigmoid(L["gate"][sl]).reshape(E, Hq, d)                          # (E,Hq,d)
        grms = np.sqrt((gt ** 2).mean(-1))                                     # (E,Hq)
        gh = grms.T.reshape(Hkv, G, E)                                         # (Hkv,G,E)
        sr = scores(q, L["k"], qpos)
        o_ref = attend(softmax(sr), L["v"])
        r_ref = (o_ref * gt).reshape(E, -1) @ Wo.T
        def errs(o):
            r = (o * gt).reshape(E, -1) @ Wo.T
            a = float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
            return a, float((np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean())
        t = np.arange(T)[None, :]
        A = (t < 4) | ((t > qpos[:, None] - 128) & (t <= qpos[:, None]))
        gs = gain_shape_cache_v2(L, P, block=128, smooth=SMOOTH)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc); Kf = Mk + (gs["Kf"] - Mk) / np.sqrt(1 - Df)
        Vc = Mv + (gs["Vc"] - Mv) / (1 - Dc); Vf = Mv + (gs["Vf"] - Mv) / (1 - Df)
        sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
        lp = np.where(A[None, None], -np.inf, logprob(np.where(np.isfinite(sr), sc, -np.inf)))   # (Hkv,G,E,T)
        k = k_for(qpos, KF)                                                    # (E,)
        # estimated tail mass per query head at the uniform budget (coarse logits; FoveaKV error model: mass x code error)
        pc = softmax(np.where(np.isfinite(sr), sc, -np.inf))
        g_grp = gh.mean(1)                                                     # (Hkv,E) group gate rms
        gst.append((C, Li, grms.mean(), float((grms < 0.1).mean()), float((grms < 0.25).mean()),
                    float(np.std(np.log(grms), 0).mean()), float(np.corrcoef(grms.ravel(), np.linalg.norm(o_ref, axis=-1).ravel())[0, 1])))
        votes = {"max log-prob (E17)": lp.max(1),
                 "gate-weighted vote": (lp + np.log(gh)[..., None]).max(1)}
        budgets = {"uniform k": np.broadcast_to(k, (Hkv, E))}
        def realloc(w):                                                        # w (Hkv,E) >= 0, same total per query
            kk = np.floor(w / w.sum(0, keepdims=True) * Hkv * k[None]).astype(int)
            return np.minimum(np.maximum(kk, 1), (qpos + 1)[None])
        budgets["k ~ gate"] = realloc(g_grp)
        # tail mass at the uniform set, per KV head (gate-weighted mean over the group)
        hot_u = topk_rows(lp.max(1), budgets["uniform k"])
        tail = (pc * ~(hot_u[:, None] | A[None, None])).sum(-1)                # (Hkv,G,E)
        budgets["k ~ gate x tail"] = realloc((gh * tail).mean(1) + 1e-6)
        budgets["k ~ tail (no gate)"] = realloc(tail.mean(1) + 1e-6)
        for vn, v in votes.items():
            for bn, kk in budgets.items():
                if vn != "max log-prob (E17)" and bn != "uniform k":
                    continue
                hot = topk_rows(v, kk)
                a, r = errs(tiered(sc, sf, sr, Vc, Vf, L["v"], hot, A))
                res.append((C, Li, vn, bn, a, r))
        hot = topk_rows(votes["gate-weighted vote"], budgets["k ~ gate x tail"])
        a, r = errs(tiered(sc, sf, sr, Vc, Vf, L["v"], hot, A))
        res.append((C, Li, "gate-weighted vote", "k ~ gate x tail", a, r))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
Gs = np.array(gst, dtype=object)
np.save(os.path.join(RES, "e22_raw.npy"), R)
rows = []
for key in dict.fromkeys(map(tuple, R[:, 2:4])):
    a = R[(R[:, 2] == key[0]) & (R[:, 3] == key[1])]
    rows.append((*key, float(a[:, 4].astype(float).mean()), float(a[:, 5].astype(float).mean())))
grows = [(f"L{Li}", *[float(Gs[Gs[:, 1] == Li][:, j].astype(float).mean()) for j in range(2, 7)]) for Li in LAYERS]
write("22_gate.md", f"""# E22 The output gate in FoveaKV's selection and budget (k = 5% mean per KV head)

## Gate statistics (rms of sigmoid(g) per query and head)
""" + md_table(["layer", "mean", "share < 0.1", "share < 0.25", "std of log rms across heads", "corr(gate, |o|)"], grows) + """
## Errors (resid = after gate and o_proj, what the next layer sees)
""" + md_table(["hot-set vote", "budget per KV head", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
