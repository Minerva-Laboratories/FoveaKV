"""E23 Sparse attention from the error model, run backwards: given an error budget, prune as many (query head, token)
pairs as possible. Dropping a set D of tokens from a head's softmax changes its output by
    o' - o = sum_{i in D} p_i (o - v_i) / (1 - m_D)
and the next layer sees it through the output gate and o_proj, so each pair gets the cost
    c_{h,i} = p_{h,i} * || w_h * (v_i - o_h) ||,   w_{h,c} = sigmoid(g)_{h,c} * ||W_o[:, h, c]||
(gate known before attention; column norms static). By the triangle inequality the sum of the dropped costs bounds the
change of the layer's residual contribution (up to the 1/(1-m) renormalization). Per query, all (head, token) pairs of
the layer are pooled and the cheapest are dropped until the summed cost reaches eps * ||W_o (g * o)||: the budget flows
to heads with closed gates, flat attention or values close to the head's output. A head whose tokens are all dropped
is skipped (output 0).

Arms (FLOPs = kept (head, token) pairs / all causal pairs, QK^T and PV alike):
  top-k      per query head, the k largest exact logits (k a fraction of the context; sinks and window not forced)
  top-p      per query head, smallest set with mass >= 1 - tau (adaptive top-k)
  gate heads full attention, the heads with the smallest gate rms skipped (gate only, no error model)
  backward   the error-model greedy above with exact p and o (upper bound of what the model allows)
  backward/est  same with p and o from FoveaKV's coarse 2-bit keys and values (a realistic estimator; its own pass
             costs ~1/8 of dense attention at 2-bit, not included in FLOPs)
Errors: attn = relative error of the head outputs; resid = relative error of the layer's residual contribution."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.scheme import gain_shape_cache_v2, block_means
from foveakv.quant import NestedCodebook
from foveakv.load import model_weights

t0 = time.time()
CB = NestedCodebook()
Dc = CB.mse_coarse
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
STRIDE = int(os.environ.get("FOVEAKV_E23_STRIDE", "4"))
EPS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
KFR = (0.01, 0.02, 0.05, 0.1, 0.2)
TAU = (0.003, 0.01, 0.03, 0.1, 0.3)
HFR = (0.25, 0.5)
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]


def pair_costs(p, o, V, w):
    """p (H,G,E,T), o (H,G,E,d), V (T,H,d), w (H,G,E,d) -> c (H,G,E,T) = p * ||w * (v - o)||."""
    w2 = w * w
    Vh = V.transpose(1, 0, 2)[:, None]                                       # (H,1,T,d)
    a = np.matmul(w2, (Vh * Vh).transpose(0, 1, 3, 2))                        # sum w^2 v^2
    b = np.matmul(w2 * o, Vh.transpose(0, 1, 3, 2))                           # sum w^2 v o
    c = (w2 * o * o).sum(-1, keepdims=True)
    return p * np.sqrt(np.maximum(a - 2 * b + c, 0))


def backward_keep(c, budget):
    """c (H,G,E,T) costs (-inf / nan-free, 0 for masked), budget (E,) -> keep mask (H,G,E,T). Pools all heads of a query."""
    H, G, E, T = c.shape
    flat = c.transpose(2, 0, 1, 3).reshape(E, -1)
    order = np.argsort(flat, -1)
    cs = np.cumsum(np.take_along_axis(flat, order, -1), -1)
    ndrop = (cs <= budget[:, None]).sum(-1)
    keep = np.ones_like(flat, bool)
    rows = np.repeat(np.arange(E), flat.shape[1]).reshape(E, -1)
    drop = np.arange(flat.shape[1])[None] < ndrop[:, None]
    keep[rows[drop], order[drop]] = False
    return keep.reshape(E, H, G, T).transpose(1, 2, 0, 3)


res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float32)
        sl = slice(0, None, STRIDE)
        q, qpos = L["q"][sl], L["qpos"][sl]
        E, Hq, d = q.shape
        T, H, _ = L["k"].shape
        G = Hq // H
        gt = sigmoid(L["gate"][sl]).reshape(E, Hq, d) if "gate" in L else np.ones((E, Hq, d), np.float32)
        ncol = np.linalg.norm(Wo, axis=0).reshape(Hq, d)                     # ||W_o[:, h, c]||
        w = (gt * ncol[None]).reshape(E, H, G, d).transpose(1, 2, 0, 3)       # (H,G,E,d)
        sr = scores(q, L["k"], qpos)
        valid = np.isfinite(sr)
        pr = softmax(sr)
        o_ref = attend(pr, L["v"])
        r_ref = (o_ref * gt).reshape(E, -1) @ Wo.T
        rn = np.linalg.norm(r_ref, axis=-1)
        npairs = valid.sum() / (H * G * E) * Hq                               # causal pairs per query (all heads)

        def run(keep, arm, setting):
            ps = np.where(keep, pr, 0.0)
            m = ps.sum(-1, keepdims=True)
            ps = np.divide(ps, m, out=np.zeros_like(ps), where=m > 0)        # renormalized; fully pruned head -> 0
            o = attend(ps, L["v"])
            r = (o * gt).reshape(E, -1) @ Wo.T
            a = float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
            rr = float((np.linalg.norm(r - r_ref, axis=-1) / rn).mean())
            fl = float((keep & valid).sum() / valid.sum())
            heads = float((m[..., 0] == 0).mean())
            res.append((C, Li, arm, setting, fl, heads, a, rr))

        if os.environ.get("FOVEAKV_E23_ABL"):                          # where does the gain come from?
            oh = o_ref.reshape(E, H, G, d).transpose(1, 2, 0, 3)
            ones = np.broadcast_to(ncol.mean() * np.sqrt((gt ** 2).mean()), w.shape)
            c_nog = np.where(valid, pair_costs(pr, oh, L["v"], ones), 0.0)     # no gate / o_proj weighting
            c_p = np.where(valid, pr, 0.0)                                      # mass only (pooled top-p)
            c_full = np.where(valid, pair_costs(pr, oh, L["v"], w), 0.0)
            for eps in EPS:
                run(backward_keep(c_nog, eps * rn) & valid, "backward, no gate/W_o weights", f"eps={eps:g}")
                run(backward_keep(c_p, eps * rn / np.median(rn) * 0.05) & valid, "pooled top-p (mass only)", f"eps={eps:g}")
                # per-head budgets: same total, split evenly over the heads (no pooling across heads)
                kh = np.ones((H, G, E, T), bool)
                for h in range(H):
                    for g in range(G):
                        kh[h, g] = backward_keep(c_full[h:h + 1, g:g + 1], eps * rn / Hq)[0, 0]
                run(kh & valid, "backward, per-head budget", f"eps={eps:g}")
            print(C, Li, f"{time.time()-t0:.0f}s", flush=True)
            continue
        srt = -np.sort(-np.where(valid, sr, -np.inf), -1)
        for f in KFR:
            k = np.maximum(1, np.ceil(f * (qpos + 1)).astype(int))
            thr = np.take_along_axis(srt, (k - 1)[None, None, :, None], -1)
            run((sr >= thr) & valid, "top-k", f"k={f:g}")
        ps_ = -np.sort(-pr, -1); cm = np.cumsum(ps_, -1)
        for tau in TAU:
            n = (cm < 1 - tau).sum(-1, keepdims=True) + 1
            thr = np.take_along_axis(ps_, n - 1, -1)
            run((pr >= thr) & valid, "top-p", f"tau={tau:g}")
        grms = np.sqrt((gt ** 2).mean(-1)).reshape(E, H, G).transpose(1, 2, 0)   # (H,G,E)
        for hf in HFR:
            cut = np.quantile(grms, hf, axis=(0, 1), keepdims=True)
            run(np.broadcast_to((grms > cut)[..., None], sr.shape) & valid, "gate heads", f"skip {hf:g}")
        oh = o_ref.reshape(E, H, G, d).transpose(1, 2, 0, 3)
        c_true = np.where(valid, pair_costs(pr, oh, L["v"], w), 0.0)
        # estimator: coarse 2-bit keys / values (FoveaKV nested code, block-centred, unbiased rescale)
        P = prefill_len(L)
        gs = gain_shape_cache_v2(L, P, block=128, smooth=SMOOTH)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc)
        Vc = Mv + (gs["Vc"] - Mv) / (1 - Dc)
        pc = softmax(scores(q, Kc, qpos))
        oc = attend(pc, Vc).reshape(E, H, G, d).transpose(1, 2, 0, 3)
        c_est = np.where(valid, pair_costs(pc, oc, Vc, w), 0.0)
        rc = np.linalg.norm((attend(pc, Vc) * gt).reshape(E, -1) @ Wo.T, axis=-1)
        for eps in EPS:
            run(backward_keep(c_true, eps * rn) & valid, "backward (exact p)", f"eps={eps:g}")
            run(backward_keep(c_est, eps * rc) & valid, "backward (coarse est.)", f"eps={eps:g}")
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e23_raw" + ("_abl" if os.environ.get("FOVEAKV_E23_ABL") else "") + ".npy"), R)
rows = []
for key in dict.fromkeys(map(tuple, R[:, 2:4])):
    a = R[(R[:, 2] == key[0]) & (R[:, 3] == key[1])]
    rows.append((*key, *[float(a[:, j].astype(float).mean()) for j in (4, 5, 6, 7)]))
write("23_backward_sparse" + ("_ablation" if os.environ.get("FOVEAKV_E23_ABL") else "") + ".md", f"""# E23 Sparse attention from the error model, run backwards (gate- and o_proj-weighted pair costs)

FLOPs = kept (query head, token) pairs / all causal pairs. heads off = share of (query, head) skipped entirely.
resid = relative error of the layer's residual contribution (after gate and o_proj), what the next layer sees.

""" + md_table(["arm", "setting", "FLOPs kept", "heads off", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
