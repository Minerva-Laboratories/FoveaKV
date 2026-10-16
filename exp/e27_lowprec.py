"""E27 The dense low-precision baseline for attention compute (SageAttention-style), and a precision ladder on top.

Logits from quantized Q and K (keys smoothed by their per-channel mean over the context, which leaves softmax unchanged;
symmetric per-block scales: K per 64 keys, Q per 16-query tile), probabilities and values in FP8 E4M3 (V per-channel
scale) for the PV product. Arms:
  int8 QK / fp8 PV, int4 QK / fp8 PV                      dense, every pair at low precision
  int4 QK + exact top-rho                                 the rho fraction of tokens with the largest int4 logits
                                                           (per tile and GQA group) recomputed in bf16, rest int4/fp8
Relative compute (tensor-core throughput vs bf16: int8 2x, int4 4x, fp8 2x; QK and PV half of the FLOPs each):
  int8/fp8 0.5, int4/fp8 0.375, ladder 0.375 + rho. Errors as in E26 (16-query tiles over the prefill positions)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.load import model_weights

t0 = time.time()
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
TQ, NT, WIN = 16, int(os.environ.get("FOVEAKV_E26_TILES", "24")), 128
RHO = (0.02, 0.05, 0.1)
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS = CORPORA[-1:], LAYERS[1:2]


def qsym(x, bits, axis_blocks):
    """Symmetric quantization with one scale per block of rows (x (..., n, d), blocks along n)."""
    lv = 2 ** (bits - 1) - 1
    n = x.shape[-2]
    pad = (-n) % axis_blocks
    xp = np.concatenate([x, np.zeros(x.shape[:-2] + (pad, x.shape[-1]))], -2) if pad else x
    xb = xp.reshape(xp.shape[:-2] + (-1, axis_blocks, x.shape[-1]))
    sc = np.abs(xb).max((-1, -2), keepdims=True) / lv + 1e-30
    return (np.clip(np.round(xb / sc), -lv, lv) * sc).reshape(xp.shape)[..., :n, :]


def fp8(x):
    """Round to FP8 E4M3 (3 mantissa bits, max 448)."""
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    m = np.round(a / 2 ** e * 8) / 8
    return np.sign(x) * np.minimum(m * 2 ** e, 448.0)


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
        starts = QP[0] + np.arange(0, len(QP) - TQ + 1, TQ)
        starts = starts[starts >= 1024]
        pick = starts[np.linspace(0, len(starts) - 1, NT).astype(int)]
        acc = {}
        for t0_ in pick:
            rows = np.arange(t0_ - QP[0], t0_ - QP[0] + TQ)
            q, qpos, gt = Q[rows], QP[rows], GT[rows]
            tmax = int(qpos[-1]) + 1
            Kc, Vc = K[:tmax], V[:tmax]
            sr = scores(q, Kc, qpos)
            o_ref = attend(softmax(sr), Vc)
            r_ref = (o_ref * gt).reshape(TQ, -1) @ Wo.T
            dense = (qpos + 1).sum()
            Ks = Kc - Kc.mean(0, keepdims=True)                                 # smoothing: softmax-invariant
            vs = np.abs(Vc).max(0, keepdims=True) / 448 + 1e-30                 # per-channel FP8 scale for V
            V8 = fp8(Vc / vs) * vs
            vi = np.abs(Vc).max(0, keepdims=True) / 127 + 1e-30                # per-channel INT8 scale for V
            Vi8 = np.round(Vc / vi) * vi
            def run(name, qbits, rho, pv8=True, pvint=False):
                Kq = qsym(Ks.transpose(1, 0, 2), qbits, 64).transpose(1, 0, 2) if qbits < 16 else Ks
                qq = qsym(q.transpose(1, 0, 2), qbits, TQ).transpose(1, 0, 2) if qbits < 16 else q
                mu = Kc.mean(0)                                                     # (H,d): add back s q.mu so the
                off = s * np.einsum("ehgd,hd->hge", q.reshape(TQ, H, G, d), mu)[..., None]   # exact pairs mix consistently
                sl = scores(qq, Kq, qpos) + off
                ex = np.zeros(sl.shape, bool)
                if rho:
                    t = np.arange(tmax)[None, :]
                    win = (t > qpos[:, None] - WIN) & (t <= qpos[:, None]) | (t < 4)
                    sc_ = np.where(win[None, None], np.inf, sl).max((1, 2))                 # (H, T): per tile x group
                    kk = int(rho * tmax) + WIN + 4
                    thr = -np.sort(-sc_, -1)[:, kk - 1:kk]
                    ex = np.broadcast_to((sc_ >= thr)[:, None, None, :], sl.shape) & np.isfinite(sr)
                lg = np.where(ex, sr, sl)
                p = softmax(lg)
                pe = p * ex
                if pvint:                                                           # P in INT8, per-row max scale
                    pr_ = p - pe
                    mx = pr_.max(-1, keepdims=True) / 127 + 1e-30
                    o = attend(pe, Vc) + attend(np.round(pr_ / mx) * mx, Vi8)
                else:
                    pl = fp8((p - pe) * 448) / 448 if pv8 else p - pe               # P in FP8, scaled to its range
                    o = attend(pe, Vc) + attend(pl, V8 if pv8 else Vc)
                r = (o * gt).reshape(TQ, -1) @ Wo.T
                a_ = acc.setdefault(name, [0.0, 0.0, [], []])
                cost = ({8: 0.25, 4: 0.125, 16: 0.5}[qbits] + (0.25 if pv8 else 0.5)) * dense + (ex[:, :, :, :].sum() / (H * G) if rho else 0)
                a_[0] += cost; a_[1] += dense
                a_[2].append((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
                a_[3].append((np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean())
            run("int8 QK / fp8 PV", 8, 0)
            run("int8 QK / bf16 PV", 8, 0, pv8=False)
            run("bf16 QK / fp8 PV", 16, 0)
            run("int8 QK / int8 PV (Ampere)", 8, 0, pvint=True)
            for rho in RHO:
                run(f"int8 QK / int8 PV + exact top {rho:g} (+window)", 8, rho, pvint=True)
            run("int4 QK / fp8 PV", 4, 0)
            for rho in RHO:
                run(f"int4 QK / fp8 PV + exact top {rho:g} (+window)", 4, rho)
        for name, a_ in acc.items():
            res.append((C, Li, name, a_[0] / a_[1], float(np.mean(a_[2])), float(np.mean(a_[3]))))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e27_raw.npy"), R)
rows = [(k, *[float(R[R[:, 2] == k][:, j].astype(float).mean()) for j in (3, 4, 5)]) for k in dict.fromkeys(R[:, 2])]
write("27_lowprec.md", f"""# E27 Dense low-precision attention (SageAttention-style) and an exact-refinement ladder, prefill tiles

compute = relative tensor-core time vs bf16 (int8 2x, int4 4x, fp8 2x; QK and PV half each) + exact refined pairs.

""" + md_table(["arm", "compute", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
