"""E28 Prefill attention computed in the code domain of the nested gain-shape codes (FoveaKV's 2-bit + 2-bit residual).

Keys and values are stored as k = mu_blk + R^T (g / sqrt(d)) lev[idx]  (RHT R, per-token gain g, Lloyd-Max levels lev,
mean per 128-token block). Because R is orthogonal, the logit is
    q.k = (R q) . lev[idx] * g / sqrt(d) + q . mu_blk
so QK^T is an integer GEMM between the rotated query (quantized per 16-query tile and head) and the code levels, with the
gain as a per-key column scale; PV is an integer GEMM between (p * g) (quantized per row) and the code levels, rotated back
once per tile. The 16 Lloyd-Max levels fit an int8 grid (the 4-bit index is expanded through a 16-entry table, no other
dequantization); int4 MMA needs integer levels, so the s4 arms use a uniform 16-level code (step 0.3352, the N(0,1)
optimum) with the coarse level at the nearest fine level.

Ladder (all per 16-query tile x GQA group, decided from the cheap pass itself with the E23 error model
c_i = sum p_i ||w (v_i - o)||, w = gate x o_proj column norms):
    every token on the coarse 2-bit code -> residual (fine 4-bit) for the top rho1 -> exact bf16 for the top rho0 (+ the
    4 sinks and the 128-token window always exact).
Relative tensor-core time (bf16 = 1 per pair, QK and PV half each): s8 GEMM 1/2 of bf16, s4 1/4. Refining a token to the
fine code costs one more s8 pass over it (the residual plane), exact costs 1. Not counted: encoding (RHT + search, O(d log d)
per token), the per-tile rotation of q and o (O(d^2) per query), level expansion, gain scaling."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.load import model_weights
from foveakv.quant import NestedCodebook, GainShape, gaussian_mse
from foveakv.rotate import rht_matrix
from foveakv.scheme import block_means

t0 = time.time()
ST = model_weights()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
TQ, NT, WIN, BLK = 16, int(os.environ.get("FOVEAKV_E26_TILES", "24")), 128, 128
if os.environ.get("FOVEAKV_E21_SMOKE"):
    CORPORA, LAYERS, NT = CORPORA[-1:], LAYERS[1:2], 6

CB = NestedCodebook(4, 2)                                     # Lloyd-Max levels on an int8 grid (s8 arms)
CB.lut16 = np.round(CB.lut16 / CB.lut16.max() * 127) * CB.lut16.max() / 127
CB.lut4 = np.round(CB.lut4 / CB.lut16.max() * 127) * CB.lut16.max() / 127
CU = NestedCodebook(4, 2)                                     # uniform 16 levels for int4 MMA (s4 arms)
CU.lut16 = (np.arange(16) - 7.5) * 0.3352
CU.bnd16 = (CU.lut16[1:] + CU.lut16[:-1]) / 2
CU.lut4 = CU.lut16[np.abs(CB.lut4[:, None] - CU.lut16[None]).argmin(1)]
CU.mse_fine = gaussian_mse(CU.bnd16, CU.lut16)


def qsym(x, bits, axis=None):
    lv = 2 ** (bits - 1) - 1
    sc = np.abs(x).max(axis, keepdims=True) / lv + 1e-30
    return np.clip(np.round(x / sc), -lv, lv) * sc


def code(X, P, R, cb):
    """X (T,H,d) -> mean, rotated-domain reconstructions (fine, coarse) as (T,H,d), per token gains (T,H)."""
    T, H, d = X.shape
    M = block_means(X, BLK).astype(np.float64)
    F, C, Gs = np.empty((T, H, d)), np.empty((T, H, d)), np.empty((T, H))
    for h in range(H):
        gs = GainShape(R, cb, 8, center=False).fit(X[:P, h] - M[:P, h])
        idx, gq, _, _ = gs.encode(X[:, h] - M[:, h])
        F[:, h], C[:, h] = cb.fine(idx), cb.coarse(idx)                  # levels (integer grid up to a constant)
        Gs[:, h] = gq / np.sqrt(d)
    return M, F, C, Gs


res = []
for C_ in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C_, Li)
        Wo = ST.get(kvdata.weight_name(Li)).astype(np.float32)
        K, V = L["k"].astype(np.float64), L["v"].astype(np.float64)
        T, H, d = K.shape
        Q, QP = L["q"].astype(np.float64), L["qpos"]
        Hq = Q.shape[1]
        G, s = Hq // H, d ** -0.5
        GT = sigmoid(L["gate"]).reshape(len(QP), Hq, d) if "gate" in L else np.ones((len(QP), Hq, d))
        R = rht_matrix(d, 0)
        P0 = 1024
        codes = {nm: (code(K, P0, R, cb), code(V, P0, R, cb)) for nm, cb in (("lm", CB), ("uni", CU))}
        starts = QP[0] + np.arange(0, len(QP) - TQ + 1, TQ)
        starts = starts[starts >= 1024]
        pick = starts[np.linspace(0, len(starts) - 1, NT).astype(int)]
        ncol = np.linalg.norm(Wo, axis=0).reshape(Hq, d)
        Vi = np.abs(V).max(0, keepdims=True) / 127 + 1e-30
        acc = {}
        for t0_ in pick:
            rows = np.arange(t0_ - QP[0], t0_ - QP[0] + TQ)
            q, qpos, gt = Q[rows], QP[rows], GT[rows]
            tmax = int(qpos[-1]) + 1
            Kc, Vc = K[:tmax], V[:tmax]
            sr = scores(q, Kc, qpos)
            o_ref = attend(softmax(sr), Vc)
            r_ref = (o_ref * gt).reshape(TQ, -1) @ Wo.T
            dense = (qpos + 1).sum() * H * G                                    # (pair) count, all heads
            mask = ~np.isfinite(sr)
            wq = (gt * ncol[None]).reshape(TQ, H, G, d).transpose(1, 2, 0, 3)  # (H,G,TQ,d)
            t = np.arange(tmax)[None, :]
            win = ((t > qpos[:, None] - WIN) & (t <= qpos[:, None])) | (t < 4)  # (TQ,T)
            qh = q.reshape(TQ, H, G, d)

            def logits(lev, cd, qbits):
                """lev (T,H,d) rotated levels; integer GEMM of the quantized rotated query against them."""
                M, _, _, Gs = cd
                qr = qh @ R.T                                                     # (TQ,H,G,d) rotated queries
                qr = np.stack([qsym(qr[:, h], qbits) for h in range(H)], 1) if qbits < 16 else qr
                dot = np.einsum("ehgd,thd->hget", qr, lev[:tmax]) * Gs[:tmax].T[:, None, None, :]
                mu = np.einsum("ehgd,thd->hget", qh, M[:tmax])
                return np.where(mask, -np.inf, s * (dot + mu))

            def pv(p, lev, cd, pbits=8):
                """(p * g) quantized per row, integer GEMM against the value levels, rotated back; plus p . mu."""
                M, _, _, Gs = cd
                w = p * Gs[:tmax].T[:, None, None, :]                            # (H,G,TQ,T)
                w = qsym(w, pbits, -1) if pbits < 16 else w
                o_rot = np.matmul(w, lev[:tmax].transpose(1, 0, 2)[:, None])     # (H,G,TQ,d)
                o = o_rot @ R + np.matmul(p, M[:tmax].transpose(1, 0, 2)[:, None])
                return o.transpose(2, 0, 1, 3).reshape(TQ, Hq, d)

            def errsel(p0, o0, Vh, rho):
                """E23 error model from the cheap pass: per tile x group cost of approximating each token."""
                w2 = wq * wq
                Vt = Vh.transpose(1, 0, 2)[:, None]                               # (H,1,T,d)
                o0h = o0.reshape(TQ, H, G, d).transpose(1, 2, 0, 3)
                dev2 = (np.matmul(w2, (Vt ** 2).transpose(0, 1, 3, 2)) - 2 * np.matmul(w2 * o0h, Vt.transpose(0, 1, 3, 2))
                        + (w2 * o0h * o0h).sum(-1, keepdims=True))
                c = np.where(win[None, None], np.inf, p0 * np.sqrt(np.maximum(dev2, 0))).sum((1, 2))   # (H,T)
                kk = min(int(rho * tmax) + WIN + 4, tmax)
                thr = -np.sort(-c, -1)[:, kk - 1:kk]
                return np.broadcast_to((c >= thr)[:, None, None, :], sr.shape) & ~mask

            def run(name, cb, qbits, base, rho1, rho0, qkcost, pvcost):
                """base 'c' (coarse) or 'f' (fine) for every token; then fine for the top rho1, exact for the top rho0."""
                (cK, cV) = codes[cb]
                levK = cK[1] if base == "f" else cK[2]
                levV = cV[1] if base == "f" else cV[2]
                sl = logits(levK, cK, qbits)
                p0 = softmax(sl)
                o0 = pv(p0, levV, cV)
                Vhat = (np.einsum("thd,de->the", levV, R) * cV[3][..., None] + cV[0])[:tmax]
                ex = errsel(p0, o0, Vhat, rho0) if rho0 else np.zeros(sr.shape, bool)
                fi = (errsel(p0, o0, Vhat, rho0 + rho1) & ~ex) if rho1 else np.zeros(sr.shape, bool)
                if rho1:
                    sl = np.where(fi, logits(cK[1], cK, qbits), sl)
                lg = np.where(ex, sr, sl)
                p = softmax(lg)
                pe, pf = p * ex, p * fi
                pc = p - pe - pf
                o = attend(pe, Vc) + pv(pc, levV, cV) + (pv(pf, cV[1], cV) if rho1 else 0)
                r = (o * gt).reshape(TQ, -1) @ Wo.T
                cost = (qkcost + pvcost) * dense + 0.5 * fi.sum() + ex.sum() * 1.0
                a_ = acc.setdefault(name, [0.0, 0.0, [], []])
                a_[0] += cost; a_[1] += dense
                a_[2].append((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
                a_[3].append((np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)).mean())

            # reference points: dense codes in bf16 arithmetic (code error only), then integer arithmetic
            run("fine 4b code, bf16 arithmetic (code error only)", "lm", 16, "f", 0, 0, 0.5, 0.5)
            run("coarse 2b code, bf16 arithmetic", "lm", 16, "c", 0, 0, 0.5, 0.5)
            run("fine 4b code, s8 QK / s8 PV", "lm", 8, "f", 0, 0, 0.25, 0.25)
            run("coarse 2b code, s8 / s8", "lm", 8, "c", 0, 0, 0.25, 0.25)
            for r0 in (0.05, 0.1):
                run(f"fine s8/s8 + exact {r0:g}", "lm", 8, "f", 0, r0, 0.25, 0.25)
            for r1, r0 in ((0.1, 0.05), (0.2, 0.05), (0.3, 0.05), (0.2, 0.1)):
                run(f"coarse s8/s8 + fine {r1:g} + exact {r0:g}", "lm", 8, "c", r1, r0, 0.25, 0.25)
            # int4 QK (uniform code, q int4), PV s8 (P int8 x levels)
            run("uniform fine, s4 QK / s8 PV", "uni", 4, "f", 0, 0, 0.125, 0.25)
            for r0 in (0.05, 0.1):
                run(f"uniform fine s4/s8 + exact {r0:g}", "uni", 4, "f", 0, r0, 0.125, 0.25)
            run("uniform fine, s8 QK / s8 PV", "uni", 8, "f", 0, 0, 0.25, 0.25)
        for name, a_ in acc.items():
            res.append((C_, Li, name, a_[0] / a_[1], float(np.mean(a_[2])), float(np.mean(a_[3]))))
        print(C_, Li, f"{time.time()-t0:.0f}s", flush=True)

R_ = np.array(res, dtype=object)
np.save(os.path.join(RES, "e28_raw.npy"), R_)
rows = [(k, *[float(R_[R_[:, 2] == k][:, j].astype(float).mean()) for j in (3, 4, 5)]) for k in dict.fromkeys(R_[:, 2])]
write("28_code_domain.md", f"""# E28 Prefill attention in the code domain of the nested gain-shape codes (16-query tiles, as E26/E27)

compute = relative tensor-core time vs bf16 (s8 GEMM 1/2, s4 1/4; QK and PV half each) + 0.5 per pair refined to the fine
code + 1 per exact pair. Uniform-code fine MSE {CU.mse_fine:.4f} vs Lloyd-Max {CB.mse_fine:.4f} (per N(0,1) coordinate).
E27 reference (same tiles): int8/int8 + exact 5% by error model 0.58 -> 0.0092; int8 QK / bf16 PV 0.75 -> 0.0137.

""" + md_table(["arm", "compute", "attn err", "resid err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
