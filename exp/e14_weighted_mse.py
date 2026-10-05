"""E14 Verify that attention-output error is (to first order) an importance-weighted MSE:
   o_hat - o ≈ Σ_i p_i ε_i (v_i - o) + Σ_i p_i δ_i                         (linearization, 'lin')
   E||o_hat - o||² ≈ Σ_i p_i² ( ε_i² ||v_i - o||² + ||δ_i||² )             (independence -> weighted MSE, 'wmse')
   E||o_hat - o||² ≈ Σ_i p_i² ( ||q||² g_ki² D/d² ||v_i - o||² + g_vi² D )   (statistical model, no error knowledge, 'model')
with ε_i = q·(k̂_i - k_i)/√d, δ_i = v̂_i - v_i. Compared per (query, head) with the actual ||o_hat - o||² for several schemes.
Also verifies the SDPA identity Σ p_i² x_i = exp(lse(2s) - 2 lse(s)) · softmax(2s)·x (NumPy here; GPU flash check separately)."""
from common import *
from scipy.stats import spearmanr
from foveakv.attention import scores, softmax, attend
from foveakv.select import select
from foveakv.scheme import gain_shape_cache_v2, tiered_attention, turboquant_mse
from foveakv.quant import NestedCodebook

t0 = time.time()
CB = NestedCodebook()
W_REC = 128
rows = []
ident_err = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, 4)
        q, qpos = L["q"][sl], L["qpos"][sl]
        E, T = len(qpos), L["T"]
        Hkv, d = L["k"].shape[1], L["k"].shape[2]
        sr = scores(q, L["k"], qpos)
        p = softmax(sr)
        o = attend(p, L["v"])                                                   # (E,Hq,d)
        oh = o.reshape(E, Hkv, -1, d).transpose(1, 2, 0, 3)                     # (Hkv,G,E,d)
        qn2 = (q.reshape(E, Hkv, -1, d) ** 2).sum(-1).transpose(1, 2, 0)        # (Hkv,G,E)
        # ||v_i - o||^2 per (h,g,e,i)
        vv = (L["v"] ** 2).sum(-1).T[:, None, None, :]
        dist = vv - 2 * np.einsum("tkd,kged->kget", L["v"], oh) + (oh ** 2).sum(-1)[..., None]
        # SDPA identity check (NumPy, fp64): Σ p² x = exp(lse(2s) - 2 lse(s)) softmax(2s) x
        x = L["v"][:, :, 0].astype(np.float64)                                  # arbitrary per-token scalar
        s64 = sr.astype(np.float64)
        lse = lambda z: np.logaddexp.reduce(z, axis=-1)
        lhs = np.einsum("kget,tk->kge", p.astype(np.float64) ** 2, x)
        rhs = np.exp(lse(2 * s64) - 2 * lse(s64)) * np.einsum("kget,tk->kge", softmax(2 * s64), x)
        ident_err.append(float(np.abs(lhs - rhs).max() / np.abs(lhs).max()))

        gs = gain_shape_cache_v2(L, P, block=128)
        gk = np.stack([gs["diag"][h]["k"]["g"] for h in range(Hkv)], 1).T     # (Hkv,T)
        gv = np.stack([gs["diag"][h]["v"]["g"] for h in range(Hkv)], 1).T
        t = np.arange(T)[None, :]
        A = (t > qpos[:, None] - W_REC) & (t <= qpos[:, None])
        sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
        hot = select(np.where(A[None, None], -np.inf, sc), qpos, 0.05, 0, 0)
        tq = turboquant_mse(L, 2)
        schemes = {
            "GS-2b (all coarse)": (gs["Kc"], gs["Vc"], np.zeros((Hkv, E, T), bool), np.zeros_like(A), CB.mse_coarse),
            "GS-4b (all fine)": (gs["Kf"], gs["Vf"], np.zeros((Hkv, E, T), bool), np.zeros_like(A), CB.mse_fine),
            "DynReq k=5% +A": (None, None, hot, A, None),
        }
        for name, (Kq, Vq, hmask, win, Dconst) in schemes.items():
            if Kq is None:     # tiered: build effective per-token reconstructions
                h4 = hmask[:, None] & ~win[None, None]
                w4 = np.broadcast_to(win[None, None], h4.shape)
                s_hat = np.where(w4, sr, np.where(h4, sf, sc))
                eps = s_hat - sr
                o_hat = tiered_attention(q, qpos, sc, sf, sr, gs["Vc"], gs["Vf"], L["v"], hmask, win)
                dV_c = ((gs["Vc"] - L["v"]) ** 2).sum(-1).T[:, None, None, :]
                dV_f = ((gs["Vf"] - L["v"]) ** 2).sum(-1).T[:, None, None, :]
                dV2 = np.where(w4, 0.0, np.where(h4, dV_f, dV_c))
                Dtok = np.where(w4, 0.0, np.where(h4, CB.mse_fine, CB.mse_coarse))
                # linearized vector prediction
                Vc_d, Vf_d = gs["Vc"] - L["v"], gs["Vf"] - L["v"]
                lin_val = (np.einsum("kget,tkd->kged", p * (h4 & ~w4), Vf_d) + np.einsum("kget,tkd->kged", p * (~h4 & ~w4), Vc_d))
            else:
                s_hat = scores(q, Kq, qpos)
                eps = s_hat - sr
                o_hat = attend(softmax(s_hat), Vq)
                dV2 = ((Vq - L["v"]) ** 2).sum(-1).T[:, None, None, :]
                Dtok = Dconst
                lin_val = np.einsum("kget,tkd->kged", p, Vq - L["v"])
            eps = np.where(np.isfinite(eps), eps, 0.0)
            ohat_h = o_hat.reshape(E, Hkv, -1, d).transpose(1, 2, 0, 3)
            actual = ((ohat_h - oh) ** 2).sum(-1)                                          # (Hkv,G,E)
            lin_key = np.einsum("kget,kget,tkd->kged", p, eps, L["v"]) - (p * eps).sum(-1)[..., None] * oh
            lin = ((lin_key + lin_val) ** 2).sum(-1)
            wmse = (p ** 2 * (eps ** 2 * dist + dV2)).sum(-1)
            model = (p ** 2 * Dtok * (qn2[..., None] * (gk ** 2)[:, None, None, :] * dist / d ** 2 + (gv ** 2)[:, None, None, :])).sum(-1)
            for nm, pred in (("lin", lin), ("wmse", wmse), ("model", model)):
                ratio = np.median(pred / actual)
                rho = spearmanr(np.log(pred.ravel()), np.log(actual.ravel())).correlation
                rows.append((C, Li, name, nm, float(ratio), float(rho)))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

A_ = np.array(rows, dtype=object)
np.save(os.path.join(RES, "e14_raw.npy"), A_)
tab = []
for name in dict.fromkeys(A_[:, 2]):
    for nm in ("lin", "wmse", "model"):
        a = A_[(A_[:, 2] == name) & (A_[:, 3] == nm)]
        tab.append((name, nm, float(np.median(a[:, 4].astype(float))), float(a[:, 4].astype(float).min()),
                    float(a[:, 4].astype(float).max()), float(np.mean(a[:, 5].astype(float))), float(a[:, 5].astype(float).min())))
g = {(r[0], r[1]): r for r in tab}
d5 = "DynReq k=5% +A"
write("14_weighted_mse.md", f"""# E14 Attention-output error = importance-weighted MSE (verification)

**TL;DR:** the first-order weighted-MSE prediction Σ_i p_i²(ε_i²||v_i−o||² + ||δ_i||²) tracks the actual squared attention-output error:
for DynReq k=5% +A, median predicted/actual ratio {g[(d5,'wmse')][2]:.2f} (per corpus×layer range {g[(d5,'wmse')][3]:.2f}–{g[(d5,'wmse')][4]:.2f}),
Spearman ρ over (query, head) {g[(d5,'wmse')][5]:.2f}; the purely statistical model (gains, D, no error knowledge) gives ratio
{g[(d5,'model')][2]:.2f}, ρ {g[(d5,'model')][5]:.2f}. SDPA identity Σp²x = exp(lse(2s)−2lse(s))·softmax(2s)x holds to {max(ident_err):.1e} (fp64).

## Config
5 corpora × 6 layers, 128 decode queries per layer (stride 4), int8 reference. 'lin' = norm² of the linearized error vector
(tests linearization incl. cross terms); 'wmse' = diagonal (independent-errors) weighted MSE with the actual per-token errors;
'model' = statistical model Eq. (errmodel) with D_c/D_f and per-token gains. Ratio: median over (query, head) of predicted/actual.

""" + md_table(["scheme", "prediction", "median ratio", "min (corpus,layer)", "max", "Spearman ρ mean", "ρ min"], tab) + f"""
runtime {time.time()-t0:.0f}s
""")
