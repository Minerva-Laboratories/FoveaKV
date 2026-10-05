"""E15 Logit shrinkage correction. Conditional-mean (MMSE) reconstruction is unbiased given the code but shrinks the
true value: E[z_hat | z] ≈ (1 - D) z (regression to the mean, since Cov(z_hat, z) = E[z_hat²] = 1 - D). Hence the residual
logit q·(k_hat - mu)/sqrt(d) ≈ (1 - D) × true residual logit, a coherent temperature increase that leaks attention mass.
Correction: scale the key-residual logits by alpha = 1/(1 - D) (coarse 1.137, fine 1.0096) — one multiply on the rotated query.
Values keep the MMSE reconstruction. Tested at several alphas on the E11 configuration."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import select
from foveakv.scheme import gain_shape_cache_v2, tiered_attention, block_means, turboquant_mse
from foveakv.quant import NestedCodebook, lloyd_max, gaussian_mse

t0 = time.time()
CB = NestedCodebook()
AC = 1 / (1 - CB.mse_coarse)
AF = 1 / (1 - CB.mse_fine)
b2, l2 = lloyd_max(4); D_TQ2 = gaussian_mse(b2, l2)
b4, l4 = lloyd_max(16); D_TQ4 = gaussian_mse(b4, l4)
W_REC = 128
res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos = L["q"][sl], L["qpos"][sl]
        E, T = len(qpos), L["T"]
        sr = scores(q, L["k"], qpos)
        o_ref = attend(softmax(sr), L["v"])
        t = np.arange(T)[None, :]
        A = (t < 4) | ((t > qpos[:, None] - W_REC) & (t <= qpos[:, None]))
        gs = gain_shape_cache_v2(L, P, block=128)
        Mk = block_means(L["k"], 128)
        err = lambda o: float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
        nohot = np.zeros((L["k"].shape[1], E, T), bool)
        for ac_name, ac in (("1", 1.0), ("1/sqrt(1-Dc)", 1 / np.sqrt(1 - CB.mse_coarse)), ("1/(1-Dc)", AC), ("1.20", 1.2)):
            Kc = Mk + ac * (gs["Kc"] - Mk)
            Kf = Mk + (AF if ac != 1.0 else 1.0) * (gs["Kf"] - Mk)
            sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
            hot = select(np.where(A[None, None], -np.inf, sc), qpos, 0.05, 0, 0)
            res.append((C, Li, f"DynReq k=5% alpha_c={ac_name}", err(tiered_attention(q, qpos, sc, sf, sr, gs["Vc"], gs["Vf"], L["v"], hot, A))))
            res.append((C, Li, f"GS-2b +A alpha={ac_name}", err(tiered_attention(q, qpos, sc, sc, sr, gs["Vc"], gs["Vc"], L["v"], nohot, A))))
        # value residual scaling on top of the best key scaling (renormalized key shape, alpha_k = 1/sqrt(1-D))
        Mv = block_means(L["v"], 128)
        ak_c, ak_f = 1 / np.sqrt(1 - CB.mse_coarse), 1 / np.sqrt(1 - CB.mse_fine)
        Kc = Mk + ak_c * (gs["Kc"] - Mk); Kf = Mk + ak_f * (gs["Kf"] - Mk)
        sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
        hot = select(np.where(A[None, None], -np.inf, sc), qpos, 0.05, 0, 0)
        for av_name, pw in (("1", 0.0), ("1/sqrt(1-D)", 0.5), ("1/(1-D)", 1.0)):
            Vc = Mv + (gs["Vc"] - Mv) / (1 - CB.mse_coarse) ** pw
            Vf = Mv + (gs["Vf"] - Mv) / (1 - CB.mse_fine) ** pw
            res.append((C, Li, f"DynReq k=5% alpha_k=renorm alpha_v={av_name}", err(tiered_attention(q, qpos, sc, sf, sr, Vc, Vf, L["v"], hot, A))))
            res.append((C, Li, f"GS-2b +A alpha_k=renorm alpha_v={av_name}", err(tiered_attention(q, qpos, sc, sc, sr, Vc, Vc, L["v"], nohot, A))))
        for ff_name, ff in (("1", 1.0), ("1/(1-Df)", AF)):
            Kf = Mk + ff * (gs["Kf"] - Mk)
            sf = scores(q, Kf, qpos)
            res.append((C, Li, f"GS-4b +A alpha={ff_name}", err(tiered_attention(q, qpos, sf, sf, sr, gs["Vf"], gs["Vf"], L["v"], nohot, A))))
        for b, D in ((2, D_TQ2), (4, D_TQ4)):
            tq = turboquant_mse(L, b)
            for nm, a in (("1", 1.0), ("1/(1-D)", 1 / (1 - D))):
                s = scores(q, a * tq["K"], qpos)
                res.append((C, Li, f"TQ-{b}b +A alpha={nm}", err(tiered_attention(q, qpos, s, s, sr, tq["V"], tq["V"], L["v"], nohot, A))))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e15_raw.npy"), R)
rows = [(n, float(R[R[:, 2] == n][:, 3].astype(float).mean()), float(R[(R[:, 2] == n) & (R[:, 1] == 3)][:, 3].astype(float).mean()))
        for n in dict.fromkeys(R[:, 2])]
g = {r[0]: r[1] for r in rows}
write("15_logit_shrinkage.md", f"""# E15 Logit shrinkage correction alpha = 1/(1-D) (free: one multiply on the rotated query)

**TL;DR:** DynReq k=5% (+A, block means): attn err {g['DynReq k=5% alpha_c=1']:.4f} → {g['DynReq k=5% alpha_c=1/(1-Dc)']:.4f} with alpha_c = 1/(1−D_c) = {AC:.3f}
(alpha_f = {AF:.4f}); GS-2b {g['GS-2b +A alpha=1']:.4f} → {g['GS-2b +A alpha=1/(1-Dc)']:.4f}; TurboQuant-MSE 2b {g['TQ-2b +A alpha=1']:.4f} → {g['TQ-2b +A alpha=1/(1-D)']:.4f},
4b {g['TQ-4b +A alpha=1']:.4f} → {g['TQ-4b +A alpha=1/(1-D)']:.4f}; GS-4b {g['GS-4b +A alpha=1']:.4f} → {g['GS-4b +A alpha=1/(1-Df)']:.4f}.

## Why
MMSE reconstruction satisfies E[z | z_hat] = z_hat but E[z_hat | z] ≈ (1−D) z: high-attention tokens (large true logits) are
reconstructed with logits shrunk by (1−D), which raises the softmax temperature coherently across tokens (a correlated error that
the independent-error weighted-MSE model misses; E14). Scaling the key-residual logits (not the block-mean offsets) by 1/(1−D)
undoes the shrinkage on average; values keep the MMSE reconstruction (their error enters linearly).

""" + md_table(["scheme / alpha", "attn err", "layer-3 attn err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
