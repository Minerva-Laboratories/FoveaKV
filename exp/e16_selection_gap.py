"""E16 The selection gap: coarse-estimated vs oracle hot sets (shrinkage-corrected caches, window A, block means).

Characterization:
  - oracle: rank by the true attention; est: rank by coarse logits (current rule).
  - noise model: the coarse logit is the true logit plus noise with per-token std sigma_i = ||q|| g_i sqrt(D_c) / d
    (gain-dependent, computable from stored data). 'sim' ranks by true logits + Gaussian noise of exactly that std:
    if sim ≈ est, the gap is fully explained by quantization noise of the coarse plane.
  - budget inflation: k the estimated selector needs to match the oracle at 5%.
  - misses: true-rank distribution of the oracle tokens that the estimate misses.
Attempted fix: uncertainty-aware ranking. E[e^{s} | s_hat] = e^{s_hat + sigma²/2} (Gaussian noise), so rank by
s_hat + c·sigma² (c = 0.5 for expected weight; c = 2 for expected squared weight)."""
from common import *
from foveakv.attention import scores, softmax, attend, logprob
from foveakv.select import topk_rows, k_for
from foveakv.scheme import gain_shape_cache_v2, tiered_attention, block_means
from foveakv.quant import NestedCodebook

t0 = time.time()
CB = NestedCodebook()
Dc, Df = CB.mse_coarse, CB.mse_fine
W_REC = 128
rng = np.random.default_rng(0)
res, miss = [], []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos = L["q"][sl], L["qpos"][sl]
        E, T = len(qpos), L["T"]
        Hkv, d = L["k"].shape[1], L["k"].shape[2]
        G = q.shape[1] // Hkv
        sr = scores(q, L["k"], qpos)
        p_true = softmax(sr)
        o_ref = attend(p_true, L["v"])
        t = np.arange(T)[None, :]
        A = (t < 4) | ((t > qpos[:, None] - W_REC) & (t <= qpos[:, None]))
        gs = gain_shape_cache_v2(L, P, block=128, smooth=SMOOTH)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc); Kf = Mk + (gs["Kf"] - Mk) / np.sqrt(1 - Df)
        Vc = Mv + (gs["Vc"] - Mv) / (1 - Dc); Vf = Mv + (gs["Vf"] - Mv) / (1 - Df)
        sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
        gk = np.stack([gs["diag"][h]["k"]["g"] for h in range(Hkv)], 0)             # (Hkv,T)
        qn = np.sqrt((q.reshape(E, Hkv, G, d) ** 2).sum(-1)).transpose(1, 2, 0)      # (Hkv,G,E)
        sigma = qn[..., None] * gk[:, None, None, :] * np.sqrt(Dc) / d               # (Hkv,G,E,T) predicted noise std
        fin = np.isfinite(sr)
        actual_noise = np.where(fin & ~A[None, None], sc - sr, np.nan)
        z = actual_noise / sigma
        noise_ratio = float(np.sqrt(np.nanmean(z ** 2)))                              # actual / predicted std

        def err_of(hot):
            o = tiered_attention(q, qpos, sc, sf, sr, Vc, Vf, L["v"], hot, A)
            return float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())

        def rule(s):     # max over the group of log-prob (window tokens excluded from selection)
            lp = logprob(np.where(A[None, None], np.where(fin, sr, -np.inf), s))
            return np.where(A[None], -np.inf, lp.max(1))

        sim = sr + rng.standard_normal(sr.shape).astype(np.float32) * sigma
        # persistent key noise: one isotropic perturbation per key (variance g² D_c / d per coordinate), shared by all
        # query heads of the group and all decode steps — the structure of real quantization error
        ek = rng.standard_normal(L["k"].shape).astype(np.float32) * (gk.T[..., None] * np.sqrt(Dc / d))
        s_keysim = scores(q, L["k"] + ek, qpos)
        s_mmse = scores(q, gs["Kc"], qpos)                                           # raw centroid logits
        s_unb = scores(q, Mk + (gs["Kc"] - Mk) / (1 - Dc), qpos)                     # unbiased in the true logit
        crit = {"oracle": rule(sr), "est": rule(sc), "sim (true + predicted noise)": rule(np.where(fin, sim, -np.inf)),
                "sim (persistent key noise)": rule(s_keysim), "est, MMSE logits (no scaling)": rule(s_mmse),
                "est, unbiased logits x1/(1-Dc)": rule(s_unb), "est from fine (4-bit) keys": rule(sf)}
        for c in (0.5, 1.0, 2.0):
            crit[f"est + {c}·sigma²"] = rule(sc + c * sigma ** 2)
        for kf in (0.02, 0.05):
            k = k_for(qpos, kf)
            for name, cr in crit.items():
                res.append((C, Li, f"{kf*100:.0f}%", name, err_of(topk_rows(cr, k)), noise_ratio))
        for kf in (0.065, 0.075, 0.08, 0.10):
            res.append((C, Li, "5%", f"est at k={kf*100:g}%", err_of(topk_rows(crit["est"], k_for(qpos, kf))), noise_ratio))
        # two-stage: 2k candidates by coarse logits -> read their KEY residuals -> re-rank by fine logits -> top-k get VALUE
        # residuals. Reads 2 bits x 2k (K) + 2 bits x k (V) = same as plain selection at 1.5k.
        for kf in (0.02, 0.05):
            k = k_for(qpos, kf)
            cand = topk_rows(crit["est"], 2 * k)                                        # (Hkv,E,T)
            s2 = np.where(cand[:, None], sf, -np.inf)
            hot = topk_rows(rule(s2), k) & cand
            w = A[None, None]
            ck = cand[:, None] & ~w
            hv = hot[:, None] & ~w
            pr = softmax(np.where(w, sr, np.where(ck, sf, sc)))
            o = attend(pr * w, L["v"]) + attend(pr * hv, Vf) + attend(pr * ~(w | hv), Vc)
            res.append((C, Li, f"{kf*100:.0f}%", "two-stage (2k fine keys, k fine values)",
                        float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean()), noise_ratio))
        # miss analysis at 5%: true rank (in units of k) of oracle tokens missed by est, mass missed
        k = k_for(qpos, 0.05)
        ho, he = topk_rows(crit["oracle"], k), topk_rows(crit["est"], k)
        pm = p_true.max(1)                                                            # (Hkv,E,T) group-max weight
        missed = ho & ~he
        mass_missed = (pm * missed).sum(-1) / np.maximum((pm * ho).sum(-1), 1e-12)
        order = np.argsort(-np.where(A[None], -np.inf, np.where(np.isfinite(crit["oracle"]), crit["oracle"], -np.inf)), -1)
        rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(T)[None, None, :], -1)
        rel_rank = rank / k[None, :, None]
        r_m = rel_rank[missed]
        miss.append((C, Li, float(mass_missed.mean()), float(np.mean(r_m > 0.8)) if r_m.size else 0.0,
                     float(np.mean(r_m > 0.5)) if r_m.size else 0.0, float(missed.sum(-1).mean() / k.mean())))
        print(C, Li, f"{time.time()-t0:.0f}s noise ratio {noise_ratio:.2f}", flush=True)

R = np.array(res, dtype=object)
M_ = np.array(miss, dtype=object)
np.savez(os.path.join(RES, "e16_raw.npz"), res=R, miss=M_)
rows = []
for kk in ("2%", "5%"):
    for name in dict.fromkeys(R[R[:, 2] == kk][:, 3]):
        a = R[(R[:, 2] == kk) & (R[:, 3] == name)]
        rows.append((kk, name, float(a[:, 4].astype(float).mean()), float(a[a[:, 1] == LAYERS[0]][:, 4].astype(float).mean())))
g = {(r[0], r[1]): r[2] for r in rows}
mm = M_[:, 2:].astype(float).mean(0)
gap = g[("5%", "est")] - g[("5%", "oracle")]
write("16_selection_gap.md", f"""# E16 The selection gap (shrinkage-corrected caches, window A, block means)

**TL;DR:** at k=5%: oracle {g[('5%','oracle')]:.4f}, coarse estimate {g[('5%','est')]:.4f}, simulated true-logits + predicted Gaussian noise
{g[('5%','sim (true + predicted noise)')]:.4f} (independent per head and step) and {g[('5%','sim (persistent key noise)')]:.4f} (one perturbation per key, shared by heads and steps)
— persistent key noise {'explains' if abs(g[('5%','sim (persistent key noise)')] - g[('5%','est')]) < 0.3 * gap else 'does NOT explain'} the gap
(measured/predicted noise std {float(np.mean(R[:, 5].astype(float))):.2f}). The estimate misses {mm[3]*100:.1f}% of the oracle tokens but only {mm[0]*100:.1f}% of their mass;
{mm[1]*100:.0f}% of the missed tokens rank in the last 20% of the oracle set. Uncertainty bonus (rank by s_hat + c·sigma²): c=0.5 {g[('5%','est + 0.5·sigma²')]:.4f},
c=1 {g[('5%','est + 1.0·sigma²')]:.4f}, c=2 {g[('5%','est + 2.0·sigma²')]:.4f}. Matching the oracle at 5% needs k≈{', '.join(f"{n.split('=')[1]}: {g[('5%', n)]:.4f}" for n in ('est at k=6.5%', 'est at k=8%', 'est at k=10%'))}.

## Config
sigma_i = ||q|| g_k,i sqrt(D_c)/d (per head and token). 'sim' replaces the coarse logits by true logits + N(0, sigma_i²).
Rule: max over the GQA group of log-prob; window tokens excluded; error = attention-output rel err vs int8 reference.

""" + md_table(["k", "selector", "attn err", f"layer-{LAYERS[0]} attn err"], rows) + f"""
## Misses at k=5% (mean over corpora/layers)
| missed tokens / k | missed mass / oracle mass | missed with rank > 0.8k | missed with rank > 0.5k |
|---|---|---|---|
| {mm[3]:.3f} | {mm[0]:.4f} | {mm[1]:.2f} | {mm[2]:.2f} |

runtime {time.time()-t0:.0f}s
""")
