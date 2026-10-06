"""E13 Selection criteria derived from the first-order error model (paper §Method):
   E||o_hat - o||^2 ≈ sum_i p_i^2 [ Var(eps_i) ||v_i - o||^2 + E||dv_i||^2 ],
   Var(eps_i) = ||q||^2 g_{k,i}^2 D / d^2  (logit noise),  E||dv_i||^2 = g_{v,i}^2 D  (value noise),
   so refining token i removes (D_c - D_f) * w_i with  w_i = sum_h p_{h,i}^2 [ ||q_h||^2 g_{k,i}^2 ||v_i - o_h||^2 / d^2 + g_{v,i}^2 ].
Criteria compared at equal k (all with the exact window A and 128-token block means, coarse-estimated p, o, v):
   max-logp (used so far), sum-p2 (GQA sum of squared probabilities), sum-p, expected-error w_i, oracle (true p, sum-p2)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import topk_rows, sort_desc, k_for
from foveakv.scheme import gain_shape_cache_v2, tiered_attention

t0 = time.time()
W_REC, S_SINK = 128, 4
res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos = L["q"][sl], L["qpos"][sl]
        E, T = len(qpos), L["T"]
        Hkv, d = L["k"].shape[1], L["k"].shape[2]
        sr = scores(q, L["k"], qpos)
        p_true = softmax(sr)
        o_ref = attend(p_true, L["v"])
        t = np.arange(T)[None, :]
        A = (t < S_SINK) | ((t > qpos[:, None] - W_REC) & (t <= qpos[:, None]))
        gs = gain_shape_cache_v2(L, P, block=128, smooth=SMOOTH)
        sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
        # coarse estimates (exact for A tokens)
        se = np.where(A[None, None], sr, sc)
        pe = softmax(se)                                                    # (Hkv,G,E,T)
        Ve = gs["Vc"]
        oe = attend(pe, Ve).reshape(E, Hkv, -1, d)                          # (E,Hkv,G,d)
        # gains of the coarse codes (norm of mean-removed reconstruction ~ quantized gain)
        dk = [gs["diag"][h]["k"]["g"] for h in range(Hkv)]                 # true gains of (k - mu)
        dv = [gs["diag"][h]["v"]["g"] for h in range(Hkv)]
        gk = np.stack(dk, 1)                                                # (T,Hkv)
        gv = np.stack(dv, 1)
        qn2 = (q.reshape(E, Hkv, -1, d) ** 2).sum(-1)                       # (E,Hkv,G)
        masked = lambda x: np.where(A[None], -np.inf, x)                   # (Hkv,E,T)
        crit = {}
        crit["max-logp"] = masked(np.log(np.maximum(pe, 1e-30)).max(1))
        crit["sum-p2"] = masked((pe ** 2).sum(1))
        crit["sum-p"] = masked(pe.sum(1))
        # expected error weight; ||v_i - o_h||^2 = ||v_i||^2 - 2 v_i.o_h + ||o_h||^2 (coarse estimates)
        vv = (Ve ** 2).sum(-1).T                                           # (Hkv,T)
        vo = np.einsum("tkd,ekgd->kget", Ve, oe)                            # (Hkv,G,E,T)
        oo = (oe ** 2).sum(-1).transpose(1, 2, 0)[..., None]               # (Hkv,G,E,1)
        dist = vv[:, None, None, :] - 2 * vo + oo
        wk = qn2.transpose(1, 2, 0)[..., None] * (gk.T ** 2)[:, None, None, :] * dist / d ** 2
        crit["expected-error"] = masked((pe ** 2 * (wk + (gv.T ** 2)[:, None, None, :])).sum(1))
        crit["oracle sum-p2"] = masked((p_true ** 2).sum(1))
        for kf in (0.02, 0.05):
            k = k_for(qpos, kf)
            for name, cr in crit.items():
                hot = topk_rows(cr, k)
                o = tiered_attention(q, qpos, sc, sf, sr, gs["Vc"], gs["Vf"], L["v"], hot, A)
                e = np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)
                res.append((C, Li, kf, name, float(e.mean())))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

Aa = np.array(res, dtype=object)
np.save(os.path.join(RES, "e13_raw.npy"), Aa)
rows = []
for kf in (0.02, 0.05):
    for name in dict.fromkeys(Aa[:, 3]):
        a = Aa[(Aa[:, 2] == kf) & (Aa[:, 3] == name)]
        rows.append((f"{kf*100:.0f}%", name, float(a[:, 4].astype(float).mean()), float(a[a[:, 1] == LAYERS[0]][:, 4].astype(float).mean())))
g = {(r[0], r[1]): r[2] for r in rows}
write("13_selection_criteria.md", f"""# E13 Selection criteria from the first-order error model (window A + block means, int8 reference)

**TL;DR:** at k=5%: max-logp {g[('5%','max-logp')]:.4f}, GQA sum of squared probabilities {g[('5%','sum-p2')]:.4f},
expected-error weight {g[('5%','expected-error')]:.4f}, oracle (true p, sum-p2) {g[('5%','oracle sum-p2')]:.4f}.
At k=2%: {g[('2%','max-logp')]:.4f} / {g[('2%','sum-p2')]:.4f} / {g[('2%','expected-error')]:.4f} / oracle {g[('2%','oracle sum-p2')]:.4f}.

## Config
Same caches as E11 (DYN +A +blk128). Criteria rank non-window tokens per KV head; p, o, v are coarse estimates (exact for A).
expected-error: w_i = Σ_h p_hi² ( ||q_h||² g_ki² ||v_i − o_h||² / d² + g_vi² ), from the first-order error model.

""" + md_table(["k", "criterion", "attn err", f"layer-{LAYERS[0]} attn err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
