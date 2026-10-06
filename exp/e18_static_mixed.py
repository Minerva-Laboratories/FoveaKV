"""E18 Static vs per-step (query-dependent) mixed precision at equal bits.
Static (ZipCache/MiKV-style): one 4-bit token set chosen once from the mean attention of 64 probe queries (the first 64 decode
positions, excluded from evaluation), the rest 2-bit. FoveaKV: a new 4-bit set per decode step from the coarse keys.
Same nested code (shrinkage-corrected), same exact window, same number of 4-bit tokens; int8 reference."""
from common import *
from foveakv.attention import scores, softmax, attend, logprob
from foveakv.select import topk_rows, k_for
from foveakv.scheme import gain_shape_cache_v2, tiered_attention, block_means
from foveakv.quant import NestedCodebook

t0 = time.time()
CB = NestedCodebook()
Dc, Df = CB.mse_coarse, CB.mse_fine
NPROBE = 64
res = []
for C in CORPORA:
    for Li, L in ((Li, Lc) for Li in LAYERS for Lc in head_chunks(kvdata.load_layer(C, Li))):   # chunks of <= 2 KV heads
        P = prefill_len(L)
        T = L["T"]
        gs = gain_shape_cache_v2(L, P, block=128, smooth=SMOOTH)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc); Kf = Mk + (gs["Kf"] - Mk) / np.sqrt(1 - Df)
        Vc = Mv + (gs["Vc"] - Mv) / (1 - Dc); Vf = Mv + (gs["Vf"] - Mv) / (1 - Df)
        # probe: first NPROBE decode queries -> static saliency (mean attention over probe queries and the GQA group)
        qp, pp = L["q"][:NPROBE], L["qpos"][:NPROBE]
        sal = softmax(scores(qp, L["k"], pp)).mean(axis=(1, 2))               # (Hkv, T)
        # evaluation queries
        q, qpos = L["q"][NPROBE::2], L["qpos"][NPROBE::2]
        E = len(qpos)
        sr = scores(q, L["k"], qpos)
        o_ref = attend(softmax(sr), L["v"])
        t = np.arange(T)[None, :]
        A = (t < 4) | ((t > qpos[:, None] - 128) & (t <= qpos[:, None]))
        sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
        fin = np.isfinite(sr)
        rule = lambda s: np.where(A[None], -np.inf, logprob(np.where(A[None, None], np.where(fin, sr, -np.inf), s)).max(1))
        err = lambda hot: float((np.linalg.norm(tiered_attention(q, qpos, sc, sf, sr, Vc, Vf, L["v"], hot, A) - o_ref, axis=-1)
                                 / np.linalg.norm(o_ref, axis=-1)).mean())
        for kf in (0.02, 0.05, 0.10):
            k = k_for(qpos, kf)
            # static set: fixed over evaluation queries (tokens up to the probe end), broadcast to every query
            kstat = int(round(kf * (pp[-1] + 1)))
            sal_m = np.where(np.arange(T)[None, :] <= pp[-1], sal, -np.inf)
            stat = topk_rows(sal_m[:, None, :], np.array([kstat]))[:, 0]       # (Hkv, T)
            hot_static = np.broadcast_to(stat[:, None, :], (sal.shape[0], E, T)) & ~A[None]
            res.append((C, Li, kf, "static (probe-chosen) 4-bit set", err(hot_static)))
            res.append((C, Li, kf, "per-step (FoveaKV)", err(topk_rows(rule(sc), k))))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e18_raw.npy"), R)
rows = [(f"{kf*100:g}%", n, float(R[(R[:, 2] == kf) & (R[:, 3] == n)][:, 4].astype(float).mean()))
        for kf in (0.02, 0.05, 0.10) for n in ("static (probe-chosen) 4-bit set", "per-step (FoveaKV)")]
g = {(r[0], r[1]): r[2] for r in rows}
write("18_static_vs_dynamic.md", f"""# E18 Static vs per-step mixed precision at equal bits

**TL;DR:** with the same nested code and the same number of 4-bit tokens, a static set chosen from 64 probe queries
(ZipCache/MiKV-style) gives attn err {g[('5%','static (probe-chosen) 4-bit set')]:.4f} at 5%, the per-step query-dependent set {g[('5%','per-step (FoveaKV)')]:.4f}
(2%: {g[('2%','static (probe-chosen) 4-bit set')]:.4f} vs {g[('2%','per-step (FoveaKV)')]:.4f}; 10%: {g[('10%','static (probe-chosen) 4-bit set')]:.4f} vs {g[('10%','per-step (FoveaKV)')]:.4f}).

## Config
Probe = first {NPROBE} decode positions (excluded from evaluation); evaluation = remaining positions (stride 2). Shrinkage-corrected
nested code, block means, exact 128-token window, int8 reference, {len(CORPORA)} corpora x {len(LAYERS)} layers.

""" + md_table(["4-bit tokens", "selection", "attn err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
