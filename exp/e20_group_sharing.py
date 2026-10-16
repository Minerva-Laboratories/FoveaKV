"""E20 Why does SpeCache-style refinement lose to FoveaKV on GQA-group-8 models? Both share one hot set per KV head
(chosen by the max over the group of estimated log-probs, as in E17). Here the hot set is shared by sub-groups of s query
heads (s = 1 .. G, each sub-group gets its own k tokens), so the effect of group size is measured inside one model.
Also swaps the parts of the two designs (selector, tail code, hot-token precision) to see which one carries the gap:
  FoveaKV      coarse nested-code selector, 2-bit gain-shape tail, 4-bit hot tokens
  SpeCache     KIVI-2b selector,            KIVI-2b tail,           exact hot tokens
  Spe+GS tail  KIVI-2b selector,            2-bit gain-shape tail,  exact hot tokens
  Fov+exact    coarse selector,             2-bit gain-shape tail,  exact hot tokens
'tail mass' = true attention mass on tokens outside the hot set and the exact window (mean over query heads)."""
from common import *
from foveakv.attention import scores, softmax, attend, logprob
from foveakv.select import topk_rows, k_for
from foveakv.scheme import gain_shape_cache_v2, block_means, kivi_cache
from foveakv.quant import NestedCodebook

t0 = time.time()
CB = NestedCodebook()
Dc, Df = CB.mse_coarse, CB.mse_fine
KF = float(os.environ.get("FOVEAKV_E20_K", "0.05"))


def tiered(sc, sf, sr, Vc, Vf, Vr, hot, win):
    """hot (Hkv, G, E, T): per query head; otherwise as scheme.tiered_attention."""
    w = win[None, None]
    h = hot & ~w
    p = softmax(np.where(w, sr, np.where(h, sf, sc)))
    pw, ph = p * w, p * h
    return attend(p - pw - ph, Vc) + attend(ph, Vf) + attend(pw, Vr)


def shared_sets(s_est, A, k, s):
    """One top-k set per sub-group of s query heads, by the max over the sub-group of estimated log-probs."""
    Hkv, G, E, T = s_est.shape
    lp = np.where(A[None, None], -np.inf, logprob(s_est))
    lp = lp.reshape(Hkv, G // s, s, E, T).max(2)                          # (Hkv, G/s, E, T)
    hot = topk_rows(lp, k)
    return np.repeat(hot, s, axis=1)                                       # (Hkv, G, E, T)


res = []
for C in CORPORA:
    for Li, L in ((Li, Lc) for Li in LAYERS for Lc in head_chunks(kvdata.load_layer(C, Li))):
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos = L["q"][sl], L["qpos"][sl]
        T = L["T"]
        G = q.shape[1] // L["k"].shape[1]
        sr = scores(q, L["k"], qpos)
        pr = softmax(sr)
        o_ref = attend(pr, L["v"])
        t = np.arange(T)[None, :]
        A = (t < 4) | ((t > qpos[:, None] - 128) & (t <= qpos[:, None]))
        err = lambda o: float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
        fin = np.isfinite(sr)
        gs = gain_shape_cache_v2(L, P, block=128, smooth=SMOOTH)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc); Kf = Mk + (gs["Kf"] - Mk) / np.sqrt(1 - Df)
        Vc = Mv + (gs["Vc"] - Mv) / (1 - Dc); Vf = Mv + (gs["Vf"] - Mv) / (1 - Df)
        sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
        kv = kivi_cache(L, 2, 128)
        sk = scores(q, kv["Kq"], qpos)
        sc_, sk_ = np.where(fin, sc, -np.inf), np.where(fin, sk, -np.inf)
        k = k_for(qpos, KF)
        for s in [x for x in (1, 2, 4, 8, 16) if x <= G and G % x == 0]:
            hc, hk = shared_sets(sc_, A, k, s), shared_sets(sk_, A, k, s)
            # distinct hot tokens per KV head per step, in units of k (traffic of the union of the sub-group sets)
            uc = float((hc.any(1).sum(-1) / k[None]).mean()); uk = float((hk.any(1).sum(-1) / k[None]).mean())
            tail = lambda h: float((pr * ~(h | A[None, None])).sum(-1).mean())
            for name, h, u, args in (
                    ("FoveaKV", hc, uc, (sc, sf, sr, Vc, Vf, L["v"])),
                    ("SpeCache", hk, uk, (sk, sr, sr, kv["Vq"], L["v"], L["v"])),
                    ("Spe+GS tail", hk, uk, (sc, sr, sr, Vc, L["v"], L["v"])),
                    ("Fov+exact", hc, uc, (sc, sr, sr, Vc, L["v"], L["v"]))):
                res.append((C, Li, name, G, s, u, tail(h), err(tiered(*args, h, A))))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e20_raw.npy"), R)
G = int(R[0, 3])
rows = []
for name in dict.fromkeys(R[:, 2]):
    for s in sorted(set(R[:, 4].astype(int))):
        a = R[(R[:, 2] == name) & (R[:, 4].astype(int) == s)]
        rows.append((name, s, float(a[:, 5].astype(float).mean()), float(a[:, 6].astype(float).mean()), float(a[:, 7].astype(float).mean())))
g = {(r[0], r[1]): r[4] for r in rows}
tm = {(r[0], r[1]): r[3] for r in rows}
write("20_group_sharing.md", f"""# E20 Sharing the hot set across query heads (k = {KF*100:g}%, group size G = {G})

**TL;DR:** one set per KV head (s = {G}): FoveaKV {g[('FoveaKV', G)]:.4f}, SpeCache-style {g[('SpeCache', G)]:.4f}; one set per query head
(s = 1): FoveaKV {g[('FoveaKV', 1)]:.4f}, SpeCache-style {g[('SpeCache', 1)]:.4f}. Attention mass left on the low-bit tail (shared set):
FoveaKV selector {tm[('FoveaKV', G)]:.3f}, KIVI selector {tm[('SpeCache', G)]:.3f}. SpeCache-style with FoveaKV's 2-bit tail:
{g[('Spe+GS tail', G)]:.4f}; FoveaKV with exact hot tokens: {g[('Fov+exact', G)]:.4f}.

## Config
Same exact window (4 sink + 128 recent tokens) and k for every row; each sub-group of s query heads gets its own top-k set
(max over the sub-group of estimated log-probs). 'distinct hot / k' = tokens fetched per KV head per step, in units of k.

""" + md_table(["scheme", "heads per set (s)", "distinct hot / k", "tail mass", "attn err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
