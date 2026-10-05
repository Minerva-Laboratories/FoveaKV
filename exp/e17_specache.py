"""E17 FoveaKV vs a SpeCache-style design (Jie et al., ICML 2025): a 2-bit KIVI copy of all tokens on the GPU, top-k
selection from the low-bit keys, and the selected tokens replaced by their exact (reference) K/V fetched from CPU.
FoveaKV refines selected tokens with a 2-bit residual instead (4-bit result), so per hot token it moves 2 bits per value
instead of 16 (reference bf16 copy). Compared at equal refinement traffic (SpeCache k' = k/8) and at equal token count.
Both get the same exact recent window; reference = int8 cache (SpeCache's 'exact' copy = reference precision)."""
from common import *
from foveakv.attention import scores, softmax, attend, logprob
from foveakv.select import topk_rows, k_for
from foveakv.scheme import gain_shape_cache_v2, tiered_attention, block_means, kivi_cache
from foveakv.quant import NestedCodebook

t0 = time.time()
CB = NestedCodebook()
Dc, Df = CB.mse_coarse, CB.mse_fine
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
        A = (t < 4) | ((t > qpos[:, None] - 128) & (t <= qpos[:, None]))
        err = lambda o: float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)).mean())
        fin = np.isfinite(sr)
        rule = lambda s: np.where(A[None], -np.inf, logprob(np.where(A[None, None], np.where(fin, sr, -np.inf), s)).max(1))
        # FoveaKV (shrinkage-corrected)
        gs = gain_shape_cache_v2(L, P, block=128)
        Mk, Mv = block_means(L["k"], 128), block_means(L["v"], 128)
        Kc = Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc); Kf = Mk + (gs["Kf"] - Mk) / np.sqrt(1 - Df)
        Vc = Mv + (gs["Vc"] - Mv) / (1 - Dc); Vf = Mv + (gs["Vf"] - Mv) / (1 - Df)
        sc, sf = scores(q, Kc, qpos), scores(q, Kf, qpos)
        # SpeCache-style: KIVI-2b copy for all, exact for selected
        kv = kivi_cache(L, 2, 128)
        sk = scores(q, kv["Kq"], qpos)
        for kf in (0.01, 0.02, 0.05, 0.10):
            k = k_for(qpos, kf)
            hot = topk_rows(rule(sc), k)
            res.append((C, Li, "FoveaKV", kf, 2.16, 2 * kf, err(tiered_attention(q, qpos, sc, sf, sr, Vc, Vf, L["v"], hot, A))))
            for kp_name, kp in (("equal traffic (k/8)", kf / 8), ("equal tokens (k)", kf)):
                hk = topk_rows(rule(sk), k_for(qpos, kp))
                res.append((C, Li, f"SpeCache-style, {kp_name}", kf, kv["bits"], 16 * kp,
                            err(tiered_attention(q, qpos, sk, sr, sr, kv["Vq"], L["v"], L["v"], hk, A))))
        res.append((C, Li, "KIVI-2b only (no refinement)", 0.0, kv["bits"], 0.0,
                    err(tiered_attention(q, qpos, sk, sk, sr, kv["Vq"], kv["Vq"], L["v"], np.zeros_like(hot), A))))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

R = np.array(res, dtype=object)
np.save(os.path.join(RES, "e17_raw.npy"), R)
rows = []
for name in dict.fromkeys(R[:, 2]):
    for kf in sorted(set(R[R[:, 2] == name][:, 3].astype(float))):
        a = R[(R[:, 2] == name) & (R[:, 3].astype(float) == kf)]
        rows.append((name, f"{kf*100:g}%", float(a[0, 4]), float(a[0, 5]), float(a[:, 6].astype(float).mean()), float(a[a[:, 1] == 3][:, 6].astype(float).mean())))
g = {(r[0], r[1]): r[4] for r in rows}
write("17_specache.md", f"""# E17 FoveaKV vs SpeCache-style refinement (offline, int8 reference, shared exact window)

**TL;DR:** at 5% hot tokens FoveaKV gives {g[('FoveaKV','5%')]:.4f} while moving 0.10 bits/value/step of refinement data; SpeCache-style
refinement with the same traffic (0.625% of tokens, 16-bit) gives {g[('SpeCache-style, equal traffic (k/8)','5%')]:.4f}, and with the same token count (5%, 8x the
traffic) {g[('SpeCache-style, equal tokens (k)','5%')]:.4f}. KIVI-2b alone: {g[('KIVI-2b only (no refinement)','0%')]:.4f}.

## Config
SpeCache-style = KIVI-2b (group 128, fp16 scale/zero) resident for all tokens; top-k by max-over-group estimated log-prob from the
KIVI keys; selected tokens use the exact reference K/V (16-bit copy in CPU memory). FoveaKV = shrinkage-corrected nested code.
'refinement bits' = bits per value moved per step for hot tokens (FoveaKV: 2·k; SpeCache: 16·k').

""" + md_table(["method", "k", "resident bits", "refinement bits/step", "attn err", "layer-3 attn err"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
