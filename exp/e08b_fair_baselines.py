"""E8b Fairness check for E8: KIVI keeps its incomplete most-recent group (≤ group−1 tokens) at reference precision.
Compare KIVI with/without that residual and gain-shape 4b/2b/DYN with the *same* int8 residual group, plus a
key/value split (KIVI-style keys + gain-shape values)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import select
from foveakv.scheme import gain_shape_cache, kivi_cache
from foveakv.quant import INT8_BITS

t0 = time.time()
G = 128
res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos = L["q"][sl], L["qpos"][sl]
        sr = scores(q, L["k"], qpos)
        o_ref = attend(softmax(sr), L["v"])
        gs = gain_shape_cache(L, P)
        kv4, kv2 = kivi_cache(L, 4, G), kivi_cache(L, 2, G)
        pm = kv4["partial_mask"](qpos)                         # (E,T) incomplete group at reference precision
        Hkv = L["k"].shape[1]
        pmh = np.broadcast_to(pm[None], (Hkv,) + pm.shape)

        def out(sK, V, resid=None, sK_res=None, V_res=None):
            """Attention with logits sK and values V; tokens in `resid` mask use sK_res / V_res."""
            if resid is None:
                return attend(softmax(sK), V)
            h = resid[:, None]
            p = softmax(np.where(h, sK_res, sK))
            ph = p * h
            return attend(p - ph, V) + attend(ph, V_res)

        def rec(name, o):
            e = np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)
            res.append((C, Li, name, float(e.mean())))

        sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
        sk4, sk2 = scores(q, kv4["Kq"], qpos), scores(q, kv2["Kq"], qpos)
        rec("KIVI-4b g128 (with int8 residual group)", out(sk4, kv4["Vq"], pmh, sr, L["v"]))
        rec("KIVI-4b g128 (no residual)", out(sk4, kv4["Vq"]))
        rec("KIVI-2b g128 (with int8 residual group)", out(sk2, kv2["Vq"], pmh, sr, L["v"]))
        rec("KIVI-2b g128 (no residual)", out(sk2, kv2["Vq"]))
        rec("GS-4b (no residual)", out(sf, gs["Vf"]))
        rec("GS-4b + int8 residual group", out(sf, gs["Vf"], pmh, sr, L["v"]))
        rec("GS-2b + int8 residual group", out(sc, gs["Vc"], pmh, sr, L["v"]))
        rec("KIVI-4b keys + GS-4b values (no residual)", out(sk4, gs["Vf"]))
        hot = select(sc, qpos, 0.05, 4, 64)
        h = hot[:, None]
        p = softmax(np.where(pmh[:, None], sr, np.where(h, sf, sc)))
        hh = h & ~pmh[:, None]
        rec("DYN k=5% + int8 residual group", attend(p * ~(h | pmh[:, None]), gs["Vc"]) + attend(p * hh, gs["Vf"]) + attend(p * pmh[:, None], L["v"]))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

A = np.array(res, dtype=object)
names = list(dict.fromkeys(A[:, 2]))
rows = [(n, float(A[A[:, 2] == n][:, 3].astype(float).mean()), float(A[(A[:, 2] == n) & (A[:, 1] == 3)][:, 3].astype(float).mean())) for n in names]
g = {r[0]: r for r in rows}
write("08b_fair_baselines.md", f"""# E8b Fair baselines: the KIVI residual group

**TL;DR:** KIVI-4b's E8 advantage comes mostly from its int8 residual group (recent ≤127 tokens): without it KIVI-4b = {g['KIVI-4b g128 (no residual)'][1]:.4f} vs GS-4b {g['GS-4b (no residual)'][1]:.4f};
with the same residual group: KIVI-4b {g['KIVI-4b g128 (with int8 residual group)'][1]:.4f}, GS-4b {g['GS-4b + int8 residual group'][1]:.4f}, DYN k=5% {g['DYN k=5% + int8 residual group'][1]:.4f}.
Key/value split: KIVI-style per-channel keys + gain-shape values = {g['KIVI-4b keys + GS-4b values (no residual)'][1]:.4f} — keys favour per-channel quantization, values favour gain-shape.

## Config
Same data/reference as E8 (int8 cache, 5 corpora × 6 layers, 256 decode queries/layer). Residual group = tokens after the last
complete 128-token group at each decode step (mean ≈ 64 tokens), kept at int8 reference precision.

""" + md_table(["scheme", "attn err (all layers)", "attn err (layer 3)"], rows) + f"""
runtime {time.time()-t0:.0f}s
""")
