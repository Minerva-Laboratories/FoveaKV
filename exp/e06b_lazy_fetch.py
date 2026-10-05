"""E6b Diagnose the low temporal reuse of E6: are the misses high-mass or boundary churn?
Policy 'lazy fetch': at step t only hot tokens whose residual is already on GPU (LRU after step t-1, plus sinks and
recent window) use the fine plane; missing residuals are fetched for step t+1 (no stall). Compares attention error
vs the ideal (all hot fine) and reports the mass-weighted hit rate. Walk-forward, 512 consecutive steps."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import select, always_hot
from foveakv.scheme import gain_shape_cache

t0 = time.time()
KF, S, W = 0.05, 4, 64
CAPS = [1, 2, 4]
res = []


def lru_available(hots):
    """hots (E,T) dynamic hot sets. Returns dict cap_x -> (E,T) bool residuals on GPU at the start of each step."""
    E, T = hots.shape
    k_dyn = max(1, int(hots.sum(-1).mean()))
    out = {}
    for x in CAPS:
        cap = x * k_dyn
        last = np.full(T, -1)
        avail = np.zeros((E, T), bool)
        for t in range(E):
            used = np.flatnonzero(last >= 0)
            keep = used[np.argpartition(-last[used], cap - 1)[:cap]] if len(used) > cap else used
            avail[t, keep] = True
            last[hots[t]] = t                      # misses fetched now, usable from step t+1
        out[x] = avail
    return out


for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P, q, qpos, T = prefill_len(L), L["q"], L["qpos"], L["T"]
        p_true = softmax(scores(q, L["k"], qpos))
        o_ref = attend(p_true, L["v"])
        gs = gain_shape_cache(L, P)
        sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
        ah = always_hot(qpos, T, S, W)
        hot = select(sc, qpos, KF, S, W)
        dyn = hot & ~ah[None]

        def err(hmask):
            h = hmask[:, None]
            pm = softmax(np.where(h, sf, sc))
            o = attend(pm * ~h, gs["Vc"]) + attend(pm * h, gs["Vf"])
            return float((np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1))[1:].mean())

        e_ideal, e_coarse_dyn = err(hot), err(hot & ah[None])
        row = [C, Li, e_ideal, e_coarse_dyn]
        mass_dyn = (p_true * dyn[:, None]).sum(-1)                              # (Hkv,G,E)
        for x in CAPS:
            avail = np.stack([lru_available(dyn[h])[x] for h in range(dyn.shape[0])])
            eff = (dyn & avail) | (hot & ah[None])
            mhit = (p_true * (dyn & avail)[:, None]).sum(-1) / np.maximum(mass_dyn, 1e-12)
            chit = (dyn & avail).sum(-1) / np.maximum(dyn.sum(-1), 1)
            row += [float(chit[:, 1:].mean()), float(mhit[..., 1:].mean()), err(eff)]
        res.append(tuple(row))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

A = np.array(res, dtype=object)
cols = ["ideal (all hot fine)", "no dynamic residuals (only S+W fine)"] + sum(
    [[f"count hit LRU{x}×", f"mass hit LRU{x}×", f"err lazy LRU{x}×"] for x in CAPS], [])
mean = A[:, 2:].astype(float).mean(0)
pl = [(Li,) + tuple(A[A[:, 1] == Li][:, 2:].astype(float).mean(0)) for Li in LAYERS]
write("06b_lazy_fetch.md", f"""# E6b Lazy fetch: is the low step-to-step reuse harmful? (k=5%, S=4, W=64)

**TL;DR:** count hit-rates LRU 1×/2×/4×: {mean[2]:.3f} / {mean[5]:.3f} / {mean[8]:.3f}; *mass-weighted* hit-rates {mean[3]:.3f} / {mean[6]:.3f} / {mean[9]:.3f}.
Attention error with lazy fetch (misses served coarse this step, fetched for the next): {mean[4]:.4f} / {mean[7]:.4f} / {mean[10]:.4f} vs ideal {mean[0]:.4f} (and {mean[1]:.4f} with no dynamic residuals at all).
Lazy fetch with LRU 2× recovers {(mean[1]-mean[7])/max(mean[1]-mean[0],1e-12)*100:.0f}% of the error reduction that ideal dynamic residuals give over S+W-only.

## Config
Walk-forward over 512 consecutive decode steps; per KV head LRU; capacities in units of the mean dynamic hot-set size.
Reference = exact attention on int8 cache; GS planes as in E8 (post-RoPE, RHT, 8-bit gain).

""" + md_table(["metric", "value"], list(zip(cols, mean))) + """
## Per layer
""" + md_table(["layer", "ideal", "S+W only"] + sum([[f"cnt{x}", f"mass{x}", f"err{x}"] for x in CAPS], []), pl) + f"""
runtime {time.time()-t0:.0f}s
""")
