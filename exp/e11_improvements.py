"""E11 Low-hanging improvements (offline, walk-forward, int8 reference) + TurboQuant-MSE baseline.
  A. always-exact set A = sinks (4) + recent window (128) at reference precision, OUTSIDE the hot budget
  B. closed-form per-channel key/query smoothing before the RHT (prefill statistics only)
  C. adaptive k per query head (smallest k with estimated mass >= tau), union over the GQA group
Metrics as E8 (attn rel err, residual-contribution rel err); bits at 8K and in the long-context limit."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import select, topk_rows, sort_desc, k_for
from foveakv.scheme import gain_shape_cache, gain_shape_cache_v2, turboquant_mse, tiered_attention, kivi_cache
from foveakv.model_hybrid import sigmoid
from foveakv.load import SafeTensors, snapshot_dir
from foveakv.quant import INT8_BITS

t0 = time.time()
ST = SafeTensors(snapshot_dir())
S_SINK, W_REC = 4, 128
res = []

for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, 2)
        q, qpos, gate = L["q"][sl], L["qpos"][sl], L["gate"][sl]
        E, T = len(qpos), L["T"]
        Wo = ST.get(f"model.language_model.layers.{Li}.self_attn.o_proj.weight")
        sr = scores(q, L["k"], qpos)
        o_ref = attend(softmax(sr), L["v"])
        r_ref = (o_ref.reshape(E, -1) * sigmoid(gate)) @ Wo.T
        t = np.arange(T)[None, :]
        A = (t < S_SINK) | ((t > qpos[:, None] - W_REC) & (t <= qpos[:, None]))          # (E,T)
        nA = A.sum(-1).mean()
        n = qpos.mean() + 1

        def rec(name, o, b8k, binf):
            ea = np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)
            r = (o.reshape(E, -1) * sigmoid(gate)) @ Wo.T
            er = np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)
            res.append((C, Li, name, b8k, binf, float(ea.mean()), float(er.mean())))

        bitsA = lambda b: (b * (n - nA) + INT8_BITS * nA) / n
        # ---------------- baselines
        for b in (2, 3, 4):
            tq = turboquant_mse(L, b)
            stq = scores(q, tq["K"], qpos)
            rec(f"TQ-{b}b", attend(softmax(stq), tq["V"]), tq["bits"], tq["bits"])
            nohot = np.zeros((L["k"].shape[1], E, T), bool)
            rec(f"TQ-{b}b +A", tiered_attention(q, qpos, stq, stq, sr, tq["V"], tq["V"], L["v"], nohot, A), bitsA(tq["bits"]), tq["bits"])
        kv = kivi_cache(L, 4, 128)
        pm = kv["partial_mask"](qpos)
        nohot = np.zeros((L["k"].shape[1], E, T), bool)
        skv = scores(q, kv["Kq"], qpos)
        rec("KIVI-4b g128 (+own int8 residual)", tiered_attention(q, qpos, skv, skv, sr, kv["Vq"], kv["Vq"], L["v"], nohot, pm), kv["bits"], kv["bits"])

        for smooth, block in ((False, None), (True, None), (False, 128), (True, 128)):
            tag = (" +smooth" if smooth else "") + (f" +blk{block}" if block else "")
            gs = gain_shape_cache_v2(L, P, smooth=smooth, block=block)
            bc, bf = gs["bits_coarse"], gs["bits_fine"]
            sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
            rec("GS-4b" + tag, attend(softmax(sf), gs["Vf"]), bf, bf)
            rec("GS-4b +A" + tag, tiered_attention(q, qpos, sf, sf, sr, gs["Vf"], gs["Vf"], L["v"], nohot, A), bitsA(bf), bf)
            rec("GS-2b +A" + tag, tiered_attention(q, qpos, sc, sc, sr, gs["Vc"], gs["Vc"], L["v"], nohot, A), bitsA(bc), bc)
            if not smooth and not block:     # old configuration (E8): S/W inside the budget, 4-bit window
                hot = select(sc, qpos, 0.05, 4, 64)
                f = hot.sum(-1).mean() / n
                h = hot[:, None]
                p = softmax(np.where(h, sf, sc))
                rec("DYN k=5% (E8 config)", attend(p * ~h, gs["Vc"]) + attend(p * h, gs["Vf"]), bc + 2 * f, bc + 2 * f)
            # dynamic tokens exclude A; estimator = coarse logits
            sel = np.where(A[None, None], -np.inf, sc)
            for kf in (0.02, 0.05, 0.10):
                hot = select(sel, qpos, kf, 0, 0)
                f = hot.sum(-1).mean() / n
                rec(f"DYN k={kf*100:.0f}% +A" + tag, tiered_attention(q, qpos, sc, sf, sr, gs["Vc"], gs["Vf"], L["v"], hot, A),
                    bitsA(bc) + 2 * f, bc + 2 * f)
            # adaptive k (estimated mass incl. the exact A tokens)
            pe = softmax(np.where(A[None, None], sr, sc))                                 # (Hkv,G,E,T)
            mA = (pe * A[None, None]).sum(-1)
            pn = np.where(A[None, None], -1.0, pe)
            srt = sort_desc(pn)
            cum = np.cumsum(np.maximum(srt, 0), -1)
            for tau in (0.8, 0.9, 0.95):
                kh = ((cum + mA[..., None]) < tau).sum(-1) + 1
                kh = np.minimum(kh, np.ceil(0.2 * (qpos + 1)).astype(int)[None, None])
                kh = np.where(mA >= tau, 0, kh)
                hot = topk_rows(pn, np.maximum(kh, 1), srt) & (kh[..., None] > 0) & ~A[None, None]
                hot = hot.any(1)
                f = hot.sum(-1).mean() / n
                rec(f"DYN adaptive τ={tau} +A" + tag, tiered_attention(q, qpos, sc, sf, sr, gs["Vc"], gs["Vf"], L["v"], hot, A),
                    bitsA(bc) + 2 * f, bc + 2 * f)
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

Aarr = np.array(res, dtype=object)
np.save(os.path.join(RES, "e11_raw.npy"), Aarr)
names = list(dict.fromkeys(Aarr[:, 2]))
rows = []
for nm in names:
    a = Aarr[Aarr[:, 2] == nm]
    f = lambda j: float(a[:, j].astype(float).mean())
    rows.append((nm, f(3), f(4), f(5), float(a[:, 5].astype(float).max()), f(6), float(a[(a[:, 1] == 3)][:, 5].astype(float).mean())))
rows.sort(key=lambda r: r[2])
g = {r[0]: r for r in rows}
pl = []
for Li in LAYERS:
    for nm in ("GS-4b +A", "DYN k=5% +A", "DYN k=5% +A +blk128", "DYN adaptive τ=0.9 +A +blk128", "TQ-4b +A", "TQ-2b +A"):
        a = Aarr[(Aarr[:, 2] == nm) & (Aarr[:, 1] == Li)]
        pl.append((Li, nm, float(a[:, 5].astype(float).mean()), float(a[:, 6].astype(float).mean())))

fig, ax = plt.subplots(figsize=(8, 5))
fam = {"TQ-": ("k^", "TurboQuant-MSE"), "GS-": ("rD", "gain-shape uniform"), "KIVI": ("ms", "KIVI"), "DYN k=": ("bo", "DynReq fixed k"),
       "DYN adaptive": ("g*", "DynReq adaptive k")}
for key, (mk, lab) in fam.items():
    pts = [r for r in rows if r[0].startswith(key)]
    ax.plot([r[2] for r in pts], [r[3] for r in pts], mk, label=lab, alpha=.8)
    for r in pts:
        ax.annotate(r[0].replace("DYN ", "").replace(" (E8 config)", "*"), (r[2], r[3]), fontsize=6)
ax.set_xlabel("bits/elem read per step (long-context limit)"); ax.set_ylabel("attention-output rel err (vs int8 ref)")
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "e11_frontier.png"), dpi=100)

best = g["DYN k=5% +A +blk128"]
write("11_improvements.md", f"""# E11 Low-hanging improvements + TurboQuant-MSE baseline (offline, int8 reference, 5 corpora × 6 layers)

**TL;DR:** E8 config DYN k=5%: attn err {g['DYN k=5% (E8 config)'][3]:.4f}. Adding (A) an exact sink+recent set outside the budget → {g['DYN k=5% +A'][3]:.4f};
plus (C) block-local centering (128-token fp16 means) → {best[3]:.4f} at {best[2]:.2f} bits (long-context limit); (B) key smoothing alone: {g['DYN k=5% +A +smooth'][3]:.4f}. References: GS-4b +A {g['GS-4b +A'][3]:.4f} ({g['GS-4b +A'][2]:.2f} b),
TurboQuant-MSE 2/3/4b (+A): {g['TQ-2b +A'][3]:.4f} / {g['TQ-3b +A'][3]:.4f} / {g['TQ-4b +A'][3]:.4f}, KIVI-4b {g['KIVI-4b g128 (+own int8 residual)'][3]:.4f} ({g['KIVI-4b g128 (+own int8 residual)'][2]:.2f} b).
Adaptive k (τ=0.9, +A, +blk128): {g['DYN adaptive τ=0.9 +A +blk128'][3]:.4f} at {g['DYN adaptive τ=0.9 +A +blk128'][2]:.2f} bits. GS-4b +A +blk128: {g['GS-4b +A +blk128'][3]:.4f}.

## Config
- A = first 4 tokens + last 128 tokens at reference (int8) precision, not counted in k. bits@8K include A (8.06 b for ~132 of ~8K tokens);
  the long-context limit drops A (→ coarse + 2·hot fraction).
- Smoothing: s_c = (E[(k_c−μ_c)²]/E[q_c²])^(1/4) per KV head from prefill keys and prefill queries (group-averaged); keys stored as (k−μ)/s, queries scaled by s.
- TurboQuant-MSE: RHT + Lloyd-Max N(0,1/d) codebook + fp16 norm, no mean removal, no QJL; b + 16/256 bits.
- Block centering: per-channel mean of each complete 128-token block (fp16), subtracted from K and V before RHT; +0.125 bits/elem.
- Adaptive: per query head smallest k s.t. estimated mass (coarse logits; exact for A) ≥ τ, capped at 20%; union over the 4 heads of a group.

""" + md_table(["scheme", "bits @8K", "bits (long ctx)", "attn err", "attn err worst", "resid err", "layer-3 attn err"], rows) + """
## Per layer
""" + md_table(["layer", "scheme", "attn err", "resid err"], pl) + f"""
![frontier](plots/e11_frontier.png)

runtime {time.time()-t0:.0f}s
""")
