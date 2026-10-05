"""E8 Per-layer output error vs exact attention on the int8 reference cache, for all schemes, with bits.
Metrics per (query, query-head): rel L2 error of the attention output (pre-gate); and per query: rel L2 error of
the layer's residual contribution (o_proj(attn * sigmoid(gate))). Walk-forward, 256 decode queries/layer (stride 2)."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import select
from foveakv.scheme import gain_shape_cache, kivi_cache
from foveakv.model_hybrid import sigmoid
from foveakv.load import SafeTensors, snapshot_dir
from foveakv.quant import INT8_BITS

t0 = time.time()
ST = SafeTensors(snapshot_dir())
KS = [0.01, 0.02, 0.05, 0.10, 0.20]
STRIDE = 2
D = 256
res = []          # (corpus, layer, scheme, group, resident_bits, read_bits, err_attn, err_resid)


def mix(sc, sf, Vc, Vf, hot):
    h = hot[:, None]
    p = softmax(np.where(h, sf, sc))
    ph = p * h
    return attend(p - ph, Vc) + attend(ph, Vf)


for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, STRIDE)
        q, qpos, gate = L["q"][sl], L["qpos"][sl], L["gate"][sl]
        E = len(qpos)
        Wo = ST.get(f"model.language_model.layers.{Li}.self_attn.o_proj.weight")
        o_ref = attend(softmax(scores(q, L["k"], qpos)), L["v"])
        r_ref = (o_ref.reshape(E, -1) * sigmoid(gate)) @ Wo.T

        def rec(scheme, group, res_bits, read_bits, o):
            ea = np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)
            r = (o.reshape(E, -1) * sigmoid(gate)) @ Wo.T
            er = np.linalg.norm(r - r_ref, axis=-1) / np.linalg.norm(r_ref, axis=-1)
            res.append((C, Li, scheme, group, res_bits, read_bits, float(ea.mean()), float(er.mean())))

        rec("int8 (reference)", "ref", INT8_BITS, INT8_BITS, o_ref)

        def gs_family(tag, group, rope="post", rht=True, gain_bits=8, dyn=True, sw_sweep=False):
            gs = gain_shape_cache(L, P, rope=rope, rht=rht, gain_bits=gain_bits)
            bc, bf = gs["bits_coarse"], gs["bits_fine"]
            sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
            rec(f"GS-2b {tag}", group, bc, bc, attend(softmax(sc), gs["Vc"]))
            rec(f"GS-4b {tag}", group, bf, bf, attend(softmax(sf), gs["Vf"]))
            if not dyn:
                return
            ss = scores(q, gs["Ks"], qpos)
            for sel_name, s_sel in (("coarse", sc), ("sign", ss)):
                for kf in (KS if group == "main" else [0.05]):
                    hot = select(s_sel, qpos, kf, 4, 64)
                    f = hot.sum(-1).mean() / (qpos.mean() + 1)                       # actual hot fraction of valid tokens
                    rec(f"DYN sel={sel_name} k={kf*100:.0f}% {tag}", group, bc, bc + 2 * f, mix(sc, sf, gs["Vc"], gs["Vf"], hot))
            if group == "main":
                hot = select(sc, qpos, 0.05, 4, 64, mode="union")
                f = hot.sum(-1).mean() / (qpos.mean() + 1)
                rec(f"DYN sel=coarse k=5%/head UNION {tag}", group, bc, bc + 2 * f, mix(sc, sf, gs["Vc"], gs["Vf"], hot))
            if sw_sweep:
                for S, W in ((0, 0), (4, 0), (0, 64), (16, 256)):
                    for kf in (0.02, 0.05):
                        hot = select(sc, qpos, kf, S, W)
                        f = hot.sum(-1).mean() / (qpos.mean() + 1)
                        rec(f"DYN sel=coarse k={kf*100:.0f}% S={S} W={W}", "sw", bc, bc + 2 * f, mix(sc, sf, gs["Vc"], gs["Vf"], hot))

        gs_family("post-RoPE RHT g8", "main", sw_sweep=True)
        gs_family("pre-RoPE RHT g8", "rope", rope="pre")
        gs_family("post-RoPE noRHT g8", "rht", rht=False)
        for gb in (4, 6):
            gs_family(f"post-RoPE RHT g{gb}", "gain", gain_bits=gb)
        for bits in (2, 4):
            for G in (32, 128):
                kv = kivi_cache(L, bits, G)
                pm = kv["partial_mask"](qpos)                                     # incomplete group: reference precision
                frac_ref = pm.sum(-1).mean() / (qpos.mean() + 1)
                b = kv["bits"] * (1 - frac_ref) + INT8_BITS * frac_ref
                sk = scores(q, kv["Kq"], qpos)
                sr = scores(q, L["k"], qpos)
                hot = np.broadcast_to(pm[None], (L["k"].shape[1],) + pm.shape)
                rec(f"KIVI-{bits}b g{G}", "main", b, b, mix(sk, sr, kv["Vq"], L["v"], hot))
        del L
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

A = np.array(res, dtype=object)
np.save(os.path.join(RES, "e08_raw.npy"), A)


def agg(rows_sel):
    out = {}
    for r in rows_sel:
        out.setdefault(r[2], []).append(r)
    tab = []
    for name, rs in out.items():
        rs = np.array(rs, dtype=object)
        tab.append((name, float(np.mean(rs[:, 4].astype(float))), float(np.mean(rs[:, 5].astype(float))),
                    float(np.mean(rs[:, 6].astype(float))), float(np.max(rs[:, 6].astype(float))),
                    float(np.mean(rs[:, 7].astype(float))), float(np.max(rs[:, 7].astype(float)))))
    return tab


hdr = ["scheme", "resident bits/elem", "read bits/elem/step", "attn err mean", "attn err worst (corpus,layer)", "resid err mean", "resid err worst"]
main = agg([r for r in res if r[3] in ("main", "ref")])
main.sort(key=lambda r: r[2])
get = lambda name: [r for r in main if r[0] == name][0]
u4, u2 = get("GS-4b post-RoPE RHT g8"), get("GS-2b post-RoPE RHT g8")
d5, d10 = get("DYN sel=coarse k=5% post-RoPE RHT g8"), get("DYN sel=coarse k=10% post-RoPE RHT g8")
ds5 = get("DYN sel=sign k=5% post-RoPE RHT g8")
perlayer = []
for Li in LAYERS:
    for name in ("GS-2b post-RoPE RHT g8", "DYN sel=coarse k=5% post-RoPE RHT g8", "DYN sel=coarse k=10% post-RoPE RHT g8",
                 "GS-4b post-RoPE RHT g8", "KIVI-4b g128"):
        rs = np.array([r for r in res if r[1] == Li and r[2] == name], dtype=object)
        perlayer.append((Li, name, float(rs[:, 6].astype(float).mean()), float(rs[:, 7].astype(float).mean())))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for j, (col, lab) in enumerate(((3, "attention-output rel err"), (5, "residual-contribution rel err"))):
    for fam, mk in (("DYN sel=coarse", "o-"), ("DYN sel=sign", "s--")):
        pts = sorted([r for r in main if r[0].startswith(fam) and "UNION" not in r[0]], key=lambda r: r[2])
        ax[j].plot([r[2] for r in pts], [r[col] for r in pts], mk, label=fam)
    for r in main:
        if not r[0].startswith("DYN"):
            ax[j].plot(r[2], r[col], "k^" if "KIVI" in r[0] else "rD")
            ax[j].annotate(r[0].replace(" post-RoPE RHT g8", ""), (r[2], r[col]), fontsize=7)
    ax[j].set_xlabel("bits read per element per decode step"); ax[j].set_ylabel(lab); ax[j].set_yscale("log"); ax[j].grid(alpha=.3)
    ax[j].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "e08_err_vs_bits.png"), dpi=90)

txt = f"""# E8 Per-layer output error vs bits (Qwen3.5-0.8B, 6 full-attn layers, 5 corpora, T=8192, 256 decode queries/layer)

**TL;DR:** uniform gain-shape 4-bit: attn err {u4[3]:.4f} at {u4[2]:.2f} bits; 2-bit: {u2[3]:.4f} at {u2[2]:.2f} bits.
Dynamic (coarse select, S=4, W=64): k=5% → {d5[3]:.4f} at {d5[2]:.2f} read bits ({d5[1]:.2f} resident); k=10% → {d10[3]:.4f} at {d10[2]:.2f}; sign-plane select k=5% → {ds5[3]:.4f}.
Residual-stream contribution error: GS-4b {u4[5]:.4f}, DYN k=5% {d5[5]:.4f}, DYN k=10% {d10[5]:.4f}, GS-2b {u2[5]:.4f}.

## Config
- Reference: exact attention on the int8 cache (8.06 bits). Errors: mean over corpora/layers of the per-(query, head) rel L2 error.
- GS = gain-shape (mean removal from prefill, RHT seed 0, 8-bit log gain calibrated on prefill, Gaussian Lloyd-Max 4b with nested 2b).
- DYN: budget mode — hot set shared by the GQA group, ranked by max over the group of estimated log-prob; k counts all hot tokens incl. sinks S=4 and window W=64.
- Bits: resident = coarse planes + gains; read/step = coarse (all) + 2·(hot fraction) for the residual plane. KIVI: bits + 32/group (fp16 scale+zero), incomplete trailing group kept at int8.
- mu vectors (one per head) are not counted (≈0.0005 bits/elem at 8K).

## Main comparison
""" + md_table(hdr, main) + """
## Per layer (mean over corpora)
""" + md_table(["layer", "scheme", "attn err", "resid err"], perlayer) + """
## RoPE: pre vs post (k=5% for DYN)
""" + md_table(hdr, agg([r for r in res if r[3] == "rope"] + [r for r in res if r[3] == "main" and r[2] in ("GS-2b post-RoPE RHT g8", "GS-4b post-RoPE RHT g8", "DYN sel=coarse k=5% post-RoPE RHT g8")])) + """
## RHT on/off
""" + md_table(hdr, agg([r for r in res if r[3] == "rht"])) + """
## Gain bits
""" + md_table(hdr, agg([r for r in res if r[3] == "gain"])) + """
## Sinks / recent window (coarse select, budget includes S+W)
""" + md_table(hdr, agg([r for r in res if r[3] == "sw"] + [r for r in res if r[2] in ("DYN sel=coarse k=2% post-RoPE RHT g8", "DYN sel=coarse k=5% post-RoPE RHT g8")])) + f"""
## Plot
![err vs bits](plots/e08_err_vs_bits.png)

## Caveats
- Per-layer only: no error propagation across layers (offline cache). End-to-end PPL needs E9.
- Read-bits for DYN assume the residual plane is read only for hot tokens; on-GPU LRU hits (E6) do not cost PCIe but do cost HBM reads (counted here).
runtime {time.time()-t0:.0f}s
"""
write("08_layer_error.md", txt)
