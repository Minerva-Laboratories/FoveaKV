"""E19 Why does key smoothing matter on Qwen3 and not on Qwen3.5? Pipeline check + characterization.
Runs over several captures at once (FOVEAKV_KV_DIRS = name=dir[@ref],... with ref int8 (default) or fp16); NumPy only.

 (A) pipeline check: offline exact attention (int8 cache, our RoPE) vs the model's own o_proj input (captured attn_in).
     If RoPE/scale/layout were wrong for a model, this error would be large and every result on it would be invalid.
 (B) per (layer, KV head), from prefill statistics of post-RoPE keys (mean removed) and queries:
       k2_c = Var(k_c), q2_c = E[q_c^2] (group mean).
     After the rotation the quantization error is ~isotropic, so the logit error variance is ∝ ||q||^2 ||k-mu||^2 =
     (Σ q2)(Σ k2); with smoothing s_c it is (Σ q2 s²)(Σ k2 / s²), minimized at (Σ sqrt(q2 k2))^2. Predicted gain
       Gpred = (Σ q2)(Σ k2) / (Σ sqrt(q2 k2))^2 ≥ 1   (= 1 iff q2 ∝ k2 across channels).
     Measured: MSE of 2-bit (coarse) logits vs true, without/with smoothing, on decode queries; and the logit error
     relative to the spread of true logits (what the softmax sees)."""
import os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

t0 = time.time()
CAPS = [kv.split("=") for kv in os.environ["FOVEAKV_KV_DIRS"].split(",")]
RES = os.environ.get("FOVEAKV_RESULTS", os.path.join(ROOT, "results"))
os.makedirs(os.path.join(RES, "plots"), exist_ok=True)
CORPORA = os.environ.get("E19_CORPORA", "prose,code").split(",")
rows_a, rows_b, prof = [], [], {}


def md_table(header, rows, fmt="{:.3f}"):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(fmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |\n"
    return s


REFS = {}
for name, kvdir in CAPS:
    kvdir, _, ref = kvdir.partition("@")
    REFS[name] = ref or "int8"
    os.environ["FOVEAKV_KV_DIR"] = kvdir
    os.environ["FOVEAKV_REF"] = REFS[name]
    for m in [m for m in list(sys.modules) if m.startswith("foveakv")]:
        del sys.modules[m]                                   # kvdata reads FOVEAKV_KV_DIR at import
    from foveakv import kvdata
    from foveakv.attention import scores, softmax, attend
    from foveakv.scheme import gain_shape_cache_v2, key_smoothing_scales
    meta = kvdata.meta(CORPORA[0])
    layers = meta["attn_layers"]
    if len(layers) > 4:
        layers = [layers[i] for i in np.linspace(0, len(layers) - 1, 4).round().astype(int)]
    for C in CORPORA:
        for Li in layers:
            L = kvdata.load_layer(C, Li)
            E, T = L["E"], L["T"]
            P = T - E
            Hkv, d = L["k"].shape[1], L["k"].shape[2]
            G = L["q"].shape[1] // Hkv
            # (A) pipeline check on a subset of queries
            sl = slice(0, None, 8)
            o = attend(softmax(scores(L["q"][sl], L["k"], L["qpos"][sl])), L["v"]).reshape(len(L["qpos"][sl]), -1)
            ref = L["attn_in"][sl]
            if "gate" in L:
                o = o / (1 + np.exp(-L["gate"][sl]))
            rows_a.append((name, C, Li, float(np.linalg.norm(o - ref) / np.linalg.norm(ref))))
            # (B) per KV head
            S = key_smoothing_scales(L, P, "post")
            qs = slice(0, None, 4)
            q, qpos = L["q"][qs], L["qpos"][qs]
            for h0 in range(0, Hkv, 2):
                hk = slice(h0, h0 + 2)
                Lc = dict(L)
                Lc["k"], Lc["k_pre"], Lc["v"] = L["k"][:, hk], L["k_pre"][:, hk], L["v"][:, hk]
                Lc["q2_post"], Lc["q2_pre"] = L["q2_post"][h0 * G:(h0 + 2) * G], L["q2_pre"][h0 * G:(h0 + 2) * G]
                qc = q[:, h0 * G:(h0 + 2) * G]
                st = scores(qc, Lc["k"], qpos)
                valid = np.isfinite(st)
                err = {}
                for sm in (False, True):
                    gs = gain_shape_cache_v2(Lc, P, block=128, smooth=sm)
                    sc = scores(qc, gs["Kc"], qpos)
                    err[sm] = ((sc - st)[valid] ** 2).reshape(-1)
                    del gs
                for j in range(hk.stop - hk.start if hk.stop <= Hkv else Hkv - h0):
                    h = h0 + j
                    x = L["k"][:P, h].astype(np.float64)
                    k2 = ((x - x.mean(0)) ** 2).mean(0)
                    q2 = L["q2_post"][h * G:(h + 1) * G].mean(0).astype(np.float64)
                    gpred = q2.sum() * k2.sum() / np.sqrt(q2 * k2).sum() ** 2
                    stj = st[j][valid[j]]
                    nv = valid[j].sum()
                    # per-head slices of the flattened error arrays
                    off = valid[:j].sum()
                    e0, e1 = err[False][off:off + nv].mean(), err[True][off:off + nv].mean()
                    spread = stj.std()
                    rho = spearmanr(q2, k2)[0]
                    ka = np.abs(L["k"][:, h]).mean(0)
                    rows_b.append((name, C, Li, h, gpred, e0 / e1, np.sqrt(e0) / spread, np.sqrt(e1) / spread,
                                   k2.max() / np.median(k2), q2.max() / np.median(q2), rho, ka.max() / np.median(ka)))
                    if C == CORPORA[0]:
                        prof[(name, Li, h)] = (q2, k2, S[h], gpred)
            print(f"{name} {C} L{Li} {time.time()-t0:.0f}s", flush=True)

B = np.array(rows_b, dtype=object)
models = [n for n, _ in CAPS]
agg = []
for n in models:
    b = B[B[:, 0] == n]
    f = lambda i: b[:, i].astype(float)
    agg.append((n, len(b), float(np.median(f(4))), float(f(4).max()), float(np.median(f(5))), float(f(5).max()),
                float(np.median(f(6))), float(np.median(f(7))), float(np.median(f(8))), float(np.median(f(9))),
                float(np.median(f(10))), float(np.median(f(11))), float(f(11).max())))
gp, gm = B[:, 4].astype(float), B[:, 5].astype(float)
rho_pm = spearmanr(gp, gm)[0]

# profile plots: worst (largest Gpred) head per model
fig, axs = plt.subplots(len(models), 1, figsize=(10, 3.2 * len(models)))
axs = np.atleast_1d(axs)
for ax, n in zip(axs, models):
    key = max((k for k in prof if k[0] == n), key=lambda k: prof[k][3])
    q2, k2, s, g = prof[key]
    ax.semilogy(q2 / q2.mean(), label="E[q_c²] (norm.)", lw=1)
    ax.semilogy(k2 / k2.mean(), label="Var(k_c) (norm.)", lw=1)
    ax.set_title(f"{n} layer {key[1]} head {key[2]}: predicted smoothing gain {g:.1f}x")
    ax.set_xlabel("channel (post-RoPE)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "e19_channel_profiles.png"), dpi=90)
plt.figure(figsize=(5, 4))
for n in models:
    b = B[B[:, 0] == n]
    plt.loglog(b[:, 4].astype(float), b[:, 5].astype(float), "o", ms=4, label=n)
lim = [1, max(gp.max(), gm.max()) * 1.2]
plt.plot(lim, lim, "k--", lw=0.8)
plt.xlabel("predicted gain (prefill stats)"); plt.ylabel("measured logit-MSE gain (2-bit)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "e19_gain_pred_vs_meas.png"), dpi=90)

A = np.array(rows_a, dtype=object)
amax = {n: float(A[A[:, 0] == n][:, 3].astype(float).max()) for n in models}
txt = f"""# E19 Why key smoothing matters on some models (pipeline check + characterization)

**TL;DR:** pipeline check (offline exact attention vs the model's own o_proj input), max rel. error: {', '.join(f'{n} {amax[n]:.3f}' for n in models)}.
Median predicted smoothing gain (Σq²·Σk²/(Σ√(q²k²))², from prefill stats): {', '.join(f'{a[0]} {a[2]:.2f}x' for a in agg)};
measured 2-bit logit-MSE gain: {', '.join(f'{a[0]} {a[4]:.2f}x' for a in agg)} (Spearman predicted vs measured over heads {rho_pm:.2f}).

## Config
Reference per model: {REFS}. Corpora {CORPORA}; up to 4 attention layers per model; all KV heads; 2-bit (coarse) gain-shape keys with 128-token block means.
Logit error is measured on every 4th decode query over all causal positions. k² = per-channel variance of post-RoPE prefill keys,
q² = per-channel mean square of post-RoPE prefill queries (averaged over the GQA group). runtime {time.time()-t0:.0f}s.

## (A) Pipeline check
{md_table(["model", "corpus", "layer", "rel err vs model"], rows_a, "{:.4f}")}
## (B) Summary per model (medians over layer × head × corpus)
{md_table(["model", "n heads", "Gpred median", "Gpred max", "measured gain median", "measured max", "logit err/spread (no smooth)", "logit err/spread (smooth)", "k² max/median", "q² max/median", "Spearman(q², k²)", "outlier: max/median mean|k| (median)", "(max)"], agg)}
## (B) Per head
{md_table(["model", "corpus", "layer", "head", "Gpred", "measured gain", "err/spread no smooth", "err/spread smooth", "k² max/med", "q² max/med", "ρ(q²,k²)", "max/med mean|k|"], rows_b)}
## Plots
![profiles](plots/e19_channel_profiles.png) — per-channel query energy and key variance for the head with the largest predicted gain.
![pred vs meas](plots/e19_gain_pred_vs_meas.png)
"""
open(os.path.join(RES, "19_smoothing.md"), "w").write(txt)
print(txt[:3000])
