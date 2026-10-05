"""E1 Gaussianization: kurtosis of z = u*sqrt(d) and empirical shape MSE vs the Gaussian table,
for raw / mean-removed / mean-removed+RHT, keys pre- vs post-RoPE, and values."""
from common import *
from foveakv.quant import NestedCodebook
from foveakv.rotate import rht_matrix

t0 = time.time()
CB = NestedCodebook()
R = rht_matrix(256, 0)
VARIANTS = {"raw": (False, False), "mean": (True, False), "mean+RHT": (True, True)}
rows, flags = [], []
hist = {}
for C in ("prose", "code"):
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        for tname, X in (("K-post", L["k"]), ("K-pre", L["k_pre"]), ("V", L["v"])):
            for h in range(X.shape[1]):
                x = X[:, h].astype(np.float64)
                for vname, (center, rot) in VARIANTS.items():
                    y = x - x[:P].mean(0) if center else x
                    if rot:
                        y = y @ R.T
                    z = y / np.linalg.norm(y, axis=1, keepdims=True) * 16.0
                    kurt = float(np.mean(z ** 4) / np.mean(z ** 2) ** 2)
                    idx = CB.encode(z)
                    m4 = float(np.mean((CB.fine(idx) - z) ** 2))
                    m2 = float(np.mean((CB.coarse(idx) - z) ** 2))
                    g4, g2 = 10 * np.log10(m4 / CB.mse_fine), 10 * np.log10(m2 / CB.mse_coarse)
                    rows.append((C, Li, tname, h, vname, kurt, m2, g2, m4, g4))
                    if vname == "mean+RHT" and max(g2, g4) > 0.5:
                        flags.append((C, Li, tname, h, f"{g2:+.2f}", f"{g4:+.2f}", f"{kurt:.2f}"))
                    if C == "prose" and h == 0 and Li in (3, 15) and tname != "K-pre":
                        hist[(Li, tname, vname)] = z[:, ::8].ravel()

# aggregate: median over (corpus, layer, head) for each tensor/variant
agg = []
for tname in ("K-post", "K-pre", "V"):
    for vname in VARIANTS:
        sel = [r for r in rows if r[2] == tname and r[4] == vname]
        a = np.array([r[5:] for r in sel])
        agg.append((tname, vname, np.median(a[:, 0]), a[:, 0].max(), np.median(a[:, 2]), a[:, 2].max(),
                    np.median(a[:, 4]), a[:, 4].max()))

fig, axs = plt.subplots(2, 4, figsize=(16, 6))
xs = np.linspace(-5, 5, 200)
for ax, (key, z) in zip(axs.ravel(), sorted(hist.items())):
    ax.hist(np.clip(z, -6, 6), bins=120, density=True, alpha=0.7)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1)
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_title(f"L{key[0]} {key[1]} {key[2]}", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "e01_hist.png"), dpi=90)

kpost = [a for a in agg if a[0] == "K-post" and a[1] == "mean+RHT"][0]
v = [a for a in agg if a[0] == "V" and a[1] == "mean+RHT"][0]
raw = [a for a in agg if a[0] == "K-post" and a[1] == "raw"][0]
txt = f"""# E1 Gaussianization (Qwen3.5-0.8B, int8 reference cache, corpora prose+code, 6 attn layers × 2 KV heads)

**TL;DR:** mean removal + RHT brings key-shape kurtosis from median {raw[2]:.1f} (raw, max {raw[3]:.1f}) to {kpost[2]:.2f} (max {kpost[3]:.2f}; Gaussian = 3).
Empirical shape-MSE gap vs the Gaussian table after mean+RHT: keys post-RoPE median {kpost[4]:+.2f} dB (2b) / {kpost[6]:+.2f} dB (4b), worst {kpost[5]:+.2f} / {kpost[7]:+.2f} dB; values {v[4]:+.2f} / {v[6]:+.2f} dB.
{len(flags)} (corpus, layer, tensor, head) cases exceed the 0.5 dB gap after mean+RHT.

## Config
- z = u·sqrt(d), u = y/||y||; y = x − mu (mu from prefill tokens [0, T−512)) then optionally RHT (seed 0). d = 256.
- Table: 2b nested-coarse MSE {CB.mse_coarse:.4f}, 4b Lloyd-Max {CB.mse_fine:.4f}. gap dB = 10·log10(empirical/table) (positive = worse than Gaussian).
- Data: all 8192 tokens per corpus.

## Aggregate (median / max over corpus × layer × head)
""" + md_table(["tensor", "variant", "kurt med", "kurt max", "gap2b med dB", "gap2b max dB", "gap4b med dB", "gap4b max dB"], agg, "{:.2f}")
txt += "\n## Heads exceeding 0.5 dB after mean+RHT\n"
txt += md_table(["corpus", "layer", "tensor", "head", "gap2b dB", "gap4b dB", "kurt"], flags) if flags else "None.\n"
txt += "\n## Per layer (mean+RHT, keys post-RoPE / values, prose)\n" + md_table(
    ["corpus", "layer", "tensor", "head", "variant", "kurt", "mse2", "gap2 dB", "mse4", "gap4 dB"],
    [r for r in rows if r[0] == "prose" and r[4] == "mean+RHT" and r[2] != "K-pre"])
txt += f"""
## Plots
![hist](plots/e01_hist.png) — log-density of z (prose, head 0, layers 3/15) with N(0,1) dashed.

## Caveats
- The reference is the int8 cache (user decision); int8 noise (~1% rel) is far below 4-bit shape error, so it does not affect these gaps.
- Note that a negative gap means *better* than the Gaussian table (lighter tails / sub-Gaussian shape).

runtime {time.time()-t0:.0f}s
"""
write("01_gaussianization.md", txt)
