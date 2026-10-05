"""E16b Structure of the coarse-logit error (why selection from 2-bit keys loses a little to the oracle).
For each (head, query) row over non-window tokens: corr(err, estimate), corr(err, true logit), corr(estimate, true);
and the mean / RMS of the error normalized by its predicted std, binned by the TRUE logit percentile.
Berkson-type error (independent of the estimate, anti-correlated with the truth) means ranking by the estimate is already
near Bayes-optimal given the codes; no monotone re-scoring can help."""
from common import *
from foveakv.attention import scores
from foveakv.scheme import gain_shape_cache_v2, block_means
from foveakv.quant import NestedCodebook
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()
CB = NestedCodebook()
Dc = CB.mse_coarse
BINS = ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0))
corr_rows, bin_rows = [], []


def rowcorr(a, b, m):
    a = np.where(m, a, np.nan); b = np.where(m, b, np.nan)
    a = a - np.nanmean(a, -1, keepdims=True); b = b - np.nanmean(b, -1, keepdims=True)
    return float(np.nanmean(np.nansum(a * b, -1) / np.sqrt(np.nansum(a * a, -1) * np.nansum(b * b, -1))))


for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        q, qpos = L["q"][::4], L["qpos"][::4]
        E, T = len(qpos), L["T"]
        Hkv, d = L["k"].shape[1], L["k"].shape[2]
        sr = scores(q, L["k"], qpos)
        gs = gain_shape_cache_v2(L, P, block=128)
        Mk = block_means(L["k"], 128)
        s_mmse = scores(q, gs["Kc"], qpos)
        sc = scores(q, Mk + (gs["Kc"] - Mk) / np.sqrt(1 - Dc), qpos)
        t = np.arange(T)[None]
        m = np.isfinite(sr) & ~((t > qpos[:, None] - 128) & (t <= qpos[:, None]))[None, None]
        e = s_mmse - sr
        corr_rows.append((C, Li, rowcorr(e, s_mmse, m), rowcorr(e, sr, m), rowcorr(s_mmse, sr, m)))
        gk = np.stack([gs["diag"][h]["k"]["g"] for h in range(Hkv)], 0)
        qn = np.sqrt((q.reshape(E, Hkv, -1, d) ** 2).sum(-1)).transpose(1, 2, 0)
        sig = qn[..., None] * gk[:, None, None, :] * np.sqrt(Dc) / d
        z = np.where(m, (sc - sr) / sig, np.nan)
        srm = np.where(m, sr, -1e9)
        rank = np.argsort(np.argsort(-srm, -1), -1)
        pct = rank / m.sum(-1, keepdims=True)
        row = [C, Li]
        for lo, hi in BINS:
            sel = m & (pct >= lo) & (pct < hi)
            row += [float(np.nanmean(z[sel])), float(np.sqrt(np.nanmean(z[sel] ** 2)))]
        bin_rows.append(tuple(row))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

Cr = np.array(corr_rows, dtype=object); Br = np.array(bin_rows, dtype=object)
np.savez(os.path.join(RES, "e16b_raw.npz"), corr=Cr, bins=Br)
c = Cr[:, 2:].astype(float)
b = Br[:, 2:].astype(float)
write("16b_error_structure.md", f"""# E16b Structure of the coarse-logit error

**TL;DR:** the 2-bit logit error is mostly Berkson-type: corr(error, estimate) = {c[:,0].mean():+.2f} (range {c[:,0].min():+.2f}..{c[:,0].max():+.2f}),
corr(error, true logit) = {c[:,1].mean():+.2f} (range {c[:,1].min():+.2f}..{c[:,1].max():+.2f}); conditioned on the true logit the error has a negative mean that grows
with rank (top 1%: {b[:,0].mean():+.2f}σ, top 5–20%: {b[:,4].mean():+.2f}σ, tail: {b[:,6].mean():+.2f}σ). Ranking by the estimate is near Bayes-optimal given the codes,
so re-scoring cannot close the selection gap; more key information can (two-stage selection, E16).

## Correlations (MMSE coarse logits, mean over (head, query) rows; non-window tokens)
""" + md_table(["corpus", "layer", "corr(err, est)", "corr(err, true)", "corr(est, true)"], corr_rows) + """
## Normalized error by true-logit percentile (shrinkage-corrected logits; mean / RMS in units of predicted σ)
""" + md_table(["corpus", "layer"] + sum([[f"top {lo*100:g}-{hi*100:g}% mean", "rms"] for lo, hi in BINS], []), bin_rows, "{:.2f}") + f"""
runtime {time.time()-t0:.0f}s
""")
