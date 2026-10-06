"""E2 attention concentration, E3 selection quality (coarse vs sign, ±RHT, bound check), E4 GQA union.
Ground truth = exact attention on the int8 reference cache; walk-forward over the last 512 positions."""
from common import *
from foveakv.attention import scores, softmax, logprob
from foveakv.select import topk_rows, k_for, sort_desc
from foveakv.scheme import gain_shape_cache

t0 = time.time()
KS = [0.01, 0.02, 0.05, 0.10, 0.20]
TAUS = [0.8, 0.9, 0.95]
e2, e3, e4, adapt = [], [], [], []
bound_viol = bound_total = 0
for C in CORPORA:
    for Li, L in ((Li, Lc) for Li in LAYERS for Lc in head_chunks(kvdata.load_layer(C, Li))):   # chunks of <= 2 KV heads
        P, q, qpos = prefill_len(L), L["q"], L["qpos"]
        s_true = scores(q, L["k"], qpos)
        p = softmax(s_true)                                   # (Hkv,G,E,T)
        Hkv, G, E, T = p.shape
        psort = -np.sort(-p, -1)
        csum = np.cumsum(psort, -1)
        # ---- E2: m_opt(k) per head
        for kf in KS:
            k = k_for(qpos, kf)
            mopt = np.take_along_axis(csum, np.broadcast_to((k - 1)[:, None], (Hkv, G, E, 1)), -1)[..., 0]
            for kh in range(Hkv):
                for g in range(G):
                    e2.append((C, Li, (L["h0"] + kh) * G + g, kf, float(mopt[kh, g].mean())))
        k_opt = {kf: k_for(qpos, kf) for kf in KS}
        mopt_k = {kf: np.take_along_axis(csum, np.broadcast_to((k_opt[kf] - 1)[:, None], (Hkv, G, E, 1)), -1)[..., 0] for kf in KS}
        srt_true = sort_desc(s_true)
        # ---- E3: selection estimators
        est = {}
        for rht in (True, False):
            gs = gain_shape_cache(L, P, rope="post", rht=rht)
            est[("coarse", rht)] = scores(q, gs["Kc"], qpos)
            est[("sign", rht)] = scores(q, gs["Ks"], qpos)
            del gs
        for (name, rht), sh in est.items():
            srt_h = sort_desc(sh)
            eta = np.nanmax(np.where(np.isfinite(s_true), np.abs(sh - s_true), np.nan), -1)   # (Hkv,G,E) max logit err
            for kf in KS:
                hot = topk_rows(sh, k_opt[kf], srt_h)
                true_top = topk_rows(s_true, k_opt[kf], srt_true)
                mH = (p * hot).sum(-1)
                ratio = mH / mopt_k[kf]
                ovl = (hot & true_top).sum(-1) / k_opt[kf]
                bound = np.exp(-2 * eta) * mopt_k[kf]
                bound_viol += int((mH < bound - 1e-9).sum()); bound_total += mH.size
                e3.append((C, Li, name, rht, kf, float(ovl.mean()), float(ratio.mean()), float(np.percentile(ratio, 5)),
                           float(np.median(eta)), float(np.median(np.exp(-2 * eta)))))
        # ---- E4: GQA union of per-head top-k (true attention and the coarse+RHT estimate)
        for src, sh in (("true", s_true), ("coarse", est[("coarse", True)]), ("sign", est[("sign", True)])):
            srt_s = srt_true if src == "true" else sort_desc(sh)
            lpm = logprob(sh).max(1)
            srt_b = sort_desc(lpm)
            for kf in KS:
                hot_h = topk_rows(sh, k_opt[kf], srt_s)                # (Hkv,G,E,T)
                union = hot_h.any(1)
                usize = union.sum(-1) / k_opt[kf]                      # (Hkv,E) in units of k
                # budgeted alternative: union capped at k via max over group of estimated logprob
                bud = topk_rows(lpm, k_opt[kf], srt_b)
                m_bud = (p * bud[:, None]).sum(-1) / mopt_k[kf]
                m_uni = (p * union[:, None]).sum(-1) / mopt_k[kf]
                e4.append((C, Li, src, kf, float(usize.mean()), float(usize.max()), float(m_uni.mean()), float(m_bud.mean()),
                           float(np.percentile(m_bud, 5))))
        # ---- adaptive k rule: smallest k with estimated (coarse+RHT) mass >= tau
        ph = softmax(est[("coarse", True)])
        order = np.argsort(-ph, -1)
        pc = np.cumsum(np.take_along_axis(ph, order, -1), -1)
        for tau in TAUS:
            kk = (pc < tau).sum(-1) + 1                                 # per head/query
            sel = np.zeros_like(ph, bool)
            np.put_along_axis(sel, order, np.arange(T)[None, None, None, :] < kk[..., None], -1)
            adapt.append((C, Li, tau, float((kk / (qpos + 1)).mean()), float(np.percentile(kk / (qpos + 1), 90)),
                          float((p * sel).sum(-1).mean()), float(np.percentile((p * sel).sum(-1), 5))))
        del est, s_true, p, psort, csum, srt_true
    print(C, f"{time.time()-t0:.0f}s", flush=True)

e2a, e3a, e4a, ada = map(np.array, (e2, e3, e4, adapt), [object] * 4)
fl = lambda a, i: a[:, i].astype(float)
np.savez(os.path.join(RES, "e02_04_raw.npz"), e2=e2a, e3=e3a, e4=e4a, adapt=ada)

# ---------------- E2 report
rows2 = []
for Li in LAYERS:
    for kf in KS:
        m = fl(e2a[(e2a[:, 1] == Li) & (e2a[:, 3] == kf)], 4)
        rows2.append((Li, f"{kf*100:.0f}%", m.mean(), m.min(), m.max()))
m10 = e2a[e2a[:, 3] == 0.10]
flat = [(r[0], r[1], r[2], f"{r[4]:.3f}") for r in m10 if r[4] < 0.8]
adarows = []
for tau in TAUS:
    a = ada[ada[:, 2] == tau]
    adarows.append((tau, fl(a, 3).mean() * 100, fl(a, 4).max() * 100, fl(a, 5).mean(), fl(a, 6).min()))
ov10 = fl(m10, 4)
write("02_concentration.md", f"""# E2 Attention concentration (Qwen3.5-0.8B, 6 full-attn layers × 8 query heads, 5 corpora, T=8192, 512 decode queries)

**TL;DR:** top-10% tokens hold on average {ov10.mean():.3f} of the attention mass (min head-avg {ov10.min():.3f}); top-1% holds {fl(e2a[e2a[:,3]==0.01],4).mean():.3f}.
{len(flat)} (corpus, layer, head) cases are "flat" (m_opt(10%) < 0.8).
Adaptive-k with the coarse estimate: τ=0.9 needs on average {adarows[1][1]:.1f}% of tokens and captures {adarows[1][3]:.3f} true mass.

## Config
m_opt(k) = mass of the true top-k tokens (k = frac·(p+1)) per query head, averaged over decode queries. Reference = int8 cache.

## m_opt(k) by layer (mean/min/max over corpus × head)
""" + md_table(["layer", "k", "mean", "min", "max"], rows2) + """
## Flat heads (m_opt(10%) < 0.8)
""" + (md_table(["corpus", "layer", "q-head", "m_opt(10%)"], flat) if flat else "None.\n") + """
## Adaptive-k rule: smallest k with estimated (coarse, RHT) mass ≥ τ
""" + md_table(["τ", "mean k %", "max (p90 per layer) k %", "true mass mean", "true mass worst p5"], adarows) + f"""
## Caveats
- Contexts are 8K; concentration at 32K–256K is not measured (tails get heavier with length in general).
runtime {time.time()-t0:.0f}s
""")

# ---------------- E3 report
rows3 = []
for name in ("coarse", "sign"):
    for rht in (True, False):
        for kf in KS:
            a = e3a[(e3a[:, 2] == name) & (e3a[:, 3] == rht) & (e3a[:, 4] == kf)]
            rows3.append((name, "RHT" if rht else "no-RHT", f"{kf*100:.0f}%", fl(a, 5).mean(), fl(a, 6).mean(),
                          fl(a, 6).min(), fl(a, 7).min(), np.median(fl(a, 8)), np.median(fl(a, 9))))
perlayer3 = []
for Li in LAYERS:
    for name in ("coarse", "sign"):
        a = e3a[(e3a[:, 1] == Li) & (e3a[:, 2] == name) & (e3a[:, 3] == True) & (e3a[:, 4] == 0.05)]
        perlayer3.append((Li, name, fl(a, 5).mean(), fl(a, 6).mean(), fl(a, 7).min()))
get = lambda n, r, k: [x for x in rows3 if x[0] == n and x[1] == r and x[2] == k][0]
write("03_selection.md", f"""# E3 Selection quality (per query head, no always-hot tokens)

**TL;DR:** at k=5%, coarse(+RHT) selection captures m_H/m_opt = {get('coarse','RHT','5%')[4]:.3f} (overlap {get('coarse','RHT','5%')[3]:.3f}); sign-plane(+RHT) {get('sign','RHT','5%')[4]:.3f} (overlap {get('sign','RHT','5%')[3]:.3f}).
Without RHT: coarse {get('coarse','no-RHT','5%')[4]:.3f}, sign {get('sign','no-RHT','5%')[4]:.3f}.
Bound m_H ≥ e^(−2η) m_opt (η = max |logit error|): {bound_viol} violations / {bound_total} (query, head, k) cases; but the bound is loose (median e^(−2η) shown).

## Config
Estimated logits from the 2-bit coarse plane or the 1-bit sign plane (× exact 8-bit log gain, level sqrt(2/π)), post-RoPE keys,
mean removed using prefill stats. Top-k per query head (k = frac·(p+1)); captured mass m_H vs oracle m_opt. Reference = int8 cache.

## Aggregate (mean over {len(CORPORA)} corpora × {len(LAYERS)} layers × all query heads × 512 queries)
""" + md_table(["estimator", "rotation", "k", "overlap", "m_H/m_opt mean", "worst layer-mean", "worst p5", "median η", "median e^-2η"], rows3) + """
## Per layer at k=5% (+RHT)
""" + md_table(["layer", "estimator", "overlap", "m_H/m_opt", "worst p5"], perlayer3) + f"""
runtime {time.time()-t0:.0f}s
""")

# ---------------- E4 report
rows4 = []
for src in ("true", "coarse", "sign"):
    for kf in KS:
        a = e4a[(e4a[:, 2] == src) & (e4a[:, 3] == kf)]
        rows4.append((src, f"{kf*100:.0f}%", fl(a, 4).mean(), fl(a, 5).max(), fl(a, 6).mean(), fl(a, 7).mean(), fl(a, 8).min()))
perlayer4 = []
for Li in LAYERS:
    a = e4a[(e4a[:, 1] == Li) & (e4a[:, 2] == "true") & (e4a[:, 3] == 0.05)]
    b = e4a[(e4a[:, 1] == Li) & (e4a[:, 2] == "coarse") & (e4a[:, 3] == 0.05)]
    perlayer4.append((Li, fl(a, 4).mean(), fl(a, 5).max(), fl(b, 4).mean(), fl(b, 7).mean()))
u5 = [r for r in rows4 if r[0] == "true" and r[1] == "5%"][0]
u5c = [r for r in rows4 if r[0] == "coarse" and r[1] == "5%"][0]
_m = kvdata.meta(CORPORA[0]); GQ = _m.get("Hq", 8) // _m.get("Hkv", 2)
write("04_gqa_union.md", f"""# E4 GQA union (group size G = {GQ} query heads per KV head) — key go/no-go metric

**TL;DR:** at k=5%, the union of the {GQ} per-head true top-k sets is {u5[2]:.2f}× k on average (max {u5[3]:.2f}×); with the coarse estimate {u5c[2]:.2f}× k.
Criterion "union ≤ ~3× per-head k": {'PASS' if u5c[2] <= 3 else 'FAIL'} (G={GQ} bounds the union at {GQ}×).
A budgeted alternative (rank by max over the group of estimated log-prob, keep exactly k) captures {u5c[5]:.3f} of each head's m_opt on average.

## Config
union size in units of k; m_H/m_opt per query head when the shared hot set is the union (≤ G·k tokens) or budgeted (= k tokens).

""" + md_table(["scores", "k", "union/k mean", "union/k max", "m_H/m_opt union", "m_H/m_opt budget", "budget worst p5"], rows4) + """
## Per layer at k=5%
""" + md_table(["layer", "true union/k", "true max", "coarse union/k", "coarse budget m_H/m_opt"], perlayer4) + f"""
## Caveats
- With G={GQ} the union can never exceed {GQ}×. Larger groups (G=8–16) are not tested here.
runtime {time.time()-t0:.0f}s
""")
