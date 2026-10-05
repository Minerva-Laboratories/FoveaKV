"""E5 Pages: page sizes {1,16,32,64} with a fixed token budget; captured mass, attention error and bytes read."""
from common import *
from foveakv.attention import scores, softmax, attend
from foveakv.select import select_pages
from foveakv.scheme import gain_shape_cache

t0 = time.time()
PAGES = [1, 16, 32, 64]
KS = [0.02, 0.05, 0.10]
STRIDE = 2
res = []
for C in CORPORA:
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P = prefill_len(L)
        sl = slice(0, None, STRIDE)
        q, qpos = L["q"][sl], L["qpos"][sl]
        p_true = softmax(scores(q, L["k"], qpos))
        o_ref = attend(p_true, L["v"])
        gs = gain_shape_cache(L, P)
        sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
        for page in PAGES:
            for kf in KS:
                hot = select_pages(sc, qpos, kf, page, 4, 64)
                h = hot[:, None]
                pm = softmax(np.where(h, sf, sc))
                o = attend(pm * ~h, gs["Vc"]) + attend(pm * h, gs["Vf"])
                err = np.linalg.norm(o - o_ref, axis=-1) / np.linalg.norm(o_ref, axis=-1)
                frac = hot.sum(-1).mean() / (qpos.mean() + 1)
                mass = (p_true * h).sum(-1).mean()
                res.append((C, Li, page, kf, float(frac), float(mass), float(err.mean()), gs["bits_coarse"] + 2 * float(frac)))
        print(C, Li, f"{time.time()-t0:.0f}s", flush=True)

A = np.array(res, dtype=object)
rows = []
for page in PAGES:
    for kf in KS:
        a = A[(A[:, 2] == page) & (A[:, 3] == kf)]
        rows.append((page, f"{kf*100:.0f}%", a[:, 4].astype(float).mean() * 100, a[:, 5].astype(float).mean(),
                     a[:, 5].astype(float).min(), a[:, 6].astype(float).mean(), a[:, 7].astype(float).mean()))
fig, ax = plt.subplots(figsize=(6, 4))
for page in PAGES:
    r = [x for x in rows if x[0] == page]
    ax.plot([x[6] for x in r], [x[5] for x in r], "o-", label=f"page {page}")
ax.set_xlabel("bits read / elem / step"); ax.set_ylabel("attn rel err"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "e05_pages.png"), dpi=90)
g = lambda pg, k: [x for x in rows if x[0] == pg and x[1] == k][0]
write("05_pages.md", f"""# E5 Page-granular selection (token budget fixed; page score = max over tokens and GQA group of estimated log-prob)

**TL;DR:** at k=5%, captured true mass: page 1 → {g(1,'5%')[3]:.3f}, 16 → {g(16,'5%')[3]:.3f}, 32 → {g(32,'5%')[3]:.3f}, 64 → {g(64,'5%')[3]:.3f};
attn err {g(1,'5%')[5]:.4f} / {g(16,'5%')[5]:.4f} / {g(32,'5%')[5]:.4f} / {g(64,'5%')[5]:.4f}. Pages trade selection precision for transfer granularity.

## Config
floor(k/page) pages selected by coarse(+RHT) estimate; sinks (4) and recent window (64) added at token granularity, so the
actual hot fraction is reported. Reference = exact attention on int8 cache. 256 decode queries/layer, 5 corpora, 6 layers.

""" + md_table(["page", "k", "actual hot %", "true mass mean", "true mass worst (corpus,layer)", "attn err", "read bits/elem"], rows) + f"""
![pages](plots/e05_pages.png)

## Caveats
- Bytes are counted at element level; real PCIe efficiency for page 1 (256-dim × 2 bit = 64 B per token/head/plane) would be poor without gathering; pages ≥ 16 give ≥ 1 KB transfers.
runtime {time.time()-t0:.0f}s
""")
