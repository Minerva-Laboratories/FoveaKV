"""E6 temporal reuse (step-to-step hit rate, LRU residual cache of size {1,2,4}×k) and
E7 hybrid prefetch for attention layer L+4: predict its hot set from (a) its own previous step, (b) layer L's
current hot set, (c) a speculative query W_Q(L+4) applied to the hidden state after layer L (captured at prefill-time
forward; exact same hidden state the decode would have). Only the *dynamic* part of the hot set counts
(sinks / recent window are resident on GPU). Walk-forward over 512 consecutive decode steps."""
from common import *
from foveakv.attention import scores
from foveakv.select import select, always_hot
from foveakv.scheme import gain_shape_cache

t0 = time.time()
KS = [0.02, 0.05, 0.10]
LRU_X = [1, 2, 4]
S, W = 4, 64
e6, e7 = [], []


def lru_hits(hots, cap):
    """hots: (E, T) bool needed (dynamic) sets for consecutive steps. LRU of `cap` tokens. Returns per-step hit rate."""
    E, T = hots.shape
    last = np.full(T, -1)
    rates = []
    for t in range(E):
        need = np.flatnonzero(hots[t])
        if t > 0:
            used = np.flatnonzero(last >= 0)
            if len(used) > cap:
                keep = used[np.argpartition(-last[used], cap - 1)[:cap]]
            else:
                keep = used
            incache = np.zeros(T, bool); incache[keep] = True
            rates.append(incache[need].mean() if len(need) else 1.0)
        last[need] = t
    return np.array(rates)


for C in CORPORA:
    hot_by_layer = {}
    for Li in LAYERS:
        L = kvdata.load_layer(C, Li)
        P, q, qpos = prefill_len(L), L["q"], L["qpos"]
        T = L["T"]
        gs = gain_shape_cache(L, P)
        sc = scores(q, gs["Kc"], qpos)
        ah = always_hot(qpos, T, S, W)
        hot_by_layer[Li] = {}
        for kf in KS:
            hot = select(sc, qpos, kf, S, W) & ~ah[None]                 # dynamic part (Hkv,E,T)
            hot_by_layer[Li][kf] = hot
            for h in range(hot.shape[0]):
                hs = hot[h]
                inter = (hs[1:] & hs[:-1]).sum(-1) / np.maximum(hs[1:].sum(-1), 1)
                row = [C, Li, h, kf, float(inter.mean())]
                k_dyn = int(hs.sum(-1).mean())
                for x in LRU_X:
                    row.append(float(lru_hits(hs, x * k_dyn).mean()))
                e6.append(tuple(row))
        if "q_spec" in L and Li - 4 in hot_by_layer:                      # E7 for this layer as target L+4
            src = Li - 4
            ss = scores(L["q_spec"], gs["Kc"], qpos)
            for kf in KS:
                act = hot_by_layer[Li][kf]
                pred_c = select(ss, qpos, kf, S, W) & ~ah[None]
                pred_c2 = select(ss, qpos, 2 * kf, S, W) & ~ah[None]     # over-fetch 2x
                pred_b = hot_by_layer[src][kf]
                for h in range(act.shape[0]):
                    a = act[h][1:]
                    n = np.maximum(a.sum(-1), 1)
                    hr = lambda pr: float(((pr & a).sum(-1) / n).mean())
                    prev = act[h][:-1]
                    e7.append((C, Li, h, kf, hr(prev), hr(pred_b[h][1:]), hr(pred_c[h][1:]), hr(pred_c2[h][1:]),
                               hr(prev | pred_c[h][1:]), float((prev | pred_c[h][1:]).sum(-1).mean() / n.mean())))
        del L, gs, sc
    print(C, f"{time.time()-t0:.0f}s", flush=True)

A6, A7 = np.array(e6, dtype=object), np.array(e7, dtype=object)
np.savez(os.path.join(RES, "e06_07_raw.npz"), e6=A6, e7=A7)
r6 = []
for kf in KS:
    a = A6[A6[:, 3] == kf]
    r6.append((f"{kf*100:.0f}%",) + tuple(float(a[:, j].astype(float).mean()) for j in range(4, 8)) + (float(a[:, 4].astype(float).min()),))
pl6 = []
for Li in LAYERS:
    a = A6[(A6[:, 1] == Li) & (A6[:, 3] == 0.05)]
    pl6.append((Li,) + tuple(float(a[:, j].astype(float).mean()) for j in range(4, 8)))
g6 = r6[1]
write("06_temporal_reuse.md", f"""# E6 Temporal reuse of the hot set (dynamic part only; sinks+window resident)

**TL;DR:** at k=5%, consecutive decode steps share {g6[1]:.3f} of the dynamic hot set (worst head {g6[5]:.3f});
an on-GPU LRU of residuals of size 1×/2×/4× k gives hit rates {g6[2]:.3f} / {g6[3]:.3f} / {g6[4]:.3f}.

## Config
Budget selection (coarse+RHT, S=4, W=64, k counts all hot tokens); 512 consecutive decode steps; per KV head; 5 corpora × 6 layers.
LRU capacity in units of the mean dynamic hot-set size.

""" + md_table(["k", "step-to-step overlap", "LRU 1×", "LRU 2×", "LRU 4×", "worst head overlap"], r6) + """
## Per layer (k=5%)
""" + md_table(["layer", "overlap", "LRU 1×", "LRU 2×", "LRU 4×"], pl6) + f"""
## Caveats
- Queries come from teacher-forced text (the captured corpus), the same as real decoding of that text.
runtime {time.time()-t0:.0f}s
""")
r7 = []
for kf in KS:
    a = A7[A7[:, 3] == kf]
    r7.append((f"{kf*100:.0f}%",) + tuple(float(a[:, j].astype(float).mean()) for j in range(4, 10)))
pl7 = []
for Li in sorted(set(A7[:, 1])):
    a = A7[(A7[:, 1] == Li) & (A7[:, 3] == 0.05)]
    pl7.append((f"{Li-4}→{Li}",) + tuple(float(a[:, j].astype(float).mean()) for j in range(4, 10)))
g7 = r7[1]
write("07_prefetch.md", f"""# E7 Hybrid prefetch for attention layer L+4 (dynamic hot set; hit = fraction of the actual set predicted)

**TL;DR:** at k=5%: (a) previous step same layer {g7[1]:.3f}, (b) layer L's current hot set {g7[2]:.3f}, (c) speculative query {g7[3]:.3f}
(over-fetch 2× → {g7[4]:.3f}); (a)∪(c) {g7[5]:.3f} at {g7[6]:.2f}× the transfer of the actual set. Target ≥ 0.85.

## Config
Speculative query = q_norm(W_Q(L+4) · RMSNorm_{{L+4}}(h_L)) with RoPE at the true position, scored against the coarse keys of L+4.
The 3 linear-attention layers in between are skipped (that is the latency window for the prefetch).

""" + md_table(["k", "(a) prev step", "(b) layer L hot set", "(c) spec query", "(c) 2× over-fetch", "(a)∪(c)", "(a)∪(c) size / actual"], r7) + """
## Per layer pair (k=5%)
""" + md_table(["L→L+4", "(a)", "(b)", "(c)", "(c)2×", "(a)∪(c)", "size"], pl7) + f"""
runtime {time.time()-t0:.0f}s
""")
