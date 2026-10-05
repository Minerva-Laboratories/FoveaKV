"""E9 End-to-end: NumPy forward pass with the quantized cache in the loop (error propagates across layers).

Method (equivalent to teacher-forced token-by-token decode):
  1. Prefill tokens [0, P) once with exact attention (identical for all schemes); cache DeltaNet/conv states and the
     int8 reference K/V of the 6 attention layers to data/e09_prefill_<corpus>.npz.
  2. For each scheme, run the decode window [P, N) as one chunk from a copy of the prefill state. At every
     full-attention layer a hook replaces attention for the chunk's queries by walk-forward attention over
     int8(prefill + chunk K/V) quantized with the scheme (calibration = prefill only). Layers run in order, so
     chunk K/V at layer L already reflect quantization errors from earlier layers — exactly as in decoding.
Metrics: perplexity over the window (prose, code); needle: exact-match greedy retrieval of the 5 answer tokens
(teacher-forced answer appended after the question) and their mean NLL, needle at 10/50/90% depth."""
from common import *
import json, copy
from foveakv.model_hybrid import Qwen35, rope_cos_sin, apply_rope
from foveakv.quant import int8_absmax
from foveakv.scheme import scheme_attention
from foveakv.text import tokenizer

t0 = time.time()
WINDOW = int(os.environ.get("E9_WINDOW", 512))
SCHEMES = {
    "fp32 (no cache quant)": None,
    "int8 (reference)": dict(kind="ref"),
    "GS-4b": dict(kind="gs", bits=4),
    "GS-2b": dict(kind="gs", bits=2),
    "DYN coarse k=2%": dict(kind="dyn", sel="coarse", k=0.02),
    "DYN coarse k=5%": dict(kind="dyn", sel="coarse", k=0.05),
    "DYN coarse k=10%": dict(kind="dyn", sel="coarse", k=0.10),
    "DYN sign k=5%": dict(kind="dyn", sel="sign", k=0.05),
    "KIVI-4b g128": dict(kind="kivi", bits=4, group=128),
    "KIVI-2b g128": dict(kind="kivi", bits=2, group=128),
}
m = Qwen35()
tok = tokenizer(m.dir)
meta = json.load(open(os.path.join(ROOT, "data/corpus/meta.json")))
cos_all, sin_all = rope_cos_sin(np.arange(9000), m.rot_dim, m.theta)


def int8_heads(x):
    return int8_absmax(x.astype(np.float32))


def prefill(C, ids, P):
    path = os.path.join(ROOT, "data", f"e09_prefill_{C}_{P}_{len(ids)}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["state"].item(), z["kv"].item()
    kv = {i: ([], []) for i in m.attn_layers}

    def hook(i, q, k, v, k_pre, pos, **kw):
        kv[i][0].append(int8_heads(k_pre)); kv[i][1].append(int8_heads(v))
        return None
    st = m.new_state()
    for s0 in range(0, P, 1024):                 # chunked prefill (bounded activation memory; == full forward)
        m.forward(ids[s0:min(P, s0 + 1024)], state=st, hook=hook)
    kv = {i: (np.concatenate(a).astype(np.float32), np.concatenate(b).astype(np.float32)) for i, (a, b) in kv.items()}
    for s in st["layers"]:                      # attention layers keep their own cache here (kv); drop fp copies
        s.pop("k", None); s.pop("v", None)
    np.savez(path, state=np.array(st, dtype=object), kv=np.array(kv, dtype=object))
    return st, kv


def run_window(st0, kv0, ids, P, cfg):
    st = copy.deepcopy(st0)

    def hook(i, q, k, v, k_pre, pos, **kw):
        kp = np.concatenate([kv0[i][0], int8_heads(k_pre)])
        vv = np.concatenate([kv0[i][1], int8_heads(v)])
        T = kp.shape[0]
        L = dict(k_pre=kp, k=apply_rope(kp, cos_all[:T], sin_all[:T], m.rot_dim), v=vv, T=T)
        out, _ = scheme_attention(cfg, L, P, q, pos)
        return out.astype(np.float32)

    return m.forward(ids[P:], state=st, hook=hook)


rows = []
for C in ("prose", "code", "needle10", "needle50", "needle90"):
    ids = np.load(os.path.join(ROOT, f"data/corpus/{C}.npy")).astype(np.int64)
    if os.environ.get("E9_TRUNC"):                # smoke-test mode: last N tokens only
        ids = ids[-int(os.environ["E9_TRUNC"]):]
    is_needle = C.startswith("needle")
    if is_needle:
        ans = tok.encode(" " + meta[C]["answer"]).ids
        ids = np.concatenate([ids, ans])
        win = 64 + len(ans)                       # short window: question + answer
    else:
        win = WINDOW
    N = len(ids)
    P = N - win
    st0, kv0 = prefill(C, ids, P)
    print(C, "prefill done", f"{time.time()-t0:.0f}s", flush=True)
    for name, cfg in SCHEMES.items():
        if cfg is None:
            # sanity row: window K/V stay fp32 (model's own incremental cache); prefill K/V = stored int8 reference
            st = copy.deepcopy(st0)
            for i in m.attn_layers:
                st["layers"][i]["k"] = apply_rope(kv0[i][0], cos_all[:P], sin_all[:P], m.rot_dim)
                st["layers"][i]["v"] = kv0[i][1]
            h = m.forward(ids[P:], state=st)
        else:
            h = run_window(st0, kv0, ids, P, cfg)
        nll, arg = m.nll_and_argmax(h[:-1], ids[P + 1:])
        if is_needle:
            na = len(ans)
            em = bool(np.all(arg[-na:] == ans))
            rows.append((C, name, float(nll[-na:].mean()), em, tok.decode(arg[-na:].tolist())))
        else:
            rows.append((C, name, float(np.exp(nll.mean())), None, ""))
        print(C, name, rows[-1][2:4], f"{time.time()-t0:.0f}s", flush=True)

A = np.array(rows, dtype=object)
np.save(os.path.join(RES, "e09_raw.npy"), A)
tab = []
for name in SCHEMES:
    r = {x[0]: x for x in rows if x[1] == name}
    ppl = [r[c][2] for c in ("prose", "code")]
    tab.append((name, r["prose"][2], r["code"][2],
                " ".join("✓" if r[c][3] else "✗" for c in ("needle10", "needle50", "needle90")),
                np.mean([r[c][2] for c in ("needle10", "needle50", "needle90")])))
ref = [t for t in tab if t[0] == "int8 (reference)"][0]
u4 = [t for t in tab if t[0] == "GS-4b"][0]
d5 = [t for t in tab if t[0] == "DYN coarse k=5%"][0]
d10 = [t for t in tab if t[0] == "DYN coarse k=10%"][0]
rel = lambda a, b: (a / b - 1) * 100
tab2 = [(t[0], t[1], rel(t[1], u4[1]), t[2], rel(t[2], u4[2]), t[3], t[4]) for t in tab]
crit = max(rel(d10[1], u4[1]), rel(d10[2], u4[2]))
write("09_end_to_end.md" if not os.environ.get("E9_TRUNC") else "09_smoke.md", f"""# E9 End-to-end (quantized cache in the loop, Qwen3.5-0.8B NumPy forward)

**TL;DR:** window PPL (prose / code): int8 ref {ref[1]:.3f} / {ref[2]:.3f}; GS-4b {u4[1]:.3f} / {u4[2]:.3f}; DYN k=5% {d5[1]:.3f} / {d5[2]:.3f} ({rel(d5[1],u4[1]):+.2f}% / {rel(d5[2],u4[2]):+.2f}% vs GS-4b); DYN k=10% {rel(d10[1],u4[1]):+.2f}% / {rel(d10[2],u4[2]):+.2f}%.
Go/no-go "+1% PPL vs uniform 4-bit at ≤10% hot": {'PASS' if crit <= 1 else 'FAIL'} (worst corpus {crit:+.2f}% at k=10%). Needle exact-match (10/50/90% depth): GS-4b {u4[3]}, DYN k=5% {d5[3]}.
Only {WINDOW} decode positions per corpus at 8K context — small-sample PPL; differences < ~0.5% are within noise.

## Config
- Prefill P = N − window tokens with exact attention (shared by all schemes); window = {WINDOW} (prose, code) or question+answer (needle).
- Reference cache = int8 per-token/head absmax of the NumPy model's K/V. Schemes quantize from it; calibration (mu, gain range) on prefill only.
- "fp32 (no cache quant)" row: window K/V are fp32, prefill K/V are the int8 reference (fp32 prefill KV not stored) — near-reference sanity row.
- DYN: coarse/sign selection, budget mode (GQA-shared), S=4, W=64. KIVI: per-channel K / per-token V, group 128.

| scheme | prose PPL | Δ vs GS-4b % | code PPL | Δ vs GS-4b % | needle 10/50/90 | answer NLL |
|---|---|---|---|---|---|---|
""" + "".join(f"| {t[0]} | {t[1]:.4f} | {t[2]:+.2f} | {t[3]:.4f} | {t[4]:+.2f} | {t[5]} | {t[6]:.3f} |\n" for t in tab2) + f"""
## Needle greedy outputs
""" + "".join(f"- {r[0]} / {r[1]}: `{r[4]!r}` (NLL {r[2]:.3f})\n" for r in rows if r[0].startswith("needle")) + f"""
## Caveats
- Single model (0.8B), 8K context, 2 text corpora × {WINDOW} positions: PPL deltas below ~0.5% are not significant.
- Needle at 8K is easy for this model; harder retrieval (longer context, multiple needles) is untested.
runtime {time.time()-t0:.0f}s
""")
