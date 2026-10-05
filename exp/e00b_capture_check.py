"""Validate the offline (captured-cache, walk-forward) methodology:
 (1) offline exact attention from captured q/k/v reproduces the model's o_proj input;
 (2) int8 reference vs fp16(=bf16) cache: attention-output error, compared to a 4-bit baseline;
 (3) captured keys agree with the independent NumPy forward pass (model_hybrid)."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foveakv.kvdata import load_layer, meta, DATA
from foveakv.model_hybrid import decode_attention, sigmoid, Qwen35

ROOT = os.path.dirname(DATA)
t0 = time.time()
C = "prose"
M = meta(C)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))


def walk_forward_attn(q, k, v):
    return decode_attention(q, k, v, k.shape[-1] ** -0.5)      # query j sees keys 0..T-E+j


def q4_token(x):  # simple per-token/head absmax 4-bit (scale-of-reference only)
    s = np.abs(x).max(-1, keepdims=True) / 7
    return np.clip(np.round(x / s), -7, 7) * s


rows = []
for L in M["attn_layers"]:
    r16, r8 = load_layer(C, L, "fp16"), load_layer(C, L, "int8")
    o16 = walk_forward_attn(r16["q"], r16["k"], r16["v"])
    o8 = walk_forward_attn(r8["q"], r8["k"], r8["v"])
    o4 = walk_forward_attn(r16["q"], q4_token(r16["k"]), q4_token(r16["v"]))
    E = o16.shape[0]
    model_in = r16["attn_in"]
    recon16 = o16.reshape(E, -1) * sigmoid(r16["gate"])
    recon8 = o8.reshape(E, -1) * sigmoid(r8["gate"])
    rows.append((L, rel(recon16, model_in), rel(recon8, model_in), rel(o8, o16), rel(o4, o16),
                 rel(r8["k_pre"], r16["k_pre"]), rel(r8["v"], r16["v"])))

# (3) independent NumPy forward on the first 512 tokens: compare layer-3 and layer-23 pre-RoPE keys
m = Qwen35()
ids = np.load(os.path.join(DATA, "corpus", C + ".npy"))[:512]
got = {}
def hook(i, q, k, v, k_pre, pos, **kw):
    got[i] = k_pre
    return None
m.forward(ids, hook=hook)
kcmp = {L: rel(load_layer(C, L, "fp16")["k_pre"][:512], got[L]) for L in (3, 23)}

ok1 = max(r[1] for r in rows) < 0.05
ok2 = max(r[3] for r in rows) < 0.1 * min(r[4] for r in rows)
with open(os.path.join(ROOT, "results/00b_capture_check.md"), "w") as f:
    f.write("# 00b Offline-cache methodology check (Qwen3.5-0.8B, corpus=prose, T=8192, E=512)\n\n")
    f.write(f"**TL;DR:** (1) offline walk-forward attention reproduces the model's o_proj input: "
            f"max rel err {max(r[1] for r in rows):.4f} (fp16 cache) — {'OK' if ok1 else 'FAIL'}. "
            f"(2) int8 reference: attention-output rel err vs fp16 cache ≤ {max(r[3] for r in rows):.4f}, "
            f"vs ≥ {min(r[4] for r in rows):.4f} for naive 4-bit — {'int8 is effectively lossless' if ok2 else 'int8 NOT negligible'}. "
            f"(3) captured keys vs NumPy forward: rel diff {', '.join(f'L{k}: {v:.4f}' for k, v in kcmp.items())} "
            f"(capture is AMP bf16; NumPy is fp32).\n\n")
    f.write("## Config\n- capture: HF transformers on GPU, fp32 weights + torch.autocast(bf16); keys pre-RoPE (post k_norm)\n"
            "- int8 ref: symmetric per-token/per-head absmax, fp16 scale → 8 + 16/256 = 8.06 bits/elem\n"
            "- walk-forward: decode query at position p attends to cache[0..p]; RoPE applied in NumPy\n"
            "- 'naive 4-bit' = per-token/head absmax int4 (scale reference only, not a scheme under test)\n\n")
    f.write("| layer | offline(fp16) vs model | offline(int8) vs model | int8 vs fp16 attn out | naive-4b vs fp16 attn out | K int8 rel | V int8 rel |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | " + " | ".join(f"{x:.4f}" for x in r[1:]) + " |\n")
    f.write(f"\n## Caveats\n- The offline method has no error propagation across layers: K/V of layer L+4 are computed from "
            "unquantized upstream activations. Per-layer metrics (E1–E8, E10) are exact under this method; "
            "end-to-end perplexity/needle (E9) needs a forward pass with the quantized cache in the loop.\n"
            f"- The model-vs-offline residual reflects bf16 AMP arithmetic inside the model (SDPA in bf16).\n\nruntime {time.time()-t0:.0f}s\n")
print(open(os.path.join(ROOT, "results/00b_capture_check.md")).read())
