"""End-to-end perplexity of sequence-fitted low-rank cache projections (gpu/proj_cache.py; E29 in the model) against bf16
and FoveaKV, protocol of gpu/eval_e2e.py ppl (exact prefill of L-N tokens, N tokens teacher-forced through each
cache). The fixed ('static', MLA-like) bases come from one 8K-token calibration prefill of wikitext-2 TEST (the prose
windows are from train). Writes results/<tag>/proj_ppl.jsonl (resumable). FOVEAKV_PROJ_QUICK=1: one doc at 16K, N=64."""
import json, math, os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu.eval_e2e import Runner, prose_text, code_text, doc_windows, DEV
from gpu.proj_cache import Proj, static_bases

QUICK = bool(os.environ.get("FOVEAKV_PROJ_QUICK"))
LENS = [16384] if QUICK else [16384, 32768]
NDOC, N = (1, 64) if QUICK else (2, 256)
PROJ = {
    "k96-static": dict(kb="static", rk=96), "k96-kpca": dict(kb="kpca", rk=96), "k96-qw": dict(kb="qw", rk=96),
    "v128-static": dict(vb="static", rv=128), "v128-vpca": dict(vb="vpca", rv=128), "v128-opca": dict(vb="opca", rv=128),
    "k96v128-static": dict(kb="static", rk=96, vb="static", rv=128),
    "k96v128-qw-opca": dict(kb="qw", rk=96, vb="opca", rv=128),
    # equal bytes with FoveaKV gs2 (512 bits per key and per value): latent 4-bit, key RoPE dims 2-bit
    "eq-qw-opca-4b": dict(kb="qw", rk=96, vb="opca", rv=128, codes=(4, 2)),
    "eq-static-4b": dict(kb="static", rk=96, vb="static", rv=128, codes=(4, 2)),
}
SCHEMES = ["bf16", "gs2", "dyn5"] + list(PROJ)
out_p = os.path.join(ROOT, "results", os.environ.get("FOVEAKV_RESULTS_TAG", "qwen3.5-9b"), "proj_ppl" + ("_quick" if QUICK else "") + ".jsonl")
done = {(r["src"], r["L"], r["doc"], r["scheme"]) for r in map(json.loads, open(out_p))} if os.path.exists(out_p) else set()


@torch.no_grad()
def decode_proj(R, cfg, snap, targets, static):
    caches = {}
    for i, (k, v) in snap["kv"].items():
        c = Proj(R.Hkv, snap["T"] + 512, **cfg)
        c.prefill(k, v)
        c.build(snap["qm"][i], snap["qbuf"][i], static.get(i), layer_seed=i)
        caches[i] = c
    lin = snap["lin"].clone()
    nll = [float(-torch.log_softmax(snap["last_logits"], -1)[targets[0]])]
    pos = snap["T"]
    for t in range(len(targets) - 1):
        h = R.forward(targets[t:t + 1], pos, caches, lin, prefill=False)
        pos += 1
        nll.append(float(-torch.log_softmax(R.logits(h[-1:])[0], -1)[targets[t + 1]]))
    del caches
    return np.array(nll)


R = Runner()
t0 = time.time()
cal = R.tok(open(os.path.join(ROOT, "data", "wikitext2_test.txt")).read()[:200000], add_special_tokens=False)["input_ids"][:8192]
snap_c = R.prefill(torch.tensor(cal, device=DEV))
static = static_bases(snap_c["kv"], R.Hkv)
del snap_c
torch.cuda.empty_cache()
print(f"static bases from {len(cal)} calibration tokens ({time.time()-t0:.0f}s)", flush=True)
for src, text in (("prose", prose_text()), ("code", code_text())):
    for L in LENS:
        for d, ids in enumerate(doc_windows(R.tok, text, L, NDOC, seed=L)):
            todo = [sc for sc in SCHEMES if (src, L, d, sc) not in done]
            if not todo:
                continue
            x = torch.tensor(ids, device=DEV)
            snap = R.prefill(x[:L - N])
            for sc in todo:
                ts = time.time()
                if sc in PROJ:
                    nll = decode_proj(R, PROJ[sc], snap, x[L - N:], static)
                    bits = 16.0
                else:
                    nll, bits = R.decode_teacher(sc, snap, x[L - N:])
                rec = dict(src=src, L=L, doc=d, scheme=sc, nll=float(nll.mean()), nll_tok=[float(v) for v in nll], bits=bits)
                with open(out_p, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{time.time()-t0:6.0f}s] {src} L={L} doc{d} {sc:16s} PPL {math.exp(nll.mean()):8.4f} ({time.time()-ts:.0f}s)", flush=True)
            del snap
            torch.cuda.empty_cache()
print("PROJ_PPL_DONE")
