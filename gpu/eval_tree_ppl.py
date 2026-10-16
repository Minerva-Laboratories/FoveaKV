"""End-to-end perplexity of the Barnes-Hut tree cache (gpu/tree_cache.py) against bf16 and FoveaKV, protocol of
gpu/eval_e2e.py ppl: exact prefill of L-N tokens, then N tokens teacher-forced through each scheme's cache (decode).
Writes results/<FOVEAKV_RESULTS_TAG>/tree_ppl.jsonl (resumable). FOVEAKV_TREE_QUICK=1: one doc at 16K, N=64."""
import json, math, os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu.eval_e2e import Runner, prose_text, code_text, doc_windows, DEV
from gpu.tree_cache import Tree

QUICK = bool(os.environ.get("FOVEAKV_TREE_QUICK"))
LENS = [16384] if QUICK else [16384, 32768]
NDOC, N = (1, 64) if QUICK else (3, 256)
TREES = {"tree-flat-r10": dict(rho=0.1), "tree-flat-r20": dict(rho=0.2), "tree-pp6-r20": dict(rho=0.2, mode="tree"),
         "drop-flat-r20": dict(rho=0.2, drop=True)}
SCHEMES = ["bf16", "dyn5"] + list(TREES)
out_p = os.path.join(ROOT, "results", os.environ.get("FOVEAKV_RESULTS_TAG", "qwen3.5-9b"), "tree_ppl" + ("_quick" if QUICK else "") + ".jsonl")
done = {(r["src"], r["L"], r["doc"], r["scheme"]) for r in map(json.loads, open(out_p))} if os.path.exists(out_p) else set()


@torch.no_grad()
def decode_tree(R, cfg, snap, targets):
    caches = {}
    for i, (k, v) in snap["kv"].items():
        c = Tree(R.Hkv, snap["T"] + 512, **cfg)
        c.prefill(k, v)
        c.build_tree(snap["qm"][i])
        caches[i] = c
    lin = snap["lin"].clone()
    nll = [float(-torch.log_softmax(snap["last_logits"], -1)[targets[0]])]
    pos = snap["T"]
    for t in range(len(targets) - 1):
        h = R.forward(targets[t:t + 1], pos, caches, lin, prefill=False)
        pos += 1
        nll.append(float(-torch.log_softmax(R.logits(h[-1:])[0], -1)[targets[t + 1]]))
    comp = float(np.mean([c.compute() for c in caches.values()]))
    del caches
    return np.array(nll), comp


R = Runner()
t0 = time.time()
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
                if sc in TREES:
                    nll, comp = decode_tree(R, TREES[sc], snap, x[L - N:])
                    bits = 16.0
                else:
                    nll, bits = R.decode_teacher(sc, snap, x[L - N:])
                    comp = 1.0
                rec = dict(src=src, L=L, doc=d, scheme=sc, nll=float(nll.mean()), nll_tok=[float(v) for v in nll], bits=bits, compute=comp)
                with open(out_p, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{time.time()-t0:6.0f}s] {src} L={L} doc{d} {sc:14s} PPL {math.exp(nll.mean()):8.4f} compute {comp:.3f} ({time.time()-ts:.0f}s)", flush=True)
            del snap
            torch.cuda.empty_cache()
print("TREE_PPL_DONE")
