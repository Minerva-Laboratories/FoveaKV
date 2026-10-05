"""Measured memory of every KV-cache scheme: GPU bytes (stored cache data and per-step workspace) and pinned host bytes,
for one attention layer of Qwen3.5-0.8B (2 KV heads, d=256) at several context lengths, and for the whole model (6 layers).
Counts the allocated tensors of each cache object (gpu/foveakv_gpu.footprint)."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.foveakv_gpu import make, footprint, D

SCHEMES = ["bf16", "tq2", "tq3", "tq4", "gs4", "kivi2", "kivi4", "quest5", "dyn5", "dyn5-off", "dyn5-2stage-off",
           "specache5", "specache5-off"]
CTX = [32768, 65536, 131072]
LAYERS = 6
out = []
for T in CTX:
    K = torch.randn(T, 2, D, device="cuda"); V = torch.randn(T, 2, D, device="cuda")
    for nm in SCHEMES:
        c = make(nm, 2, T)
        c.prefill(K, V)
        f = footprint(c)
        r = dict(T=T, key=nm, name=c.name, **{k: v * LAYERS for k, v in f.items()})
        out.append(r)
        print(f"T={T:6d} {c.name:34s} GPU {r['gpu']/2**20:8.1f} MiB (+{r['gpu_workspace']/2**20:5.1f} workspace)  host {r['host']/2**20:8.1f} MiB", flush=True)
        del c
        torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gpu_memory.json"), "w"), indent=1)
