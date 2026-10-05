"""Full decode-step latency (whole model, one token) with each KV-cache scheme in the loop.
Qwen3.5-0.8B, bf16 weights + autocast, RTX 4050 Laptop GPU. A context of L tokens is prefilled once (exact, shared);
each scheme builds its cache from the snapshot and decodes N tokens; we report the median step time (CUDA events).
Everything except the 6 attention layers is identical across schemes, so differences come from the attention path.
Usage: <torch venv python> gpu/bench_e2e.py"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.eval_e2e import Runner, prose_text, DEV
from gpu.foveakv_gpu import ext, footprint

SCHEMES = ["bf16", "tq2", "tq3", "tq4", "gs4", "kivi2", "kivi4", "quest5", "quest10", "dyn2", "dyn5", "dyn10",
           "dyn5-2stage", "static5", "specache5", "specache5-eqtraffic", "dyn5-off", "dyn5-2stage-off",
           "specache5-off", "specache5-eqtraffic-off"]
LENS = [int(x) for x in os.environ.get("BENCH_LENS", "16384,32768").split(",")]
N, WARM = 32, 4
ext()
R = Runner()
ids_all = R.tok(prose_text()[:600000], add_special_tokens=False)["input_ids"]
out = []
for L in LENS:
    x = torch.tensor(ids_all[:L + N + WARM], device=DEV)
    snap = R.prefill(x[:L])
    for sc in SCHEMES:
        caches = R.build(sc, snap)
        lin = snap["lin"].clone()
        pos, ts = L, []
        for t in range(WARM + N):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            R.forward(x[L + t:L + t + 1], pos, caches, lin, prefill=False)
            b.record(); torch.cuda.synchronize()
            if t >= WARM:
                ts.append(a.elapsed_time(b))
            pos += 1
        ms = float(np.median(ts))
        bits = float(np.mean([c.read_bits() for c in caches.values()]))
        fp = [footprint(c) for c in caches.values()]
        gpu_mb, host_mb = sum(f["gpu"] + f["gpu_workspace"] for f in fp) / 2**20, sum(f["host"] for f in fp) / 2**20
        out.append(dict(L=L, scheme=sc, name=next(iter(caches.values())).name, ms=ms, bits=bits, cache_gpu_mib=gpu_mb, cache_host_mib=host_mb))
        print(f"L={L:6d} {sc:24s} {ms:7.2f} ms/token  bits {bits:.2f}  cache GPU {gpu_mb:7.1f} MiB host {host_mb:7.1f} MiB", flush=True)
        del caches, lin
        torch.cuda.empty_cache()
    del snap
    torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gpu_e2e_latency" + os.environ.get("BENCH_TAG", "") + ".json"), "w"), indent=1)
