"""Decode-attention latency per layer (batch 1, Hkv=2, G=4, D=256) vs context length, CUDA events, median of N.
Times the full attend() call (all kernels + selection/top-k + merges), i.e. what the model pays per layer per token.
Also reports bytes read per step (from read_bits) and the effective bandwidth. Random K/V (timing is data-independent
except for FoveaKV/Quest selection sizes, which are fixed by k)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.foveakv_gpu import make, ext, D

ext()
torch.manual_seed(0)
SCHEMES = ["bf16", "bf16-kernel", "tq2", "tq4", "gs4", "dyn5", "dyn10", "kivi2", "kivi4", "quest5"]
CTX = [int(x) for x in os.environ.get("BENCH_CTX", "8192,32768,65536,131072").split(",")]
N = 50
out = []
for T in CTX:
    K = torch.randn(T, 2, D, device="cuda") * 0.5 + torch.randn(1, 2, D, device="cuda")
    V = torch.randn(T, 2, D, device="cuda")
    q = torch.randn(8, D, device="cuda") * 2
    for nm in SCHEMES:
        c = make(nm, 2, T + 256)
        c.prefill(K, V)
        for _ in range(5):
            c.attend(q)
        torch.cuda.synchronize()
        ts = []
        for _ in range(N):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); c.attend(q); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        ms = float(np.median(ts))
        bits = c.read_bits()
        mb = bits * T * 2 * 2 * D / 8 / 1e6
        out.append(dict(T=T, scheme=c.name, key=nm, ms=ms, read_bits=bits, MB=mb, GBps=mb / 1e3 / (ms / 1e3)))
        print(f"T={T:6d} {c.name:26s} {ms*1e3:8.1f} us  read {bits:5.2f} b/elem = {mb:6.2f} MB  -> {mb/ms:6.1f} GB/s", flush=True)
        del c
        torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gpu_kernel_bench.json"), "w"), indent=1)
