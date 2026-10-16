"""Coarse-key scoring through a per-step query table (kernels.cu set_klut) vs the unpack + level-LUT + FMA path, in the
production decode kernel. Same schemes and timing as bench_kernels.py (batch 1, Hkv=2, G=4, D=256, CUDA events, median
of N attend() calls); output difference between the two paths on the same cache."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.foveakv_gpu import make, ext, D

E = ext()
torch.manual_seed(0)
SCHEMES = ["gs2", "tq2", "dyn5", "gs4", "kivi2", "bf16-kernel"]
CTX = [int(x) for x in os.environ.get("BENCH_CTX", "32768,131072").split(",")]
N = 50
out = []


def timed(c, q):
    for _ in range(5):
        c.attend(q)
    torch.cuda.synchronize()
    ts = []
    for _ in range(N):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); c.attend(q); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


for T in CTX:
    K = torch.randn(T, 2, D, device="cuda") * 0.5 + torch.randn(1, 2, D, device="cuda")
    V = torch.randn(T, 2, D, device="cuda")
    q = torch.randn(8, D, device="cuda") * 2
    for nm in SCHEMES:
        c = make(nm, 2, T + 256)
        c.prefill(K, V)
        row = dict(T=T, key=nm, scheme=c.name)
        outs = {}
        for lut in (0, 1):
            E.set_klut(lut)
            row[f"ms_lut{lut}"] = timed(c, q)
            outs[lut] = c.attend(q).float()
        E.set_klut(0)
        bits = c.read_bits()
        mb = bits * T * 2 * 2 * D / 8 / 1e6
        row["MB"] = mb
        row["rel_diff"] = float((outs[1] - outs[0]).norm() / outs[0].norm())
        out.append(row)
        print(f"T={T:6d} {c.name:26s} fma {row['ms_lut0']*1e3:8.1f} us ({mb/row['ms_lut0']:6.1f} GB/s)  "
              f"table {row['ms_lut1']*1e3:8.1f} us ({mb/row['ms_lut1']:6.1f} GB/s)  x{row['ms_lut0']/row['ms_lut1']:.2f}  "
              f"rel.diff {row['rel_diff']:.1e}", flush=True)
        del c
        torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gpu_lut_bench.json"), "w"), indent=1)
print("LUT_BENCH_DONE")
