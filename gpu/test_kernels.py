"""Kernel correctness on real captured tensors: every scheme's CUDA attend(q) must equal its fp64 dequantize-and-dense
reference (same hot set / pages). Run: <torch venv python> gpu/test_kernels.py"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.foveakv_gpu import make, ext
from foveakv import kvdata

t0 = time.time()
ext()
print(f"build/load {time.time()-t0:.0f}s")
L = kvdata.load_layer("prose", 11)
K = torch.tensor(L["k"], device="cuda"); V = torch.tensor(L["v"], device="cuda")
Q = torch.tensor(L["q"], device="cuda")                 # (512, 8, 256) post-RoPE, for positions 7680..8191
T0 = 7000
names = ["bf16", "bf16-kernel", "tq2", "tq3", "tq4", "gs2", "gs4", "dyn5", "dyn5-gmean", "dyn5-noshrink", "dyn5-2stage", "static5",
         "kivi2", "kivi4", "quest5", "specache5", "specache5-eqtraffic",
         "dyn5-off", "dyn5-2stage-off", "specache5-off", "specache5-eqtraffic-off"]
ok = True
for nm in names:
    c = make(nm, 2, 8192)
    c.prefill(K[:T0], V[:T0])
    errs = []
    for t in range(T0, 7300):                            # decode: append then attend with the query of position t
        c.append(K[t:t + 1], V[t:t + 1])
        if t % 50 == 0 or t == 7299:
            qi = min(max(t - 7680, 0), 511)
            q = Q[qi]
            o = c.attend(q)
            r = c.reference(q)
            errs.append(float((o.double() - r).norm() / r.norm()))
    e = max(errs)
    flag = e < (5e-3 if nm == "bf16" else 1e-4)
    ok &= flag
    print(f"{c.name:28s} n={c.n} n_comp={c.n_comp} read_bits={c.read_bits():.3f}  max rel err vs reference {e:.2e} {'OK' if flag else 'FAIL'}")
print("ALL OK" if ok else "FAILURES", f"{time.time()-t0:.0f}s")
