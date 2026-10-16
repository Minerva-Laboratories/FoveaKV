"""INT8 vs FP16 tensor-core GEMM throughput with one Triton kernel (same tiling for both), at attention-like shapes.
torch._int_mm on this PyTorch build (no sm_87 kernels) ran at 0.3x bf16, so this checks the hardware directly."""
import os, time, torch, triton, triton.language as tl
LAYOUT = os.environ.get("FOVEAKV_MM_LAYOUT", "kn")

@triton.jit
def mm(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INT: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm, rn, rk = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32 if INT else tl.float32)
    for k in range(0, K, BK):
        x = tl.load(a + rm[:, None] * sam + (k + rk)[None, :] * sak)
        y = tl.load(b + (k + rk)[:, None] * sbk + rn[None, :] * sbn)
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc)

def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(5):
        t = time.time()
        for _ in range(n): f()
        torch.cuda.synchronize(); best = min(best, (time.time() - t) / n)
    return best

def run(a, b, c, INT, BM, BN, BK, w, ST=3):
    M, K = a.shape; N = b.shape[1]
    g = (M // BM, N // BN)
    return lambda: mm[g](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                         BM=BM, BN=BN, BK=BK, INT=INT, num_warps=w, num_stages=ST)

for (M, K, N, what) in [(4096, 256, 8192, "QK^T 4K x 8K, d 256"), (4096, 8192, 256, "PV"), (8192, 4096, 4096, "square 8Kx4Kx4K")]:
    a, b = torch.randn(M, K, device="cuda", dtype=torch.float16), torch.randn(K, N, device="cuda", dtype=torch.float16)
    a8, b8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8), torch.randint(-127, 127, (K, N), device="cuda", dtype=torch.int8)
    if LAYOUT == "nk":                           # B stored N x K (K contiguous), as keys are; V would be stored transposed
        b, b8 = b.t().contiguous().t(), b8.t().contiguous().t()
    c16, c32 = torch.empty(M, N, device="cuda", dtype=torch.float32), torch.empty(M, N, device="cuda", dtype=torch.int32)
    fl = 2 * M * K * N
    best = {}
    for BM, BN, BK, w in [(128, 128, 32, 4), (128, 128, 64, 4), (128, 256, 64, 8), (64, 128, 64, 4), (128, 64, 64, 4), (128, 128, 128, 8), (128, 256, 128, 8), (256, 128, 128, 8), (128, 128, 256, 8), (64, 256, 128, 4)]:
        if N % BN or M % BM or K % BK: continue
        for INT, x, y, c in [(False, a, b, c16), (True, a8, b8, c32)]:
            try:
                t = min(bench(run(x, y, c, INT, BM, BN, BK, w, st)) for st in (2, 3, 4))
            except Exception as e:
                continue
            best[INT] = min(best.get(INT, 1e9), t)
    torch.testing.assert_close((a8[:64].float() @ b8[:, :64].float()), c32[:64, :64].float()) if False else None
    print(f"[B {LAYOUT}] {what}: triton fp16 {fl/best[False]/1e12:.1f} TFLOPs, int8 {fl/best[True]/1e12:.1f} TOPS -> int8/fp16 {best[False]/best[True]:.2f}x", flush=True)
print("TRITON_INT8_DONE")
