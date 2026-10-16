"""Tensor-core throughput on this GPU: bf16 / fp16 GEMM vs int8 GEMM (torch._int_mm), at attention-like shapes.
Checks the cost model of E27 (int8 = 2x bf16). min over rounds of synced timings."""
import time, torch
dev = "cuda"
def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(5):
        t = time.time()
        for _ in range(n): f()
        torch.cuda.synchronize(); best = min(best, (time.time() - t) / n)
    return best
for (M, K, N, what) in [(4096, 256, 8192, "QK^T (4K queries x 8K keys, d 256)"), (4096, 8192, 256, "PV"), (8192, 4096, 4096, "square-ish 8Kx4Kx4K")]:
    a16, b16 = torch.randn(M, K, device=dev, dtype=torch.bfloat16), torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    h16, g16 = a16.half(), b16.half()
    a8, b8 = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8), torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8).t()
    fl = 2 * M * K * N
    tb, th, ti = bench(lambda: a16 @ b16), bench(lambda: h16 @ g16), bench(lambda: torch._int_mm(a8, b8))
    print(f"{what}: bf16 {fl/tb/1e12:.1f} TFLOPs, fp16 {fl/th/1e12:.1f}, int8 {fl/ti/1e12:.1f} TOPS -> int8/bf16 {tb/ti:.2f}x", flush=True)
