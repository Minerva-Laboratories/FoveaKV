// Peak tensor-core throughput from registers only: mma.sync m16n8k16 f16 (f32 acc) vs m16n8k32 s8 (s32 acc).
// If int8 is 2x fp16 here but not in GEMMs, the GEMM kernels are the bottleneck, not the hardware.
#include <cstdio>
#include <cuda_runtime.h>
#define ITERS 4096
__global__ void f16(float *out) {
  unsigned a0 = threadIdx.x, a1 = a0 ^ 1, a2 = a0 ^ 2, a3 = a0 ^ 3, b0 = a0 ^ 5, b1 = a0 ^ 6;
  float c[8][4] = {};
  for (int i = 0; i < ITERS; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float s = 0; for (int j = 0; j < 8; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void s8(int *out) {
  unsigned a0 = threadIdx.x, a1 = a0 ^ 1, a2 = a0 ^ 2, a3 = a0 ^ 3, b0 = a0 ^ 5, b1 = a0 ^ 6;
  int c[8][4] = {};
  for (int i = 0; i < ITERS; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                   : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  int s = 0; for (int j = 0; j < 8; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
int main() {
  int sm; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sm * 8, threads = 256; void *buf; cudaMalloc(&buf, blocks * threads * 4);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double warps = blocks * threads / 32.0;
  for (int k = 0; k < 2; k++) {
    float best[2] = {1e9, 1e9};
    for (int r = 0; r < 5; r++) {
      cudaEventRecord(e0); if (k == 0) f16<<<blocks, threads>>>((float *)buf); else s8<<<blocks, threads>>>((int *)buf);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best[k]) best[k] = ms;
    }
    double ops = warps * ITERS * 8 * 2.0 * 16 * 8 * (k == 0 ? 16 : 32);
    printf("%s mma: %.1f T%s/s (%.2f ms)\n", k == 0 ? "f16" : "s8 ", ops / best[k] / 1e9, k == 0 ? "FLOP" : "OP", best[k]);
  }
  printf("MMA_PEAK_DONE (SMs %d)\n", sm);
}
