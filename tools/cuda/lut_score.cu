// Decode scoring pass on FoveaKV's packed 2-bit shape codes (d = 256, 4 codes per byte, GQA group of 4 queries):
// s[t][g] = gain[t] * sum_j q_g[j] * lev[code_tj].  The 2-bit decode kernels run at ~50 GB/s against ~90-120 GB/s for 4-bit
// codes, i.e. they are ALU bound (unpack + FMA per element). Three ways to do the inner product:
//   fma : unpack each 2-bit code, select its level, FMA with q (q slice in registers)        64 FMA / 16 codes / thread
//   lut4: per query pair a half2 table T[64 groups][256 bytes] = partial dot of 4 coordinates  -> 2 lookups + 2 HADD2 / byte
//   lut2: per query pair a half2 table T[128 groups][16 nibbles] (2 coordinates per lookup)    -> 4 lookups + 4 HADD2 / byte
// Layout: 16 threads per key, each thread owns 4 bytes (16 coordinates); partial sums reduced over the 16 lanes.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define D 256
#define G 4
#define BYTES (D / 4)
__constant__ float LEV[4];

__device__ __forceinline__ float red16(float x) {
#pragma unroll
  for (int o = 8; o; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}

__global__ void k_fma(const unsigned *codes, const float *gain, const float *q, float *out, int T) {
  int lane = threadIdx.x & 15, kid = (blockIdx.x * blockDim.x + threadIdx.x) >> 4, nk = (gridDim.x * blockDim.x) >> 4;
  float qr[G][16];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int c = 0; c < 16; c++) qr[g][c] = q[g * D + lane * 16 + c];
  float l0 = LEV[0], l1 = LEV[1], l2 = LEV[2], l3 = LEV[3];
  for (int t = kid; t < T; t += nk) {
    unsigned w = codes[(size_t)t * 16 + lane];
    float acc[G] = {0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < 16; c++) {
      unsigned b = (w >> (2 * c)) & 3;
      float lv = b == 0 ? l0 : b == 1 ? l1 : b == 2 ? l2 : l3;
#pragma unroll
      for (int g = 0; g < G; g++) acc[g] = fmaf(qr[g][c], lv, acc[g]);
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
      float s = red16(acc[g]);
      if (lane == g) out[(size_t)t * G + g] = s * gain[t];
    }
  }
}

// tables: tab[pair][group][entry] half2 (queries 2*pair, 2*pair+1)
template <int BITS>   // BITS = 8 (lut4: 4 coordinates per entry) or 4 (lut2: 2 coordinates per entry)
__global__ void k_lut(const unsigned *codes, const float *gain, const __half2 *tab, float *out, int T) {
  constexpr int NG = D * 2 / BITS, NE = 1 << BITS;      // groups, entries per group
  extern __shared__ __half2 sm[];
  for (int i = threadIdx.x; i < 2 * NG * NE; i += blockDim.x) sm[i] = tab[i];
  __syncthreads();
  int lane = threadIdx.x & 15, kid = (blockIdx.x * blockDim.x + threadIdx.x) >> 4, nk = (gridDim.x * blockDim.x) >> 4;
  constexpr int PER = 32 / BITS;                       // groups per 32-bit word
  for (int t = kid; t < T; t += nk) {
    unsigned w = codes[(size_t)t * 16 + lane];
    __half2 a0 = __float2half2_rn(0.f), a1 = a0;
#pragma unroll
    for (int c = 0; c < PER; c++) {
      int grp = lane * PER + c, e = (w >> (BITS * c)) & (NE - 1);
      a0 = __hadd2(a0, sm[grp * NE + e]);
      a1 = __hadd2(a1, sm[NG * NE + grp * NE + e]);
    }
    float s[G] = {__low2float(a0), __high2float(a0), __low2float(a1), __high2float(a1)};
#pragma unroll
    for (int g = 0; g < G; g++) {
      float r = red16(s[g]);
      if (lane == g) out[(size_t)t * G + g] = r * gain[t];
    }
  }
}

int main(int argc, char **argv) {
  int T = argc > 1 ? atoi(argv[1]) : 524288;           // keys (all heads of a layer, laid end to end)
  const float lev[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};   // 2-bit Lloyd-Max levels for N(0,1)
  cudaMemcpyToSymbol(LEV, lev, sizeof lev);
  unsigned *hc = (unsigned *)malloc((size_t)T * BYTES);
  float *hg = (float *)malloc(T * 4), hq[G * D];
  srand(1);
  for (size_t i = 0; i < (size_t)T * 16; i++) hc[i] = (unsigned)rand() ^ ((unsigned)rand() << 16);
  for (int i = 0; i < T; i++) hg[i] = 0.5f + (rand() % 1000) / 1000.f;
  for (int i = 0; i < G * D; i++) hq[i] = ((rand() % 2001) - 1000) / 16000.f;
  // tables on the host (fp32 sums rounded to fp16 once)
  int NG4 = 64, NG2 = 128;
  __half2 *t4 = (__half2 *)malloc(2 * NG4 * 256 * sizeof(__half2)), *t2 = (__half2 *)malloc(2 * NG2 * 16 * sizeof(__half2));
  for (int p = 0; p < 2; p++) {
    for (int g = 0; g < NG4; g++)
      for (int e = 0; e < 256; e++) {
        float s0 = 0, s1 = 0;
        for (int c = 0; c < 4; c++) { float l = lev[(e >> (2 * c)) & 3]; s0 += hq[(2 * p) * D + 4 * g + c] * l; s1 += hq[(2 * p + 1) * D + 4 * g + c] * l; }
        t4[(p * NG4 + g) * 256 + e] = __floats2half2_rn(s0, s1);
      }
    for (int g = 0; g < NG2; g++)
      for (int e = 0; e < 16; e++) {
        float s0 = 0, s1 = 0;
        for (int c = 0; c < 2; c++) { float l = lev[(e >> (2 * c)) & 3]; s0 += hq[(2 * p) * D + 2 * g + c] * l; s1 += hq[(2 * p + 1) * D + 2 * g + c] * l; }
        t2[(p * NG2 + g) * 16 + e] = __floats2half2_rn(s0, s1);
      }
  }
  unsigned *dc; float *dg, *dq, *dout; __half2 *dt4, *dt2;
  cudaMalloc(&dc, (size_t)T * BYTES); cudaMalloc(&dg, T * 4); cudaMalloc(&dq, sizeof hq); cudaMalloc(&dout, (size_t)T * G * 4);
  cudaMalloc(&dt4, 2 * NG4 * 256 * sizeof(__half2)); cudaMalloc(&dt2, 2 * NG2 * 16 * sizeof(__half2));
  cudaMemcpy(dc, hc, (size_t)T * BYTES, cudaMemcpyHostToDevice); cudaMemcpy(dg, hg, T * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, hq, sizeof hq, cudaMemcpyHostToDevice);
  cudaMemcpy(dt4, t4, 2 * NG4 * 256 * sizeof(__half2), cudaMemcpyHostToDevice);
  cudaMemcpy(dt2, t2, 2 * NG2 * 16 * sizeof(__half2), cudaMemcpyHostToDevice);
  size_t sm4 = 2 * NG4 * 256 * sizeof(__half2), sm2 = 2 * NG2 * 16 * sizeof(__half2);
  cudaFuncSetAttribute(k_lut<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm4);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  float *ref = (float *)malloc((size_t)T * G * 4), *got = (float *)malloc((size_t)T * G * 4);
  for (int t = 0; t < 4096; t++)
    for (int g = 0; g < G; g++) {
      double s = 0;
      for (int j = 0; j < D; j++) s += hq[g * D + j] * lev[(hc[(size_t)t * 16 + j / 16] >> (2 * (j % 16))) & 3];
      ref[t * G + g] = (float)(s * hg[t]);
    }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double bytes = (double)T * (BYTES + 4);
  for (int k = 0; k < 3; k++) {
    float best = 1e9; const char *nm = k == 0 ? "fma " : k == 1 ? "lut4" : "lut2";
    int bestcfg = 0;
    for (int bpsm = 1; bpsm <= 8; bpsm *= 2)
      for (int thr = 128; thr <= 512; thr *= 2) {
        if (k == 1 && bpsm > 1) continue;                     // 128 KB table: one block per SM
        int blocks = nsm * bpsm;
        for (int r = 0; r < 6; r++) {
          cudaEventRecord(e0);
          if (k == 0) k_fma<<<blocks, thr>>>(dc, dg, dq, dout, T);
          else if (k == 1) k_lut<8><<<blocks, thr, sm4>>>(dc, dg, dt4, dout, T);
          else k_lut<4><<<blocks, thr, sm2>>>(dc, dg, dt2, dout, T);
          cudaEventRecord(e1); cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms, e0, e1);
          if (r > 0 && ms < best) { best = ms; bestcfg = blocks * 1000 + thr; }
        }
      }
    cudaError_t err = cudaGetLastError();
    // correctness on the first 4096 keys with the best config's last run (same output for every config)
    cudaMemcpy(got, dout, 4096 * G * 4, cudaMemcpyDeviceToHost);
    double e = 0, nrm = 0;
    for (int i = 0; i < 4096 * G; i++) { e += (got[i] - ref[i]) * (got[i] - ref[i]); nrm += ref[i] * ref[i]; }
    printf("%s  %8.1f us  %6.1f GB/s  %7.2f Gkeys/s  rel.err %.1e  (blocks %d x %d threads) %s\n", nm, best * 1e3,
           bytes / best / 1e6, T / best / 1e6, sqrt(e / nrm), bestcfg / 1000, bestcfg % 1000, err ? cudaGetErrorString(err) : "");
  }
  // memory bound reference: a pure copy-read of the same bytes
  {
    float best = 1e9;
    for (int r = 0; r < 6; r++) {
      cudaEventRecord(e0); cudaMemsetAsync(dout, 0, (size_t)T * G * 4); cudaMemcpyAsync(dt4, dc, 4, cudaMemcpyDeviceToDevice);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
    }
    unsigned *dc2; cudaMalloc(&dc2, (size_t)T * BYTES);
    for (int r = 0; r < 6; r++) {
      cudaEventRecord(e0); cudaMemcpyAsync(dc2, dc, (size_t)T * BYTES, cudaMemcpyDeviceToDevice);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (r && ms < best) best = ms;
    }
    printf("copy  %8.1f us  %6.1f GB/s read (+ same written)\n", best * 1e3, (double)T * BYTES / best / 1e6);
  }
  printf("LUT_SCORE_DONE T=%d\n", T);
}
