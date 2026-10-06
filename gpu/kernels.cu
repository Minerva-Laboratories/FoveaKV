// DynReq CUDA decode-attention kernels (batch 1, GQA group G=4, head_dim D=FKV_D in {128, 256}, default 256).
// Split-K flash-decoding: one warp per token, each lane owns LD = D/32 consecutive dims; the 4 query heads of a KV group
// are processed together so every compressed code is read once per KV head.
//
// Formats (all per KV head h, token t; Tcap = allocated capacity):
//   BF16 : k,v (Hkv, Tcap, D) bf16                                     [dense baseline, recent window, Quest pages]
//   GS   : 2-bit planes hi/lo (Hkv, Tcap, D/4 bytes), 2*LD bits per lane,
//          gain: uint8 code + per-head LUT (256 floats, = g/sqrt(D)) or fp16 norm (TurboQuant, divided by sqrt(D)),
//          optional rotated means (Hkv, Tcap/mu_block, D) fp16 added back (mu_block = Tcap -> one global mean),
//          level LUT: 4 entries (hi plane only) or 16 entries (hi<<2|lo).
//   KIVI : codes b in {2,4} bits, K per-channel scale/zero per 128-token block, V per-token scale/zero per 128 channels.
// Pass outputs are partials (m, l, acc) per (block, h, g); combine kernels finish the softmax.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

#ifndef FKV_D
#define FKV_D 256
#endif
constexpr int D = FKV_D;
static_assert(D == 128 || D == 256, "head_dim must be 128 or 256");
constexpr int G = 4;
constexpr int LD = D / 32;       // dims per lane (8 or 4)
constexpr int WARPS = 4;         // warps per block
constexpr int RB = D / 4;        // bytes per 2-bit plane row
constexpr int RU4 = RB / 16;     // uint4 per 2-bit plane row (4 or 2)
constexpr float INV_SQRT_D = (D == 256) ? 0.0625f : 0.08838834764831845f;   // TurboQuant norms are stored unscaled

// LD consecutive 16-bit values (bf16 / fp16) of one lane
template <typename T> __device__ __forceinline__ void load_lane16(const T* src, T* dst) {
  if constexpr (LD == 8) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  else *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
}
// the 2*LD bits of 2-bit codes owned by `lane` in a plane row
__device__ __forceinline__ unsigned lane_codes2(const uint8_t* row, int lane) {
  if constexpr (LD == 8) return reinterpret_cast<const uint16_t*>(row)[lane];
  else return row[lane];
}

struct Params {
  int Hkv, Tcap;
  // token iteration: either range [t0, t0+n) or index list idx[h*idx_stride + i] (i < n, -1 = skip)
  int t0, n; const int* idx; int idx_stride;
  int tokens_per_block;
  // query: (Hkv, G, D) fp32, rotated if the format is rotated, pre-multiplied by the softmax scale
  const float* q;
  // BF16
  const __nv_bfloat16 *kb, *vb;
  // GS
  const uint8_t *khi, *klo, *vhi, *vlo;                     // 2-bit planes, RB bytes per row
  const uint8_t *kg8, *vg8; const float *kglut, *vglut;     // (Hkv,256)
  const __half *kg16, *vg16;                                 // TurboQuant norms
  const __half *kmu, *vmu; int mu_block;                     // rotated means or nullptr
  const float *klev, *vlev; int kfine, vfine;                // level LUTs (4 or 16 entries); fine = use lo plane
  const float *klev_f, *vlev_f;                              // delta pass: fine LUTs (klev/vlev = coarse LUTs)
  // KIVI
  const uint8_t *kc, *vc; int kbits;                         // codes (Hkv,Tcap,D*b/8)
  const __half *ksc, *kzp;                                   // (Hkv, Tcap/128, D)
  const __half *vsc, *vzp;                                   // (Hkv, Tcap, D/128)
  // outputs
  float *pm, *pl, *pacc;                                     // (nb, Hkv, G), (nb, Hkv, G, D)
  float* logits;                                             // optional (Hkv, G, Tcap)
  // delta pass (DynReq): reference max per head from pass 1 combine
  const float* mref;                                         // (Hkv, G)
};

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}

// ------------------------------------------------------------------------------------------- loaders
template <int FMT> struct Loader;

enum { F_BF16 = 0, F_GS = 1, F_KIVI = 2 };

template <> struct Loader<F_BF16> {
  __device__ static void k(const Params& p, int h, int t, int lane, float* x) {
    __nv_bfloat16 b[LD];
    load_lane16(p.kb + ((size_t)h * p.Tcap + t) * D + lane * LD, b);
#pragma unroll
    for (int i = 0; i < LD; i++) x[i] = __bfloat162float(b[i]);
  }
  __device__ static void v(const Params& p, int h, int t, int lane, float* x) {
    __nv_bfloat16 b[LD];
    load_lane16(p.vb + ((size_t)h * p.Tcap + t) * D + lane * LD, b);
#pragma unroll
    for (int i = 0; i < LD; i++) x[i] = __bfloat162float(b[i]);
  }
};

__device__ __forceinline__ void gs_load(const uint8_t* hi, const uint8_t* lo, const uint8_t* g8, const float* glut,
                                        const __half* g16, const __half* mu, int mu_block, const float* lev, int fine,
                                        int Tcap, int h, int t, int lane, float* x) {
  const size_t row = (size_t)h * Tcap + t;
  const unsigned a = lane_codes2(hi + row * RB, lane);
  const unsigned b = fine ? lane_codes2(lo + row * RB, lane) : 0u;
  const float g = g8 ? glut[h * 256 + g8[row]] : __half2float(g16[row]) * INV_SQRT_D;
#pragma unroll
  for (int i = 0; i < LD; i++) {
    const int c = (a >> (2 * i)) & 3;
    const int idx = fine ? ((c << 2) | ((b >> (2 * i)) & 3)) : c;
    x[i] = g * lev[idx];
  }
  if (mu) {
    const __half* m = mu + ((size_t)h * (Tcap / mu_block) + t / mu_block) * D + lane * LD;
    __half hm[LD];
    load_lane16(m, hm);
#pragma unroll
    for (int i = 0; i < LD; i++) x[i] += __half2float(hm[i]);
  }
}

template <> struct Loader<F_GS> {
  __device__ static void k(const Params& p, int h, int t, int lane, float* x) {
    gs_load(p.khi, p.klo, p.kg8, p.kglut, p.kg16, p.kmu, p.mu_block, p.klev, p.kfine, p.Tcap, h, t, lane, x);
  }
  __device__ static void v(const Params& p, int h, int t, int lane, float* x) {
    gs_load(p.vhi, p.vlo, p.vg8, p.vglut, p.vg16, p.vmu, p.mu_block, p.vlev, p.vfine, p.Tcap, h, t, lane, x);
  }
};

__device__ __forceinline__ void unpack_codes(const uint8_t* base, int bits, int lane, int* c) {
  if (bits == 2) {
    const unsigned a = lane_codes2(base, lane);
#pragma unroll
    for (int i = 0; i < LD; i++) c[i] = (a >> (2 * i)) & 3;
  } else {
    const unsigned a = (LD == 8) ? reinterpret_cast<const uint32_t*>(base)[lane] : reinterpret_cast<const uint16_t*>(base)[lane];
#pragma unroll
    for (int i = 0; i < LD; i++) c[i] = (a >> (4 * i)) & 15;
  }
}

template <> struct Loader<F_KIVI> {
  __device__ static void k(const Params& p, int h, int t, int lane, float* x) {
    int c[LD];
    const size_t row = (size_t)h * p.Tcap + t;
    unpack_codes(p.kc + row * (D * p.kbits / 8), p.kbits, lane, c);
    const size_t m = ((size_t)h * (p.Tcap / 128) + t / 128) * D + lane * LD;
    __half s[LD], z[LD];
    load_lane16(p.ksc + m, s); load_lane16(p.kzp + m, z);
#pragma unroll
    for (int i = 0; i < LD; i++) x[i] = c[i] * __half2float(s[i]) + __half2float(z[i]);
  }
  __device__ static void v(const Params& p, int h, int t, int lane, float* x) {
    int c[LD];
    const size_t row = (size_t)h * p.Tcap + t;
    unpack_codes(p.vc + row * (D * p.kbits / 8), p.kbits, lane, c);
    const size_t m = row * (D / 128) + (lane * LD) / 128;
    const float s = __half2float(p.vsc[m]), z = __half2float(p.vzp[m]);
#pragma unroll
    for (int i = 0; i < LD; i++) x[i] = c[i] * s + z;
  }
};

// ------------------------------------------------------------------------------------------- pass kernel
// Lane-per-token keys / dims-parallel values. A warp processes tiles of 32 tokens:
//  K: lane j computes the 4 head logits of token j over all 256 dims, streaming its token's codes with 16-byte loads;
//     q (pre-multiplied by per-block channel scales for KIVI) lives in a per-warp shared buffer read as broadcast float4;
//     the level LUT is replicated per lane in shared memory (conflict-free). No shuffles.
//  softmax: one max/rescale per tile per head (5 shuffles x 4 heads per 32 tokens), 4 exps per lane.
//  V: lane owns 8 dims, loops over the 32 tokens with probabilities broadcast from shared memory.
//  Per-block affine terms (GS block means, KIVI per-channel zero points) enter as one scalar per (block, head) on the
//  logits and as (sum p in block) * mu on the accumulator.
constexpr int TILE = 32;

template <int FMT>
__global__ void __launch_bounds__(32 * WARPS) attn_pass(Params p) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  // shared memory carve-up (bytes). qs: block-shared query (GS/BF16) or per-warp scaled query (KIVI).
  constexpr int QS_WARPS = (FMT == F_KIVI) ? WARPS : 1;
  constexpr int VROW = (FMT == F_BF16) ? 0 : D / 2;          // staged V bytes per token (hi+lo planes / KIVI codes)
  constexpr int SMEM_NEED = QS_WARPS * D * 16 + WARPS * TILE * VROW + 2 * 16 * 32 * 4 + 2 * 256 * 4 + 2 * WARPS * TILE * 16;
  constexpr int SMEM_BYTES = SMEM_NEED > WARPS * G * D * 4 ? SMEM_NEED : WARPS * G * D * 4;
  __shared__ __align__(16) unsigned char smem[SMEM_BYTES];
  float4 (*qs)[D] = reinterpret_cast<float4(*)[D]>(smem);
  unsigned char* vbuf = smem + QS_WARPS * D * 16;                                   // [w][j][VROW]
  float (*lutk)[32] = reinterpret_cast<float(*)[32]>(vbuf + WARPS * TILE * VROW);
  float (*lutv)[32] = lutk + 16;
  float* kglut_s = reinterpret_cast<float*>(lutv + 16);
  float* vglut_s = kglut_s + 256;
  float4 (*ps)[TILE] = reinterpret_cast<float4(*)[TILE]>(vglut_s + 256);              // probabilities per token
  float4 (*vaux)[TILE] = ps + WARPS;                                                   // per-token V gain / scale,zero
  float4* qw = qs[(FMT == F_KIVI) ? w : 0];

  if (FMT == F_GS) {
    for (int e = threadIdx.x; e < 16 * 32; e += blockDim.x) {
      const int i = e / 32;            // index = (hi << 2) | lo; coarse tables ignore lo
      lutk[i][e % 32] = p.klev[p.kfine ? i : (i >> 2)];
      lutv[i][e % 32] = p.vlev[p.vfine ? i : (i >> 2)];
    }
    if (p.kg8)
      for (int i = threadIdx.x; i < 256; i += blockDim.x) { kglut_s[i] = p.kglut[h * 256 + i]; vglut_s[i] = p.vglut[h * 256 + i]; }
  }
  float qr[G][LD];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int i = 0; i < LD; i++) qr[g][i] = p.q[(h * G + g) * D + lane * LD + i];
  if (FMT != F_KIVI && w == 0)
#pragma unroll
    for (int i = 0; i < LD; i++) qw[lane * LD + i] = make_float4(qr[0][i], qr[1][i], qr[2][i], qr[3][i]);
  __syncthreads();

  float m_run[G], l_run[G], acc[G][LD];
#pragma unroll
  for (int g = 0; g < G; g++) { m_run[g] = -INFINITY; l_run[g] = 0.f;
#pragma unroll
    for (int i = 0; i < LD; i++) acc[g][i] = 0.f; }
  const bool blockwise = (!p.idx) && ((FMT == F_GS && p.kmu) || FMT == F_KIVI);
  const int ablk = (FMT == F_KIVI) ? 128 : p.mu_block;
  int cur_b = -1;
  float qoff[G], l_b[G], vmu[LD];
#pragma unroll
  for (int g = 0; g < G; g++) { qoff[g] = 0.f; l_b[g] = 0.f; }
#pragma unroll
  for (int i = 0; i < LD; i++) vmu[i] = 0.f;

  const int per_warp = (p.tokens_per_block + WARPS - 1) / WARPS;
  const int b0 = blockIdx.x * p.tokens_per_block + w * per_warp;
  const int b1 = min(min(b0 + per_warp, (int)(blockIdx.x + 1) * p.tokens_per_block), p.n);
  unsigned char* vb_w = vbuf + w * TILE * VROW;

  for (int it0 = b0; it0 < b1; it0 += TILE) {
    if (blockwise) {
      const int bb = (p.t0 + it0) / ablk;
      if (bb != cur_b) {
        if (cur_b >= 0 && FMT == F_GS) {
#pragma unroll
          for (int g = 0; g < G; g++)
#pragma unroll
            for (int i = 0; i < LD; i++) acc[g][i] += l_b[g] * vmu[i];
        }
#pragma unroll
        for (int g = 0; g < G; g++) l_b[g] = 0.f;
        cur_b = bb;
        float part[G] = {0.f, 0.f, 0.f, 0.f};
        if (FMT == F_GS) {
          const size_t mo = ((size_t)h * (p.Tcap / p.mu_block) + bb) * D + lane * LD;
          __half hk[LD], hv[LD];
          load_lane16(p.kmu + mo, hk); load_lane16(p.vmu + mo, hv);
#pragma unroll
          for (int i = 0; i < LD; i++) {
            const float km = __half2float(hk[i]); vmu[i] = __half2float(hv[i]);
#pragma unroll
            for (int g = 0; g < G; g++) part[g] += qr[g][i] * km;
          }
        } else {
          const size_t mo = ((size_t)h * (p.Tcap / 128) + bb) * D + lane * LD;
          __half hs[LD], hz[LD];
          load_lane16(p.ksc + mo, hs); load_lane16(p.kzp + mo, hz);
          __syncwarp();
#pragma unroll
          for (int i = 0; i < LD; i++) {
            const float sc = __half2float(hs[i]), zp = __half2float(hz[i]);
            qw[lane * LD + i] = make_float4(qr[0][i] * sc, qr[1][i] * sc, qr[2][i] * sc, qr[3][i] * sc);
#pragma unroll
            for (int g = 0; g < G; g++) part[g] += qr[g][i] * zp;
          }
          __syncwarp();
        }
#pragma unroll
        for (int g = 0; g < G; g++) qoff[g] = warp_sum(part[g]);
      }
    }
    // ---------------- loads (lane-per-token): K codes to registers, V codes staged to shared memory
    const int it = it0 + lane;
    int t = -1;
    if (it < b1) t = p.idx ? p.idx[h * p.idx_stride + it] : p.t0 + it;
    const size_t row = (size_t)h * p.Tcap + (t < 0 ? 0 : t);
    uint4 ka[D / 32];                                          // KIVI key codes (2-bit: RU4 used, 4-bit: 2*RU4)
    float kgain = 0.f;
    if (FMT == F_GS && t >= 0) {
      kgain = p.kg8 ? kglut_s[p.kg8[row]] : __half2float(p.kg16[row]) * INV_SQRT_D;
      uint4* dst = reinterpret_cast<uint4*>(vb_w + lane * VROW);
      const uint4* vh = reinterpret_cast<const uint4*>(p.vhi + row * RB);
#pragma unroll
      for (int c = 0; c < RU4; c++) dst[c] = vh[c];
      if (p.vfine) { const uint4* vl = reinterpret_cast<const uint4*>(p.vlo + row * RB);
#pragma unroll
        for (int c = 0; c < RU4; c++) dst[RU4 + c] = vl[c]; }
      vaux[w][lane] = make_float4(p.vg8 ? vglut_s[p.vg8[row]] : __half2float(p.vg16[row]) * INV_SQRT_D, 0.f, 0.f, 0.f);
    } else if (FMT == F_KIVI && t >= 0) {
      const uint4* kp = reinterpret_cast<const uint4*>(p.kc + row * (D * p.kbits / 8));
      const uint4* vp = reinterpret_cast<const uint4*>(p.vc + row * (D * p.kbits / 8));
      uint4* dst = reinterpret_cast<uint4*>(vb_w + lane * VROW);
#pragma unroll
      for (int c = 0; c < RU4; c++) { ka[c] = kp[c]; dst[c] = vp[c]; }
      if (p.kbits == 4)
#pragma unroll
        for (int c = RU4; c < 2 * RU4; c++) { ka[c] = kp[c]; dst[c] = vp[c]; }
      constexpr int NG = D / 128;                              // value channel groups per token (1 or 2)
      const __half* vs = p.vsc + row * NG; const __half* vz = p.vzp + row * NG;
      vaux[w][lane] = make_float4(__half2float(vs[0]), __half2float(vs[NG - 1]), __half2float(vz[0]), __half2float(vz[NG - 1]));
    }
    // ---------------- K: logits
    float s[G] = {0.f, 0.f, 0.f, 0.f};
    if (t >= 0) {
      if (FMT == F_GS) {
        const uint4* hp = reinterpret_cast<const uint4*>(p.khi + row * RB);
        const uint4* lp = reinterpret_cast<const uint4*>(p.klo + row * RB);
#pragma unroll 1
        for (int c = 0; c < RU4; c++) {
          const uint4 a = hp[c];
          const uint4 b = p.kfine ? lp[c] : make_uint4(0, 0, 0, 0);
          const unsigned aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
          for (int wd = 0; wd < 4; wd++)
#pragma unroll
            for (int i = 0; i < 16; i++) {
              const int code = (aw[wd] >> (2 * i)) & 3;
              const float lv = lutk[(code << 2) | ((bw[wd] >> (2 * i)) & 3)][lane];   // coarse LUT replicated over lo
              const float4 qq = qw[c * 64 + wd * 16 + i];
              s[0] += qq.x * lv; s[1] += qq.y * lv; s[2] += qq.z * lv; s[3] += qq.w * lv;
            }
        }
#pragma unroll
        for (int gg = 0; gg < G; gg++) s[gg] *= kgain;
        if (p.kmu && p.idx) {
          const __half* mm = p.kmu + ((size_t)h * (p.Tcap / p.mu_block) + t / p.mu_block) * D;
          for (int d = 0; d < D; d++) { const float km = __half2float(mm[d]); const float4 qq = qw[d];
            s[0] += qq.x * km; s[1] += qq.y * km; s[2] += qq.z * km; s[3] += qq.w * km; }
        }
      } else if (FMT == F_KIVI) {
        if (p.kbits == 2) {
#pragma unroll
          for (int c = 0; c < RU4; c++) {
            const unsigned aw[4] = {ka[c].x, ka[c].y, ka[c].z, ka[c].w};
#pragma unroll
            for (int wd = 0; wd < 4; wd++)
#pragma unroll
              for (int i = 0; i < 16; i++) {
                const float cv = (float)((aw[wd] >> (2 * i)) & 3);
                const float4 qq = qw[c * 64 + wd * 16 + i];
                s[0] += qq.x * cv; s[1] += qq.y * cv; s[2] += qq.z * cv; s[3] += qq.w * cv;
              }
          }
        } else {
#pragma unroll
          for (int c = 0; c < 2 * RU4; c++) {
            const uint4 a = ka[c];
            const unsigned aw[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int wd = 0; wd < 4; wd++)
#pragma unroll
              for (int i = 0; i < 8; i++) {
                const float cv = (float)((aw[wd] >> (4 * i)) & 15);
                const float4 qq = qw[c * 32 + wd * 8 + i];
                s[0] += qq.x * cv; s[1] += qq.y * cv; s[2] += qq.z * cv; s[3] += qq.w * cv;
              }
          }
        }
      } else {
        const uint4* kp = reinterpret_cast<const uint4*>(p.kb + row * D);
#pragma unroll 4
        for (int c = 0; c < D / 8; c++) {
          const uint4 r = kp[c];
          const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&r);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const float kv = __bfloat162float(bf[i]);
            const float4 qq = qw[c * 8 + i];
            s[0] += qq.x * kv; s[1] += qq.y * kv; s[2] += qq.z * kv; s[3] += qq.w * kv;
          }
        }
      }
#pragma unroll
      for (int g = 0; g < G; g++) s[g] += qoff[g];
      if (p.logits)
#pragma unroll
        for (int g = 0; g < G; g++) p.logits[((size_t)h * G + g) * p.Tcap + t] = s[g];
    } else {
#pragma unroll
      for (int g = 0; g < G; g++) s[g] = -INFINITY;
    }
    // ---------------- online softmax, once per tile
    float pr[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
      float mt = s[g];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, o));
      const float mn = fmaxf(m_run[g], mt);
      const float sc = (mn == -INFINITY) ? 1.f : __expf(m_run[g] - mn);
      pr[g] = (s[g] == -INFINITY) ? 0.f : __expf(s[g] - mn);
      const float psum = warp_sum(pr[g]);
      l_run[g] = l_run[g] * sc + psum;
      l_b[g] = l_b[g] * sc + psum;
      m_run[g] = mn;
#pragma unroll
      for (int i = 0; i < LD; i++) acc[g][i] *= sc;
    }
    ps[w][lane] = make_float4(pr[0], pr[1], pr[2], pr[3]);
    __syncwarp();
    // ---------------- V
    const int nt = min(TILE, b1 - it0);
    if (FMT == F_GS) {
#pragma unroll 4
      for (int j = 0; j < nt; j++) {
        const float4 pj = ps[w][j];
        const float g = vaux[w][j].x;
        const unsigned a = lane_codes2(vb_w + j * VROW, lane);
        const unsigned b = p.vfine ? lane_codes2(vb_w + j * VROW + RB, lane) : 0u;
#pragma unroll
        for (int i = 0; i < LD; i++) {
          const int code = (a >> (2 * i)) & 3;
          const float x = g * lutv[(code << 2) | ((b >> (2 * i)) & 3)][lane];
          acc[0][i] += pj.x * x; acc[1][i] += pj.y * x; acc[2][i] += pj.z * x; acc[3][i] += pj.w * x;
        }
      }
    } else if (FMT == F_KIVI) {
#pragma unroll 4
      for (int j = 0; j < nt; j++) {
        const float4 pj = ps[w][j];
        const float4 sz = vaux[w][j];
        const float sc = (lane < 16) ? sz.x : sz.y, zp = (lane < 16) ? sz.z : sz.w;
        int c[LD];
        unpack_codes(vb_w + j * VROW, p.kbits, lane, c);
#pragma unroll
        for (int i = 0; i < LD; i++) {
          const float x = c[i] * sc + zp;
          acc[0][i] += pj.x * x; acc[1][i] += pj.y * x; acc[2][i] += pj.z * x; acc[3][i] += pj.w * x;
        }
      }
    } else {
      for (int j = 0; j < nt; j++) {
        const int tj = p.idx ? p.idx[h * p.idx_stride + it0 + j] : p.t0 + it0 + j;
        if (tj < 0) continue;
        const float4 pj = ps[w][j];
        float x[LD];
        Loader<F_BF16>::v(p, h, tj, lane, x);
#pragma unroll
        for (int i = 0; i < LD; i++) {
          acc[0][i] += pj.x * x[i]; acc[1][i] += pj.y * x[i]; acc[2][i] += pj.z * x[i]; acc[3][i] += pj.w * x[i];
        }
      }
    }
    __syncwarp();
  }
  if (blockwise && FMT == F_GS && cur_b >= 0)
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
      for (int i = 0; i < LD; i++) acc[g][i] += l_b[g] * vmu[i];
  // block reduction over warps (sacc aliases the start of smem: needs WARPS*G*D*4 = 16 KB)
  __shared__ float sm[WARPS][G], sl[WARPS][G];
  __syncthreads();
  float (*sacc)[G][D] = reinterpret_cast<float(*)[G][D]>(smem);
  static_assert(sizeof(smem) >= WARPS * G * D * 4, "smem too small for sacc alias");
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; g++) { sm[w][g] = m_run[g]; sl[w][g] = l_run[g]; }
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int i = 0; i < LD; i++) sacc[w][g][lane * LD + i] = acc[g][i];
  __syncthreads();
  const size_t ob = (size_t)blockIdx.x * p.Hkv + h;
  for (int e = threadIdx.x; e < G * D; e += blockDim.x) {
    const int g = e / D, dd = e % D;
    float M = -INFINITY;
    for (int ww = 0; ww < WARPS; ww++) M = fmaxf(M, sm[ww][g]);
    float L = 0.f, A = 0.f;
    if (M > -INFINITY)
      for (int ww = 0; ww < WARPS; ww++) {
        if (sm[ww][g] == -INFINITY) continue;
        const float f = __expf(sm[ww][g] - M); L += sl[ww][g] * f; A += sacc[ww][g][dd] * f;
      }
    p.pacc[(ob * G + g) * D + dd] = A;
    if (dd == 0) { p.pm[ob * G + g] = M; p.pl[ob * G + g] = L; }
  }
}

// ------------------------------------------------------------------------------------------- delta pass (DynReq)
// For hot tokens: replace the coarse contribution by the fine one, relative to the pass-1 max mref.
// Outputs partial (dl, dacc) per block in pl / pacc (pm unused).
__global__ void delta_pass(Params p) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  float q[G][LD], mr[G];
#pragma unroll
  for (int g = 0; g < G; g++) { mr[g] = p.mref[h * G + g];
#pragma unroll
    for (int i = 0; i < LD; i++) q[g][i] = p.q[(h * G + g) * D + lane * LD + i]; }
  float dl[G], dacc[G][LD];
#pragma unroll
  for (int g = 0; g < G; g++) { dl[g] = 0.f;
#pragma unroll
    for (int i = 0; i < LD; i++) dacc[g][i] = 0.f; }
  const int per_warp = (p.tokens_per_block + WARPS - 1) / WARPS;
  const int b0 = blockIdx.x * p.tokens_per_block + w * per_warp;
  const int b1 = min(min(b0 + per_warp, (int)(blockIdx.x + 1) * p.tokens_per_block), p.n);
  float kc[LD], kf[LD], vc[LD], vf[LD];
  for (int it = b0; it < b1; it++) {
    const int t = p.idx[h * p.idx_stride + it];
    if (t < 0) continue;
    gs_load(p.khi, p.klo, p.kg8, p.kglut, p.kg16, p.kmu, p.mu_block, p.klev, 0, p.Tcap, h, t, lane, kc);
    gs_load(p.khi, p.klo, p.kg8, p.kglut, p.kg16, p.kmu, p.mu_block, p.klev_f, 1, p.Tcap, h, t, lane, kf);
    gs_load(p.vhi, p.vlo, p.vg8, p.vglut, p.vg16, p.vmu, p.mu_block, p.vlev, 0, p.Tcap, h, t, lane, vc);
    gs_load(p.vhi, p.vlo, p.vg8, p.vglut, p.vg16, p.vmu, p.mu_block, p.vlev_f, 1, p.Tcap, h, t, lane, vf);
#pragma unroll
    for (int g = 0; g < G; g++) {
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int i = 0; i < LD; i++) { a += q[g][i] * kc[i]; b += q[g][i] * kf[i]; }
      a = warp_sum(a); b = warp_sum(b);
      const float ec = __expf(a - mr[g]), ef = __expf(b - mr[g]);
      dl[g] += ef - ec;
#pragma unroll
      for (int i = 0; i < LD; i++) dacc[g][i] += ef * vf[i] - ec * vc[i];
    }
  }
  __shared__ float sl[WARPS][G];
  __shared__ float sacc[WARPS][G][D];
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; g++) sl[w][g] = dl[g];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int i = 0; i < LD; i++) sacc[w][g][lane * LD + i] = dacc[g][i];
  __syncthreads();
  const size_t ob = (size_t)blockIdx.x * p.Hkv + h;
  for (int e = threadIdx.x; e < G * D; e += blockDim.x) {
    const int g = e / D, dd = e % D;
    float L = 0.f, A = 0.f;
    for (int ww = 0; ww < WARPS; ww++) { L += sl[ww][g]; A += sacc[ww][g][dd]; }
    p.pacc[(ob * G + g) * D + dd] = A;
    if (dd == 0) p.pl[ob * G + g] = L;
  }
}

// ------------------------------------------------------------------------------------------- combine kernels
// combine partials (nb, Hkv*G) -> m, l, acc (Hkv*G, D); grid Hkv*G, block D threads
__global__ void combine(const float* pm, const float* pl, const float* pacc, int nb, int HG,
                        float* m_out, float* l_out, float* acc_out, float* out) {
  const int hg = blockIdx.x, dd = threadIdx.x;
  float M = -INFINITY;
  for (int b = 0; b < nb; b++) M = fmaxf(M, pm[b * HG + hg]);
  float L = 0.f, A = 0.f;
  for (int b = 0; b < nb; b++) {
    const float f = (pm[b * HG + hg] > -INFINITY) ? __expf(pm[b * HG + hg] - M) : 0.f;
    L += pl[b * HG + hg] * f; A += pacc[((size_t)b * HG + hg) * D + dd] * f;
  }
  if (acc_out) acc_out[hg * D + dd] = A;
  if (out) out[hg * D + dd] = A / L;
  if (dd == 0) { if (m_out) m_out[hg] = M; if (l_out) l_out[hg] = L; }
}

__global__ void combine_delta(const float* l1, const float* acc1, const float* pl, const float* pacc, int nb, int HG, float* out) {
  const int hg = blockIdx.x, dd = threadIdx.x;
  float L = l1[hg], A = acc1[hg * D + dd];
  for (int b = 0; b < nb; b++) { L += pl[b * HG + hg]; A += pacc[((size_t)b * HG + hg) * D + dd]; }
  out[hg * D + dd] = A / L;
}

// ------------------------------------------------------------------------------------------- host side
static const void* ptr(const c10::optional<torch::Tensor>& t) { return (t.has_value() && t->defined()) ? t->data_ptr() : nullptr; }

// Generic pass launcher. fmt: 0 bf16, 1 gs, 2 kivi. Writes partials into pm/pl/pacc starting at block offset `boff`.
void pass(int64_t fmt, torch::Tensor q, int64_t Hkv, int64_t Tcap, int64_t t0, int64_t n,
          c10::optional<torch::Tensor> idx, int64_t tokens_per_block, int64_t boff,
          torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc,
          c10::optional<torch::Tensor> kb, c10::optional<torch::Tensor> vb,
          c10::optional<torch::Tensor> khi, c10::optional<torch::Tensor> klo, c10::optional<torch::Tensor> vhi, c10::optional<torch::Tensor> vlo,
          c10::optional<torch::Tensor> kg8, c10::optional<torch::Tensor> vg8, c10::optional<torch::Tensor> kglut, c10::optional<torch::Tensor> vglut,
          c10::optional<torch::Tensor> kg16, c10::optional<torch::Tensor> vg16,
          c10::optional<torch::Tensor> kmu, c10::optional<torch::Tensor> vmu, int64_t mu_block,
          c10::optional<torch::Tensor> klev, c10::optional<torch::Tensor> vlev, int64_t kfine, int64_t vfine,
          c10::optional<torch::Tensor> kc, c10::optional<torch::Tensor> vc, int64_t kbits,
          c10::optional<torch::Tensor> ksc, c10::optional<torch::Tensor> kzp, c10::optional<torch::Tensor> vsc, c10::optional<torch::Tensor> vzp,
          c10::optional<torch::Tensor> logits) {
  Params p{};
  p.Hkv = Hkv; p.Tcap = Tcap; p.t0 = t0; p.n = n;
  p.idx = (const int*)ptr(idx); p.idx_stride = idx.has_value() ? idx->size(1) : 0;
  p.tokens_per_block = tokens_per_block; p.q = q.data_ptr<float>();
  p.kb = (const __nv_bfloat16*)ptr(kb); p.vb = (const __nv_bfloat16*)ptr(vb);
  p.khi = (const uint8_t*)ptr(khi); p.klo = (const uint8_t*)ptr(klo); p.vhi = (const uint8_t*)ptr(vhi); p.vlo = (const uint8_t*)ptr(vlo);
  p.kg8 = (const uint8_t*)ptr(kg8); p.vg8 = (const uint8_t*)ptr(vg8); p.kglut = (const float*)ptr(kglut); p.vglut = (const float*)ptr(vglut);
  p.kg16 = (const __half*)ptr(kg16); p.vg16 = (const __half*)ptr(vg16);
  p.kmu = (const __half*)ptr(kmu); p.vmu = (const __half*)ptr(vmu); p.mu_block = mu_block;
  p.klev = (const float*)ptr(klev); p.vlev = (const float*)ptr(vlev); p.kfine = kfine; p.vfine = vfine;
  p.kc = (const uint8_t*)ptr(kc); p.vc = (const uint8_t*)ptr(vc); p.kbits = kbits;
  p.ksc = (const __half*)ptr(ksc); p.kzp = (const __half*)ptr(kzp); p.vsc = (const __half*)ptr(vsc); p.vzp = (const __half*)ptr(vzp);
  const int HG = Hkv * G;
  p.pm = pm.data_ptr<float>() + boff * HG; p.pl = pl.data_ptr<float>() + boff * HG; p.pacc = pacc.data_ptr<float>() + boff * HG * D;
  p.logits = (float*)ptr(logits);
  const int nb = (n + tokens_per_block - 1) / tokens_per_block;
  if (nb == 0) return;
  dim3 grid(nb, Hkv), block(32 * WARPS);
  auto st = c10::cuda::getCurrentCUDAStream();
  if (fmt == 0) attn_pass<F_BF16><<<grid, block, 0, st>>>(p);
  else if (fmt == 1) attn_pass<F_GS><<<grid, block, 0, st>>>(p);
  else attn_pass<F_KIVI><<<grid, block, 0, st>>>(p);
}

void delta(torch::Tensor q, int64_t Hkv, int64_t Tcap, torch::Tensor idx, int64_t tokens_per_block, torch::Tensor mref,
           torch::Tensor pl, torch::Tensor pacc, torch::Tensor khi, torch::Tensor klo, torch::Tensor vhi, torch::Tensor vlo,
           torch::Tensor kg8, torch::Tensor vg8, torch::Tensor kglut, torch::Tensor vglut,
           c10::optional<torch::Tensor> kmu, c10::optional<torch::Tensor> vmu, int64_t mu_block,
           torch::Tensor klev4, torch::Tensor klev16, torch::Tensor vlev4, torch::Tensor vlev16) {
  Params p{};
  p.Hkv = Hkv; p.Tcap = Tcap; p.n = idx.size(1); p.idx = idx.data_ptr<int>(); p.idx_stride = idx.size(1);
  p.tokens_per_block = tokens_per_block; p.q = q.data_ptr<float>(); p.mref = mref.data_ptr<float>();
  p.khi = (const uint8_t*)khi.data_ptr(); p.klo = (const uint8_t*)klo.data_ptr();
  p.vhi = (const uint8_t*)vhi.data_ptr(); p.vlo = (const uint8_t*)vlo.data_ptr();
  p.kg8 = kg8.data_ptr<uint8_t>(); p.vg8 = vg8.data_ptr<uint8_t>(); p.kglut = kglut.data_ptr<float>(); p.vglut = vglut.data_ptr<float>();
  p.kmu = (const __half*)ptr(kmu); p.vmu = (const __half*)ptr(vmu); p.mu_block = mu_block;
  p.klev = klev4.data_ptr<float>(); p.klev_f = klev16.data_ptr<float>();
  p.vlev = vlev4.data_ptr<float>(); p.vlev_f = vlev16.data_ptr<float>();
  p.pl = pl.data_ptr<float>(); p.pacc = pacc.data_ptr<float>();
  const int nb = (p.n + tokens_per_block - 1) / tokens_per_block;
  if (nb == 0) return;
  delta_pass<<<dim3(nb, Hkv), 32 * WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
}

std::vector<torch::Tensor> combine_partials(torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc, int64_t nb, bool finalize) {
  const int HG = pm.size(1) * pm.size(2);
  auto o = torch::empty({HG, D}, pacc.options());
  auto m = torch::empty({HG}, pm.options()), l = torch::empty({HG}, pm.options()), a = torch::empty({HG, D}, pacc.options());
  combine<<<HG, D, 0, c10::cuda::getCurrentCUDAStream()>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(), nb, HG,
                                                            m.data_ptr<float>(), l.data_ptr<float>(), a.data_ptr<float>(),
                                                            finalize ? o.data_ptr<float>() : nullptr);
  return {o, m, l, a};
}

torch::Tensor combine_with_delta(torch::Tensor l1, torch::Tensor acc1, torch::Tensor pl, torch::Tensor pacc, int64_t nb) {
  const int HG = l1.size(0);
  auto o = torch::empty({HG, D}, acc1.options());
  combine_delta<<<HG, D, 0, c10::cuda::getCurrentCUDAStream()>>>(l1.data_ptr<float>(), acc1.data_ptr<float>(), pl.data_ptr<float>(),
                                                                  pacc.data_ptr<float>(), nb, HG, o.data_ptr<float>());
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pass_", &pass); m.def("delta", &delta);
  m.def("combine_partials", &combine_partials); m.def("combine_with_delta", &combine_with_delta);
}
