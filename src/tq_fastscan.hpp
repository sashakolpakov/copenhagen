// tq_fastscan.hpp — SIMD nibble-LUT "fast scan" kernel for TurboQuantizer.
//
// The scalar TurboQuantizer::score_ip (turbo_quant.hpp) walks one vector across
// `dim` dimensions, doing a dependent float load + add per dimension. At 4 bits
// the per-dimension table has n_levels=16 entries — exactly one register-width
// byte table. The fast-scan reorganization (FAISS / Quick-ADC style) flips the
// loop: process P vectors at once, one dimension at a time, replacing P scalar
// loads with a single in-register table shuffle (vqtbl1q_u8 / pshufb).
//
// Two prerequisites vs the scalar path:
//   1. The float query table is affine-quantized to uint8 (table ≈ q*step + lo).
//      Precision loss is absorbed by the exact BLAS rerank already in the search
//      path — fast scan only has to surface the right rerank candidates.
//   2. Codes are stored transposed in blocks of P: packed[block][dim][P], so the
//      P codes for a given dimension are contiguous and load in one instruction.
//
// Reconstruction: with S = Σ_d qlut[d*16 + code_d] (the uint accumulator),
//   <q,v> ≈ vec_scale · (bias + step·S + dim·lo),   ‖q-v‖² ≈ q_sq + ‖v‖² - 2<q,v>
// which is the scalar estimator with table[i] = qlut[i]·step + lo substituted in.
//
// ISA: AArch64 NEON (P=32, two 16-byte tbl lanes) and x86-64 AVX2 (P=32, two
// pshufb lanes), with a scalar fallback. AVX2 vs scalar is chosen at runtime via
// __builtin_cpu_supports; NEON is baseline on arm64 so it's compile-time.

#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
  #define CPH_FS_X86 1
  #include <immintrin.h>
#elif defined(__aarch64__)
  #define CPH_FS_ARM 1
  #include <arm_neon.h>
#endif

namespace cph {
namespace fastscan {

// Vectors processed per block. 32 on both supported ISAs (two 16-byte lanes).
#if defined(CPH_FS_X86) || defined(CPH_FS_ARM)
static constexpr int kWidth = 32;
#else
static constexpr int kWidth = 16;
#endif

// Per-dimension table stride in the quantized LUT, fixed at the register table
// width (one vqtbl1q_u8 / pshufb lookup). Holds 16 entries; tables with fewer
// levels (bits < 4) are zero-padded to this stride. All kernels assume it.
static constexpr int kLutStride = 16;

// True when this build can run a SIMD kernel (vs scalar-only fallback).
inline bool simd_available() {
#if defined(CPH_FS_ARM)
    return true;                                  // NEON is baseline on arm64
#elif defined(CPH_FS_X86)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

// ------------------------------------------------------------------ LUT quant
// Affine-quantize a float query table (dim rows of n_levels entries) to uint8:
// table[i] ≈ qlut[i]*step + lo, using one global (step, lo) for the whole table.
// Global (rather than per-dimension) keeps the kernel's dequant to a single
// scale; the exact rerank covers the resulting estimate error. The output is
// written at the fixed kLutStride (16) per dimension, zero-padding rows when
// n_levels < 16, so the kernels can assume a uniform stride. Requires
// n_levels <= kLutStride. `qlut` must hold dim*kLutStride bytes.
inline void quantize_lut(const float* table, int dim, int n_levels,
                         uint8_t* qlut, float& step, float& lo) {
    size_t N = (size_t)dim * n_levels;
    float mn = table[0], mx = table[0];
    for (size_t i = 1; i < N; i++) { mn = std::min(mn, table[i]); mx = std::max(mx, table[i]); }
    lo = mn;
    float range = mx - mn;
    step = range > 0.0f ? range / 255.0f : 1.0f;
    float inv = 1.0f / step;
    std::fill(qlut, qlut + (size_t)dim * kLutStride, (uint8_t)0);
    for (int d = 0; d < dim; d++) {
        const float* row = table + (size_t)d * n_levels;
        uint8_t* out = qlut + (size_t)d * kLutStride;
        for (int l = 0; l < n_levels; l++) {
            int q = (int)std::lround((row[l] - mn) * inv);
            out[l] = (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q));
        }
    }
}

// --------------------------------------------------------------- code packing
// Transpose row-major codes[n][dim] into blocked layout packed[nblocks][dim][P],
// zero-padding the final partial block. Returns the block count.
inline int n_blocks(int n) { return (n + kWidth - 1) / kWidth; }

inline void pack_codes(const uint8_t* codes, int n, int dim,
                       std::vector<uint8_t>& packed) {
    int nb = n_blocks(n);
    packed.assign((size_t)nb * dim * kWidth, 0);
    for (int b = 0; b < nb; b++) {
        int base = b * kWidth;
        int cnt = std::min(kWidth, n - base);
        uint8_t* blk = packed.data() + (size_t)b * dim * kWidth;
        for (int v = 0; v < cnt; v++) {
            const uint8_t* src = codes + (size_t)(base + v) * dim;
            for (int d = 0; d < dim; d++) blk[(size_t)d * kWidth + v] = src[d];
        }
    }
}

// ------------------------------------------------------------- block kernels
// Each computes, for one packed block, the uint32 code-sum S[v] for all kWidth
// lanes: S[v] = Σ_d qlut[d*16 + code(d,v)]. uint16 lane accumulators are flushed
// to uint32 every 256 dims (256*255 < 65535) to stay overflow-free at any dim.

inline void block_scalar(const uint8_t* blk, const uint8_t* qlut, int dim,
                         uint32_t* sums) {
    for (int v = 0; v < kWidth; v++) {
        uint32_t s = 0;
        for (int d = 0; d < dim; d++) s += qlut[(size_t)d * kLutStride + blk[(size_t)d * kWidth + v]];
        sums[v] = s;
    }
}

#if defined(CPH_FS_ARM)
inline void block_neon(const uint8_t* blk, const uint8_t* qlut, int dim,
                       uint32_t* sums) {
    // 8 uint32 lane-group accumulators cover all 32 vectors.
    uint32x4_t a[8];
    for (int i = 0; i < 8; i++) a[i] = vdupq_n_u32(0);
    uint16x8_t s0 = vdupq_n_u16(0), s1 = vdupq_n_u16(0),   // lanes 0..15
               s2 = vdupq_n_u16(0), s3 = vdupq_n_u16(0);   // lanes 16..31
    int flush = 0;
    for (int d = 0; d < dim; d++) {
        const uint8_t* cp = blk + (size_t)d * kWidth;
        uint8x16_t lut = vld1q_u8(qlut + (size_t)d * 16);   // 16-entry table
        uint8x16_t c_lo = vld1q_u8(cp);                     // vectors 0..15
        uint8x16_t c_hi = vld1q_u8(cp + 16);                // vectors 16..31
        uint8x16_t r_lo = vqtbl1q_u8(lut, c_lo);
        uint8x16_t r_hi = vqtbl1q_u8(lut, c_hi);
        s0 = vaddw_u8(s0, vget_low_u8(r_lo));
        s1 = vaddw_u8(s1, vget_high_u8(r_lo));
        s2 = vaddw_u8(s2, vget_low_u8(r_hi));
        s3 = vaddw_u8(s3, vget_high_u8(r_hi));
        if (++flush == 256 || d == dim - 1) {
            a[0] = vaddq_u32(a[0], vmovl_u16(vget_low_u16(s0)));
            a[1] = vaddq_u32(a[1], vmovl_u16(vget_high_u16(s0)));
            a[2] = vaddq_u32(a[2], vmovl_u16(vget_low_u16(s1)));
            a[3] = vaddq_u32(a[3], vmovl_u16(vget_high_u16(s1)));
            a[4] = vaddq_u32(a[4], vmovl_u16(vget_low_u16(s2)));
            a[5] = vaddq_u32(a[5], vmovl_u16(vget_high_u16(s2)));
            a[6] = vaddq_u32(a[6], vmovl_u16(vget_low_u16(s3)));
            a[7] = vaddq_u32(a[7], vmovl_u16(vget_high_u16(s3)));
            s0 = s1 = s2 = s3 = vdupq_n_u16(0);
            flush = 0;
        }
    }
    for (int i = 0; i < 8; i++) vst1q_u32(sums + i * 4, a[i]);
}
#endif

#if defined(CPH_FS_X86)
__attribute__((target("avx2")))
inline void block_avx2(const uint8_t* blk, const uint8_t* qlut, int dim,
                       uint32_t* sums) {
    // uint16 lane accumulators: 32 lanes = two __m256i (16xu16 each).
    __m256i s_lo = _mm256_setzero_si256();   // vectors 0..15
    __m256i s_hi = _mm256_setzero_si256();   // vectors 16..31
    // uint32 spill accumulators: 32 lanes = four __m256i (8xu32 each).
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    int flush = 0;
    for (int d = 0; d < dim; d++) {
        const uint8_t* cp = blk + (size_t)d * kWidth;
        // Broadcast the 16-byte table into both 128-bit lanes (pshufb is per-lane).
        __m128i lut128 = _mm_loadu_si128((const __m128i*)(qlut + (size_t)d * 16));
        __m256i lut = _mm256_broadcastsi128_si256(lut128);
        __m256i codes = _mm256_loadu_si256((const __m256i*)cp);  // 32 codes
        __m256i r = _mm256_shuffle_epi8(lut, codes);             // 32 looked-up bytes
        // Widen bytes -> uint16 and accumulate. unpack interleaves within lanes,
        // but the mapping is fixed and consistent, so spilled sums stay correct
        // per spill group; we resolve lane identity only at store time below.
        s_lo = _mm256_add_epi16(s_lo, _mm256_unpacklo_epi8(r, zero));
        s_hi = _mm256_add_epi16(s_hi, _mm256_unpackhi_epi8(r, zero));
        if (++flush == 256 || d == dim - 1) {
            a0 = _mm256_add_epi32(a0, _mm256_unpacklo_epi16(s_lo, zero));
            a1 = _mm256_add_epi32(a1, _mm256_unpackhi_epi16(s_lo, zero));
            a2 = _mm256_add_epi32(a2, _mm256_unpacklo_epi16(s_hi, zero));
            a3 = _mm256_add_epi32(a3, _mm256_unpackhi_epi16(s_hi, zero));
            s_lo = _mm256_setzero_si256();
            s_hi = _mm256_setzero_si256();
            flush = 0;
        }
    }
    // Resolve the unpack permutation back to vector order. _mm256_shuffle_epi8
    // preserves byte position (byte j -> vector j). The two-stage byte->u16->u32
    // unpack runs per 128-bit lane, which interleaves the 256-bit halves. Tracing
    // it through (low lane = v0..15, high lane = v16..31) yields:
    //   a0 = [v0,v1,v2,v3,   v16,v17,v18,v19]
    //   a1 = [v4,v5,v6,v7,   v20,v21,v22,v23]
    //   a2 = [v8,v9,v10,v11, v24,v25,v26,v27]
    //   a3 = [v12,v13,v14,v15,v28,v29,v30,v31]
    alignas(32) uint32_t t0[8], t1[8], t2[8], t3[8];
    _mm256_store_si256((__m256i*)t0, a0);
    _mm256_store_si256((__m256i*)t1, a1);
    _mm256_store_si256((__m256i*)t2, a2);
    _mm256_store_si256((__m256i*)t3, a3);
    for (int i = 0; i < 4; i++) {
        sums[0  + i] = t0[i];   sums[16 + i] = t0[4 + i];
        sums[4  + i] = t1[i];   sums[20 + i] = t1[4 + i];
        sums[8  + i] = t2[i];   sums[24 + i] = t2[4 + i];
        sums[12 + i] = t3[i];   sums[28 + i] = t3[4 + i];
    }
}
#endif

// ------------------------------------------------------------------ dispatch
using block_fn = void (*)(const uint8_t*, const uint8_t*, int, uint32_t*);

inline block_fn select_block_fn() {
#if defined(CPH_FS_ARM)
    return &block_neon;
#elif defined(CPH_FS_X86)
    return __builtin_cpu_supports("avx2") ? &block_avx2 : &block_scalar;
#else
    return &block_scalar;
#endif
}

// ------------------------------------------------------------- full L2 scan
// Score `n` packed vectors against a quantized query LUT, writing ‖q-v‖²
// estimates to out[n] (in original vector order). `fn` from select_block_fn().
inline void scan_l2sq(const uint8_t* packed, int n, int dim,
                      const uint8_t* qlut, float step, float lo, float bias,
                      const float* scales, const float* sqnorms, float q_sq,
                      float* out, block_fn fn) {
    int nb = n_blocks(n);
    uint32_t sums[kWidth];
    float dim_lo = (float)dim * lo;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = packed + (size_t)b * dim * kWidth;
        fn(blk, qlut, dim, sums);
        int base = b * kWidth;
        int cnt = std::min(kWidth, n - base);
        for (int v = 0; v < cnt; v++) {
            int idx = base + v;
            float ip = (bias + step * (float)sums[v] + dim_lo) * scales[idx];
            out[idx] = q_sq + sqnorms[idx] - 2.0f * ip;
        }
    }
}

}  // namespace fastscan
}  // namespace cph
