// Numerics for the native HIP fast chain.
// H() = f16 RNE, F() = E4M3FN RNE (OCP, bias 7, max 448).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#ifndef __HIP_DEVICE_COMPILE__
#include <cstring>
#endif

namespace dlss5 {

using u32 = uint32_t;
using u8  = uint8_t;

__host__ __device__ inline u32 as_u32(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(v);
#else
    u32 u;
    std::memcpy(&u, &v, 4);
    return u;
#endif
}
__host__ __device__ inline float as_f32(u32 u) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(u);
#else
    float v;
    std::memcpy(&v, &u, 4);
    return v;
#endif
}

// Software f16 RNE (HLSL bit recipe, used when NATIVE_HW_H=0).
__host__ __device__ inline float H_sw(float v) {
    u32 b = as_u32(v), sg = b & 0x80000000u, a = b & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return v;
    if (a < 0x38800000u) {
        float q = nearbyintf(fabsf(v) * 16777216.0f) * 5.9604644775390625e-8f;
        return sg ? -q : q;
    }
    u32 r = (a + 0xfffu + ((a >> 13) & 1u)) & 0xffffe000u;
    return as_f32(sg | (r >= 0x47800000u ? 0x7f800000u : r));
}

// Software E4M3FN RNE (HLSL Ffast / NativeFastFp8).
__host__ __device__ inline float F_sw(float v) {
    u32 bits = as_u32(v), a = bits & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return v;
    float sg = v < 0 ? -1.f : 1.f;
    if (a < 0x3c800000u)
        return copysignf(nearbyintf(fabsf(v) * 512.f) / 512.f, v);
    if (a >= 0x43e00000u)
        return sg * 448.f;
    u32 r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    float m = as_f32(r);
    return sg * (m > 448.f ? 448.f : m);
}

__host__ __device__ inline float H_hw(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __half2float(__float2half_rn(v));
#else
    return H_sw(v);
#endif
}
__host__ __device__ inline float F_hw(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return float(__hip_fp8_e4m3(v));
#else
    return F_sw(v);
#endif
}

#ifndef DLSS5_HW_H
#define DLSS5_HW_H 1
#endif
#if DLSS5_HW_H
#define H H_hw
#define F F_hw
#else
#define H H_sw
#define F F_sw
#endif

__host__ __device__ inline float ActivatePoly(float v) {
    float g = fminf(fmaxf(v, -4.f), 4.f);
    return v * (g * (fabsf(g) * (-0.055908203125f) + 0.447265625f) + 0.89453125f);
}
__host__ __device__ inline float Activate(float v) { return F(ActivatePoly(v)); }

// v_rcp_f32 + two Newton–Raphson FMAs. Exhaustive gfx1201 check: every finite
// divisor in [1/256, 624] matches IEEE 1.f/x (144 441 345 values). Window
// softmax sums 64 positive half exponents into that interval, so replacing
// 1.f/sum is bit-identical. Do not use for ViT (640-key sums can exceed 624).
__host__ __device__ inline float norm_inverse(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    float r = __builtin_amdgcn_rcpf(x);
    r = __builtin_fmaf(r, __builtin_fmaf(-x, r, 1.f), r);
    return __builtin_fmaf(r, __builtin_fmaf(-x, r, 1.f), r);
#else
    return 1.f / x;
#endif
}

// native_c32_ffn_fused.hlsli precise q/p: no multiply-add contraction.
// Keep the legacy polynomial for other shader families and public operators.
__host__ __device__ inline float ActivatePolyC32(float v) {
    float g = fminf(fmaxf(v, -4.f), 4.f);
    // Opaque f32 multiplies keep each rounding separate without the volatile
    // private-memory round-trips that spilled 20 B/lane of scratch on gfx12.
#if defined(__HIP_DEVICE_COMPILE__)
    float qmul, pmul;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(qmul) : "v"(fabsf(g)), "v"(-0.055908203125f));
    float q = qmul + 0.447265625f;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(pmul) : "v"(g), "v"(q));
    float p = pmul + 0.89453125f;
    return v * p;
#else
    volatile float qmul = fabsf(g) * (-0.055908203125f);
    volatile float q = qmul + 0.447265625f;
    volatile float pmul = g * q;
    volatile float p = pmul + 0.89453125f;
    return v * p;
#endif
}

// Saturating E4M3FN round-to-nearest-even, including exponent carries.
// Preserve NaNs instead of silently treating corrupt data as a finite value.
// On gfx12 the hardware E4M3 conversion (cvt_pk_fp8_f32 after an fmed3f
// clamp to +/-448) is bit-identical to the software recipe for every
// finite input (verified: all 256 codes, all RNE midpoints, denormal scan
// and 4M random values, hip/tests/test_e4m3_hw.hip); the software path
// stays as the non-finite fallback.
__host__ __device__ inline u8 e4m3_byte(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    // Branchless for finite + Inf: the +/-448 clamp maps +/-Inf to the finite max, so
    // the hardware cvt is bit-identical to the software recipe for finite and Inf
    // inputs (verified in test_e4m3_hw.hip). NaN is selected to the E4M3 NaN code via
    // a cndmask (no divergent branch); production values are finite so the select is
    // uniform. This drops the old finiteness branch + the dead software path per call.
    u32 bits = as_u32(v);
    u8 hw = u8(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_amdgcn_fmed3f(v, 448.f, -448.f), 0.f, 0, false));
    return ((bits & 0x7fffffffu) > 0x7f800000u) ? u8(0x7f | (bits & 0x80u)) : hw;
#else
    u32 b = as_u32(v), a = b & 0x7fffffffu;
    u8 sg = u8((b >> 24) & 0x80u);
    if (a > 0x7f800000u)
        return u8(sg | 0x7f);
    if (a >= 0x43e00000u)
        return u8(sg | 0x7e);
    if (a < 0x3c800000u)
        return u8(sg | u8(nearbyintf(fabsf(v) * 512.f)));
    u32 rounded = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    u32 code = ((rounded >> 23) - 120u) * 8u + ((rounded >> 20) & 7u);
    return u8(sg | (code > 126u ? 126u : code));
#endif
}

// Fast E4M3 for verified-finite operands: drops the defensive NaN-sign select of
// e4m3_byte (4 instructions/call). Bit-identical to e4m3_byte for every finite
// input (same fmed3f clamp + cvt_pk_fp8_f32, verified in test_e4m3_hw.hip); on
// NaN input it returns the hardware NaN code 0x7f instead of 0x7f|sign. Use only
// where all production operands are finite (C32/ViT paths, covered by
// test_graph_vit + test_raw_pipeline).
__host__ __device__ inline u8 e4m3_byte_fast(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return u8(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_amdgcn_fmed3f(v, 448.f, -448.f), 0.f, 0, false));
#else
    return e4m3_byte(v);
#endif
}

__host__ __device__ inline float from_e4m3(u8 b) {
    u32 e = (b >> 3) & 15u, m = b & 7u;
    if (e == 15u && m == 7u)
        return as_f32(0x7fc00000u | (u32(b & 0x80u) << 24));
    float v = e ? as_f32(((e + 120u) << 23) | (m << 20)) : float(m) * 0.001953125f;
    return (b & 0x80u) ? -v : v;
}

__host__ __device__ inline u32 pcg(u32 s) {
    u32 w = ((s >> ((s >> 28) + 4)) ^ s) * 0x108ef2d9u;
    return (w >> 22) ^ w;
}
__host__ __device__ inline float uniform24(u32 s) {
    u32 w = ((s >> ((s >> 28) + 4)) ^ s) * 0x108ef2d9u;
    return float(((w >> 30) ^ (w >> 8)) + 1) * 5.9604644775390625e-8f;
}

// 512-byte B-tile packing: [K=32][N=16] row-major from row-major [N][K].
inline void pack_tiled_e4m3(u8* dst, const float* src, size_t N, size_t K) {
    for (size_t t = 0; t < N / 16; t++)
        for (size_t g = 0; g < K / 32; g++)
            for (size_t k = 0; k < 32; k++)
                for (size_t j = 0; j < 16; j++)
                    dst[(t * (K / 32) + g) * 512 + k * 16 + j] =
                        e4m3_byte(src[(t * 16 + j) * K + g * 32 + k]);
}

// Within each 512-byte tile, pack the 8 K-bytes of one B fragment contiguously.
// Same bytes the C256 tiled gather reads at stride 16 (k = half*16+group*8+e).
inline void permute_tiled_tiles_to_frag(u8* bytes, size_t nbytes) {
    for (size_t base = 0; base < nbytes; base += 512) {
        u8 tmp[512];
        for (int half = 0; half < 2; ++half)
            for (int group = 0; group < 2; ++group)
                for (int j = 0; j < 16; ++j)
                    for (int e = 0; e < 8; ++e) {
                        int k = half * 16 + group * 8 + e;
                        tmp[((half * 2 + group) * 16 + j) * 8 + e] = bytes[base + k * 16 + j];
                    }
        for (int i = 0; i < 512; ++i)
            bytes[base + i] = tmp[i];
    }
}

} // namespace dlss5
