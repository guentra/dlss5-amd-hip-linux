// Fused QKV GEMM + per-head scaling + E4M3 quantization.
// Replaces k_linear_f32 (mode 3, ordered 0) followed by k_normalize_qkv for the
// window blocks: the f32 QKV tensor never touches global memory.
//
// Numerics are bit-identical to the split pair: the GEMM accumulation order is
// the same per-wave K loop, the per-row square sum keeps its strict channel
// order (one thread per row-head), and the E4M3 quantization sees the same
// f32 values (mode 3 applies no rounding between the two passes).
//
// Grid: (m/16, 3C/(32*NH)). A block covers NH heads (32*NH columns) with
// 2*NH waves; each wave owns one 16-column tile. One 32-channel head spans
// two adjacent waves.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// Per-channel squares exactly as the split k_normalize_qkv computes them, without
// the volatile private-memory round trip. v_fma_mixlo_f16 with f32 operands
// multiplies at f32 precision and RNE-rounds the result to f16 — the bit-exact
// H(v*v) the split kernel compiles to (f16(v^2), not f16(f16(v)^2)). The opaque
// asm also blocks FMA contraction of the following accumulate.
__device__ inline float sq_half_f32(float v) {
    float t;
    asm("v_fma_mixlo_f16 %0, %1, %1, 0" : "=v"(t) : "v"(v));
    return __half2float(__ushort_as_half((unsigned short)(unsigned)__builtin_bit_cast(unsigned, t)));
}
__device__ inline float sq_f32(float v) {
    float t;
    asm("v_mul_f32_e32 %0, %1, %1" : "=v"(t) : "v"(v));
    return t;
}

template <int C, int NH, bool F8IN = false, bool HIN = false, bool VONLY = false>
__global__ void k_qkv_norm_f32(const float* input, const u8* input_f8, const u8* weights,
                               u8* output, uint m, uint k, const float* scales, float qgain,
                               bool half_squares, const __half* input_h = nullptr,
                               uint head_base = 0u) {
    using namespace linalg;
    constexpr uint HEADS = C / 32u;
    constexpr uint COLS = 32u * NH;
    const uint first = blockIdx.x * 16u;
    const uint head0 = head_base + blockIdx.y * NH;
    const uint wave = threadIdx.x / 32u;
    // Single LDS buffer: the A-tile staging (512B) aliases the start of the
    // stage buffer. The tile is only read inside the K loop; stage is only
    // written after it (with a barrier between), so the aliasing is safe.
    // Cuts LDS 512+16*COLS*4 -> 16*COLS*4 (e.g. C32/NH1 2560B->2048B).
    // 66 (not 64) floats/row: the 16 ds_load_b128 norm reads stride 264 B (2 words
    // per row) so the 16 rows hit distinct LDS bank sets instead of 8-way conflicts.
    constexpr uint STAGE_COLS = VONLY ? 128u : (C == 32 && NH == 3) ? 66u : COLS;
    __shared__ float stage[16 * STAGE_COLS];
    u8* tile = reinterpret_cast<u8*>(stage);
    C32 acc = C32::Splat(0.f);
    const uint col = head0 * 32u + wave * 16u;
    // blockDim = 2*NH*32, 8 threads per input row (float4). C32 uses NH=3 so one
    // CTA computes Q+K+V and reads the input tile once; then 192 threads exist but
    // only 16*8 load, one row each. NH=1 still needs two rows per thread.
    constexpr uint LOADERS = 16u * 8u;
    constexpr uint THREADS = 2u * NH * 32u;
    constexpr uint ROWS_PER_THREAD = (LOADERS + THREADS - 1u) / THREADS;
    for (uint g = 0; g < k / 32; ++g) {
        if (threadIdx.x < LOADERS) {
            const uint c4 = (threadIdx.x % 8u) * 4u;
            const uint base = threadIdx.x / 8u;
            for (uint rr = 0; rr < ROWS_PER_THREAD; ++rr) {
                const uint rlocal = base * ROWS_PER_THREAD + rr;
                const uint row = first + rlocal;
                if (F8IN) {
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + c4) =
                        row < m
                            ? *reinterpret_cast<const u32*>(input_f8 + row * k + g * 32u + c4)
                            : 0u;
                } else if (HIN) {
                    const __half2* hp = reinterpret_cast<const __half2*>(
                        input_h + row * k + g * 32u + c4);
                    __half2 h0 = row < m ? hp[0] : __half2(0.f, 0.f);
                    __half2 h1 = row < m ? hp[1] : __half2(0.f, 0.f);
                    float x = __half2float(__low2half(h0));
                    float y = __half2float(__high2half(h0));
                    float z = __half2float(__low2half(h1));
                    float w = __half2float(__high2half(h1));
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + c4) =
                        u32(e4m3_byte_fast(x)) | u32(e4m3_byte_fast(y)) << 8 |
                        u32(e4m3_byte_fast(z)) << 16 | u32(e4m3_byte_fast(w)) << 24;
                } else {
                    const float4 v = row < m
                        ? *reinterpret_cast<const float4*>(input + row * k + g * 32u + c4)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + c4) =
                        u32(e4m3_byte_fast(v.x)) | u32(e4m3_byte_fast(v.y)) << 8 |
                        u32(e4m3_byte_fast(v.z)) << 16 | u32(e4m3_byte_fast(v.w)) << 24;
                }
            }
        }
        __syncthreads();
        A8 a = A8::Load(tile, 0, 32);
        B8 b = B8::LoadRow(weights, ((col / 16u) * (k / 32u) + g) * 512u, 16);
        acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    const uint norm_heads = head0 < 2u * HEADS
                                ? (NH < 2u * HEADS - head0 ? NH : 2u * HEADS - head0)
                                : 0u;
    const uint norm_waves = 2u * norm_heads;
    if (wave < norm_waves)
        acc.StoreF32(stage, wave * 16u * 4u, STAGE_COLS * 4u);
    __syncthreads();
    // 16 rows x normalized heads; one thread per row-head keeps the strict channel sum order.
    const uint item = threadIdx.x;
    const uint ROW_THREADS = 16 * norm_heads;
    if (item < ROW_THREADS) {
        const uint row = item / norm_heads;
        const uint headg = head0 + (item % norm_heads);
        const float* s0 = &stage[row * STAGE_COLS + (item % norm_heads) * 32u];
        const float* s1 = s0 + 16u;
        float inv = 1.f;
        float sum = 0.f;
        const float4* q0 = reinterpret_cast<const float4*>(s0);
        const float4* q1 = reinterpret_cast<const float4*>(s1);
        #pragma unroll
        for (uint q = 0; q < 4u; ++q) {
            float4 a = q0[q];
            sum += half_squares ? sq_half_f32(a.x) : sq_f32(a.x);
            sum += half_squares ? sq_half_f32(a.y) : sq_f32(a.y);
            sum += half_squares ? sq_half_f32(a.z) : sq_f32(a.z);
            sum += half_squares ? sq_half_f32(a.w) : sq_f32(a.w);
        }
        #pragma unroll
        for (uint q = 0; q < 4u; ++q) {
            float4 b = q1[q];
            sum += half_squares ? sq_half_f32(b.x) : sq_f32(b.x);
            sum += half_squares ? sq_half_f32(b.y) : sq_f32(b.y);
            sum += half_squares ? sq_half_f32(b.z) : sq_f32(b.z);
            sum += half_squares ? sq_half_f32(b.w) : sq_f32(b.w);
        }
        inv = rsqrtf(fmaxf(sum, 6.198883056640625e-5f));
        if (headg < HEADS)
            inv *= scales[headg] * qgain;
        u32 w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint ch = 0; ch < 16u; ++ch)
            w[ch >> 2] |= u32(e4m3_byte_fast(s0[ch] * inv)) << ((ch & 3u) * 8u);
        for (uint ch = 0; ch < 16u; ++ch)
            w[4u + (ch >> 2)] |= u32(e4m3_byte_fast(s1[ch] * inv)) << ((ch & 3u) * 8u);
        u8* out = output + (size_t)(first + row) * (3u * C) + headg * 32u;
        uint4* o4 = reinterpret_cast<uint4*>(out);
        o4[0] = make_uint4(w[0], w[1], w[2], w[3]);
        o4[1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
    if (wave >= norm_waves) {
        const uint headg = head0 + wave / 2u;
        u8* out = output + (size_t)first * (3u * C) + headg * 32u + (wave % 2u) * 16u;
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < m)
                out[(size_t)rc.x * (3u * C) + rc.y] = e4m3_byte_fast(acc.Get(i));
        }
    }
}

// Fused ViT QKV (f16 weights, raw f32 acc) + per-head RMS + E4M3.
// Bit-identical to launch_linear_half_out(..., 640, 3072, 1024, 1, 0, 2)
// followed by launch_normalize_qkv(..., 1024, 5.65625f): same K order, same
// f32→f16 input casts, same serial 32-channel square sum, same V path (no RMS).
__global__ void __launch_bounds__(64) k_vit_qkv_norm(const float* input, const __half* weights,
                                                     u8* output, uint m, const float* scales,
                                                     float qgain) {
    using namespace linalg;
    constexpr uint K = 1024u, N = 3072u, HEADS = 32u;
    const uint first = blockIdx.x * 16u;
    const uint head0 = blockIdx.y;
    const uint part = head0 / HEADS;
    const uint head = head0 % HEADS;
    const uint wave = threadIdx.x / 32u;
    const uint col = head0 * 32u + wave * 16u;
    __shared__ float stage[16 * 32];
    __half* tile = reinterpret_cast<__half*>(stage);
    C32 acc = C32::Splat(0.f);
    for (uint g = 0; g < K / 32u; ++g) {
        const uint c4 = (threadIdx.x % 8u) * 4u;
        for (uint rlocal = threadIdx.x / 8u; rlocal < 16u; rlocal += blockDim.x / 8u) {
            const uint row = first + rlocal;
            const float4 v = row < m
                ? *reinterpret_cast<const float4*>(input + row * K + g * 32u + c4)
                : make_float4(0.f, 0.f, 0.f, 0.f);
            const __half h0 = __float2half(v.x), h1 = __float2half(v.y);
            const __half h2 = __float2half(v.z), h3 = __float2half(v.w);
            *reinterpret_cast<uint2*>(tile + rlocal * 32u + c4) =
                uint2(u32(__half_as_ushort(h0)) | u32(__half_as_ushort(h1)) << 16,
                      u32(__half_as_ushort(h2)) | u32(__half_as_ushort(h3)) << 16);
        }
        __syncthreads();
        A16 a = A16::Load(tile, 0, 64);
        B16 b = B16::LoadRowMajorOut(weights, col, g * 32u, K);
        acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    if (part < 2u) {
        acc.StoreF32(stage, wave * 16u * 4u, 32u * 4u);
        __syncthreads();
        if (threadIdx.x < 16u) {
            const uint row = threadIdx.x;
            const float* s0 = &stage[row * 32u];
            const float* s1 = s0 + 16u;
            float sum = 0.f;
            const float4* q0 = reinterpret_cast<const float4*>(s0);
            const float4* q1 = reinterpret_cast<const float4*>(s1);
#pragma unroll
            for (uint q = 0; q < 4u; ++q) {
                float4 a = q0[q];
                sum += sq_half_f32(a.x);
                sum += sq_half_f32(a.y);
                sum += sq_half_f32(a.z);
                sum += sq_half_f32(a.w);
            }
#pragma unroll
            for (uint q = 0; q < 4u; ++q) {
                float4 b = q1[q];
                sum += sq_half_f32(b.x);
                sum += sq_half_f32(b.y);
                sum += sq_half_f32(b.z);
                sum += sq_half_f32(b.w);
            }
            float inv = rsqrtf(fmaxf(sum, 6.198883056640625e-5f));
            if (part == 0u)
                inv *= scales[head] * qgain;
            u32 w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint ch = 0; ch < 16u; ++ch)
                w[ch >> 2] |= u32(e4m3_byte_fast(s0[ch] * inv)) << ((ch & 3u) * 8u);
            for (uint ch = 0; ch < 16u; ++ch)
                w[4u + (ch >> 2)] |= u32(e4m3_byte_fast(s1[ch] * inv)) << ((ch & 3u) * 8u);
            u8* out = output + (size_t)(first + row) * N + part * K + head * 32u;
            uint4* o4 = reinterpret_cast<uint4*>(out);
            o4[0] = make_uint4(w[0], w[1], w[2], w[3]);
            o4[1] = make_uint4(w[4], w[5], w[6], w[7]);
        }
    } else {
        u8* out = output + (size_t)first * N + part * K + head * 32u + wave * 16u;
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < m)
                out[(size_t)rc.x * N + rc.y] = e4m3_byte_fast(acc.Get(i));
        }
    }
}

} // namespace dlss5
