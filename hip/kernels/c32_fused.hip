#pragma once
#include "dlss5_linalg.hpp"
#include "qkv_norm.hip"
#include "attention.hip"

namespace dlss5 {

namespace {
__device__ inline void c32_sync_window() { __syncthreads(); }
__device__ inline void c32_wave_sync() { __syncthreads(); }
} // namespace

template <bool CHAIN, bool WINDOWED, bool WRITE_FFN, bool WRITE_OUT>
__global__ void __launch_bounds__(128) k_c32_fused(const __half* in_h, const u8* ffn_w,
                                                    const float* ffn_s, const u8* qkv_w,
                                                    const float* qscale, const float* bias,
                                                    const u8* proj_w, const float* proj_s,
                                                    __half* ffn_h, __half* out_h,
                                                    __half* out_h_raster, uint w, uint h, uint sw,
                                                    uint sh, uint px, uint py) {
    using i2 = int __attribute__((ext_vector_type(2)));
    using f8 = float __attribute__((ext_vector_type(8)));
    constexpr uint PACKED_BYTES = 3u * 64u * 36u;
    constexpr uint RAW_STRIDE = 36u;
    constexpr uint SCRATCH_BYTES = 64u * RAW_STRIDE * 4u;
    constexpr uint LDS_BYTES = PACKED_BYTES + SCRATCH_BYTES;
    constexpr uint HIDDEN_STRIDE = 132u;

    __shared__ u8 lds[LDS_BYTES];
    u8* packed = lds;
    u8* ffn8 = packed + 128u * 36u;
    u8* hidden = lds + PACKED_BYTES;
    float* raw = reinterpret_cast<float*>(hidden);
    __half* scores = reinterpret_cast<__half*>(hidden);
    // Probabilities alias scores after the row-sum (same pattern as k_window_attention).
    u8* p8 = reinterpret_cast<u8*>(scores);
    // f16 residual lives in the unused K/V packed bytes during FFN; V is only
    // written after a group barrier, once every wave has finished those reads.
    __half* in16 = reinterpret_cast<__half*>(packed + 64u * 36u);

    const uint window = blockIdx.x;
    const uint tid = threadIdx.x;
    const uint wave = tid / 32u;
    const uint lane = tid & 31u;
    const uint first = wave * 16u;
    const uint rc = lane & 15u;
    const uint gr = (lane >> 4) & 1u;
    const float fs0 = ffn_s[rc];
    const float fs1 = ffn_s[16u + rc];
    const float ps0 = proj_s[rc];
    const float ps1 = proj_s[16u + rc];
    const float q0s = qscale[0];

    uint2 hv[4];
    uint tok[4], c4s[4];
#pragma unroll
    for (uint it = 0; it < 4u; ++it) {
        const uint i4 = tid + it * 128u;
        tok[it] = i4 / 8u;
        c4s[it] = (i4 % 8u) * 4u;
        hv[it] = *reinterpret_cast<const uint2*>(in_h + (window * 64u + tok[it]) * 32u + c4s[it]);
    }
#pragma unroll
    for (uint it = 0; it < 4u; ++it) {
        const __half2* hp = reinterpret_cast<const __half2*>(&hv[it]);
        *reinterpret_cast<__half2*>(in16 + tok[it] * 34u + c4s[it]) = hp[0];
        *reinterpret_cast<__half2*>(in16 + tok[it] * 34u + c4s[it] + 2u) = hp[1];
        *reinterpret_cast<u32*>(packed + tok[it] * 36u + c4s[it]) =
            u32(e4m3_byte_fast(__half2float(__low2half(hp[0])))) |
            u32(e4m3_byte_fast(__half2float(__high2half(hp[0])))) << 8 |
            u32(e4m3_byte_fast(__half2float(__low2half(hp[1])))) << 16 |
            u32(e4m3_byte_fast(__half2float(__high2half(hp[1])))) << 24;
    }
    c32_sync_window();

    {
        using namespace linalg;
#pragma unroll
        for (uint col = 0; col < 128u; col += 16u) {
            C32 ex = C32::Splat(0.f);
            A8 a = A8::Load(packed + first * 36u, 0, 36);
            B8 b = B8::LoadRowMajorOut(ffn_w, col, 0, 32);
            ex.MultiplyAccumulate(a, b);
            const uint hcol = col + rc;
            const uint grow = (lane >= 16u ? 8u : 0u);
#pragma unroll
            for (uint i = 0; i < 8u; ++i)
                hidden[(first + grow + i) * HIDDEN_STRIDE + hcol] =
                    e4m3_byte_fast(ActivatePolyC32(ex.Get(i)));
        }
    }
    c32_sync_window();

    linalg::C32 acc0 = linalg::C32::Splat(0.f), acc1 = linalg::C32::Splat(0.f);
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        uint2 rc2 = acc0.GetCoordinate(i);
        float iv0 = __half2float(in16[(first + rc2.x) * 34u + rc2.y]);
        float iv1 = __half2float(in16[(first + rc2.x) * 34u + 16u + rc2.y]);
        if constexpr (CHAIN) {
            iv0 = from_e4m3(e4m3_byte_fast(iv0));
            iv1 = from_e4m3(e4m3_byte_fast(iv1));
        }
        acc0.Set(i, iv0 * fs0);
        acc1.Set(i, iv1 * fs1);
    }
#pragma unroll
    for (uint g = 0; g < 4u; ++g) {
        linalg::A8 a = linalg::A8::Load(hidden + first * HIDDEN_STRIDE, g * 32u, HIDDEN_STRIDE);
        linalg::B8 b0 = linalg::B8::LoadRowMajorOut(ffn_w + 4096u, 0, g * 32u, 128);
        linalg::B8 b1 = linalg::B8::LoadRowMajorOut(ffn_w + 4096u, 16, g * 32u, 128);
        acc0.MultiplyAccumulate(a, b0);
        acc1.MultiplyAccumulate(a, b1);
    }
    c32_sync_window();

    __half saved0[8], saved1[8];
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        uint2 rc2 = acc0.GetCoordinate(i);
        uint row = first + rc2.x;
        saved0[i] = __float2half_rn(acc0.Get(i));
        ffn8[row * 36u + rc2.y] = e4m3_byte_fast(__half2float(saved0[i]));
        if constexpr (WRITE_FFN)
            ffn_h[(window * 64u + row) * 32u + rc2.y] = saved0[i];
        uint2 rc3 = acc1.GetCoordinate(i);
        saved1[i] = __float2half_rn(acc1.Get(i));
        ffn8[row * 36u + 16u + rc3.y] = e4m3_byte_fast(__half2float(saved1[i]));
        if constexpr (WRITE_FFN)
            ffn_h[(window * 64u + row) * 32u + 16u + rc3.y] = saved1[i];
    }
    c32_sync_window();

    for (uint part = 0; part < 3u; ++part) {
        {
            using namespace linalg;
            A8 a = A8::Load(ffn8 + first * 36u, 0, 36);
            B8 b0 = B8::LoadRowMajorOut(qkv_w, part * 32u, 0, 32);
            B8 b1 = B8::LoadRowMajorOut(qkv_w, part * 32u + 16u, 0, 32);
            C32 q0 = C32::Splat(0.f), q1 = C32::Splat(0.f);
            q0.MultiplyAccumulate(a, b0);
            q1.MultiplyAccumulate(a, b1);
            for (uint i = 0; i < q0.Length(); ++i) {
                uint2 rc2 = q0.GetCoordinate(i);
                raw[(first + rc2.x) * RAW_STRIDE + part * 32u + rc2.y] = q0.Get(i);
                uint2 rc3 = q1.GetCoordinate(i);
                raw[(first + rc3.x) * RAW_STRIDE + part * 32u + 16u + rc3.y] = q1.Get(i);
            }
        }
        c32_sync_window();
        if (part < 2u) {
            if (lane < 16u) {
                const uint row = first + lane;
                const float* s0 = raw + row * RAW_STRIDE + part * 32u;
                const float* s1 = s0 + 16u;
                float sum = 0.f;
                const float4* q0f = reinterpret_cast<const float4*>(s0);
                const float4* q1f = reinterpret_cast<const float4*>(s1);
#pragma unroll
                for (uint q = 0; q < 4u; ++q) {
                    float4 v = q0f[q];
                    sum += sq_half_f32(v.x);
                    sum += sq_half_f32(v.y);
                    sum += sq_half_f32(v.z);
                    sum += sq_half_f32(v.w);
                }
#pragma unroll
                for (uint q = 0; q < 4u; ++q) {
                    float4 v = q1f[q];
                    sum += sq_half_f32(v.x);
                    sum += sq_half_f32(v.y);
                    sum += sq_half_f32(v.z);
                    sum += sq_half_f32(v.w);
                }
                float inv = rsqrtf(fmaxf(sum, 6.198883056640625e-5f));
                if (part == 0u)
                    inv *= q0s;
                u8* dst = packed + (part * 64u + row) * 36u;
                for (uint ch = 0; ch < 32u; ++ch)
                    dst[ch] = e4m3_byte_fast(s0[ch] * inv);
            }
        } else {
            for (uint col = 0; col < 32u; col += 16u)
                for (uint e = 0; e < 8u; ++e) {
                    const uint row = first + gr * 8u + e;
                    const uint c = col + rc;
                    packed[(128u + row) * 36u + c] = e4m3_byte_fast(raw[row * RAW_STRIDE + part * 32u + c]);
                }
        }
        c32_sync_window();
    }
    c32_sync_window();

    {
        using namespace linalg;
        A8 q = A8::Load(packed + first * 36u, 0, 36);
#pragma unroll
        for (uint k0 = 0; k0 < 64u; k0 += 16u) {
            B8 k = B8::LoadCol(packed, (64u + k0) * 36u, 36u);
            C32 s = Multiply(q, k);
#pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                uint2 rc2 = s.GetCoordinate(i);
                uint qi = first + rc2.x, kj = k0 + rc2.y;
                scores[qi * 68u + kj] =
                    __float2half_rn(fast_exp_raw(s.Get(i) + bias[qi * 64u + kj]));
            }
        }
    }
    c32_sync_window();

    uint4 p8r[4];
    if (lane < 16u) {
        const uint qi = first + lane;
        float sum = 0.f;
#pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            uint2 v = *reinterpret_cast<const uint2*>(&scores[qi * 68u + j * 4u]);
            const __half2* hp = reinterpret_cast<const __half2*>(&v);
            sum += __half2float(__low2half(hp[0]));
            sum += __half2float(__high2half(hp[0]));
            sum += __half2float(__low2half(hp[1]));
            sum += __half2float(__high2half(hp[1]));
        }
        const float inv = norm_inverse(fmaxf(sum, 1e-20f));
#pragma unroll
        for (uint q4 = 0; q4 < 4u; ++q4) {
            uint4 v;
            u8* b = reinterpret_cast<u8*>(&v);
#pragma unroll
            for (uint t = 0; t < 16u; ++t)
                b[t] = e4m3_byte_fast(__half2float(scores[qi * 68u + q4 * 16u + t]) * inv);
            p8r[q4] = v;
        }
    }
    c32_sync_window();
    if (lane < 16u) {
        const uint qi = first + lane;
        uint4* row = reinterpret_cast<uint4*>(p8 + qi * 64u);
#pragma unroll
        for (uint q4 = 0; q4 < 4u; ++q4)
            row[q4] = p8r[q4];
    }
    c32_sync_window();

    {
        using namespace linalg;
        C32 rav0 = C32::Splat(0.f), rav1 = C32::Splat(0.f);
#pragma unroll
        for (uint k0 = 0; k0 < 64u; k0 += 32u) {
            A8 p = A8::Load(p8 + first * 64u, k0, 64);
            B8 v0 = B8::LoadRow(packed + (128u + k0) * 36u, 0, 36);
            B8 v1 = B8::LoadRow(packed + (128u + k0) * 36u, 16u, 36);
            rav0.MultiplyAccumulate(p, v0);
            rav1.MultiplyAccumulate(p, v1);
        }
        for (uint i = 0; i < rav0.Length(); ++i) {
            uint2 rc2 = rav0.GetCoordinate(i);
            packed[(first + rc2.x) * 36u + rc2.y] = e4m3_byte_fast(rav0.Get(i));
            uint2 rc3 = rav1.GetCoordinate(i);
            packed[(first + rc3.x) * 36u + 16u + rc3.y] = e4m3_byte_fast(rav1.Get(i));
        }
    }
    c32_sync_window();

    {
        using namespace linalg;
        A8 a = A8::Load(packed + first * 36u, 0, 36);
        B8 b0 = B8::LoadRowMajorOut(proj_w, 0, 0, 32);
        B8 b1 = B8::LoadRowMajorOut(proj_w, 16, 0, 32);
        C32 p0 = C32::Splat(0.f), p1 = C32::Splat(0.f);
        p0.MultiplyAccumulate(a, b0);
        p1.MultiplyAccumulate(a, b1);
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            uint2 rc2 = p0.GetCoordinate(i);
            uint row = first + rc2.x;
            float v = p0.Get(i) + __half2float(saved0[i]) * ps0;
            __half hv = __float2half_rn(v);
            if constexpr (WRITE_OUT)
                out_h[(window * 64u + row) * 32u + rc2.y] = hv;
            if constexpr (WINDOWED) {
                uint x = (window % (sw / 8u)) * 8u + (row % 8u);
                uint y = (window / (sw / 8u)) * 8u + (row / 8u);
                if (x >= px && y >= py && x < w + px && y < h + py)
                    out_h_raster[((y - py) * w + (x - px)) * 32u + rc2.y] = hv;
            } else
                out_h_raster[(window * 64u + row) * 32u + rc2.y] = hv;
        }
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            uint2 rc2 = p1.GetCoordinate(i);
            uint row = first + rc2.x;
            float v = p1.Get(i) + __half2float(saved1[i]) * ps1;
            __half hv = __float2half_rn(v);
            if constexpr (WRITE_OUT)
                out_h[(window * 64u + row) * 32u + 16u + rc2.y] = hv;
            if constexpr (WINDOWED) {
                uint x = (window % (sw / 8u)) * 8u + (row % 8u);
                uint y = (window / (sw / 8u)) * 8u + (row / 8u);
                if (x >= px && y >= py && x < w + px && y < h + py)
                    out_h_raster[((y - py) * w + (x - px)) * 32u + 16u + rc2.y] = hv;
            } else
                out_h_raster[(window * 64u + row) * 32u + 16u + rc2.y] = hv;
        }
    }
}

} // namespace dlss5
