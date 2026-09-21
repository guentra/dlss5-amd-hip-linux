// HIP port of ViT wave kernels (640 tokens × 1024, tiled E4M3).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// Generic tiled E4M3 GEMM: [M x K] @ [N x K]^T-as-B-tiles → [M x N], 16×16 output tiles.
__global__ void k_gemm_tiled_f8(const u8* A, const u8* Btiles, u8* C, uint M, uint N, uint K,
                                int activate) {
    using namespace linalg;
    uint m0 = blockIdx.x * 16u;
    uint n0 = blockIdx.y * 16u;
    if (m0 >= M || n0 >= N)
        return;
    C32 acc = C32::Splat(0.f);
    for (uint g = 0; g < K / 32; g++) {
        A8 a = A8::Load(A, m0 * K + g * 32u, K);
        uint boff = ((n0 / 16u) * (K / 32u) + g) * 512u;
        B8 b = B8::LoadRow(Btiles, boff, 16);
        acc.MultiplyAccumulate(a, b);
    }
    if (activate) {
        for (uint i = 0; i < acc.Length(); i++)
            acc.Set(i, Activate(acc.Get(i)));
    } else {
        for (uint i = 0; i < acc.Length(); i++)
            acc.Set(i, F(H(acc.Get(i))));
    }
    acc.StoreF8(C, m0 * N + n0, N);
}

// Fused ViT FFN: expand 1024->4096 (E4M3) + ActivatePoly, contract 4096->1024.
// 16 tokens/CTA, 16 waves (512 threads); each wave owns 64 hidden columns per
// quarter (expand) and 64 output columns (contract). The 16x1024 hidden quarter
// stays in LDS (single buffer, 4 quarters in ascending K order) and the E4M3
// staging of the f32 input is done once and reused by every quarter.
// Bit-exact with the two-launch flow:
//   expand   = k_linear_f32<false,true> mode=2 ordered=0 (K=1024 ascending)
//   contract = k_linear_f32<true,false>  mode=1 ordered=2
//              (total = H(residual*scale) + four K=1024 partitions, f32 adds)
__global__ void k_vit_ffn_fused(const float* input, const u8* w_expand, const u8* w_contract,
                                const float* scales, float* output) {
    using namespace linalg;
    constexpr uint CH = 1024u;
    constexpr uint HSTRIDE = 1028u; // 257 words/row: conflict-free C32 stores/loads
    const uint tid = threadIdx.x;
    const uint wave = tid / 32u;
    const uint first = blockIdx.x * 16u;
    __shared__ u8 Ast[16 * HSTRIDE];
    __shared__ u8 hid[16 * HSTRIDE];
    {
        // f32 input -> E4M3 bytes: the same per-byte conversion as the k_linear_f32
        // tile staging (e4m3_byte on the raw f32), so the MMA operands are identical.
        const uint row = tid / 32u;
        const uint k0 = (tid % 32u) * 32u;
        for (uint i = 0; i < 8u; i++) {
            const uint kk = k0 + i * 4u;
            const float4 v = *reinterpret_cast<const float4*>(input + (first + row) * CH + kk);
            *reinterpret_cast<u32*>(Ast + row * HSTRIDE + kk) =
                u32(e4m3_byte_fast(v.x)) | u32(e4m3_byte_fast(v.y)) << 8 |
                u32(e4m3_byte_fast(v.z)) << 16 | u32(e4m3_byte_fast(v.w)) << 24;
        }
    }
    __syncthreads();
    // Contract running total: the ordered==2 residual init, H(residual * scale).
    C32 total[4];
    for (uint n = 0; n < 4u; n++) {
        total[n] = C32::Splat(0.f);
        const uint col = wave * 64u + n * 16u;
        for (uint i = 0; i < total[n].Length(); i++) {
            uint2 rc = total[n].GetCoordinate(i);
            total[n].Set(i, H(input[(first + rc.x) * CH + col + rc.y] * scales[col + rc.y]));
        }
    }
    for (uint q = 0; q < 4u; q++) {
        if (q)
            __syncthreads(); // quarter q-1 contract reads must finish before reuse
        // Expand this wave's 64 hidden columns of quarter q (full K=1024, ascending).
        C32 acc[4];
        for (uint n = 0; n < 4u; n++)
            acc[n] = C32::Splat(0.f);
        for (uint g = 0; g < 32u; g++) {
            A8 a = A8::Load(Ast, g * 32u, HSTRIDE);
            for (uint n = 0; n < 4u; n++) {
                const uint col = wave * 64u + n * 16u;
                B8 b = B8::LoadRow(w_expand, ((q * 64u + (col >> 4u)) * 32u + g) * 512u, 16);
                acc[n].MultiplyAccumulate(a, b);
            }
        }
        // mode==2 epilogue per byte: e4m3_byte(ActivatePoly(v)), into the hidden quarter.
        for (uint n = 0; n < 4u; n++) {
            const uint col = wave * 64u + n * 16u;
            for (uint i = 0; i < acc[n].Length(); i++) {
                uint2 rc = acc[n].GetCoordinate(i);
                hid[rc.x * HSTRIDE + col + rc.y] = e4m3_byte_fast(ActivatePoly(acc[n].Get(i)));
            }
        }
        __syncthreads();
        // Contract this wave's 64 output columns over the quarter's 1024 hidden columns.
        C32 part[4];
        for (uint n = 0; n < 4u; n++)
            part[n] = C32::Splat(0.f);
        for (uint g = 0; g < 32u; g++) {
            A8 a = A8::Load(hid, g * 32u, HSTRIDE);
            for (uint n = 0; n < 4u; n++) {
                const uint col = wave * 64u + n * 16u;
                B8 b = B8::LoadRow(w_contract, (((col >> 4u) * 128u + q * 32u + g) * 512u), 16);
                part[n].MultiplyAccumulate(a, b);
            }
        }
        // ordered==2 partition combine: plain f32 adds, ascending quarter order.
        for (uint n = 0; n < 4u; n++)
            for (uint i = 0; i < part[n].Length(); i++)
                total[n].Set(i, total[n].Get(i) + part[n].Get(i));
    }
    // mode==1 epilogue: from_e4m3(e4m3_byte(H(v))), f32 store (640x1024 row-major).
    for (uint n = 0; n < 4u; n++) {
        const uint col = wave * 64u + n * 16u;
        for (uint i = 0; i < total[n].Length(); i++) {
            uint2 rc = total[n].GetCoordinate(i);
            output[(first + rc.x) * CH + col + rc.y] = from_e4m3(e4m3_byte_fast(H(total[n].Get(i))));
        }
    }
}

// One quarter of the fused FFN. blockIdx.y is the quarter 0..3. The contract
// partial is the f32 accumulator only; the residual and the quarter adds stay
// in k_vit_ffn_reduce so the order remains H(x*scale) + q0 + q1 + q2 + q3.
__global__ void k_vit_ffn_quarter(const float* input, const u8* w_expand, const u8* w_contract,
                                  float* partial, uint tokens) {
    using namespace linalg;
    constexpr uint CH = 1024u;
    constexpr uint HSTRIDE = 1028u;
    const uint tid = threadIdx.x;
    const uint wave = tid / 32u;
    const uint first = blockIdx.x * 16u;
    const uint q = blockIdx.y;
    __shared__ u8 Ast[16 * HSTRIDE];
    __shared__ u8 hid[16 * HSTRIDE];
    {
        const uint row = tid / 32u;
        const uint k0 = (tid % 32u) * 32u;
        for (uint i = 0; i < 8u; i++) {
            const uint kk = k0 + i * 4u;
            const float4 v = *reinterpret_cast<const float4*>(input + (first + row) * CH + kk);
            *reinterpret_cast<u32*>(Ast + row * HSTRIDE + kk) =
                u32(e4m3_byte_fast(v.x)) | u32(e4m3_byte_fast(v.y)) << 8 |
                u32(e4m3_byte_fast(v.z)) << 16 | u32(e4m3_byte_fast(v.w)) << 24;
        }
    }
    __syncthreads();
    C32 acc[4];
    for (uint n = 0; n < 4u; n++)
        acc[n] = C32::Splat(0.f);
    for (uint g = 0; g < 32u; g++) {
        A8 a = A8::Load(Ast, g * 32u, HSTRIDE);
        for (uint n = 0; n < 4u; n++) {
            const uint col = wave * 64u + n * 16u;
            B8 b = B8::LoadRow(w_expand, ((q * 64u + (col >> 4u)) * 32u + g) * 512u, 16);
            acc[n].MultiplyAccumulate(a, b);
        }
    }
    for (uint n = 0; n < 4u; n++) {
        const uint col = wave * 64u + n * 16u;
        for (uint i = 0; i < acc[n].Length(); i++) {
            uint2 rc = acc[n].GetCoordinate(i);
            hid[rc.x * HSTRIDE + col + rc.y] = e4m3_byte_fast(ActivatePoly(acc[n].Get(i)));
        }
    }
    __syncthreads();
    C32 part[4];
    for (uint n = 0; n < 4u; n++)
        part[n] = C32::Splat(0.f);
    for (uint g = 0; g < 32u; g++) {
        A8 a = A8::Load(hid, g * 32u, HSTRIDE);
        for (uint n = 0; n < 4u; n++) {
            const uint col = wave * 64u + n * 16u;
            B8 b = B8::LoadRow(w_contract, (((col >> 4u) * 128u + q * 32u + g) * 512u), 16);
            part[n].MultiplyAccumulate(a, b);
        }
    }
    const uint stride = tokens * CH;
    for (uint n = 0; n < 4u; n++) {
        const uint col = wave * 64u + n * 16u;
        for (uint i = 0; i < part[n].Length(); i++) {
            uint2 rc = part[n].GetCoordinate(i);
            partial[(q * stride) + (first + rc.x) * CH + col + rc.y] = part[n].Get(i);
        }
    }
}

__global__ void k_vit_ffn_reduce(const float* input, const float* scales, const float* partial,
                                 float* output, uint tokens) {
    constexpr uint CH = 1024u;
    uint i4 = blockIdx.x * blockDim.x + threadIdx.x;
    uint n4 = (tokens * CH) >> 2;
    if (i4 >= n4)
        return;
    uint elem = i4 << 2;
    uint ch = elem % CH;
    float4 x = *reinterpret_cast<const float4*>(input + elem);
    float4 s = *reinterpret_cast<const float4*>(scales + ch);
    float4 v;
    v.x = H(x.x * s.x);
    v.y = H(x.y * s.y);
    v.z = H(x.z * s.z);
    v.w = H(x.w * s.w);
    const uint n = tokens * CH;
    for (uint q = 0; q < 4u; q++) {
        float4 p = *reinterpret_cast<const float4*>(partial + q * n + elem);
        v.x += p.x;
        v.y += p.y;
        v.z += p.z;
        v.w += p.w;
    }
    float4 o;
    o.x = from_e4m3(e4m3_byte_fast(H(v.x)));
    o.y = from_e4m3(e4m3_byte_fast(H(v.y)));
    o.z = from_e4m3(e4m3_byte_fast(H(v.z)));
    o.w = from_e4m3(e4m3_byte_fast(H(v.w)));
    *reinterpret_cast<float4*>(output + elem) = o;
}

__global__ void k_vit_gather(const u8* src, const int* map, u8* dst, uint tokens, uint src_c,
                              uint dst_c) {
    uint t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens * dst_c)
        return;
    uint ch = t % dst_c, tok = t / dst_c;
    int idx = map[tok * dst_c + ch];
    dst[t] = idx < 0 ? 0 : src[uint(idx)];
}

} // namespace dlss5
