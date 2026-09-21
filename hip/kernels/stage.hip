// Learned DS, skip-add, C512 split FFWD, ViT global attention, post70 merge.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// 2×2 average pool then Cin→Cout projection (E4M3 weights, row-major [Cout][Cin] tiled B).
__global__ void k_ds_project(const u8* src, const u8* w, u8* dst, uint width, uint height,
                             uint cin, uint cout) {
    uint p = blockIdx.x * blockDim.x + threadIdx.x;
    uint ow = width / 2, oh = height / 2;
    if (p >= ow * oh * cout)
        return;
    uint oc = p % cout, t = p / cout, x = t % ow, y = t / ow;
    float acc = 0.f;
    for (uint c = 0; c < cin; c++) {
        float s = 0.f;
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
                s += from_e4m3(src[((y * 2 + dy) * width + (x * 2 + dx)) * cin + c]);
        acc += (s * 0.25f) * from_e4m3(w[oc * cin + c]);
    }
    dst[p] = e4m3_byte(H(acc));
}

__global__ void k_add_f8(const u8* a, const u8* b, u8* dst, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = e4m3_byte(H(from_e4m3(a[i]) + from_e4m3(b[i])));
}

// Upsample 2× then add skip (same channel count).
__global__ void k_upsample_add_f8(const u8* lo, const u8* skip, u8* dst, uint w, uint h, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint dw = w * 2, dh = h * 2;
    if (i >= dw * dh * c)
        return;
    uint ch = i % c, t = i / c, x = t % dw, y = t / dw;
    float v = from_e4m3(lo[((y / 2) * w + (x / 2)) * c + ch]);
    v += from_e4m3(skip[i]);
    dst[i] = e4m3_byte(H(v));
}

// Decoder linear: out[t] = F(H(in[t] @ W[cout,cin] + skip[t])).
__global__ void k_linear_skip(const u8* in, const u8* w, const u8* skip, u8* out, uint tokens,
                              uint cin, uint cout) {
    uint p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= tokens * cout)
        return;
    uint oc = p % cout, t = p / cout;
    float acc = 0.f;
    for (uint k = 0; k < cin; k++)
        acc += from_e4m3(in[t * cin + k]) * from_e4m3(w[oc * cin + k]);
    if (skip)
        acc += from_e4m3(skip[t * cout + oc]);
    out[p] = e4m3_byte(H(acc));
}

// C512 split FFWD, 16 tokens/block. pre/expand/contract are 16×32 B-tiles (pack_tiled_e4m3).
__global__ void k_split_ffwd(const u8* in, const u8* pre, const u8* expand, const u8* contract,
                             u8* out, uint tokens) {
    using namespace linalg;
    const uint first = blockIdx.x * 16u;
    if (first >= tokens)
        return;
    __shared__ u8 mix[16 * 512];
    __shared__ u8 hid[16 * 256];
    // pre: 16×512 @ 512×512
    for (uint n0 = 0; n0 < 512; n0 += 16) {
        C32 acc = C32::Splat(0.f);
        for (uint g = 0; g < 512 / 32; g++) {
            A8 a = A8::Load(in, first * 512u + g * 32u, 512);
            uint boff = ((n0 / 16u) * (512u / 32u) + g) * 512u;
            B8 b = B8::LoadRow(pre, boff, 16);
            acc.MultiplyAccumulate(a, b);
        }
        for (uint i = 0; i < acc.Length(); i++)
            acc.Set(i, F(H(acc.Get(i))));
        acc.StoreF8(mix, n0, 512);
    }
    __syncthreads();
    for (uint grp = 0; grp < 8; grp++) {
        for (uint n0 = 0; n0 < 256; n0 += 16) {
            C32 acc = C32::Splat(0.f);
            for (uint g = 0; g < 64 / 32; g++) {
                A8 a = A8::Load(mix, grp * 64u + g * 32u, 512);
                uint boff = (grp * (256u / 16u) * (64u / 32u) + (n0 / 16u) * (64u / 32u) + g) * 512u;
                B8 b = B8::LoadRow(expand, boff, 16);
                acc.MultiplyAccumulate(a, b);
            }
            for (uint i = 0; i < acc.Length(); i++)
                acc.Set(i, Activate(acc.Get(i)));
            acc.StoreF8(hid, n0, 256);
        }
        __syncthreads();
        for (uint n0 = 0; n0 < 64; n0 += 16) {
            C32 acc = C32::Splat(0.f);
            for (uint g = 0; g < 256 / 32; g++) {
                A8 a = A8::Load(hid, g * 32u, 256);
                uint boff = (grp * (64u / 16u) * (256u / 32u) + (n0 / 16u) * (256u / 32u) + g) * 512u;
                B8 b = B8::LoadRow(contract, boff, 16);
                acc.MultiplyAccumulate(a, b);
            }
            for (uint i = 0; i < acc.Length(); i++)
                acc.Set(i, F(H(acc.Get(i))));
            acc.StoreF8(out + first * 512u + grp * 64u, n0, 512);
        }
        __syncthreads();
    }
}

// Stage one head-row of Q, K or V as E4M3 bytes. Out-of-range keys store 0
// so a partial 32-key tail (tokens % 32 != 0) matches the padded MMA.
template <bool F8IN>
__device__ void vit_stage_e4m3(u8* tile, uint rows, const float* qkv, const u8* qkv_f8,
                                uint token0, uint tokens, uint stride, uint col_base) {
    const uint col = (threadIdx.x % 8u) * 4u;
    for (uint r = threadIdx.x / 8u; r < rows; r += blockDim.x / 8u) {
        const uint tok = token0 + r;
        u32 packed = 0u;
        if (tok < tokens) {
            if constexpr (F8IN) {
                packed = *reinterpret_cast<const u32*>(qkv_f8 + tok * stride + col_base + col);
            } else {
                const float4 v = *reinterpret_cast<const float4*>(qkv + tok * stride + col_base + col);
                packed = u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                         u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
        *reinterpret_cast<u32*>(tile + r * 32u + col) = packed;
    }
}

// HLSL native_wave_vit_attention_fp8: bit-map exponents, unquantized
// denominator, FP8 exponents for PV, normalize only AFTER the value product.
// qkv is logical interleaved [token][3*dim], already normalized to the FP8 grid.
//
// The split kernel keeps every key's probability byte live (16*656 B) and is
// the DLSS5_VIT_ATTN_SPLIT=1 fallback. The default kernel scores one 32-key
// tile, adds that tile into the strict f32 denominator (key order 0..tokens-1,
// 1.f/denom — not norm_inverse), then multiplies it into PV before the next tile.
template <bool F8IN = false>
__global__ void k_vit_attention(const float* qkv, const u8* qkv_f8, u8* out, uint tokens,
                                uint dim) {
    using namespace linalg;
    const uint q0 = blockIdx.x * 16u, head = blockIdx.y;
    const uint stride = 3u * dim;
    __shared__ u8 qtile[16 * 32], ktile[16 * 32], values[32 * 32];
    // Row strides are padded so the 16 query rows do not alias one LDS bank
    // set (640/4 = 160 words = 0 mod 32 caused 8-way conflicts on every store).
    __shared__ u8 probabilities[16 * 656];
    __shared__ float ex[16 * 33], denom[16];
    // 16B vector staging: same bytes as the scalar path, fewer LSU round trips.
    // F8IN consumes the already-normalized E4M3 QKV, so e4m3_byte() is the
    // identity and the f8 -> f32 round trip before attention is redundant.
    {
        const uint col = (threadIdx.x % 8u) * 4u;
        for (uint r = threadIdx.x / 8u; r < 16; r += blockDim.x / 8u) {
            if (F8IN) {
                *reinterpret_cast<u32*>(qtile + r * 32u + col) =
                    *reinterpret_cast<const u32*>(qkv_f8 + (q0 + r) * stride + head * 32u + col);
            } else {
                const float4 v = *reinterpret_cast<const float4*>(
                    qkv + (q0 + r) * stride + head * 32u + col);
                *reinterpret_cast<u32*>(qtile + r * 32u + col) =
                    u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                    u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
    }
    if (threadIdx.x < 16) denom[threadIdx.x] = 0.f;
    __syncthreads();
    A8 q = A8::Load(qtile, 0, 32);
    for (uint k0 = 0; k0 < tokens; k0 += 32) {
        for (uint half = 0; half < 2; ++half) {
            {
                const uint col = (threadIdx.x % 8u) * 4u;
                for (uint r = threadIdx.x / 8u; r < 16; r += blockDim.x / 8u) {
                    uint k = k0 + half * 16u + r;
                    if (F8IN) {
                        *reinterpret_cast<u32*>(ktile + r * 32u + col) =
                            k < tokens
                                ? *reinterpret_cast<const u32*>(
                                      qkv_f8 + k * stride + dim + head * 32u + col)
                                : 0u;
                    } else {
                        const float4 v = k < tokens
                            ? *reinterpret_cast<const float4*>(
                                  qkv + k * stride + dim + head * 32u + col)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
                        *reinterpret_cast<u32*>(ktile + r * 32u + col) =
                            u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                            u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
                    }
                }
            }
            __syncthreads();
            B8 k = B8::LoadCol(ktile, 0, 32);
            C32 score = Multiply(q, k);
            for (uint i = 0; i < score.Length(); ++i) {
                uint2 rc = score.GetCoordinate(i);
                float affine = fminf(fmaxf(fmaf(score.Get(i),
                    __half2float(__ushort_as_half(0x2dbb)), 1.708984375f), 1.439453125f),
                                     1.9775390625f);
                uint bits = __half_as_ushort(__float2half_rn(affine));
                float e = k0 + half * 16 + rc.y < tokens
                              ? __half2float(__ushort_as_half(uint16_t((bits << 4) + 0x4000u)))
                              : 0.f;
                ex[rc.x * 33 + half * 16 + rc.y] = e;
                probabilities[rc.x * 656 + k0 + half * 16 + rc.y] = e4m3_byte(e);
            }
            __syncthreads();
        }
        if (threadIdx.x < 16) {
#pragma unroll
            for (uint j = 0; j < 32; ++j) denom[threadIdx.x] += ex[threadIdx.x * 33 + j];
        }
        __syncthreads();
    }
    C32 acc0 = C32::Splat(0.f), acc1 = C32::Splat(0.f);
    for (uint k0 = 0; k0 < tokens; k0 += 32) {
        {
            const uint col = (threadIdx.x % 8u) * 4u;
            for (uint r = threadIdx.x / 8u; r < 32; r += blockDim.x / 8u) {
                uint k = k0 + r;
                if (F8IN) {
                    *reinterpret_cast<u32*>(values + r * 32u + col) =
                        k < tokens
                            ? *reinterpret_cast<const u32*>(
                                  qkv_f8 + k * stride + 2u * dim + head * 32u + col)
                            : 0u;
                } else {
                    const float4 v = k < tokens
                        ? *reinterpret_cast<const float4*>(
                              qkv + k * stride + 2u * dim + head * 32u + col)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<u32*>(values + r * 32u + col) =
                        u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                        u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
                }
            }
        }
        __syncthreads();
        A8 p = A8::Load(probabilities, k0, 656);
        B8 v0 = B8::LoadRow(values, 0, 32), v1 = B8::LoadRow(values, 16, 32);
        acc0.MultiplyAccumulate(p, v0);
        acc1.MultiplyAccumulate(p, v1);
        __syncthreads();
    }
    for (uint i = 0; i < acc0.Length(); ++i) {
        uint2 rc = acc0.GetCoordinate(i);
        const float inv = 1.f / denom[rc.x];
        out[(q0 + rc.x) * dim + head * 32 + rc.y] = e4m3_byte(H(acc0.Get(i) * inv));
        out[(q0 + rc.x) * dim + head * 32 + 16 + rc.y] = e4m3_byte(H(acc1.Get(i) * inv));
    }
}

// One wave. LDS is a 32-key working set, not the full 640-key probability matrix:
//   [0,512)     Q/K tile, then the 32x32 V tile (V is staged only after ex is consumed)
//   [512,1568)  ex as the exact f16 bits of the f32 exponent (0 outside the token range)
//   [1568,2080) P tile, stride 32, 16-byte aligned rows for A8::Load
//   [2080,2144) denom
// Peak 2144 B. V reuses [0,1024) only after the f32 denom chain has read ex.
template <bool F8IN>
__global__ void k_vit_attention_fused(const float* qkv, const u8* qkv_f8, u8* out, uint tokens,
                                      uint dim) {
    using namespace linalg;
    constexpr uint kExBytes = 16u * 33u * 2u;
    constexpr uint kPrefix = 512u + kExBytes;
    constexpr uint kPstride = 32u;
    __shared__ alignas(16) u8 smem[kPrefix + 16u * kPstride + 16u * 4u];
    u8* tile = smem;
    uint16_t* ex = reinterpret_cast<uint16_t*>(smem + 512u);
    u8* ptile = smem + kPrefix;
    float* denom = reinterpret_cast<float*>(smem + kPrefix + 16u * kPstride);

    const uint q0 = blockIdx.x * 16u, head = blockIdx.y;
    const uint stride = 3u * dim;
    const uint qbase = head * 32u;
    const uint kbase = dim + qbase;
    const uint vbase = 2u * dim + qbase;

    vit_stage_e4m3<F8IN>(tile, 16u, qkv, qkv_f8, q0, tokens, stride, qbase);
    if (threadIdx.x < 16u) denom[threadIdx.x] = 0.f;
    __syncthreads();
    A8 q = A8::Load(tile, 0, 32);
    C32 acc0 = C32::Splat(0.f), acc1 = C32::Splat(0.f);
    for (uint k0 = 0; k0 < tokens; k0 += 32) {
        for (uint half = 0; half < 2; ++half) {
            vit_stage_e4m3<F8IN>(tile, 16u, qkv, qkv_f8, k0 + half * 16u, tokens, stride, kbase);
            __syncthreads();
            B8 k = B8::LoadCol(tile, 0, 32);
            C32 score = Multiply(q, k);
            for (uint i = 0; i < score.Length(); ++i) {
                uint2 rc = score.GetCoordinate(i);
                const uint key = k0 + half * 16u + rc.y;
                float affine = fminf(fmaxf(fmaf(score.Get(i),
                    __half2float(__ushort_as_half(0x2dbb)), 1.708984375f), 1.439453125f),
                                     1.9775390625f);
                uint bits = __half_as_ushort(__float2half_rn(affine));
                uint16_t hb = uint16_t((bits << 4) + 0x4000u);
                float e = key < tokens ? __half2float(__ushort_as_half(hb)) : 0.f;
                ex[rc.x * 33u + half * 16u + rc.y] = key < tokens ? hb : uint16_t(0);
                ptile[rc.x * kPstride + half * 16u + rc.y] = e4m3_byte(e);
            }
            __syncthreads();
        }
        if (threadIdx.x < 16u) {
#pragma unroll
            for (uint j = 0; j < 32; ++j)
                denom[threadIdx.x] += __half2float(__ushort_as_half(ex[threadIdx.x * 33u + j]));
        }
        // ex has been reduced in key order. V may now reuse the front of smem.
        __syncthreads();
        vit_stage_e4m3<F8IN>(tile, 32u, qkv, qkv_f8, k0, tokens, stride, vbase);
        __syncthreads();
        A8 p = A8::Load(ptile, 0, kPstride);
        B8 v0 = B8::LoadRow(tile, 0, 32), v1 = B8::LoadRow(tile, 16, 32);
        acc0.MultiplyAccumulate(p, v0);
        acc1.MultiplyAccumulate(p, v1);
        __syncthreads();
    }
    for (uint i = 0; i < acc0.Length(); ++i) {
        uint2 rc = acc0.GetCoordinate(i);
        const float inv = 1.f / denom[rc.x];
        out[(q0 + rc.x) * dim + head * 32 + rc.y] = e4m3_byte(H(acc0.Get(i) * inv));
        out[(q0 + rc.x) * dim + head * 32 + 16 + rc.y] = e4m3_byte(H(acc1.Get(i) * inv));
    }
}

// out = F(H(gemm + skip * scale[ch]))
__global__ void k_residual_scale(const u8* gemm, const u8* skip, const float* scale, u8* dst,
                                 uint n, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * c)
        return;
    uint ch = i % c;
    float v = from_e4m3(gemm[i]) + from_e4m3(skip[i]) * scale[ch];
    dst[i] = e4m3_byte(H(v));
}

// Decoder upsample: nearest 2× of low-res GEMM, add skip*scale at high-res.
// decoder39 uses 32×20 → 60×36 (right/bottom GEMM columns unused).
__global__ void k_decoder_scatter(const u8* lo, const u8* skip, const float* scale, u8* dst,
                                  uint in_w, uint in_h, uint out_w, uint out_h, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = out_w * out_h * c;
    if (i >= n)
        return;
    uint oc = i % c, t = i / c, ox = t % out_w, oy = t / out_w;
    uint ix = ox / 2, iy = oy / 2;
    float v = 0.f;
    if (ix < in_w && iy < in_h)
        v = from_e4m3(lo[(iy * in_w + ix) * c + oc]);
    v = H(v + from_e4m3(skip[i]) * scale[oc]);
    dst[i] = c == 32 ? e4m3_byte(v) : e4m3_byte(F(v));
}

// post70 merge: out_c = scale_main[c]*main + scale_skip[c]*skip  (32 channels, 64 scales).
__global__ void k_post70_merge(const u8* main, const u8* skip, const float* scales, u8* dst,
                               uint n, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * c)
        return;
    uint ch = i % c, p = i / c;
    float v = from_e4m3(main[i]) * scales[ch] + from_e4m3(skip[i]) * scales[c + ch];
    dst[i] = e4m3_byte(H(v));
}

} // namespace dlss5
