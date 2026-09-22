// Fused Swin window: FFN + optional proj0 + QKV, then attention + proj1.
// One block per 8×8 window; activations stay in LDS except QKV (global, window-major).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// C32 FFN, one 8×8 window, 256 threads: 4×16-token groups in parallel (8 waves).
__global__ void k_c32_ffn_win(const u8* in, const u8* ffn_w, u8* out, uint width, uint height,
                              int packed) {
    using namespace linalg;
    constexpr int C = 32, HIDDEN = 128;
    const uint nwin = (width / 8) * (height / 8);
    const uint wave = threadIdx.x / 32u;
    const uint grp = wave / 2u;
    const uint sub = wave % 2u;
    const uint t0 = grp * 16u;
    __shared__ u8 feat[64 * 32];
    __shared__ u8 hidden[4 * 16 * 128];

    for (uint window = blockIdx.x; window < nwin; window += gridDim.x) {
        for (uint i = threadIdx.x; i < 64u * 32u; i += blockDim.x) {
            uint tok = i / 32u, ch = i % 32u;
            feat[i] = packed ? in[window * 64u * 32u + i]
                             : in[window_pixel(window, tok, width) * 32u + ch];
        }
        __syncthreads();
        {
            C32 acc[4];
            for (uint n = 0; n < 4; n++)
                acc[n] = C32::Splat(0.f);
            A8 a = A8::Load(feat, t0 * 32u, 32);
            for (uint n = 0; n < 4; n++) {
                uint col = (sub * 4u + n) * 16u;
                B8 b = B8::LoadRowT(ffn_w, (col / 16u) * 512u, 16);
                acc[n].MultiplyAccumulate(a, b);
            }
            for (uint n = 0; n < 4; n++) {
                uint col = (sub * 4u + n) * 16u;
                for (uint i = 0; i < acc[n].Length(); i++)
                    acc[n].Set(i, ActivatePoly(acc[n].Get(i)));
                acc[n].StoreF8(hidden + grp * 16u * 128u, col, 128);
            }
        }
        __syncthreads();
        {
            C32 acc = C32::Splat(0.f);
            const uint col = sub * 16u;
            u8* hid = hidden + grp * 16u * 128u;
            for (uint g = 0; g < 128u / 32u; g++) {
                A8 a = A8::Load(hid, g * 32u, 128);
                B8 b = B8::LoadRowT(ffn_w, 128u * 32u + ((col / 16u) * 4u + g) * 512u, 16);
                acc.MultiplyAccumulate(a, b);
            }
            acc.StoreF8(feat, t0 * 32u + col, 32);
        }
        __syncthreads();
        for (uint i = threadIdx.x; i < 64u * 32u; i += blockDim.x) {
            uint tok = i / 32u, ch = i % 32u;
            if (packed)
                out[window * 64u * 32u + i] = feat[i];
            else
                out[window_pixel(window, tok, width) * 32u + ch] = feat[i];
        }
        __syncthreads();
    }
}

// Packed stream8: FFN + optional proj0 + QKV + direct FP8 window attention + proj
// in one dispatch. Activations stay in LDS; in/out are window-major when packed.
// Q/K/V are planar [part][64][C] so WMMA loads use stride C (same as FFN LDS).
// K/V 16×32 tiles are staged to an aligned scratch before LoadCol.
template <int C>
__global__ void k_stream8(const u8* in, const u8* ffn_w, const u8* proj0_w, const float* proj0_s,
                          const u8* qkv_w, const float* bias, const u8* proj_w, const float* proj_s,
                          u8* out, uint width, uint height, int do_proj0, int packed) {
    using namespace linalg;
    constexpr int HIDDEN = 4 * C;
    constexpr int FFN_WAVES = C / 16;
    constexpr int HEADS = C / 32;
    constexpr int NPAR = (4 / FFN_WAVES) ? (4 / FFN_WAVES) : 1;
    constexpr int NIT = 4 / NPAR;
    const uint nwin = (width / 8) * (height / 8);
    const uint wave = threadIdx.x / 32u;

    __shared__ u8 feat[64 * C];
    __shared__ u8 qkv[3 * 64 * C];
    __shared__ float scores[64 * 64];
    __shared__ u8 p8[64 * 32];
    __shared__ u8 tile[32 * 32];

    for (uint window = blockIdx.x; window < nwin; window += gridDim.x) {
        for (uint i = threadIdx.x; i < 64u * uint(C); i += blockDim.x) {
            uint tok = i / uint(C), ch = i % uint(C);
            feat[i] = packed ? in[window * 64u * uint(C) + i]
                             : in[window_pixel(window, tok, width) * uint(C) + ch];
        }
        __syncthreads();

        for (uint it = 0; it < uint(NIT); it++) {
            uint grp = (wave / uint(FFN_WAVES)) + it * uint(NPAR);
            uint sub = wave % uint(FFN_WAVES);
            uint t0 = grp * 16u;
            uint hid_off = (wave / uint(FFN_WAVES)) * 16u * uint(HIDDEN);
            if (grp < 4 && wave < uint(NPAR) * uint(FFN_WAVES)) {
                C32 acc[4];
#pragma unroll
                for (uint n = 0; n < 4; n++)
                    acc[n] = C32::Splat(0.f);
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
#pragma unroll
                    for (uint n = 0; n < 4; n++) {
                        uint col = (sub * 4u + n) * 16u;
                        uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
                        B8 b = B8::LoadRowT(ffn_w, boff, 16);
                        acc[n].MultiplyAccumulate(a, b);
                    }
                }
#pragma unroll
                for (uint n = 0; n < 4; n++) {
                    uint col = (sub * 4u + n) * 16u;
                    for (uint i = 0; i < acc[n].Length(); i++)
                        acc[n].Set(i, ActivatePoly(acc[n].Get(i)));
                    acc[n].StoreF8(qkv + hid_off, col, uint(HIDDEN));
                }
            }
            __syncthreads();
            if (grp < 4 && wave < uint(NPAR) * uint(FFN_WAVES)) {
                C32 acc = C32::Splat(0.f);
                const uint col = sub * 16u;
                for (uint g = 0; g < uint(HIDDEN / 32); g++) {
                    A8 a = A8::Load(qkv + hid_off, g * 32u, uint(HIDDEN));
                    uint boff = uint(HIDDEN) * uint(C) + ((col / 16u) * uint(HIDDEN / 32) + g) * 512u;
                    B8 b = B8::LoadRowT(ffn_w, boff, 16);
                    acc.MultiplyAccumulate(a, b);
                }
                acc.StoreF8(feat, t0 * uint(C) + col, uint(C));
            }
            __syncthreads();
        }

        if (do_proj0 && proj0_w) {
            if (wave < 4) {
                uint t0 = wave * 16u;
                for (uint n0 = 0; n0 < uint(C); n0 += 16) {
                    C32 acc = C32::Splat(0.f);
                    for (uint i = 0; i < acc.Length(); i++) {
                        uint2 rc = acc.GetCoordinate(i);
                        uint row = n0 + rc.y;
                        acc.Set(i, H(from_e4m3(feat[(t0 + rc.x) * uint(C) + row]) * proj0_s[row]));
                    }
                    for (uint g = 0; g < uint(C / 32); g++) {
                        A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                        B8 b = B8::LoadRowT(proj0_w, ((n0 / 16u) * uint(C / 32) + g) * 512u, 16);
                        acc.MultiplyAccumulate(a, b);
                    }
                    for (uint i = 0; i < acc.Length(); i++)
                        acc.Set(i, F(H(acc.Get(i))));
                    acc.StoreF8(feat, t0 * uint(C) + n0, uint(C));
                }
            }
            __syncthreads();
        }

        if (wave < 4) {
            uint t0 = wave * 16u;
            const uint gy_n = (3u * uint(C) + 63u) / 64u;
            for (uint gy = 0; gy < gy_n; gy++) {
                C32 acc[4];
#pragma unroll
                for (uint n = 0; n < 4; n++)
                    acc[n] = C32::Splat(0.f);
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
#pragma unroll
                    for (uint n = 0; n < 4; n++) {
                        uint col = (gy * 4u + n) * 16u;
                        if (col >= 3u * uint(C))
                            continue;
                        B8 b = B8::LoadRowT(qkv_w, ((col / 16u) * uint(C / 32) + g) * 512u, 16);
                        acc[n].MultiplyAccumulate(a, b);
                    }
                }
#pragma unroll
                for (uint n = 0; n < 4; n++) {
                    uint col = (gy * 4u + n) * 16u;
                    if (col >= 3u * uint(C))
                        continue;
                    uint part = col / uint(C), row = col % uint(C);
                    for (uint i = 0; i < acc[n].Length(); i++) {
                        uint2 rc = acc[n].GetCoordinate(i);
                        qkv[part * 64u * uint(C) + (t0 + rc.x) * uint(C) + row + rc.y] =
                            e4m3_byte(H(acc[n].Get(i)));
                    }
                }
            }
        }
        __syncthreads();

        const uint qtok = wave * 16u;
        for (uint head = 0; head < uint(HEADS); head++) {
            A8 q = A8::Load(qkv, qtok * uint(C) + head * 32u, uint(C));
            for (uint k0 = 0; k0 < 64; k0 += 16) {
                for (uint i = threadIdx.x; i < 16u * 32u; i += blockDim.x) {
                    uint tok = i / 32u, ch = i % 32u;
                    tile[i] = qkv[64u * uint(C) + (k0 + tok) * uint(C) + head * 32u + ch];
                }
                __syncthreads();
                B8 k = B8::LoadCol(tile, 0, 32);
                C32 s = Multiply(q, k);
                for (uint i = 0; i < s.Length(); i++) {
                    uint2 rc = s.GetCoordinate(i);
                    uint qi = qtok + rc.x, kj = k0 + rc.y;
                    scores[qi * 64 + kj] = H(s.Get(i) + bias[head * 4096u + qi * 64u + kj]);
                }
                __syncthreads();
            }
            if (threadIdx.x < 64) {
                uint qi = threadIdx.x;
                float m = scores[qi * 64];
                for (int j = 1; j < 64; j++)
                    m = fmaxf(m, scores[qi * 64 + j]);
                float sum = 0.f;
                for (int j = 0; j < 64; j++) {
                    float e = F(H(fast_exp_raw(scores[qi * 64 + j] - m)));
                    scores[qi * 64 + j] = e;
                    sum += e;
                }
                float inv = norm_inverse(fmaxf(sum, 1e-20f));
                for (int j = 0; j < 64; j++)
                    scores[qi * 64 + j] = F(H(scores[qi * 64 + j] * inv));
            }
            __syncthreads();
            C32 av0 = C32::Splat(0.f);
            C32 av1 = C32::Splat(0.f);
            for (uint k0 = 0; k0 < 64; k0 += 32) {
                if (threadIdx.x < 64)
                    for (int j = 0; j < 32; j++)
                        p8[threadIdx.x * 32 + j] = e4m3_byte(scores[threadIdx.x * 64 + k0 + j]);
                for (uint i = threadIdx.x; i < 32u * 32u; i += blockDim.x) {
                    uint tok = i / 32u, ch = i % 32u;
                    tile[i] = qkv[128u * uint(C) + (k0 + tok) * uint(C) + head * 32u + ch];
                }
                __syncthreads();
                A8 p = A8::Load(p8, qtok * 32u, 32);
                B8 v0 = B8::LoadCol(tile, 0, 32);
                B8 v1 = B8::LoadCol(tile, 16, 32);
                C32 part0 = Multiply(p, v0);
                C32 part1 = Multiply(p, v1);
                if (k0 == 0) {
                    for (uint i = 0; i < av0.Length(); i++) {
                        av0.Set(i, H(part0.Get(i)));
                        av1.Set(i, H(part1.Get(i)));
                    }
                } else {
                    for (uint i = 0; i < av0.Length(); i++) {
                        av0.Set(i, F(H(av0.Get(i) + part0.Get(i))));
                        av1.Set(i, F(H(av1.Get(i) + part1.Get(i))));
                    }
                }
                __syncthreads();
            }
            for (uint i = 0; i < av0.Length(); i++) {
                uint2 rc = av0.GetCoordinate(i);
                feat[(qtok + rc.x) * uint(C) + head * 32u + rc.y] = e4m3_byte(av0.Get(i));
                feat[(qtok + rc.x) * uint(C) + head * 32u + 16u + rc.y] = e4m3_byte(av1.Get(i));
            }
            __syncthreads();
        }

        if (wave < 4) {
            uint t0 = wave * 16u;
            for (uint n0 = 0; n0 < uint(C); n0 += 16) {
                C32 acc = C32::Splat(0.f);
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    uint row = n0 + rc.y;
                    acc.Set(i, H(from_e4m3(feat[(t0 + rc.x) * uint(C) + row]) * proj_s[row]));
                }
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    B8 b = B8::LoadRowT(proj_w, ((n0 / 16u) * uint(C / 32) + g) * 512u, 16);
                    acc.MultiplyAccumulate(a, b);
                }
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    u8 v = e4m3_byte(F(H(acc.Get(i))));
                    if (packed)
                        out[(window * 64u + t0 + rc.x) * uint(C) + n0 + rc.y] = v;
                    else
                        out[window_pixel(window, t0 + rc.x, width) * uint(C) + n0 + rc.y] = v;
                }
            }
        }
        __syncthreads();
    }
}

// One window, all stages. C32: 256 threads (8 waves × 4 token-groups). C64: 512 threads.
template <int C>
__global__ void k_fused_swin(const u8* in, const u8* ffn_w, const u8* qkv_w, const float* bias,
                             const u8* proj_w, const float* proj_s, const u8* proj0_w,
                             const float* proj0_s, u8* out, uint width, uint height, int do_proj0) {
    using namespace linalg;
    constexpr int HIDDEN = 4 * C;
    constexpr int FFN_WAVES = C / 16;
    constexpr int STRIDE = 3 * C;
    constexpr int HEADS = C / 32;
    const uint nwin = (width / 8) * (height / 8);
    const uint wave = threadIdx.x / 32u;
    const uint group = wave / uint(FFN_WAVES);
    const uint sub = wave % uint(FFN_WAVES);
    __shared__ u8 feat[64 * C];
    __shared__ u8 hidden[4 * 16 * HIDDEN];
    __shared__ u8 qkv[64 * STRIDE];
    __shared__ float scores[64 * 64];
    __shared__ u8 p8[64 * 32];

    for (uint window = blockIdx.x; window < nwin; window += gridDim.x) {
        for (uint i = threadIdx.x; i < 64u * uint(C); i += blockDim.x) {
            uint tok = i / uint(C), ch = i % uint(C);
            feat[i] = in[window_pixel(window, tok, width) * uint(C) + ch];
        }
        __syncthreads();

        if (group < 4) {
            const uint t0 = group * 16u;
            u8* hid = hidden + group * 16u * uint(HIDDEN);
            C32 acc[4];
            for (uint n = 0; n < 4; n++)
                acc[n] = C32::Splat(0.f);
            for (uint g = 0; g < uint(C / 32); g++) {
                A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                for (uint n = 0; n < 4; n++) {
                    uint col = (sub * 4u + n) * 16u;
                    uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
                    B8 b = B8::LoadRowT(ffn_w, boff, 16);
                    acc[n].MultiplyAccumulate(a, b);
                }
            }
            for (uint n = 0; n < 4; n++) {
                uint col = (sub * 4u + n) * 16u;
                for (uint i = 0; i < acc[n].Length(); i++)
                    acc[n].Set(i, ActivatePoly(acc[n].Get(i)));
                acc[n].StoreF8(hid, col, uint(HIDDEN));
            }
        }
        __syncthreads();
        if (group < 4) {
            const uint t0 = group * 16u;
            u8* hid = hidden + group * 16u * uint(HIDDEN);
            C32 acc = C32::Splat(0.f);
            const uint col = sub * 16u;
            for (uint g = 0; g < uint(HIDDEN / 32); g++) {
                A8 a = A8::Load(hid, g * 32u, uint(HIDDEN));
                uint boff = uint(HIDDEN) * uint(C) + ((col / 16u) * uint(HIDDEN / 32) + g) * 512u;
                B8 b = B8::LoadRowT(ffn_w, boff, 16);
                acc.MultiplyAccumulate(a, b);
            }
            acc.StoreF8(feat, t0 * uint(C) + col, uint(C));
        }
        __syncthreads();

        if (do_proj0 && proj0_w && wave < 4) {
            uint t0 = wave * 16u;
            for (uint n0 = 0; n0 < uint(C); n0 += 16) {
                C32 acc = C32::Splat(0.f);
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    uint row = n0 + rc.y;
                    acc.Set(i, H(from_e4m3(feat[(t0 + rc.x) * uint(C) + row]) * proj0_s[row]));
                }
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    B8 b = B8::LoadRowT(proj0_w, ((n0 / 16u) * uint(C / 32) + g) * 512u, 16);
                    acc.MultiplyAccumulate(a, b);
                }
                for (uint i = 0; i < acc.Length(); i++)
                    acc.Set(i, F(H(acc.Get(i))));
                acc.StoreF8(feat, t0 * uint(C) + n0, uint(C));
            }
        }
        __syncthreads();

        if (wave < 4) {
            uint t0 = wave * 16u;
            const uint gy_n = (3u * uint(C) + 63u) / 64u;
            for (uint gy = 0; gy < gy_n; gy++) {
                C32 acc[4];
                for (uint n = 0; n < 4; n++)
                    acc[n] = C32::Splat(0.f);
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    for (uint n = 0; n < 4; n++) {
                        uint col = (gy * 4u + n) * 16u;
                        if (col >= 3u * uint(C))
                            continue;
                        B8 b = B8::LoadRowT(qkv_w, ((col / 16u) * uint(C / 32) + g) * 512u, 16);
                        acc[n].MultiplyAccumulate(a, b);
                    }
                }
                for (uint n = 0; n < 4; n++) {
                    uint col = (gy * 4u + n) * 16u;
                    if (col >= 3u * uint(C))
                        continue;
                    uint part = col / uint(C), row = col % uint(C);
                    for (uint i = 0; i < acc[n].Length(); i++) {
                        uint2 rc = acc[n].GetCoordinate(i);
                        qkv[(t0 + rc.x) * uint(STRIDE) + part * uint(C) + row + rc.y] =
                            e4m3_byte(H(acc[n].Get(i)));
                    }
                }
            }
        }
        __syncthreads();

        const uint qtok = wave < 4 ? wave * 16u : 0u;
        for (uint head = 0; head < uint(HEADS); head++) {
            if (wave < 4) {
                A8 q = A8::Load(qkv, qtok * uint(STRIDE) + head * 32u, uint(STRIDE));
                for (uint k0 = 0; k0 < 64; k0 += 16) {
                    B8 k = B8::LoadCol(qkv, k0 * uint(STRIDE) + uint(C) + head * 32u, uint(STRIDE));
                    C32 s = Multiply(q, k);
                    for (uint i = 0; i < s.Length(); i++) {
                        uint2 rc = s.GetCoordinate(i);
                        scores[(qtok + rc.x) * 64 + k0 + rc.y] =
                            H(s.Get(i) + bias[head * 4096u + (qtok + rc.x) * 64u + k0 + rc.y]);
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x < 64) {
                uint qi = threadIdx.x;
                float m = scores[qi * 64];
                for (int j = 1; j < 64; j++)
                    m = fmaxf(m, scores[qi * 64 + j]);
                float sum = 0.f;
                for (int j = 0; j < 64; j++) {
                    float e = F(H(fast_exp_raw(scores[qi * 64 + j] - m)));
                    scores[qi * 64 + j] = e;
                    sum += e;
                }
                float inv = norm_inverse(fmaxf(sum, 1e-20f));
                for (int j = 0; j < 64; j++)
                    scores[qi * 64 + j] = F(H(scores[qi * 64 + j] * inv));
            }
            __syncthreads();
            C32 av0 = C32::Splat(0.f);
            C32 av1 = C32::Splat(0.f);
            for (uint k0 = 0; k0 < 64; k0 += 32) {
                if (threadIdx.x < 64)
                    for (int j = 0; j < 32; j++)
                        p8[threadIdx.x * 32 + j] = e4m3_byte(scores[threadIdx.x * 64 + k0 + j]);
                __syncthreads();
                if (wave < 4) {
                    A8 p = A8::Load(p8, qtok * 32u, 32);
                    const uint vbase = k0 * uint(STRIDE) + 2u * uint(C) + head * 32u;
                    B8 v0 = B8::LoadCol(qkv, vbase, uint(STRIDE));
                    B8 v1 = B8::LoadCol(qkv, vbase + 16u, uint(STRIDE));
                    C32 part0 = Multiply(p, v0);
                    C32 part1 = Multiply(p, v1);
                    if (k0 == 0) {
                        for (uint i = 0; i < av0.Length(); i++) {
                            av0.Set(i, H(part0.Get(i)));
                            av1.Set(i, H(part1.Get(i)));
                        }
                    } else {
                        for (uint i = 0; i < av0.Length(); i++) {
                            av0.Set(i, F(H(av0.Get(i) + part0.Get(i))));
                            av1.Set(i, F(H(av1.Get(i) + part1.Get(i))));
                        }
                    }
                }
                __syncthreads();
            }
            if (wave < 4) {
                for (uint i = 0; i < av0.Length(); i++) {
                    uint2 rc = av0.GetCoordinate(i);
                    feat[(qtok + rc.x) * uint(C) + head * 32u + rc.y] = e4m3_byte(av0.Get(i));
                    feat[(qtok + rc.x) * uint(C) + head * 32u + 16u + rc.y] = e4m3_byte(av1.Get(i));
                }
            }
            __syncthreads();
        }

        if (wave < 4) {
            uint t0 = wave * 16u;
            for (uint n0 = 0; n0 < uint(C); n0 += 16) {
                C32 acc = C32::Splat(0.f);
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    uint row = n0 + rc.y;
                    acc.Set(i, H(from_e4m3(feat[(t0 + rc.x) * uint(C) + row]) * proj_s[row]));
                }
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    B8 b = B8::LoadRowT(proj_w, ((n0 / 16u) * uint(C / 32) + g) * 512u, 16);
                    acc.MultiplyAccumulate(a, b);
                }
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    uint pix = window_pixel(window, t0 + rc.x, width);
                    out[pix * uint(C) + n0 + rc.y] = e4m3_byte(F(H(acc.Get(i))));
                }
            }
        }
        __syncthreads();
    }
}

template <int C>
__global__ void k_fused_ffn_qkv(const u8* in, const u8* ffn_w, const u8* proj0_w,
                                const float* proj0_s, const u8* qkv_w, u8* qkv_out, uint width,
                                uint height, int do_proj0, int packed) {
    using namespace linalg;
    constexpr int HIDDEN = 4 * C;
    constexpr int FFN_WAVES = C / 16;
    constexpr int NPAR = (4 / FFN_WAVES) ? (4 / FFN_WAVES) : 1;
    constexpr int NIT = 4 / NPAR;
    const uint nwin = (width / 8) * (height / 8);
    const uint wave = threadIdx.x / 32u;
    const uint npar = uint(NPAR);
    const uint nit = uint(NIT);
    __shared__ u8 feat[64 * C];
    __shared__ u8 hidden[NPAR * 16 * HIDDEN];
    __shared__ u8 ffn_lds[C == 32 ? 8192 : 4];
    const u8* fw = ffn_w;
    if constexpr (C == 32) {
        for (uint i = threadIdx.x; i < 8192u; i += blockDim.x)
            ffn_lds[i] = ffn_w[i];
        __syncthreads();
        fw = ffn_lds;
    }

    for (uint window = blockIdx.x; window < nwin; window += gridDim.x) {
        for (uint i = threadIdx.x; i < 64u * uint(C); i += blockDim.x) {
            uint tok = i / uint(C), ch = i % uint(C);
            feat[i] = packed ? in[window * 64u * uint(C) + i]
                             : in[window_pixel(window, tok, width) * uint(C) + ch];
        }
        __syncthreads();

        for (uint it = 0; it < nit; it++) {
            uint grp = (wave / uint(FFN_WAVES)) + it * npar;
            uint sub = wave % uint(FFN_WAVES);
            uint t0 = grp * 16u;
            uint hid_off = (wave / uint(FFN_WAVES)) * 16u * uint(HIDDEN);
            if (grp < 4 && wave < npar * uint(FFN_WAVES)) {
                C32 acc[4];
                for (uint n = 0; n < 4; n++)
                    acc[n] = C32::Splat(0.f);
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    for (uint n = 0; n < 4; n++) {
                        uint col = (sub * 4u + n) * 16u;
                        uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
                        B8 b = B8::LoadRowT(fw, boff, 16);
                        acc[n].MultiplyAccumulate(a, b);
                    }
                }
                for (uint n = 0; n < 4; n++) {
                    uint col = (sub * 4u + n) * 16u;
                    for (uint i = 0; i < acc[n].Length(); i++)
                        acc[n].Set(i, ActivatePoly(acc[n].Get(i)));
                    acc[n].StoreF8(hidden + hid_off, col, uint(HIDDEN));
                }
            }
            __syncthreads();
            if (grp < 4 && wave < npar * uint(FFN_WAVES)) {
                C32 acc = C32::Splat(0.f);
                const uint col = sub * 16u;
                for (uint g = 0; g < uint(HIDDEN / 32); g++) {
                    A8 a = A8::Load(hidden + hid_off, g * 32u, uint(HIDDEN));
                    uint boff = uint(HIDDEN) * uint(C) + ((col / 16u) * uint(HIDDEN / 32) + g) * 512u;
                    B8 b = B8::LoadRowT(fw, boff, 16);
                    acc.MultiplyAccumulate(a, b);
                }
                acc.StoreF8(feat, t0 * uint(C) + col, uint(C));
            }
            __syncthreads();
        }

        if (do_proj0 && proj0_w) {
            if (wave < 4) {
                uint t0 = wave * 16u;
                for (uint n0 = 0; n0 < uint(C); n0 += 16) {
                    C32 acc = C32::Splat(0.f);
                    for (uint i = 0; i < acc.Length(); i++) {
                        uint2 rc = acc.GetCoordinate(i);
                        uint row = n0 + rc.y;
                        float f = from_e4m3(feat[(t0 + rc.x) * uint(C) + row]);
                        acc.Set(i, H(f * proj0_s[row]));
                    }
                    for (uint g = 0; g < uint(C / 32); g++) {
                        A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                        uint boff = ((n0 / 16u) * uint(C / 32) + g) * 512u;
                        B8 b = B8::LoadRowT(proj0_w, boff, 16);
                        acc.MultiplyAccumulate(a, b);
                    }
                    for (uint i = 0; i < acc.Length(); i++)
                        acc.Set(i, F(H(acc.Get(i))));
                    acc.StoreF8(feat, t0 * uint(C) + n0, uint(C));
                }
            }
            __syncthreads();
        }

        if (wave < 4) {
            uint t0 = wave * 16u;
            const uint gy_n = (3u * uint(C) + 63u) / 64u;
            for (uint gy = 0; gy < gy_n; gy++) {
                C32 acc[4];
                for (uint n = 0; n < 4; n++)
                    acc[n] = C32::Splat(0.f);
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    for (uint n = 0; n < 4; n++) {
                        uint col = (gy * 4u + n) * 16u;
                        if (col >= 3u * uint(C))
                            continue;
                        uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
                        B8 b = B8::LoadRowT(qkv_w, boff, 16);
                        acc[n].MultiplyAccumulate(a, b);
                    }
                }
                for (uint n = 0; n < 4; n++) {
                    uint col = (gy * 4u + n) * 16u;
                    if (col >= 3u * uint(C))
                        continue;
                    uint part = col / uint(C), row = col % uint(C);
                    for (uint i = 0; i < acc[n].Length(); i++) {
                        uint2 rc = acc[n].GetCoordinate(i);
                        qkv_out[((window * 64u + t0 + rc.x) * 3u + part) * uint(C) + row + rc.y] =
                            e4m3_byte(H(acc[n].Get(i)));
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int C>
__global__ void k_fused_attn_proj(const u8* qkv, const float* bias, const u8* proj_w,
                                  const float* proj_s, u8* out, uint width, uint height,
                                  int packed) {
    using namespace linalg;
    constexpr int HEADS = C / 32;
    constexpr int STRIDE = 3 * C;
    const uint nwin = (width / 8) * (height / 8);
    const uint wave = threadIdx.x / 32u;
    const uint qtok = wave * 16u;
    __shared__ float scores[64 * 64];
    __shared__ u8 p8[64 * 32];
    __shared__ u8 feat[64 * C];

    for (uint window = blockIdx.x; window < nwin; window += gridDim.x) {
        for (uint head = 0; head < uint(HEADS); head++) {
            A8 q = A8::Load(qkv, (window * 64u + qtok) * uint(STRIDE) + head * 32u, uint(STRIDE));
            for (uint k0 = 0; k0 < 64; k0 += 16) {
                B8 k = B8::LoadCol(qkv, (window * 64u + k0) * uint(STRIDE) + uint(C) + head * 32u,
                                   uint(STRIDE));
                C32 s = Multiply(q, k);
                for (uint i = 0; i < s.Length(); i++) {
                    uint2 rc = s.GetCoordinate(i);
                    uint qi = qtok + rc.x, kj = k0 + rc.y;
                    scores[qi * 64 + kj] = H(s.Get(i) + bias[head * 4096u + qi * 64u + kj]);
                }
            }
            __syncthreads();
            if (threadIdx.x < 64) {
                uint qi = threadIdx.x;
                float m = scores[qi * 64];
                for (int j = 1; j < 64; j++)
                    m = fmaxf(m, scores[qi * 64 + j]);
                float sum = 0.f;
                for (int j = 0; j < 64; j++) {
                    float e = F(H(fast_exp_raw(scores[qi * 64 + j] - m)));
                    scores[qi * 64 + j] = e;
                    sum += e;
                }
                float inv = norm_inverse(fmaxf(sum, 1e-20f));
                for (int j = 0; j < 64; j++)
                    scores[qi * 64 + j] = F(H(scores[qi * 64 + j] * inv));
            }
            __syncthreads();
            C32 av0 = C32::Splat(0.f);
            C32 av1 = C32::Splat(0.f);
            for (uint k0 = 0; k0 < 64; k0 += 32) {
                if (threadIdx.x < 64)
                    for (int j = 0; j < 32; j++)
                        p8[threadIdx.x * 32 + j] = e4m3_byte(scores[threadIdx.x * 64 + k0 + j]);
                __syncthreads();
                A8 p = A8::Load(p8, qtok * 32u, 32);
                const uint vbase =
                    (window * 64u + k0) * uint(STRIDE) + 2u * uint(C) + head * 32u;
                B8 v0 = B8::LoadCol(qkv, vbase, uint(STRIDE));
                B8 v1 = B8::LoadCol(qkv, vbase + 16u, uint(STRIDE));
                C32 part0 = Multiply(p, v0);
                C32 part1 = Multiply(p, v1);
                if (k0 == 0) {
                    for (uint i = 0; i < av0.Length(); i++) {
                        av0.Set(i, H(part0.Get(i)));
                        av1.Set(i, H(part1.Get(i)));
                    }
                } else {
                    for (uint i = 0; i < av0.Length(); i++) {
                        av0.Set(i, F(H(av0.Get(i) + part0.Get(i))));
                        av1.Set(i, F(H(av1.Get(i) + part1.Get(i))));
                    }
                }
                __syncthreads();
            }
            for (uint i = 0; i < av0.Length(); i++) {
                uint2 rc = av0.GetCoordinate(i);
                feat[(qtok + rc.x) * uint(C) + head * 32u + rc.y] = e4m3_byte(av0.Get(i));
                feat[(qtok + rc.x) * uint(C) + head * 32u + 16u + rc.y] = e4m3_byte(av1.Get(i));
            }
            __syncthreads();
        }

        if (wave < 4) {
            uint t0 = wave * 16u;
            for (uint n0 = 0; n0 < uint(C); n0 += 16) {
                C32 acc = C32::Splat(0.f);
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    uint row = n0 + rc.y;
                    float f = from_e4m3(feat[(t0 + rc.x) * uint(C) + row]);
                    acc.Set(i, H(f * proj_s[row]));
                }
                for (uint g = 0; g < uint(C / 32); g++) {
                    A8 a = A8::Load(feat, t0 * uint(C) + g * 32u, uint(C));
                    uint boff = ((n0 / 16u) * uint(C / 32) + g) * 512u;
                    B8 b = B8::LoadRowT(proj_w, boff, 16);
                    acc.MultiplyAccumulate(a, b);
                }
                for (uint i = 0; i < acc.Length(); i++) {
                    uint2 rc = acc.GetCoordinate(i);
                    acc.Set(i, F(H(acc.Get(i))));
                    if (packed)
                        out[(window * 64u + t0 + rc.x) * uint(C) + n0 + rc.y] =
                            e4m3_byte(acc.Get(i));
                    else
                        out[window_pixel(window, t0 + rc.x, width) * uint(C) + n0 + rc.y] =
                            e4m3_byte(acc.Get(i));
                }
            }
        }
        __syncthreads();
    }
}

} // namespace dlss5
