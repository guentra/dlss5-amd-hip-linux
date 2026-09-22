// Logical FP32 storage keeps half-valued residuals distinct from FP8 operands.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// mode: 0 half raw, 1 FP8(half), 2 activated FP8, 3 unrounded FP32,
// 4 C32 final H(branch + residual*scale) without H(product), 5 direct FP8.
// ordered: 0 FP32 accumulation, 1 H after each K32, 2 four independent
// FP32 K partitions combined with H(residual*scale) (fast ViT reduce).
template <bool F8IN, bool F8OUT = false, bool HRID = false, bool HOUT = false,
          bool F8OUT_RASTER = false, bool F8RES = false, bool HOUT_RASTER = false>
__global__ void k_linear_f32(const float* input, const u8* input_f8, const u8* weights,
                             float* output, u8* output_f8, uint m, uint n, uint k,
                             const float* residual, const float* scales, int mode, int ordered,
                             bool matrix_residual = false, const __half* residual_h = nullptr,
                             __half* output_h = nullptr, u8* output_f8_raster = nullptr,
                             uint rw = 0, uint rh = 0, uint rsw = 0, uint rpx = 0, uint rpy = 0,
                             const u8* residual_f8 = nullptr, __half* output_h_raster = nullptr) {
    using namespace linalg;
    const uint first = blockIdx.x * 16, col = blockIdx.y * 64 + (threadIdx.x / 32) * 16;
    const uint rc_y = threadIdx.x & 15u;
    const float scale_c = (scales && col < n) ? scales[col + rc_y] : 0.f;
    __shared__ u8 tile[512];
    C32 acc = C32::Splat(0.f);
    if ((F8RES && residual_f8) || (residual && mode != 4))
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < m && col < n) {
                float rv;
                if (F8RES && residual_f8) {
                    // The p1 sibling is already on the E4M3 grid, so from_e4m3() is
                    // exactly the f32 residual used by the reference path.
                    rv = from_e4m3(residual_f8[(first + rc.x) * n + col + rc.y]);
                } else
                    rv = HRID ? __half2float(residual_h[(first + rc.x) * n + col + rc.y])
                              : residual[(first + rc.x) * n + col + rc.y];
                acc.Set(i, matrix_residual ? rv * scale_c : H(rv * scale_c));
            }
        }
    C32 partition = C32::Splat(0.f);
    for (uint g = 0; g < k / 32; ++g) {
        {
            const uint col = (threadIdx.x % 8u) * 4u;
            // 16 input rows; 128 threads -> 1 row/thread-group, 64 threads -> 2. The n=32
            // case only needs 2 active waves (col 0/16), so it launches with half the
            // threads; the stride keeps the staged tile byte-identical at any size.
            for (uint rlocal = threadIdx.x / 8u; rlocal < 16u; rlocal += blockDim.x / 8u) {
                const uint row = first + rlocal;
                if (F8IN) {
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + col) =
                        row < m ? *reinterpret_cast<const u32*>(input_f8 + row * k + g * 32u + col)
                                : 0u;
                } else {
                    const float4 v = row < m
                        ? *reinterpret_cast<const float4*>(input + row * k + g * 32u + col)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + col) =
                        u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                        u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
                }
            }
        }
        __syncthreads();
        A8 a = A8::Load(tile, 0, 32);
        // Four output waves share one converted input tile. In the final
        // partial group, inactive waves read a valid tile and omit stores.
        B8 b = B8::LoadRowT(weights, (((col < n ? col : 0) / 16) * (k / 32) + g) * 512, 16);
        if (ordered == 2) {
            partition.MultiplyAccumulate(a,b);
            if ((g+1) % (k/128) == 0) {
                for(uint i=0;i<acc.Length();++i) acc.Set(i,acc.Get(i)+partition.Get(i));
                partition=C32::Splat(0.f);
            }
        } else if (ordered == 1) {
            C32 part = Multiply(a, b);
            for (uint i = 0; i < acc.Length(); ++i) acc.Set(i, H(acc.Get(i) + part.Get(i)));
        } else acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    for (uint i = 0; i < acc.Length(); ++i) {
        uint2 rc = acc.GetCoordinate(i);
        float v = acc.Get(i);
        if (mode == 4 && residual && first + rc.x < m && col < n)
            v += (HRID ? __half2float(residual_h[(first + rc.x) * n + col + rc.y])
                       : residual[(first + rc.x) * n + col + rc.y]) * scale_c;
        u8 qout = 0;
        if (mode == 2) {
            u8 q = e4m3_byte(ActivatePoly(v));
            v = from_e4m3(q);
            if constexpr (F8OUT || F8OUT_RASTER) qout = q;
        } else if (mode == 1) {
            u8 q = e4m3_byte(H(v));
            v = from_e4m3(q);
            if constexpr (F8OUT || F8OUT_RASTER) qout = q;
        } else if (mode == 5) {
            u8 q = e4m3_byte(v);
            v = from_e4m3(q);
            if constexpr (F8OUT || F8OUT_RASTER) qout = q;
        } else if (mode == 0 || mode == 4) v = H(v);
        if (first + rc.x < m && col < n) {
            // HOUT stores the exact H(v) in f16 (mode 4 is already H-rounded), so
            // the crop reframe can read half the global traffic and still produce
            // the same f32 raster.
            if constexpr (HOUT) {
                output_h[(first + rc.x) * n + col + rc.y] = __float2half_rn(v);
                if constexpr (HOUT_RASTER) {
                    // C32 raster f16 sibling for the next block's pack reframe.
                    uint t = first + rc.x;
                    uint win = t / 64u, tok = t % 64u;
                    uint wx = (win % (rsw / 8u)) * 8u + (tok % 8u);
                    uint wy = (win / (rsw / 8u)) * 8u + (tok / 8u);
                    if (wx >= rpx && wy >= rpy && wx < rw + rpx && wy < rh + rpy)
                        output_h_raster[((wy - rpy) * rw + (wx - rpx)) * n + col + rc.y] =
                            __float2half_rn(v);
                }
            } else {
                if (output)
                    output[(first + rc.x) * n + col + rc.y] = v;
                if constexpr (F8OUT)
                    output_f8[(first + rc.x) * n + col + rc.y] = qout;
                if constexpr (F8OUT_RASTER) {
                    // Raster f8 sibling for the next block's pack reframe: token order is
                    // window-major, but the byte stream is stored row-major [y][x][C].
                    uint t = first + rc.x;
                    uint win = t / 64u, tok = t % 64u;
                    uint wx = (win % (rsw / 8u)) * 8u + (tok % 8u);
                    uint wy = (win / (rsw / 8u)) * 8u + (tok / 8u);
                    if (wx >= rpx && wy >= rpy && wx < rw + rpx && wy < rh + rpy)
                        output_f8_raster[((wy - rpy) * rw + (wx - rpx)) * n + col + rc.y] = qout;
                }
            }
        }
    }
}

// Fat-M variant of k_linear_f32: each wave stacks SUB 16-row subtiles on the
// same 16-column slice, so one B-tile load serves SUB row tiles. Global
// weight traffic drops 4x while the per-element accumulation order (K loop,
// ordered partitions, residual init, epilogue) is byte-for-byte unchanged.
template <bool F8IN, bool F8OUT = false, bool HRID = false, bool HOUT = false,
          bool F8OUT_RASTER = false, bool F8RES = false, bool HOUT_RASTER = false, int SUB = 4,
          bool PART = false>
__global__ void k_linear_f32_fat(const float* input, const u8* input_f8, const u8* weights,
                                 float* output, u8* output_f8, uint m, uint n, uint k,
                                 const float* residual, const float* scales, int mode, int ordered,
                                 bool matrix_residual = false, const __half* residual_h = nullptr,
                                 __half* output_h = nullptr, u8* output_f8_raster = nullptr,
                                 uint rw = 0, uint rh = 0, uint rsw = 0, uint rpx = 0, uint rpy = 0,
                                 const u8* residual_f8 = nullptr, __half* output_h_raster = nullptr) {
    using namespace linalg;
    constexpr uint ROWS = 16u * SUB;
    const uint first = blockIdx.x * ROWS;
    const uint col = blockIdx.y * 64 + (threadIdx.x / 32u) * 16u;
    const uint rc_y = threadIdx.x & 15u;
    const float scale_c = (scales && col < n) ? scales[col + rc_y] : 0.f;
    __shared__ u8 tile[ROWS * 32u];
    C32 acc[SUB];
    for (uint s = 0; s < SUB; ++s)
        acc[s] = C32::Splat(0.f);
    if ((F8RES && residual_f8) || (residual && mode != 4))
        for (uint s = 0; s < SUB; ++s)
            for (uint i = 0; i < acc[s].Length(); ++i) {
                uint2 rc = acc[s].GetCoordinate(i);
                const uint row = first + s * 16u + rc.x;
                if (row < m && col < n) {
                    float rv;
                    if (F8RES && residual_f8)
                        rv = from_e4m3(residual_f8[row * n + col + rc.y]);
                    else
                        rv = HRID ? __half2float(residual_h[row * n + col + rc.y])
                                  : residual[row * n + col + rc.y];
                    acc[s].Set(i, matrix_residual ? rv * scale_c : H(rv * scale_c));
                }
            }
    C32 partition[SUB];
    if constexpr (PART)
        for (uint s = 0; s < SUB; ++s)
            partition[s] = C32::Splat(0.f);
    for (uint g = 0; g < k / 32; ++g) {
        for (uint i4 = threadIdx.x; i4 < ROWS * 8u; i4 += blockDim.x) {
            const uint c4 = (i4 % 8u) * 4u;
            const uint row = first + i4 / 8u;
            if (F8IN) {
                *reinterpret_cast<u32*>(tile + i4 * 4u) =
                    row < m ? *reinterpret_cast<const u32*>(input_f8 + row * k + g * 32u + c4)
                            : 0u;
            } else {
                const float4 v =
                    row < m ? *reinterpret_cast<const float4*>(input + row * k + g * 32u + c4)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<u32*>(tile + i4 * 4u) =
                    u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                    u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
        __syncthreads();
        B8 b = B8::LoadRowT(weights, (((col < n ? col : 0) / 16) * (k / 32) + g) * 512, 16);
#pragma unroll
        for (uint s = 0; s < SUB; ++s) {
            A8 a = A8::Load(tile, s * 16u * 32u, 32);
            if constexpr (PART) {
                partition[s].MultiplyAccumulate(a, b);
                if ((g + 1) % (k / 128) == 0) {
                    for (uint i = 0; i < acc[s].Length(); ++i)
                        acc[s].Set(i, acc[s].Get(i) + partition[s].Get(i));
                    partition[s] = C32::Splat(0.f);
                }
            } else if (ordered == 1) {
                C32 part = Multiply(a, b);
                for (uint i = 0; i < acc[s].Length(); ++i)
                    acc[s].Set(i, H(acc[s].Get(i) + part.Get(i)));
            } else
                acc[s].MultiplyAccumulate(a, b);
        }
        __syncthreads();
    }
#pragma unroll
    for (uint s = 0; s < SUB; ++s)
        for (uint i = 0; i < acc[s].Length(); ++i) {
            uint2 rc = acc[s].GetCoordinate(i);
            const uint row = first + s * 16u + rc.x;
            float v = acc[s].Get(i);
            if (mode == 4 && residual && row < m && col < n)
                v += (HRID ? __half2float(residual_h[row * n + col + rc.y])
                           : residual[row * n + col + rc.y]) * scale_c;
            u8 qout = 0;
            if (mode == 2) {
                u8 q = e4m3_byte(ActivatePoly(v));
                v = from_e4m3(q);
                if constexpr (F8OUT || F8OUT_RASTER) qout = q;
            } else if (mode == 1) {
                u8 q = e4m3_byte(H(v));
                v = from_e4m3(q);
                if constexpr (F8OUT || F8OUT_RASTER) qout = q;
            } else if (mode == 5) {
                u8 q = e4m3_byte(v);
                v = from_e4m3(q);
                if constexpr (F8OUT || F8OUT_RASTER) qout = q;
            } else if (mode == 0 || mode == 4) v = H(v);
            if (row < m && col < n) {
                if constexpr (HOUT) {
                    output_h[row * n + col + rc.y] = __float2half_rn(v);
                    if constexpr (HOUT_RASTER) {
                        uint t = row;
                        uint win = t / 64u, tok = t % 64u;
                        uint wx = (win % (rsw / 8u)) * 8u + (tok % 8u);
                        uint wy = (win / (rsw / 8u)) * 8u + (tok / 8u);
                        if (wx >= rpx && wy >= rpy && wx < rw + rpx && wy < rh + rpy)
                            output_h_raster[((wy - rpy) * rw + (wx - rpx)) * n + col + rc.y] =
                                __float2half_rn(v);
                    }
                } else {
                    if (output)
                        output[row * n + col + rc.y] = v;
                    if constexpr (F8OUT)
                        output_f8[row * n + col + rc.y] = qout;
                    if constexpr (F8OUT_RASTER) {
                        uint t = row;
                        uint win = t / 64u, tok = t % 64u;
                        uint wx = (win % (rsw / 8u)) * 8u + (tok % 8u);
                        uint wy = (win / (rsw / 8u)) * 8u + (tok / 8u);
                        if (wx >= rpx && wy >= rpy && wx < rw + rpx && wy < rh + rpy)
                            output_f8_raster[((wy - rpy) * rw + (wx - rpx)) * n + col + rc.y] =
                                qout;
                    }
                }
            }
        }
}

template <int C, bool F8OUT = false, int GROUPS = 1, bool HOUT = false, bool F8IN = false,
          bool HIN = false>
__global__ void k_ffn_f32(const float* input, const u8* weights, const float* scales,
                          float* output, u8* output_f8, uint tokens, bool chain_residual = false,
                          bool precise_c32 = false, __half* output_h = nullptr,
                           const u8* input_f8 = nullptr, const __half* input_h = nullptr,
                           u8* dbg_hidden = nullptr, float* dbg_ex = nullptr) {
    using namespace linalg;
    constexpr uint hidden_c = 4 * C;
    constexpr uint WAVES_PER_GROUP = C / 16;
    const uint first = blockIdx.x * 16 * GROUPS;
    const uint wave = threadIdx.x / 32;
    const uint group = wave / WAVES_PER_GROUP;
    const uint sub = wave % WAVES_PER_GROUP;
    // C32 stages its E4M3 input (16*GROUPS*C bytes) in a SEPARATE buffer so the
    // hidden write cannot clobber it, removing one __syncthreads. C!=32 keeps
    // the original alias: the input staging (16*GROUPS*C bytes) aliases the start
    // of the hidden buffer (16*GROUPS*hidden_c bytes) and one barrier makes the
    // aliasing safe. C32 uses GROUPS=2 to halve CTA/weight traffic while keeping
    // 64 waves/CU.
    __shared__ u8 hidden[16 * GROUPS * hidden_c];
    __shared__ u8 inbuf[C == 32 ? 16 * GROUPS * 32u : 1u];
    u8* in = (C == 32) ? inbuf : hidden;
    if constexpr (C == 32) {
        // 4-byte input staging for C32: same E4M3 bytes, fewer LDS stores.
        const uint count4 = 16 * GROUPS * 8;
        for (uint i4 = threadIdx.x; i4 < count4; i4 += blockDim.x) {
            uint local4 = i4 % (16 * 8);
            uint row = first + (i4 / (16 * 8)) * 16 + local4 / 8;
            uint ch = (local4 % 8) * 4;
            if (F8IN) {
                uint v = row < tokens ? *reinterpret_cast<const u32*>(input_f8 + row * C + ch) : 0u;
                *reinterpret_cast<u32*>(in + i4 * 4) = v;
            } else if (HIN) {
                const __half2* hp = reinterpret_cast<const __half2*>(input_h + row * C + ch);
                __half2 h0 = row < tokens ? hp[0] : __half2(0.f, 0.f);
                __half2 h1 = row < tokens ? hp[1] : __half2(0.f, 0.f);
                *reinterpret_cast<u32*>(in + i4 * 4) =
                    u32(e4m3_byte(__half2float(__low2half(h0)))) |
                    u32(e4m3_byte(__half2float(__high2half(h0)))) << 8 |
                    u32(e4m3_byte(__half2float(__low2half(h1)))) << 16 |
                    u32(e4m3_byte(__half2float(__high2half(h1)))) << 24;
            } else {
                const float4 v = row < tokens
                                     ? *reinterpret_cast<const float4*>(input + row * C + ch)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<u32*>(in + i4 * 4) =
                    u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                    u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
    } else {
        const uint count4 = 16 * GROUPS * (C / 4);
        for (uint i4 = threadIdx.x; i4 < count4; i4 += blockDim.x) {
            uint local4 = i4 % (16 * (C / 4));
            uint row = first + (i4 / (16 * (C / 4))) * 16 + local4 / (C / 4);
            uint ch = (local4 % (C / 4)) * 4;
            if (F8IN) {
                uint v = row < tokens ? *reinterpret_cast<const u32*>(input_f8 + row * C + ch) : 0u;
                *reinterpret_cast<u32*>(in + i4 * 4) = v;
            } else if (HIN) {
                const __half2* hp = reinterpret_cast<const __half2*>(input_h + row * C + ch);
                __half2 h0 = row < tokens ? hp[0] : __half2(0.f, 0.f);
                __half2 h1 = row < tokens ? hp[1] : __half2(0.f, 0.f);
                *reinterpret_cast<u32*>(in + i4 * 4) =
                    u32(e4m3_byte(__half2float(__low2half(h0)))) |
                    u32(e4m3_byte(__half2float(__high2half(h0)))) << 8 |
                    u32(e4m3_byte(__half2float(__low2half(h1)))) << 16 |
                    u32(e4m3_byte(__half2float(__high2half(h1)))) << 24;
            } else {
                const float4 v = row < tokens
                                     ? *reinterpret_cast<const float4*>(input + row * C + ch)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<u32*>(in + i4 * 4) =
                    u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                    u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
    }
    __syncthreads();
    C32 ex[4];
    for (uint n = 0; n < 4; ++n) ex[n] = C32::Splat(0.f);
    if (group < GROUPS) {
        const uint t0 = group * 16;
        for (uint g = 0; g < C / 32; ++g) {
            A8 a = A8::Load(in + t0 * C, g * 32, C);
            for (uint n = 0; n < 4; ++n) {
                uint col = (sub * 4 + n) * 16;
                B8 b = B8::LoadRowT(weights, ((col / 16) * (C / 32) + g) * 512, 16);
                ex[n].MultiplyAccumulate(a, b);
            }
        }
        if (dbg_ex)
            for (uint n = 0; n < 4; ++n)
                for (uint i = 0; i < ex[n].Length(); ++i) {
                    uint2 c2 = ex[n].GetCoordinate(i);
                    dbg_ex[(first + t0 + c2.x) * hidden_c + (sub * 4 + n) * 16 + c2.y] = ex[n].Get(i);
                }
    }
    // For C!=32 the hidden buffer aliases the input staging; every group must
    // finish reading its input tile before any group overwrites it with hidden
    // bytes. C32 stages input separately, so this barrier is not needed (the
    // hidden write below reads only each wave's own accumulators).
    if constexpr (C != 32) __syncthreads();
    if (group < GROUPS) {
        const uint t0 = group * 16;
        for (uint n = 0; n < 4; ++n)
            for (uint i = 0; i < ex[n].Length(); ++i) ex[n].Set(i,
                C == 32 && precise_c32 ? ActivatePolyC32(ex[n].Get(i))
                                       : ActivatePoly(ex[n].Get(i)));
        if constexpr (C == 32) {
            const uint lane = threadIdx.x & 31u;
            const uint row0 = (lane >= 16u ? 8u : 0u);
            const uint base = lane & ~3u;
            const uint col0 = (lane & 15u) & ~3u;
            for (uint n = 0; n < 4; ++n) {
                const uint hcol = (sub * 4u + n) * 16u + col0;
                for (uint i = 0; i < 8u; ++i) {
                    uint b0 = e4m3_byte(ex[n].Get(i));
                    uint b1 = __shfl(b0, base + 1u);
                    uint b2 = __shfl(b0, base + 2u);
                    uint b3 = __shfl(b0, base + 3u);
                    if ((lane & 3u) == 0u)
                        *reinterpret_cast<u32*>(hidden + (t0 + row0 + i) * hidden_c + hcol) =
                            b0 | b1 << 8 | b2 << 16 | b3 << 24;
                }
            }
        } else {
            for (uint n = 0; n < 4; ++n)
                ex[n].StoreF8(hidden + t0 * hidden_c, (sub * 4 + n) * 16, hidden_c);
            if (dbg_hidden)
                for (uint n = 0; n < 4; ++n)
                    for (uint i = 0; i < ex[n].Length(); ++i) {
                        uint2 c2 = ex[n].GetCoordinate(i);
                        dbg_hidden[(first + t0 + c2.x) * hidden_c + (sub * 4 + n) * 16 + c2.y] =
                            e4m3_byte(ex[n].Get(i));
                    }
        }
    }
    __syncthreads();
    if (group < GROUPS) {
        const uint t0 = group * 16;
        C32 acc = C32::Splat(0.f);
        if constexpr (C == 32) {
            const float ffn_scale = scales[sub * 16 + (threadIdx.x & 15u)];
            for (uint i = 0; i < acc.Length(); ++i) {
                uint2 rc = acc.GetCoordinate(i);
                if (first + t0 + rc.x < tokens) {
                    float iv = HIN
                                   ? __half2float(input_h[(first + t0 + rc.x) * C + sub * 16 + rc.y])
                                   : input[(first + t0 + rc.x) * C + sub * 16 + rc.y];
                    acc.Set(i, (chain_residual ? from_e4m3(e4m3_byte(iv)) : iv) * ffn_scale);
                }
            }
        }
        for (uint g = 0; g < hidden_c / 32; ++g) {
            A8 a = A8::Load(hidden + t0 * hidden_c, g * 32, hidden_c);
            B8 b = B8::LoadRowT(weights, hidden_c * C + (sub * (hidden_c / 32) + g) * 512, 16);
            acc.MultiplyAccumulate(a, b);
        }
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + t0 + rc.x < tokens) {
                if constexpr (C == 32) {
                    // HOUT stores the exact H(acc) in f16. The f16 -> f32 round
                    // trip is lossless, so QKV/p1-residual can consume it with half
                    // the global traffic while staying bit-identical.
                    if constexpr (HOUT)
                        output_h[(first + t0 + rc.x) * C + sub * 16 + rc.y] =
                            __float2half_rn(acc.Get(i));
                    else
                        output[(first + t0 + rc.x) * C + sub * 16 + rc.y] = H(acc.Get(i));
                } else {
                    u8 q = e4m3_byte(acc.Get(i));
                    if constexpr (F8OUT)
                        output_f8[(first + t0 + rc.x) * C + sub * 16 + rc.y] = q;
                    output[(first + t0 + rc.x) * C + sub * 16 + rc.y] = from_e4m3(q);
                }
            }
        }
    }
}

// One thread per 32-channel item: 8 aligned float4 loads, the original serial
// FP32 sum (a tree reduction changes FP8 rounding), packed 2x float4 stores.
__global__ void k_normalize_qkv(const float* input, const float* scales, u8* output,
                                uint tokens, uint c, float qgain, bool half_squares = true) {
    const uint count = tokens * (c / 32) * 3;
    const uint item = blockIdx.x * 256u + threadIdx.x;
    if (item >= count)
        return;
    const float4* row = reinterpret_cast<const float4*>(input + item * 32u);
    float4 v0 = row[0], v1 = row[1], v2 = row[2], v3 = row[3];
    float4 v4 = row[4], v5 = row[5], v6 = row[6], v7 = row[7];
    float vals[32];
    vals[0] = v0.x; vals[1] = v0.y; vals[2] = v0.z; vals[3] = v0.w;
    vals[4] = v1.x; vals[5] = v1.y; vals[6] = v1.z; vals[7] = v1.w;
    vals[8] = v2.x; vals[9] = v2.y; vals[10] = v2.z; vals[11] = v2.w;
    vals[12] = v3.x; vals[13] = v3.y; vals[14] = v3.z; vals[15] = v3.w;
    vals[16] = v4.x; vals[17] = v4.y; vals[18] = v4.z; vals[19] = v4.w;
    vals[20] = v5.x; vals[21] = v5.y; vals[22] = v5.z; vals[23] = v5.w;
    vals[24] = v6.x; vals[25] = v6.y; vals[26] = v6.z; vals[27] = v6.w;
    vals[28] = v7.x; vals[29] = v7.y; vals[30] = v7.z; vals[31] = v7.w;
    uint head = item % (c / 32), part = (item / (c / 32)) % 3;
    float sum = 0.f;
    if (part < 2)
        for (uint ch = 0; ch < 32; ++ch) {
            // Opaque register multiplies (sq_f32/sq_half_f32) block the FMA
            // contraction of sum += v*v without the volatile scratch round trip
            // that spilled 40 B/lane on gfx12.
            float square = half_squares ? sq_half_f32(vals[ch]) : sq_f32(vals[ch]);
            sum += square;
        }
    float inv = part == 2 ? 1.f : rsqrtf(fmaxf(sum, 6.198883056640625e-5f)) *
                (part == 0 ? scales[head] * qgain : 1.f);
    u8 b[32];
    for (uint ch = 0; ch < 32; ch++)
        b[ch] = e4m3_byte(vals[ch] * inv);
    u8* out = output + item * 32u;
    for (uint k = 0; k < 8; k++)
        *reinterpret_cast<u32*>(out + k * 4u) =
            u32(b[k * 4]) | u32(b[k * 4 + 1]) << 8 | u32(b[k * 4 + 2]) << 16 |
            u32(b[k * 4 + 3]) << 24;
}

// Four channels per thread: the 4 lanes are contiguous, so win/tok are shared
// and both ends are 16B vector accesses.
template <bool HIN = false>
__global__ void k_reframe_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                               uint px, uint py, uint c, int crop, int quantize,
                               const __half* src_h = nullptr) {
    uint i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i4 + 4u > (crop ? w * h : sw * sh) * c) return;
    uint ch = i4 % c, t = i4 / c;
    uint x = t % (crop ? w : sw), y = t / (crop ? w : sw);
    uint wx = crop ? x + px : x, wy = crop ? y + py : y;
    uint win = (wy / 8) * (sw / 8) + wx / 8, tok = (wy % 8) * 8 + wx % 8;
    float4 v;
    if (HIN) {
        // The C32 p1 output is H(v), exactly representable in f16; reading it as
        // half halves the crop traffic while producing the same f32 raster.
        const __half2* hp = reinterpret_cast<const __half2*>(src_h + (win * 64u + tok) * c + ch);
        __half2 h0 = hp[0], h1 = hp[1];
        v = make_float4(__half2float(__low2half(h0)), __half2float(__high2half(h0)),
                        __half2float(__low2half(h1)), __half2float(__high2half(h1)));
    } else if (crop)
        v = *reinterpret_cast<const float4*>(src + (win * 64u + tok) * c + ch);
    else
        v = x >= px && y >= py && x < w + px && y < h + py
            ? *reinterpret_cast<const float4*>(src + ((y - py) * w + x - px) * c + ch)
            : make_float4(0.f, 0.f, 0.f, 0.f);
    if (quantize)
        v = float4(from_e4m3(e4m3_byte(v.x)), from_e4m3(e4m3_byte(v.y)),
                   from_e4m3(e4m3_byte(v.z)), from_e4m3(e4m3_byte(v.w)));
    if (crop)
        *reinterpret_cast<float4*>(dst + i4) = v;
    else
        *reinterpret_cast<float4*>(dst + (win * 64u + tok) * c + ch) = v;
}

// Pack a raster E4M3 sibling (previous block p1) into the current block's
// window-major E4M3 input. Values are already on the FP8 grid, so this is a
// byte permute with padding zeros.
__global__ void k_reframe_f8in_f8out(const u8* src, u8* dst, uint w, uint h, uint sw, uint sh,
                                     uint px, uint py, uint c) {
    uint i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i4 + 4u > sw * sh * c) return;
    uint ch = i4 % c, t = i4 / c;
    uint x = t % sw, y = t / sw;
    uint win = (y / 8) * (sw / 8) + x / 8, tok = (y % 8) * 8 + x % 8;
    u32 v = x >= px && y >= py && x < w + px && y < h + py
                ? *reinterpret_cast<const u32*>(src + ((y - py) * w + x - px) * c + ch)
                : 0u;
    *reinterpret_cast<u32*>(dst + (win * 64u + tok) * c + ch) = v;
}

// Pack a raster f32 input that is already on the f16 grid into window-major f16.
__global__ void k_reframe_f32_to_h(const float* src, __half* dst, uint w, uint h, uint sw,
                                   uint sh, uint px, uint py, uint c) {
    uint i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i4 + 4u > sw * sh * c) return;
    uint ch = i4 % c, t = i4 / c;
    uint x = t % sw, y = t / sw;
    uint win = (y / 8) * (sw / 8) + x / 8, tok = (y % 8) * 8 + x % 8;
    float4 v = x >= px && y >= py && x < w + px && y < h + py
                   ? *reinterpret_cast<const float4*>(src + ((y - py) * w + x - px) * c + ch)
                   : make_float4(0.f, 0.f, 0.f, 0.f);
    __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
    __half2* hp = reinterpret_cast<__half2*>(dst + (win * 64u + tok) * c + ch);
    hp[0] = h0;
    hp[1] = h1;
}

// Pack a raster f16 C32 sibling into the current block's window-major f16 input.
__global__ void k_reframe_hin_hout(const __half* src, __half* dst, uint w, uint h, uint sw,
                                   uint sh, uint px, uint py, uint c) {
    uint i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i4 + 4u > sw * sh * c) return;
    uint ch = i4 % c, t = i4 / c;
    uint x = t % sw, y = t / sw;
    uint win = (y / 8) * (sw / 8) + x / 8, tok = (y % 8) * 8 + x % 8;
    uint2 v = x >= px && y >= py && x < w + px && y < h + py
                  ? *reinterpret_cast<const uint2*>(src + ((y - py) * w + x - px) * c + ch)
                  : make_uint2(0u, 0u);
    *reinterpret_cast<uint2*>(dst + (win * 64u + tok) * c + ch) = v;
}

// 16B vectorized device copy. The driver's hipMemcpyAsync D2D path runs at
// ~150GB/s on gfx1201 (copy engine), a plain kernel hits HBM bandwidth.
__global__ void k_copy_f32(const float* src, float* dst, uint n4) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4)
        reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[i];
}

__global__ void k_pool_f32(const float* src, float* dst, uint w, uint h, uint c, uint ow, uint oh,
                           __half* dst_h = nullptr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * oh * c) return;
    uint ch = i % c, t = i / c, x = t % ow, y = t / ow;
    float v;
    if (x >= w / 2 || y >= h / 2) v = 0.f;
    else {
        uint base = (y * 2 * w + x * 2) * c + ch;
        float top = H(src[base] + src[base + c]);
        float bottom = H(src[base + w * c] + src[base + (w + 1) * c]);
        v = from_e4m3(e4m3_byte(H(H(top + bottom) * .25f)));
    }
    if (dst)
        dst[i] = v;
    if (dst_h)
        dst_h[i] = __float2half_rn(v);
}

// True half operands for ViT QKV, decoder entry and pool/head matrices.
// partitions reproduce the fast shader's independent K reductions; ordered
// rounds after each K32 only for the head/DS reference path.
template <bool HIN = false, bool DIRECT_OUT = false>
__global__ void k_linear_half(const float* input,const __half* weights,float* output,
  uint m,uint n,uint k,uint partitions,int ordered,int raw,__half* output_h=nullptr) {
 using namespace linalg;
 uint first=blockIdx.x*16,col=blockIdx.y*16;
 __shared__ __half tile[512];
 C32 total=C32::Splat(0.f);
    for(uint part=0;part<partitions;++part){
    C32 acc=C32::Splat(0.f);
    for(uint g=part*(k/32/partitions);g<(part+1)*(k/32/partitions);++g){
     {
      const uint col=(threadIdx.x%8u)*4u;
      for(uint r=threadIdx.x/8u;r<16;r+=4){
       const uint row=first+r;
       if constexpr (HIN) {
        const uint2 packed=row<m
          ? *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(input)+row*k+g*32u+col)
          : uint2{0u,0u};
        *reinterpret_cast<uint2*>(tile+r*32u+col)=packed;
       } else {
        const float4 v=row<m?*reinterpret_cast<const float4*>(input+row*k+g*32u+col):make_float4(0.f,0.f,0.f,0.f);
        const __half h0=__float2half(v.x),h1=__float2half(v.y),h2=__float2half(v.z),h3=__float2half(v.w);
        *reinterpret_cast<uint2*>(tile+r*32u+col)=
         uint2(u32(__half_as_ushort(h0))|u32(__half_as_ushort(h1))<<16,
               u32(__half_as_ushort(h2))|u32(__half_as_ushort(h3))<<16);
       }
      }
     }
     __syncthreads();
    A16 a=A16::Load(tile,0,64);
     B16 b = DIRECT_OUT ? B16::LoadRowMajorOut(weights, col, g * 32u, k)
                        : B16::LoadRow(weights, ((col / 16) * (k / 32) + g) * 1024, 32);
    if(ordered){C32 p=Multiply(a,b);for(uint i=0;i<acc.Length();++i)acc.Set(i,H(acc.Get(i)+p.Get(i)));}
    else acc.MultiplyAccumulate(a,b);
   __syncthreads();
  }
  for(uint i=0;i<total.Length();++i)total.Set(i,total.Get(i)+acc.Get(i));
 }
  for(uint i=0;i<total.Length();++i){uint2 rc=total.GetCoordinate(i);if(first+rc.x<m){float v=total.Get(i);float out=raw==2?v:raw==1?H(v):from_e4m3(e4m3_byte(H(v)));if(output)output[(first+rc.x)*n+col+rc.y]=out;if(output_h)output_h[(first+rc.x)*n+col+rc.y]=__float2half_rn(out);}}
}

// Split FFWD group parallel, logical window-major input/output. No opaque
// tiled activation format crosses a launch boundary.
template <bool F8OUT, bool F8IN = false, bool HIN = false>
__global__ void k_split_f32(const float* in,const u8* pre,const u8* expand,const u8* contract,
                             float* out,u8* out_f8,uint tokens,const u8* in_f8=nullptr,
                             const __half* in_h=nullptr){
  using namespace linalg;
  uint first=blockIdx.x*16,group=blockIdx.y,wave=threadIdx.x/32;
  __shared__ u8 tile[512],mix[16*64],hid[16*256];
   C32 a=C32::Splat(0.f);
   for(uint g=0;g<16;++g){
    {
     const uint col=(threadIdx.x%8u)*4u, row=first+threadIdx.x/8u;
     if constexpr (F8IN)
         *reinterpret_cast<u32*>(tile+(threadIdx.x/8u)*32u+col) =
             row < tokens ? *reinterpret_cast<const u32*>(in_f8 + row * 512u + g * 32u + col) : 0u;
     else if constexpr (HIN) {
         // The upstream f16 sibling is already on the f16 grid; quantize to E4M3.
         const __half2* hp = reinterpret_cast<const __half2*>(in_h + row * 512u + g * 32u + col);
         __half2 h0 = row < tokens ? hp[0] : __half2(0.f, 0.f);
         __half2 h1 = row < tokens ? hp[1] : __half2(0.f, 0.f);
         *reinterpret_cast<u32*>(tile+(threadIdx.x/8u)*32u+col) =
             u32(e4m3_byte(__half2float(__low2half(h0)))) |
             u32(e4m3_byte(__half2float(__high2half(h0)))) << 8 |
             u32(e4m3_byte(__half2float(__low2half(h1)))) << 16 |
             u32(e4m3_byte(__half2float(__high2half(h1)))) << 24;
     } else {
         const float4 v=row<tokens?*reinterpret_cast<const float4*>(in+row*512u+g*32u+col):make_float4(0.f,0.f,0.f,0.f);
         *reinterpret_cast<u32*>(tile+(threadIdx.x/8u)*32u+col)=
          u32(e4m3_byte(v.x))|u32(e4m3_byte(v.y))<<8|u32(e4m3_byte(v.z))<<16|u32(e4m3_byte(v.w))<<24;
     }
    }
   __syncthreads();
  a.MultiplyAccumulate(A8::Load(tile,0,32),B8::LoadRowT(pre,((group*4+wave)*16+g)*512,16));__syncthreads();
 }
 for(uint i=0;i<a.Length();++i)a.Set(i,H(a.Get(i)));a.StoreF8(mix,wave*16,64);__syncthreads();
 for(uint j=0;j<4;++j){C32 acc=C32::Splat(0.f);uint col=wave*64+j*16;
  for(uint g=0;g<2;++g)acc.MultiplyAccumulate(A8::Load(mix,g*32,64),B8::LoadRowT(expand,group*16384+((col/16)*2+g)*512,16));
  for(uint i=0;i<acc.Length();++i)acc.Set(i,ActivatePoly(H(acc.Get(i))));acc.StoreF8(hid,col,256);
 }
 __syncthreads();C32 acc=C32::Splat(0.f);
 for(uint g=0;g<8;++g)acc.MultiplyAccumulate(A8::Load(hid,g*32,256),B8::LoadRowT(contract,group*16384+(wave*8+g)*512,16));
  for(uint i=0;i<acc.Length();++i){uint2 rc=acc.GetCoordinate(i);if(first+rc.x<tokens){u8 q=e4m3_byte(H(acc.Get(i)));if constexpr(F8OUT)out_f8[(first+rc.x)*512+group*64+wave*16+rc.y]=q;out[(first+rc.x)*512+group*64+wave*16+rc.y]=from_e4m3(q);}}
}

__global__ void k_rgb_graph(const float* in,float* raster,float* tiles,uint w,uint h,uint ow,uint oh){
 uint p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=ow*oh)return;
 uint x=p%ow,y=p/ow,rx=x%(2*(w-1)),ry=y%(2*(h-1));rx=rx<w?rx:2*(w-1)-rx;ry=ry<h?ry:2*(h-1)-ry;
 uint t=((y/8)*(ow/8)+x/8)*64+(y%8)*8+x%8;
 for(uint c=0;c<4;++c){float v=in[(ry*w+rx)*4+c];raster[p*4+c]=v;tiles[t*4+c]=v;}
}
__global__ void k_gather_f32(const float* in,const int* map,float* out,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=in[map[i]];}
__global__ void k_up_f32(const float* lo,const float* skip,const float* scale,float* out,uint iw,uint ow,uint oh,uint c,__half* out_h=nullptr){
  uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ow*oh*c)return;
  uint ch=i%c,p=i/c,x=p%ow,y=p/ow;
  float v=H(lo[((y/2)*iw+x/2)*c+ch]+from_e4m3(e4m3_byte(skip[i]))*scale[ch]);
  float r=c==32?v:from_e4m3(e4m3_byte(v));
  if(out)out[i]=r;
  if(out_h)out_h[i]=__float2half_rn(r);
}
// v ends on the f16 grid (outer H round). When the f16 sibling is published
// (out_h != null) the f32 write is pure traffic: no consumer reads it (block 70
// takes the f16 path, the head reads the f16 raster), so skip it.
// Eight channels per thread: the 32-wide pixel rows stay 16B-aligned, so the
// f16 publish is one 16B store and every input load is 16B (the 1-channel
// version paid read-modify-write on sub-cache-line accesses).
__global__ void k_post_merge_f32(const float* low,const float* pre,const float* scales,float* out,uint w,uint h,bool main8_low = false,__half* out_h=nullptr){
  uint i8=(blockIdx.x*blockDim.x+threadIdx.x)*8u;if(i8+8u>w*h*32u)return;
  uint ch=i8&31u,p=i8/32u,lp=((p/w)/2u)*(w/2u)+(p%w)/2u;
  const float* lo_ptr=low+lp*32u+ch;
  float4 lo0,lo1;
  if(main8_low){
    lo0=make_float4(from_e4m3(e4m3_byte(lo_ptr[0])),from_e4m3(e4m3_byte(lo_ptr[1])),
                    from_e4m3(e4m3_byte(lo_ptr[2])),from_e4m3(e4m3_byte(lo_ptr[3])));
    lo1=make_float4(from_e4m3(e4m3_byte(lo_ptr[4])),from_e4m3(e4m3_byte(lo_ptr[5])),
                    from_e4m3(e4m3_byte(lo_ptr[6])),from_e4m3(e4m3_byte(lo_ptr[7])));
  }else{
    lo0=*reinterpret_cast<const float4*>(lo_ptr);
    lo1=*reinterpret_cast<const float4*>(lo_ptr+4);
  }
  float4 pr0=*reinterpret_cast<const float4*>(pre+i8);
  float4 pr1=*reinterpret_cast<const float4*>(pre+i8+4);
  float4 v0,v1;
  v0.x=H(H(lo0.x*scales[ch])+from_e4m3(e4m3_byte(pr0.x))*scales[32u+ch]);
  v0.y=H(H(lo0.y*scales[ch+1u])+from_e4m3(e4m3_byte(pr0.y))*scales[32u+ch+1u]);
  v0.z=H(H(lo0.z*scales[ch+2u])+from_e4m3(e4m3_byte(pr0.z))*scales[32u+ch+2u]);
  v0.w=H(H(lo0.w*scales[ch+3u])+from_e4m3(e4m3_byte(pr0.w))*scales[32u+ch+3u]);
  v1.x=H(H(lo1.x*scales[ch+4u])+from_e4m3(e4m3_byte(pr1.x))*scales[32u+ch+4u]);
  v1.y=H(H(lo1.y*scales[ch+5u])+from_e4m3(e4m3_byte(pr1.y))*scales[32u+ch+5u]);
  v1.z=H(H(lo1.z*scales[ch+6u])+from_e4m3(e4m3_byte(pr1.z))*scales[32u+ch+6u]);
  v1.w=H(H(lo1.w*scales[ch+7u])+from_e4m3(e4m3_byte(pr1.w))*scales[32u+ch+7u]);
  if(!out_h){
    *reinterpret_cast<float4*>(out+i8)=v0;
    *reinterpret_cast<float4*>(out+i8+4)=v1;
  }
  if(out_h){
    __half2 h0=__floats2half2_rn(v0.x,v0.y),h1=__floats2half2_rn(v0.z,v0.w),
            h2=__floats2half2_rn(v1.x,v1.y),h3=__floats2half2_rn(v1.z,v1.w);
    uint4 o;
    o.x=*reinterpret_cast<const uint*>(&h0);
    o.y=*reinterpret_cast<const uint*>(&h1);
    o.z=*reinterpret_cast<const uint*>(&h2);
    o.w=*reinterpret_cast<const uint*>(&h3);
    *reinterpret_cast<uint4*>(out_h+i8)=o;
  }
}
// Head over block 70's windowed f16 output: same (px,py)-shifted win/tok remap as
// k_reframe_f32<HIN>; the values are f16-exact, so this is bit-identical to
// k_head_f32 on the reframed f32 raster.
__global__ void k_head_f16(const __half* feat,const float* weights,const float* color,float* out,uint w,uint h,uint sw,uint sh,uint px,uint py){
 uint p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=w*h)return;
 uint x=p%w,y=p/w,wx=x+px,wy=y+py;
 uint win=(wy/8)*(sw/8)+wx/8,tok=(wy%8)*8+wx%8;
 const __half* f=feat+(win*64u+tok)*32u;
#pragma unroll
 for(uint row=0;row<3;++row){float acc=0;
#pragma unroll
  for(uint c=0;c<32;++c)acc+=__half2float(f[c])*weights[row*32+c];
  float base=color[p*4+row]*.125f-.0625f;float encoded=H(acc)*.03125f+base;out[p*3+row]=fminf(fmaxf(encoded*8.f+.5f,0.f),1.f);}
}
__global__ void k_head_f32(const float* feat,const float* weights,const float* color,float* out,uint n){
 uint p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 const float* f=feat+p*32;
#pragma unroll
 for(uint row=0;row<3;++row){float acc=0;
#pragma unroll
  for(uint c=0;c<32;++c)acc+=f[c]*weights[row*32+c];
  float base=color[p*4+row]*.125f-.0625f;float encoded=H(acc)*.03125f+base;out[p*3+row]=fminf(fmaxf(encoded*8.f+.5f,0.f),1.f);}
}
// Per-lane trace reductions avoid atomics and CPU synchronization in the graph.
__global__ void k_trace_f32(const float* src,uint n,float* stats,uint* bits){
 uint lane=threadIdx.x;float lo=INFINITY,hi=-INFINITY,sum=0;uint bad=0,hash=0;
 for(uint i=lane;i<n;i+=256){float v=src[i];if(!isfinite(v))++bad;else{lo=fminf(lo,v);hi=fmaxf(hi,v);sum+=v;}uint z=__float_as_uint(v)^((i+1)*0x9e3779b9u);z=(z^(z>>16))*0x85ebca6bu;z=(z^(z>>13))*0xc2b2ae35u;hash^=z^(z>>16);}
 stats[lane*3]=lo;stats[lane*3+1]=hi;stats[lane*3+2]=sum;bits[lane*2]=bad;bits[lane*2+1]=hash;
}

} // namespace dlss5
