#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include <cstdlib>
#include "../kernels/ffn_fused.hip"
#include "../kernels/qkv.hip"
#include "../kernels/qkv_norm.hip"
#include "../kernels/project.hip"
#include "../kernels/prefix.hip"
#include "../kernels/attention.hip"
#include "../kernels/layout.hip"
#include "../kernels/vit.hip"
#include "../kernels/stage.hip"
#include "../kernels/fused.hip"
#include "../kernels/ops.hip"
#include "../kernels/c32_fused.hip"

namespace dlss5 {

static thread_local hipStream_t g_stream;

void set_launch_stream(hipStream_t s) { g_stream = s; }

#define LAUNCH(name, grid, block, ...)                                                             \
    do {                                                                                           \
        name<<<grid, block, 0, g_stream>>>(__VA_ARGS__);                                            \
    } while (0)

static uint grid_cap(uint n) { return n ? n : 1u; }

void launch_ffn_f32(int c, const float* in, const u8* w, const float* scales, float* out, uint tokens,
                    bool chain_residual, bool precise_c32) {
    if (!tokens || in == out || (c == 32 && !scales)) throw std::runtime_error("FFN extent/alias/scales");
    uint groups = (tokens + 15) / 16;
    switch (c) {
    case 32: LAUNCH((k_ffn_f32<32, false, 1>), (tokens + 15) / 16, 64, in, w, scales, out, nullptr,
                    tokens, chain_residual, precise_c32); break;
    case 64: LAUNCH((k_ffn_f32<64, false>), groups, 128, in, w, scales, out, nullptr, tokens); break;
    case 128: LAUNCH((k_ffn_f32<128, false>), groups, 256, in, w, scales, out, nullptr, tokens); break;
    case 256: LAUNCH((k_ffn_f32<256, false>), groups, 512, in, w, scales, out, nullptr, tokens); break;
    default: throw std::runtime_error("FFN channels");
    }
}
void launch_ffn_f32_hout(int c, const float* in, const u8* w, const float* scales, __half* out_h,
                          uint tokens, bool chain_residual, bool precise_c32) {
    if (!tokens || c != 32 || !scales || !out_h)
        throw std::runtime_error("FFN f16 extent/alias");
    LAUNCH((k_ffn_f32<32, false, 1, true>), (tokens + 15) / 16, 64, in, w, scales, nullptr,
           nullptr, tokens, chain_residual, precise_c32, out_h);
}
void launch_ffn_f32_hin_hout(int c, const __half* in_h, const u8* w, const float* scales,
                             __half* out_h, uint tokens, bool chain_residual, bool precise_c32) {
    if (!tokens || c != 32 || !in_h || !scales || !out_h)
        throw std::runtime_error("FFN f16 in extent/alias");
    LAUNCH((k_ffn_f32<32, false, 1, true, false, true>), (tokens + 15) / 16, 64, nullptr, w,
           scales, nullptr, nullptr, tokens, chain_residual, precise_c32, out_h, nullptr, in_h);
}
void launch_ffn_f32_f8(int c, const float* in, const u8* w, const float* scales, float* out_f32,
                        u8* out_f8, uint tokens, bool chain_residual, bool precise_c32) {
    if (!tokens || c == 32 || !out_f32 || !out_f8)
        throw std::runtime_error("FFN f8 extent/alias");
    uint groups = (tokens + 15) / 16;
    switch (c) {
    case 64: LAUNCH((k_ffn_f32<64, true>), groups, 128, in, w, scales, out_f32, out_f8, tokens); break;
    case 128: LAUNCH((k_ffn_f32<128, true>), groups, 256, in, w, scales, out_f32, out_f8, tokens); break;
    case 256: LAUNCH((k_ffn_f32<256, true>), groups, 512, in, w, scales, out_f32, out_f8, tokens); break;
    default: throw std::runtime_error("FFN f8 channels");
    }
}
void launch_ffn_f32_f8in_f8out(int c, const u8* in_f8, const u8* w, const float* scales,
                               float* out_f32, u8* out_f8, uint tokens, bool chain_residual,
                               bool precise_c32) {
    if (!tokens || c == 32 || !in_f8 || !out_f32 || !out_f8)
        throw std::runtime_error("FFN f8in extent/alias");
    uint groups = (tokens + 15) / 16;
    switch (c) {
    case 64: LAUNCH((k_ffn_f32<64, true, 1, false, true>), groups, 128, nullptr, w, scales, out_f32,
                    out_f8, tokens, chain_residual, precise_c32, nullptr, in_f8); break;
    case 128: LAUNCH((k_ffn_f32<128, true, 1, false, true>), groups, 256, nullptr, w, scales, out_f32,
                     out_f8, tokens, chain_residual, precise_c32, nullptr, in_f8); break;
    case 256: LAUNCH((k_ffn_f32<256, true, 1, false, true>), groups, 512, nullptr, w, scales, out_f32,
                     out_f8, tokens, chain_residual, precise_c32, nullptr, in_f8); break;
    default: throw std::runtime_error("FFN f8in channels");
    }
}
 void launch_ffn_f32_hin_f8out(int c, const __half* in_h, const u8* w, const float* scales,
                               float* out_f32, u8* out_f8, uint tokens, bool chain_residual,
                               bool precise_c32, unsigned char* dbg_hidden, float* dbg_ex) {
     if (!tokens || c == 32 || c == 512 || !in_h || !out_f32 || !out_f8)
         throw std::runtime_error("FFN f16in extent/alias");
     uint groups = (tokens + 15) / 16;
     switch (c) {
     case 64: LAUNCH((k_ffn_f32<64, true, 1, false, false, true>), groups, 128, nullptr, w, scales,
                     out_f32, out_f8, tokens, chain_residual, precise_c32, nullptr, nullptr, in_h, dbg_hidden, dbg_ex); break;
    case 128: LAUNCH((k_ffn_f32<128, true, 1, false, false, true>), groups, 256, nullptr, w, scales,
                     out_f32, out_f8, tokens, chain_residual, precise_c32, nullptr, nullptr, in_h, dbg_hidden, dbg_ex); break;
    case 256: LAUNCH((k_ffn_f32<256, true, 1, false, false, true>), groups, 512, nullptr, w, scales,
                     out_f32, out_f8, tokens, chain_residual, precise_c32, nullptr, nullptr, in_h, dbg_hidden, dbg_ex); break;
    default: throw std::runtime_error("FFN f16in channels");
    }
}

void launch_normalize_qkv(const float* in, const float* scales, u8* out,
                          uint tokens, uint c, float qgain, bool half_squares) {
    if (!tokens || !c || c % 32 || !scales) throw std::runtime_error("QKV extent/scales");
    uint count = tokens * (c / 32) * 3;
    LAUNCH(k_normalize_qkv, (count + 255) / 256, 256, in, scales, out, tokens, c, qgain, half_squares);
}

static bool vit_qkv_split() {
    const char* e = std::getenv("DLSS5_VIT_QKV_SPLIT");
    return e && e[0] == '1' && e[1] == '\0';
}
void launch_qkv_norm_half(const float* in, const __half* w, u8* out, uint m, const float* scales,
                          float qgain) {
    if (!m || !in || !w || !out || !scales) throw std::runtime_error("ViT QKV extent/scales");
    // One CTA owns Q+K+V of one head, so the f16 input tile is staged once.
    // DLSS5_VIT_QKV_SPLIT=1 restores one CTA per Q/K/V part.
    if (vit_qkv_split())
        LAUNCH(k_vit_qkv_norm, dim3((m + 15) / 16, 96u), 64, in, w, out, m, scales, qgain);
    else
        LAUNCH(k_vit_qkv_norm_fused, dim3((m + 15) / 16, 32u), 64, in, w, out, m, scales, qgain);
}

// GEMM + normalize + quantize in one launch (bit-identical to the split pair).
void launch_qkv_norm_f32(int c, const float* in, const u8* w, u8* out, uint m,
                         const float* scales, float qgain, bool half_squares) {
    if (!m || !c || c % 32 || !scales) throw std::runtime_error("QKV norm extent/scales");
    switch (c) {
    case 32:
        // NH=3: one CTA computes Q+K+V for 16 rows, so the f32/FP8 input tile is
        // staged once instead of once per Q/K/V CTA.
        LAUNCH((k_qkv_norm_f32<32, 3, false>), dim3((m + 15) / 16, 1u), 192, in, nullptr, w, out,
               m, c, scales, qgain, half_squares);
        break;
    case 64: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<64, 2, false>), dim3((m + 15) / 16, heads), 128, in, nullptr, w,
                out, m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<64, 2, false, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                in, nullptr, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    case 128: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<128, 2, false>), dim3((m + 15) / 16, heads), 128, in, nullptr, w,
                out, m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<128, 2, false, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                in, nullptr, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    case 256: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<256, 2, false>), dim3((m + 15) / 16, heads), 128, in, nullptr, w,
                out, m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<256, 2, false, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                in, nullptr, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    case 512: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<512, 2, false>), dim3((m + 15) / 16, heads), 128, in, nullptr, w,
                out, m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<512, 2, false, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                in, nullptr, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    default:
        throw std::runtime_error("QKV norm extent");
    }
}
void launch_qkv_norm_f32_f8in(int c, const u8* in, const u8* w, u8* out, uint m,
                              const float* scales, float qgain, bool half_squares) {
    if (!m || c == 32 || !in || !out || !scales)
        throw std::runtime_error("QKV norm f8 extent");
    switch (c) {
    case 64: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<64, 2, true>), dim3((m + 15) / 16, heads), 128, nullptr, in, w, out,
                m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<64, 2, true, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                nullptr, in, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    case 128: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<128, 2, true>), dim3((m + 15) / 16, heads), 128, nullptr, in, w, out,
                m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<128, 2, true, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                nullptr, in, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    case 256: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<256, 2, true>), dim3((m + 15) / 16, heads), 128, nullptr, in, w, out,
                m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<256, 2, true, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                nullptr, in, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    case 512: {
        uint heads = c / 32u;
        LAUNCH((k_qkv_norm_f32<512, 2, true>), dim3((m + 15) / 16, heads), 128, nullptr, in, w, out,
                m, c, scales, qgain, half_squares);
        LAUNCH((k_qkv_norm_f32<512, 2, true, false, true>), dim3((m + 15) / 16, heads / 2u), 128,
                nullptr, in, w, out, m, c, scales, qgain, half_squares, nullptr, 2u * heads);
        break;
    }
    default:
        throw std::runtime_error("QKV norm f8 channels");
    }
}
void launch_qkv_norm_f32_hin(int c, const __half* in, const u8* w, u8* out, uint m,
                             const float* scales, float qgain, bool half_squares) {
    if (!m || c != 32 || !in || !out || !scales)
        throw std::runtime_error("QKV norm f16 extent");
    // Same NH=3 CTA as the f32 path; the f16 input is the exact H(acc) grid.
    LAUNCH((k_qkv_norm_f32<32, 3, false, true>), dim3((m + 15) / 16, 1u), 192, nullptr, nullptr,
           w, out, m, c, scales, qgain, half_squares, in);
}
void launch_reframe_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                        uint px, uint py, uint c, int crop, int quantize) {
    if (!w || !h || sw % 8 || sh % 8 || w + px > sw || h + py > sh || src == dst)
        throw std::runtime_error("reframe extent/alias");
    uint count = (crop ? w * h : sw * sh) * c;
    LAUNCH((k_reframe_f32<false>), (count / 4 + 255) / 256, 256, src, dst, w, h, sw, sh, px, py, c,
           crop, quantize);
}
void launch_reframe_f32_hin(const __half* src_h, float* dst, uint w, uint h, uint sw, uint sh,
                             uint px, uint py, uint c, int crop, int quantize) {
    if (!w || !h || sw % 8 || sh % 8 || w + px > sw || h + py > sh || !src_h || !dst)
        throw std::runtime_error("reframe f16 extent/alias");
    uint count = (crop ? w * h : sw * sh) * c;
    LAUNCH((k_reframe_f32<true>), (count / 4 + 255) / 256, 256, nullptr, dst, w, h, sw, sh, px, py,
           c, crop, quantize, src_h);
}
void launch_reframe_f8in_f8out(const u8* src, u8* dst, uint w, uint h, uint sw, uint sh,
                               uint px, uint py, uint c) {
    if (!w || !h || sw % 8 || sh % 8 || c % 4 || w + px > sw || h + py > sh || !src || !dst)
        throw std::runtime_error("reframe f8 extent/alias");
    uint count = sw * sh * c;
    LAUNCH(k_reframe_f8in_f8out, (count / 4 + 255) / 256, 256, src, dst, w, h, sw, sh, px, py, c);
}
void launch_reframe_f32_to_h(const float* src, __half* dst, uint w, uint h, uint sw, uint sh,
                             uint px, uint py, uint c) {
    if (!w || !h || sw % 8 || sh % 8 || c % 4 || w + px > sw || h + py > sh || !src || !dst)
        throw std::runtime_error("reframe f32->f16 extent/alias");
    uint count = sw * sh * c;
    LAUNCH(k_reframe_f32_to_h, (count / 4 + 255) / 256, 256, src, dst, w, h, sw, sh, px, py, c);
}
void launch_reframe_hin_hout(const __half* src, __half* dst, uint w, uint h, uint sw, uint sh,
                             uint px, uint py, uint c) {
    if (!w || !h || sw % 8 || sh % 8 || c % 4 || w + px > sw || h + py > sh || !src || !dst)
        throw std::runtime_error("reframe f16 extent/alias");
    uint count = sw * sh * c;
    LAUNCH(k_reframe_hin_hout, (count / 4 + 255) / 256, 256, src, dst, w, h, sw, sh, px, py, c);
}
void launch_pool_f32(const float* src, float* dst, uint w, uint h, uint c, uint ow, uint oh) {
    if (!w || !h || w % 2 || h % 2 || ow < w / 2 || oh < h / 2 || src == dst)
        throw std::runtime_error("pool extent/alias");
    LAUNCH(k_pool_f32, (ow * oh * c + 255) / 256, 256, src, dst, w, h, c, ow, oh);
}
void launch_pool_f32_h(const float* src, float* dst, __half* dst_h, uint w, uint h, uint c,
                        uint ow, uint oh) {
    if (!w || !h || w % 2 || h % 2 || ow < w / 2 || oh < h / 2 || src == dst || !dst_h)
        throw std::runtime_error("pool f16 extent/alias");
    LAUNCH(k_pool_f32, (ow * oh * c + 255) / 256, 256, src, dst, w, h, c, ow, oh, dst_h);
}
// The C32 pool inputs are the f16 window-major siblings (s.ah / skip byte copy),
// remapped through the source block's (px,py) shift; values are f16-exact.
void launch_pool_f16win_h(const __half* src, float* dst, __half* dst_h, uint w, uint h, uint c,
                           uint ow, uint oh, uint sw, uint sh, uint px, uint py) {
    if (!w || !h || w % 2 || h % 2 || ow < w / 2 || oh < h / 2 || !dst_h)
        throw std::runtime_error("pool f16win extent/alias");
    LAUNCH(k_pool_f16win_h, (ow * oh * c + 255) / 256, 256, src, dst, w, h, c, ow, oh, sw, sh, px, py, dst_h);
}

// Fat-M dispatch: one B-tile load per K step serves SUB 16-row subtiles per
// wave. Same numerics as k_linear_f32 (identical K order, partitions,
// residual init and epilogue), less global weight traffic. Falls back to the
// 16-row CTA kernel when m is too small for a full fat grid.
template <bool F8IN, bool F8OUT, bool HRID, bool HOUT, bool F8OUT_RASTER, bool F8RES,
          bool HOUT_RASTER, int SUB, bool PART>
static void launch_linear_fat(const float* in, const u8* in8, const u8* w, float* out, u8* out8,
                              uint m, uint n, uint k, const float* residual, const float* scales,
                              int mode, int ordered, bool matrix_residual,
                              const __half* residual_h, __half* out_h, u8* out8_raster, uint rw,
                              uint rh, uint rsw, uint rpx, uint rpy, const u8* residual_f8,
                              __half* out_h_raster) {
    if (m >= 16u * SUB)
        LAUNCH((k_linear_f32_fat<F8IN, F8OUT, HRID, HOUT, F8OUT_RASTER, F8RES, HOUT_RASTER, SUB,
                                  PART>),
               dim3((m + 16u * SUB - 1u) / (16u * SUB), (n + 63) / 64), n < 64 ? 64 : 128, in, in8, w,
               out, out8, m, n, k, residual, scales, mode, ordered, matrix_residual, residual_h,
               out_h, out8_raster, rw, rh, rsw, rpx, rpy, residual_f8, out_h_raster);
    else
        LAUNCH((k_linear_f32<F8IN, F8OUT, HRID, HOUT, F8OUT_RASTER, F8RES, HOUT_RASTER>),
               dim3((m + 15) / 16, (n + 63) / 64), n < 64 ? 64 : 128, in, in8, w, out, out8, m, n, k,
               residual, scales, mode, ordered, matrix_residual, residual_h, out_h, out8_raster,
               rw, rh, rsw, rpx, rpy, residual_f8, out_h_raster);
}
template <bool F8IN, bool F8OUT, bool HRID, bool HOUT, bool F8OUT_RASTER, bool F8RES,
          bool HOUT_RASTER, int SUB>
static void launch_linear_fat_ord(const float* in, const u8* in8, const u8* w, float* out, u8* out8,
                                  uint m, uint n, uint k, const float* residual,
                                  const float* scales, int mode, int ordered, bool matrix_residual,
                                  const __half* residual_h, __half* out_h, u8* out8_raster, uint rw,
                                  uint rh, uint rsw, uint rpx, uint rpy, const u8* residual_f8,
                                  __half* out_h_raster) {
    if (ordered == 2)
        launch_linear_fat<F8IN, F8OUT, HRID, HOUT, F8OUT_RASTER, F8RES, HOUT_RASTER, SUB, true>(
            in, in8, w, out, out8, m, n, k, residual, scales, mode, ordered, matrix_residual,
            residual_h, out_h, out8_raster, rw, rh, rsw, rpx, rpy, residual_f8, out_h_raster);
    else
        launch_linear_fat<F8IN, F8OUT, HRID, HOUT, F8OUT_RASTER, F8RES, HOUT_RASTER, SUB, false>(
            in, in8, w, out, out8, m, n, k, residual, scales, mode, ordered, matrix_residual,
            residual_h, out_h, out8_raster, rw, rh, rsw, rpx, rpy, residual_f8, out_h_raster);
}

void launch_linear_f32(const float* in, const u8* w, float* out, uint m, uint n, uint k,
                       const float* residual, const float* scales, int mode, int ordered, bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || in == out || ordered < 0 || ordered > 2 ||
        (ordered == 2 && k % 128))
        throw std::runtime_error("linear geometry/alias");
    launch_linear_fat_ord<false, false, false, false, false, false, false, 4>(
        in, nullptr, w, out, nullptr, m, n, k, residual, scales, mode, ordered, matrix_residual,
        nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr, nullptr);
}
void launch_linear_f32_f8out(const float* in, const u8* w, float* out, u8* out_f8, uint m,
                             uint n, uint k, const float* residual, const float* scales, int mode,
                             int ordered, bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || in == out || ordered < 0 || ordered > 2 ||
        (ordered == 2 && k % 128) || !out_f8)
        throw std::runtime_error("linear geometry/alias");
    launch_linear_fat_ord<false, true, false, false, false, false, false, 4>(
        in, nullptr, w, out, out_f8, m, n, k, residual, scales, mode, ordered, matrix_residual,
        nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr, nullptr);
}
void launch_linear_f32_f8in(const u8* in, const u8* w, float* out, uint m, uint n, uint k,
                            const float* residual, const float* scales, int mode, int ordered,
                            bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out))
        throw std::runtime_error("linear geometry/alias");
    launch_linear_fat_ord<true, false, false, false, false, false, false, 4>(
        nullptr, in, w, out, nullptr, m, n, k, residual, scales, mode, ordered, matrix_residual,
        nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr, nullptr);
}
void launch_linear_f32_f8in_hrid(const u8* in, const u8* w, float* out, uint m, uint n, uint k,
                                 const __half* residual_h, const float* scales, int mode, int ordered,
                                 bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual_h && !scales) || ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out))
        throw std::runtime_error("linear geometry/alias");
    // residual_h doubles as the non-null enable flag for the residual path.
    launch_linear_fat_ord<true, false, true, false, false, false, false, 4>(
        nullptr, in, w, out, nullptr, m, n, k, reinterpret_cast<const float*>(residual_h), scales,
        mode, ordered, matrix_residual, residual_h, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
        nullptr);
}
void launch_linear_f32_f8in_hrid_hout(const u8* in, const u8* w, __half* out_h, uint m, uint n,
                                      uint k, const __half* residual_h, const float* scales,
                                      int mode, int ordered, bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual_h && !scales) || ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) || !out_h)
        throw std::runtime_error("linear geometry/alias");
    launch_linear_fat_ord<true, false, true, true, false, false, false, 4>(
        nullptr, in, w, nullptr, nullptr, m, n, k, reinterpret_cast<const float*>(residual_h),
        scales, mode, ordered, matrix_residual, residual_h, out_h, nullptr, 0, 0, 0, 0, 0, nullptr,
        nullptr);
}
void launch_linear_f32_f8in_hrid_hout_raster(const u8* in, const u8* w, __half* out_h,
                                              __half* out_h_raster, uint m, uint n, uint k,
                                              const __half* residual_h, const float* scales, int mode,
                                              int ordered, bool matrix_residual, uint rw, uint rh,
                                              uint rsw, uint rpx, uint rpy) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 || (residual_h && !scales) ||
        ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) || !out_h || !out_h_raster ||
        rsw % 8 || rw + rpx > rsw || rh + rpy > rsw)
        throw std::runtime_error("linear raster f16 extent/alias");
    launch_linear_fat_ord<true, false, true, true, false, false, true, 4>(
        nullptr, in, w, nullptr, nullptr, m, n, k, reinterpret_cast<const float*>(residual_h),
        scales, mode, ordered, matrix_residual, residual_h, out_h, nullptr, rw, rh, rsw, rpx, rpy,
        reinterpret_cast<const u8*>(residual_h), out_h_raster);
}
void launch_linear_f32_f8in_f8out(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                  uint n, uint k, const float* residual, const float* scales,
                                  int mode, int ordered, bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out) || !out_f8)
        throw std::runtime_error("linear geometry/alias");
    launch_linear_fat_ord<true, true, false, false, false, false, false, 4>(
        nullptr, in, w, out, out_f8, m, n, k, residual, scales, mode, ordered, matrix_residual,
        nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr, nullptr);
}
void launch_linear_f32_f8in_f8out_f8res(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                        uint n, uint k, const u8* residual_f8,
                                        const float* scales, int mode, int ordered,
                                        bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 || (!residual_f8 && !scales) ||
        ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out) || !out_f8)
        throw std::runtime_error("linear f8res extent/alias");
    launch_linear_fat_ord<true, true, false, false, false, true, false, 4>(
        nullptr, in, w, out, out_f8, m, n, k, reinterpret_cast<const float*>(residual_f8), scales,
        mode, ordered, matrix_residual, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, residual_f8,
        nullptr);
}
void launch_linear_f32_f8in_f8out_hrid(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                       uint n, uint k, const __half* residual_h,
                                       const float* scales, int mode, int ordered,
                                       bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 || (residual_h && !scales) ||
        ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out) || !out_f8)
        throw std::runtime_error("linear f16res extent/alias");
    launch_linear_fat_ord<true, true, true, false, false, false, false, 4>(
        nullptr, in, w, out, out_f8, m, n, k, reinterpret_cast<const float*>(residual_h), scales,
        mode, ordered, matrix_residual, residual_h, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
        nullptr);
}
void launch_linear_f32_f8in_f8out_raster(const u8* in, const u8* w, float* out, u8* out_f8_raster,
                                          uint m, uint n, uint k, const float* residual,
                                          const float* scales, int mode, int ordered,
                                          bool matrix_residual, uint rw, uint rh, uint rsw,
                                          uint rpx, uint rpy) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 || (residual && !scales) ||
        ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out) || !out_f8_raster ||
        rsw % 8 || rw + rpx > rsw || rh + rpy > rsw)
        throw std::runtime_error("linear raster f8 extent/alias");
    launch_linear_fat_ord<true, false, false, false, true, false, false, 4>(
        nullptr, in, w, out, nullptr, m, n, k, residual, scales, mode, ordered, matrix_residual,
        nullptr, nullptr, out_f8_raster, rw, rh, rsw, rpx, rpy, nullptr, nullptr);
}


static bool vit_ffn_lds() {
    const char* e = std::getenv("DLSS5_VIT_FFN_LDS");
    return e && e[0] == '1' && e[1] == '\0';
}
void launch_vit_ffn_fused(const float* in, const u8* w_expand, const u8* w_contract,
                          const float* scales, float* out, uint tokens, float* partial) {
    if (!tokens || tokens % 16 || tokens % 4 || !in || !w_expand || !w_contract || !scales ||
        !out || in == out)
        throw std::runtime_error("vit ffn extent/alias");
    // Four quarter CTAs (160 at 640 tokens) so every CU gets work. The f32
    // quarter partials are added in order by k_vit_ffn_reduce.
    // DLSS5_VIT_FFN_LDS=1 keeps the single 16-wave CTA.
    if (vit_ffn_lds())
        LAUNCH(k_vit_ffn_fused, tokens / 16u, 512, in, w_expand, w_contract, scales, out);
    else {
        if (!partial || partial == in || partial == out)
            throw std::runtime_error("vit ffn partial buffer");
        LAUNCH(k_vit_ffn_quarter, dim3(tokens / 16u, 4u), 512, in, w_expand, w_contract, partial,
               tokens);
        LAUNCH(k_vit_ffn_reduce, ((tokens * 1024u / 4u) + 255u) / 256u, 256, in, scales, partial, out,
               tokens);
    }
}

void launch_ffn(int c, const u8* in, const u8* w, u8* out, uint tokens) {
    uint groups = grid_cap(tokens / 16);
    dim3 block(32 * (c / 16));
    switch (c) {
    case 32:
        LAUNCH(k_ffn_fused<32>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 64:
        LAUNCH(k_ffn_fused<64>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 128:
        LAUNCH(k_ffn_fused<128>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 256:
        LAUNCH(k_ffn_fused<256>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 512:
        LAUNCH(k_ffn_fused<512>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    default:
        throw std::runtime_error("ffn channels");
    }
}

void launch_qkv(int c, const u8* in, const u8* w, u8* out, uint tokens) {
    dim3 grid(grid_cap(tokens / 16), (3 * c + 63) / 64);
    switch (c) {
    case 32:
        LAUNCH(k_qkv<32>, grid, 32, in, w, out, tokens, 1);
        break;
    case 64:
        LAUNCH(k_qkv<64>, grid, 32, in, w, out, tokens, 1);
        break;
    case 128:
        LAUNCH(k_qkv<128>, grid, 32, in, w, out, tokens, 1);
        break;
    case 256:
        LAUNCH(k_qkv<256>, grid, 32, in, w, out, tokens, 1);
        break;
    case 512:
        LAUNCH(k_qkv<512>, grid, 32, in, w, out, tokens, 1);
        break;
    default:
        throw std::runtime_error("qkv channels");
    }
}

void launch_project(int c, const u8* in, const u8* w, const float* scales, u8* out, uint tokens,
                    const u8* residual) {
    if (out == in)
        throw std::runtime_error("projection input/output alias");
    if (!residual) residual = in;
    dim3 grid(grid_cap(tokens / 16), c / 64);
    switch (c) {
    case 32:
        LAUNCH((k_project<32, 2>), dim3(grid_cap(tokens / 16), c / 32), 32, in, w, scales, out, tokens,
               1, residual);
        break;
    case 64:
        LAUNCH((k_project<64, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    case 128:
        LAUNCH((k_project<128, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    case 256:
        LAUNCH((k_project<256, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    case 512:
        LAUNCH((k_project<512, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    default:
        throw std::runtime_error("project channels");
    }
}

void launch_attention(int c, const u8* qkv, const float* bias, u8* out, uint w, uint h,
                      uint packed, bool direct_f8) {
    dim3 grid(grid_cap((w / 8) * (h / 8)), c / 32);
    switch (c) {
    case 32:
        LAUNCH(k_window_attention<32>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 64:
        LAUNCH(k_window_attention<64>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 128:
        LAUNCH(k_window_attention<128>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 256:
        LAUNCH(k_window_attention<256>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 512:
        LAUNCH(k_window_attention<512>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    default:
        throw std::runtime_error("attn channels");
    }
}

void launch_prefix(const float* in, const float* w, const float* temporal, float* out, uint seed,
                    uint width, uint height, uint temporal_on) {
    uint tokens = width * height;
    LAUNCH(k_prefix, (tokens + 15) / 16, 32, in, w, temporal, out, seed, width, height, temporal_on);
}
void launch_prefix_h(const float* in, const float* w, const float* temporal, float* out,
                     __half* out_h, uint seed, uint width, uint height, uint temporal_on) {
    uint tokens = width * height;
    LAUNCH(k_prefix, (tokens + 15) / 16, 32, in, w, temporal, out, seed, width, height,
            temporal_on, out_h);
}

void launch_gemm_tiled(const u8* a, const u8* b, u8* c, uint M, uint N, uint K, int activate) {
    LAUNCH(k_gemm_tiled_f8, dim3(M / 16, N / 16), 32, a, b, c, M, N, K, activate);
}

void launch_f32_to_f8(const float* s, u8* d, uint n) {
    LAUNCH(k_f32_to_f8, (n + 255) / 256, 256, s, d, n);
}
void launch_f8_to_f32(const u8* s, float* d, uint n) {
    LAUNCH(k_f8_to_f32, (n / 4 + 255) / 256, 256, s, d, n);
}
void launch_pool2x2(const u8* s, float* d, uint w, uint h, uint c) {
    uint n = (w / 2) * (h / 2) * c;
    LAUNCH(k_pool2x2_f8, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_upsample2x(const u8* s, u8* d, uint w, uint h, uint c) {
    uint n = (w * 2) * (h * 2) * c;
    LAUNCH(k_upsample2x_f8, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_rgb_reflect(const float* rgb, float* out, uint w, uint h, uint ow, uint oh) {
    LAUNCH(k_rgb_reflect, dim3((ow + 31) / 32, oh), 32, rgb, out, w, h, ow, oh);
}
void launch_head_rgb(const u8* feat, const float* w, const float* color, float* rgb, uint n, uint c,
                     float scale) {
    LAUNCH(k_head_rgb, (n + 63) / 64, 64, feat, w, color, rgb, n, c, scale);
}
void launch_vit_gather(const u8* src, const int* map, u8* dst, uint tokens, uint src_c, uint dst_c) {
    uint n = tokens * dst_c;
    LAUNCH(k_vit_gather, (n + 255) / 256, 256, src, map, dst, tokens, src_c, dst_c);
}
void launch_ds(const u8* s, const u8* w, u8* d, uint wdt, uint h, uint cin, uint cout) {
    uint n = (wdt / 2) * (h / 2) * cout;
    LAUNCH(k_ds_project, (n + 255) / 256, 256, s, w, d, wdt, h, cin, cout);
}
void launch_add_f8(const u8* a, const u8* b, u8* d, uint n) {
    LAUNCH(k_add_f8, (n + 255) / 256, 256, a, b, d, n);
}
void launch_upsample_add(const u8* lo, const u8* skip, u8* d, uint w, uint h, uint c) {
    uint n = (w * 2) * (h * 2) * c;
    LAUNCH(k_upsample_add_f8, (n + 255) / 256, 256, lo, skip, d, w, h, c);
}
void launch_linear_skip(const u8* in, const u8* w, const u8* skip, u8* out, uint tokens, uint cin,
                        uint cout) {
    uint n = tokens * cout;
    LAUNCH(k_linear_skip, (n + 255) / 256, 256, in, w, skip, out, tokens, cin, cout);
}
void launch_split_ffwd(const u8* in, const u8* pre, const u8* expand, const u8* contract, u8* out,
                       uint tokens) {
    LAUNCH(k_split_ffwd, tokens / 16, 32, in, pre, expand, contract, out, tokens);
}
static bool vit_attn_split() {
    const char* e = std::getenv("DLSS5_VIT_ATTN_SPLIT");
    return e && e[0] == '1' && e[1] == '\0';
}
void launch_vit_attention(const float* qkv, u8* out, uint tokens, uint dim) {
    if (tokens % 16 || dim % 32)
        throw std::runtime_error("vit attention extent");
    // Fused path: 1 wave, 32-key LDS tile. DLSS5_VIT_ATTN_SPLIT=1 keeps the
    // full-matrix kernel (14.7KB, 4 CTAs/CU) for A/B.
    if (vit_attn_split())
        LAUNCH((k_vit_attention<false>), dim3(tokens / 16, dim / 32), 32, qkv, nullptr, out, tokens,
               dim);
    else
        LAUNCH((k_vit_attention_fused<false>), dim3(tokens / 16, dim / 32), 32, qkv, nullptr, out,
               tokens, dim);
}
void launch_vit_attention_f8(const u8* qkv_f8, u8* out, uint tokens, uint dim) {
    if (tokens % 16 || dim % 32)
        throw std::runtime_error("vit attention extent");
    if (vit_attn_split())
        // 64 threads: staging covers every row (step blockDim.x/8). Both waves
        // execute the MMA; kept only as the split fallback.
        LAUNCH((k_vit_attention<true>), dim3(tokens / 16, dim / 32), 64, nullptr, qkv_f8, out,
               tokens, dim);
    else
        LAUNCH((k_vit_attention_fused<true>), dim3(tokens / 16, dim / 32), 32, nullptr, qkv_f8, out,
               tokens, dim);
}
void launch_post70_merge(const u8* main, const u8* skip, const float* scales, u8* dst, uint n,
                         uint c) {
    LAUNCH(k_post70_merge, (n * c + 255) / 256, 256, main, skip, scales, dst, n, c);
}
void launch_shift_pack_f32(const float* s, float* d, uint w, uint h, uint sw, uint sh, uint px,
                           uint py, uint c) {
    uint n = sw * sh * c;
    LAUNCH(k_shift_pack_f32, (n + 255) / 256, 256, s, d, w, h, sw, sh, px, py, c);
}
void launch_shift_pack_f8(const u8* s, u8* d, uint w, uint h, uint sw, uint sh, uint px, uint py,
                          uint c) {
    uint n = sw * sh * c;
    LAUNCH(k_shift_pack_f8, (n + 255) / 256, 256, s, d, w, h, sw, sh, px, py, c);
}
void launch_crop_f8(const u8* s, u8* d, uint w, uint h, uint sw, uint sh, uint px, uint py, uint c) {
    uint n = w * h * c;
    LAUNCH(k_crop_f8, (n + 255) / 256, 256, s, d, w, h, sw, sh, px, py, c);
}
void launch_pack_windows(const u8* s, u8* d, uint w, uint h, uint c) {
    uint n = w * h * c;
    LAUNCH(k_pack_windows, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_unpack_windows(const u8* s, u8* d, uint w, uint h, uint c) {
    uint n = w * h * c;
    LAUNCH(k_unpack_windows, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_residual_scale(const u8* gemm, const u8* skip, const float* scale, u8* dst, uint n,
                           uint c) {
    LAUNCH(k_residual_scale, (n * c + 255) / 256, 256, gemm, skip, scale, dst, n, c);
}
void launch_decoder_scatter(const u8* lo, const u8* skip, const float* scale, u8* dst, uint in_w,
                            uint in_h, uint out_w, uint out_h, uint c) {
    uint n = out_w * out_h * c;
    LAUNCH(k_decoder_scatter, (n + 255) / 256, 256, lo, skip, scale, dst, in_w, in_h, out_w, out_h,
           c);
}

void launch_fused_mh(int c, const u8* in, const u8* ffn, const u8* proj0, const float* proj0_s,
                     const u8* qkv, const float* bias, const u8* proj1, const float* proj1_s,
                     u8* qkv_tmp, u8* attn_tmp, u8* out, uint w, uint h, int do_proj0, int packed) {
    // Branch activations and residual features are DIFFERENT tensors. Never
    // overwrite a matrix input while another output-channel CTA still reads it.
    if (!packed) throw std::runtime_error("Swin requires explicit window packing");
    const uint tokens = w * h;
    launch_ffn(c, in, ffn, attn_tmp, tokens);
    if (do_proj0)
        launch_project(c, attn_tmp, proj0, proj0_s, out, tokens, in);
    else
        throw std::runtime_error("C32 requires its learned FFN residual coefficients");
    launch_qkv(c, out, qkv, qkv_tmp, tokens);
    launch_attention(c, qkv_tmp, bias, attn_tmp, w, h, 1);
    launch_project(c, attn_tmp, proj1, proj1_s, out, tokens, out);
}

hipStream_t get_launch_stream() { return g_stream; }
void launch_linear_half(const float* in,const __half* w,float* out,uint m,uint n,uint k,uint parts,int ordered,int raw){
  if(!m||!n||!k||n%16||k%32||!parts||(k/32)%parts||in==out||raw<0||raw>2)throw std::runtime_error("half linear geometry/alias");
  LAUNCH((k_linear_half<false>),dim3((m+15)/16,n/16),32,in,w,out,m,n,k,parts,ordered,raw);
}
void launch_linear_half_h(const float* in,const __half* w,float* out,__half* out_h,uint m,uint n,uint k,uint parts,int ordered,int raw){
  if(!m||!n||!k||n%16||k%32||!parts||(k/32)%parts||in==out||raw<0||raw>2||!out_h)throw std::runtime_error("half linear f16 geometry/alias");
  LAUNCH((k_linear_half<false>),dim3((m+15)/16,n/16),32,in,w,out,m,n,k,parts,ordered,raw,out_h);
}
void launch_linear_half_hin(const __half* in,const __half* w,float* out,__half* out_h,uint m,uint n,uint k,uint parts,int ordered,int raw){
  if(!m||!n||!k||n%16||k%32||!parts||(k/32)%parts||!in||raw<0||raw>2)throw std::runtime_error("half linear f16-in geometry/alias");
  LAUNCH((k_linear_half<true>),dim3((m+15)/16,n/16),32,reinterpret_cast<const float*>(in),w,out,m,n,k,parts,ordered,raw,out_h);
}
void launch_linear_half_out(const float* in,const __half* w,float* out,uint m,uint n,uint k,uint parts,int ordered,int raw){
  if(!m||!n||!k||n%16||k%32||!parts||(k/32)%parts||in==out||raw<0||raw>2)throw std::runtime_error("half linear out-major geometry/alias");
  LAUNCH((k_linear_half<false,true>),dim3((m+15)/16,n/16),32,in,w,out,m,n,k,parts,ordered,raw);
}
void launch_linear_half_hin_out(const __half* in,const __half* w,float* out,__half* out_h,uint m,uint n,uint k,uint parts,int ordered,int raw){
  if(!m||!n||!k||n%16||k%32||!parts||(k/32)%parts||!in||raw<0||raw>2)throw std::runtime_error("half linear f16-in out-major geometry/alias");
  LAUNCH((k_linear_half<true,true>),dim3((m+15)/16,n/16),32,reinterpret_cast<const float*>(in),w,out,m,n,k,parts,ordered,raw,out_h);
}
void launch_split_f32(const float* in,const u8* pre,const u8* ex,const u8* ct,float* out,uint n){
  if(!n||in==out)throw std::runtime_error("split geometry/alias");LAUNCH((k_split_f32<false>),dim3((n+15)/16,8),128,in,pre,ex,ct,out,nullptr,n);
}
void launch_split_f32_f8(const float* in,const u8* pre,const u8* ex,const u8* ct,float* out,u8* out_f8,uint n){
  if(!n||in==out||!out_f8)throw std::runtime_error("split f8 geometry/alias");LAUNCH((k_split_f32<true>),dim3((n+15)/16,8),128,in,pre,ex,ct,out,out_f8,n);
}
void launch_split_f32_f8in_f8(const u8* in_f8,const u8* pre,const u8* ex,const u8* ct,float* out,u8* out_f8,uint n){
  if(!n||!in_f8||!out_f8)throw std::runtime_error("split f8in geometry/alias");
  LAUNCH((k_split_f32<true, true>),dim3((n+15)/16,8),128,nullptr,pre,ex,ct,out,out_f8,n,in_f8);
}
void launch_split_f32_hin_f8(const __half* in_h,const u8* pre,const u8* ex,const u8* ct,float* out,u8* out_f8,uint n){
  if(!n||!in_h||!out_f8)throw std::runtime_error("split f16in geometry/alias");
  LAUNCH((k_split_f32<true, false, true>),dim3((n+15)/16,8),128,nullptr,pre,ex,ct,out,out_f8,n,nullptr,in_h);
}
void launch_rgb_graph(const float* in,float* raster,float* tiles,uint w,uint h,uint ow,uint oh){
 if(w<2||h<2||ow%8||oh%8||ow<w||oh<h)throw std::runtime_error("RGB geometry");LAUNCH(k_rgb_graph,(ow*oh+255)/256,256,in,raster,tiles,w,h,ow,oh);
}
void launch_gather_f32(const float* in,const int* map,float* out,uint n){if(in==out)throw std::runtime_error("gather alias");LAUNCH(k_gather_f32,(n+255)/256,256,in,map,out,n);}
void launch_up_f32(const float* lo,const float* skip,const float* scale,float* out,uint iw,uint ih,uint ow,uint oh,uint c){
  if(!iw||!ih||ow>2*iw||oh>2*ih||out==lo||out==skip)throw std::runtime_error("upsample geometry/alias");LAUNCH(k_up_f32,(ow*oh*c+255)/256,256,lo,skip,scale,out,iw,ow,oh,c);
}
void launch_up_f32_h(const float* lo,const float* skip,const float* scale,float* out,__half* out_h,uint iw,uint ih,uint ow,uint oh,uint c){
  if(!iw||!ih||ow>2*iw||oh>2*ih||out==lo||out==skip||!out_h)throw std::runtime_error("upsample f16 geometry/alias");LAUNCH(k_up_f32,(ow*oh*c+255)/256,256,lo,skip,scale,out,iw,ow,oh,c,out_h);
}
// The skip arrives as the C32 block's f16 window-major sibling (the skip byte
// copy of s.ah): remap through the source block's (px,py) shift.
void launch_up_f32_skip_h(const float* lo,const __half* skip,const float* scale,float* out,__half* out_h,
                          uint iw,uint ih,uint ow,uint oh,uint c,uint sw,uint sh,uint px,uint py){
  if(!iw||!ih||ow>2*iw||oh>2*ih||!out_h)throw std::runtime_error("upsample f16-skip geometry/alias");
  LAUNCH(k_up_f32_skip_h,(ow*oh*c+255)/256,256,lo,skip,scale,out,iw,ow,oh,c,sw,sh,px,py,out_h);
}
void launch_copy_f32(const float* src,float* dst,size_t bytes){
  if(bytes%16)return;LAUNCH(k_copy_f32,(uint)((bytes/16+255u)/256u),256,src,dst,(uint)(bytes/16));
}
void launch_post_merge_f32(const float* lo,const float* pre,const float* scales,float* out,uint w,uint h,bool main8_low){LAUNCH(k_post_merge_f32,(w*h*32/8+255)/256,256,lo,pre,scales,out,w,h,main8_low);}
void launch_post_merge_f32_h(const float* lo,const float* pre,const float* scales,float* out,__half* out_h,uint w,uint h,bool main8_low){LAUNCH(k_post_merge_f32,(w*h*32/8+255)/256,256,lo,pre,scales,out,w,h,main8_low,out_h);}
// Both inputs are f16 window-major siblings (low: the last C32 block's s.ah,
// pre: block 0's skip byte copy), each remapped through its (px,py) shift.
void launch_post_merge_f16(const __half* low,const __half* pre,const float* scales,float* out,__half* out_h,
                           uint w,uint h,uint lsw,uint lsh,uint lpx,uint lpy,uint psw,uint psh,uint ppix,uint ppy){
  LAUNCH(k_post_merge_f16,(w*h*32/8+255)/256,256,low,pre,scales,out,w,h,lsw,lsh,lpx,lpy,psw,psh,ppix,ppy,out_h);
}
void launch_head_f32(const float* in,const float* w,const float* color,float* out,uint n){LAUNCH(k_head_f32,(n+63)/64,64,in,w,color,out,n);}
// The head reads block 70's windowed f16 output directly: block 70's values sit on
// the f16 grid (H() round), so the f16 read is bit-identical to the f32 raster
// while halving traffic and skipping the f32 reframe.
void launch_head_f16(const __half* in,const float* w,const float* color,float* out,uint wsize,uint hsize,uint sw,uint sh,uint px,uint py){
  LAUNCH(k_head_f16,(wsize*hsize+63)/64,64,in,w,color,out,wsize,hsize,sw,sh,px,py);}
void launch_trace_f32(const float* in,uint n,float* stats,uint* bits){LAUNCH(k_trace_f32,1,256,in,n,stats,bits);}
void launch_c32_fused(const __half* in_h, const u8* ffn_w, const float* ffn_s, const u8* qkv_w,
                      const float* qscale, const float* bias, const u8* proj_w, const float* proj_s,
                      __half* ffn_h, __half* out_h, __half* out_h_raster, uint w, uint h, uint sw,
                      uint sh, uint px, uint py, bool chain, bool windowed, bool write_out,
                      bool raster_in, bool in8, bool out8) {
    if (!in_h || !ffn_w || !ffn_s || !qkv_w || !qscale || !bias || !proj_w || !proj_s ||
        !out_h || sw % 8 || sh % 8 ||
        (windowed && (w + px > sw || h + py > sh)))
        throw std::runtime_error("C32 fused extent/alias");
    const uint nwin = sw * sh / 64u;
    const bool write_ffn = ffn_h != nullptr;
    // raster_in only makes sense for windowed blocks: the prefix-fed window is
    // window-major (block 0).
    raster_in = raster_in && windowed;
    in8 = in8 && raster_in;
    out8 = out8 && out_h_raster != nullptr;
    #define C32_FUSED_LAUNCH_W(C, WR, WO)                                                      \
        do {                                                                                   \
            if (raster_in)                                                                     \
                LAUNCH((k_c32_fused<C, true, WR, WO, true>), nwin, 128, in_h, ffn_w, ffn_s,    \
                       qkv_w, qscale, bias, proj_w, proj_s, ffn_h, out_h, out_h_raster, w, h,  \
                       sw, sh, px, py, in8, out8);                                            \
            else                                                                               \
                LAUNCH((k_c32_fused<C, true, WR, WO, false>), nwin, 128, in_h, ffn_w, ffn_s,   \
                       qkv_w, qscale, bias, proj_w, proj_s, ffn_h, out_h, out_h_raster, w, h,  \
                       sw, sh, px, py, false, out8);                                          \
        } while (0)
    #define C32_FUSED_LAUNCH_NW(C, WR, WO)                                                     \
        LAUNCH((k_c32_fused<C, false, WR, WO, false>), nwin, 128, in_h, ffn_w, ffn_s, qkv_w,   \
               qscale, bias, proj_w, proj_s, ffn_h, out_h, out_h_raster, w, h, sw, sh, px, py, \
               false, out8)
    if (chain) {
        if (windowed) {
            if (write_ffn) {
                if (write_out)
                    C32_FUSED_LAUNCH_W(true, true, true);
                else
                    C32_FUSED_LAUNCH_W(true, true, false);
            } else {
                if (write_out)
                    C32_FUSED_LAUNCH_W(true, false, true);
                else
                    C32_FUSED_LAUNCH_W(true, false, false);
            }
        } else {
            if (write_ffn) {
                if (write_out)
                    C32_FUSED_LAUNCH_NW(true, true, true);
                else
                    C32_FUSED_LAUNCH_NW(true, true, false);
            } else {
                if (write_out)
                    C32_FUSED_LAUNCH_NW(true, false, true);
                else
                    C32_FUSED_LAUNCH_NW(true, false, false);
            }
        }
    } else {
        if (windowed) {
            if (write_ffn) {
                if (write_out)
                    C32_FUSED_LAUNCH_W(false, true, true);
                else
                    C32_FUSED_LAUNCH_W(false, true, false);
            } else {
                if (write_out)
                    C32_FUSED_LAUNCH_W(false, false, true);
                else
                    C32_FUSED_LAUNCH_W(false, false, false);
            }
        } else {
            if (write_ffn) {
                if (write_out)
                    C32_FUSED_LAUNCH_NW(false, true, true);
                else
                    C32_FUSED_LAUNCH_NW(false, true, false);
            } else {
                if (write_out)
                    C32_FUSED_LAUNCH_NW(false, false, true);
                else
                    C32_FUSED_LAUNCH_NW(false, false, false);
            }
        }
    }
    #undef C32_FUSED_ARGS
    #undef C32_FUSED_LAUNCH_W
    #undef C32_FUSED_LAUNCH_NW
}

// Production upstream MH chain (C64/128/256): FFN + p0 + QKV + norm in one kernel, then the
// attention+projection kernel (it recomputes QK/softmax/AV from the QKV bytes, adds the p1
// projection + residual, and stores with crop). Inputs are windowed (linear tokens x C);
// weights are the prepacked prod-layout tables. The f8 raster (out8) and the windowed f32
// (out_f32, run() reframes it) are independent consumers; launch each when requested.
void launch_mh_prod(const float* in32, const u8* in8, const __half* in16, const float* fw_mh,
                    const float* aw_mh, u8* feat8, u8* qkv8, u8* out8, float* out_f32, uint c,
                    uint tokens, uint sw, uint sh, uint w, uint h, uint px, uint py, bool raw,
                    unsigned char* dbg_hidden, unsigned char* dbg_d, float* dbg_ex) {
    if ((!in32 && !in8 && !in16) || !fw_mh || !aw_mh || !feat8 || !qkv8 || (!out8 && !out_f32) ||
        tokens % 16 || sw % 8 || sh % 8 || w + px > sw || h + py > sh)
        throw std::runtime_error("MH prod extent/alias");
    const uint blocks = (tokens + 15) / 16;
    switch (c) {
    case 64:
        if (in8)
            LAUNCH((mh_ffn_fused_c64_project_g128_qkv_bytein_fb), blocks, 128,
                   reinterpret_cast<const float*>(in8), fw_mh, aw_mh, feat8, qkv8, tokens);
        else LAUNCH((mh_ffn_fused_c64_hin_project_g128_qkv_fb), blocks, 128,
                    reinterpret_cast<const float*>(in16), fw_mh, aw_mh, feat8, qkv8, tokens,
                    dbg_hidden, dbg_d, dbg_ex);
        break;
    case 128:
        if (in8)
            LAUNCH((mh_ffn_fused_c128_project_g128_qkv_bytein_fb), blocks, 256,
                   reinterpret_cast<const float*>(in8), fw_mh, aw_mh, feat8, qkv8, tokens);
        else LAUNCH((mh_ffn_fused_c128_hin_project_g128_qkv_fb), blocks, 256,
                    reinterpret_cast<const float*>(in16), fw_mh, aw_mh, feat8, qkv8, tokens,
                    dbg_hidden, dbg_ex, dbg_d);
        break;
    default:
        if (in8)
            LAUNCH((mh_ffn_fused_c256_tiled_project_g128_qkv_bytein_fb), blocks, 512,
                   reinterpret_cast<const float*>(in8), fw_mh, aw_mh, feat8, qkv8, tokens);
        else LAUNCH((mh_ffn_fused_c256_hin_project_g128_qkv_fb), blocks, 512,
                    reinterpret_cast<const float*>(in16), fw_mh, aw_mh, feat8, qkv8, tokens,
                    dbg_ex, dbg_d);
        break;
    }
    const uint pw = (sw / 8) * (sh / 8);
    if (out8) {
        // Cropped f8 raster consumed by the next block's reframe; F(v) matches mode-5 stores.
        switch (c) {
        case 64:
            LAUNCH((c64_attention_project_fb_bout), pw, 256, qkv8, aw_mh, feat8, out8, sw, sh, 4u, w, h,
                   px, py);
            break;
        case 128:
            LAUNCH((c128_attention_project_fb_bout), pw, 512, qkv8, aw_mh, feat8, out8, sw, sh, 4u, w, h,
                   px, py);
            break;
        default:
            LAUNCH((c256_attention_project_fb_bout), pw, 512, qkv8, aw_mh, feat8, out8, sw, sh, 4u, w, h,
                   px, py);
            break;
        }
    }
    if (out_f32) {
        // Windowed f32 (run() reframes to the raster): raw blocks store H(v) (split mode 0),
        // non-raw blocks store F(v) (split mode 5).
        const uint post32 = raw ? 6u : 4u;
        switch (c) {
        case 64:
            LAUNCH((c64_attention_project_fb), pw, 256, qkv8, aw_mh, feat8, out_f32, sw, sh, post32,
                   0u, 0u, 0u, 0u);
            break;
        case 128:
            LAUNCH((c128_attention_project_fb), pw, 512, qkv8, aw_mh, feat8, out_f32, sw, sh, post32,
                   0u, 0u, 0u, 0u);
            break;
        default:
            LAUNCH((c256_attention_project_fb), pw, 512, qkv8, aw_mh, feat8, out_f32, sw, sh, post32,
                   0u, 0u, 0u, 0u);
            break;
        }
    }
}

// Production upstream C32 FAST4 chain: one kernel per window block (FFN + QKV + norm +
// attention + projection + residual). Weights are the prepacked prod-layout tables.
void launch_c32_prod(const __half* in_h, const float* fw, const float* wgt, __half* out_h,
                     __half* out_h_raster, uint w, uint h, uint sw, uint sh, uint px, uint py,
                     bool chain) {
    if (!in_h || !fw || !wgt || !out_h || sw % 8 || sh % 8 || w + px > sw || h + py > sh)
        throw std::runtime_error("C32 prod extent/alias");
    const uint nwin = sw * sh / 64u;
    if (out_h_raster)
        LAUNCH((c32_fast_ffn_attention_fused_half_chain_raster), nwin, 128,
               reinterpret_cast<const float*>(in_h), fw, wgt, reinterpret_cast<float*>(out_h),
               reinterpret_cast<float*>(out_h_raster), nwin, chain ? 3u : 0u, 0u, w, h, px, py, sw,
               px, py);
    else
        LAUNCH((c32_fast_ffn_attention_fused_half_chain), nwin, 128,
               reinterpret_cast<const float*>(in_h), fw, wgt, reinterpret_cast<float*>(out_h),
               nwin, chain ? 3u : 0u, 0u, w, h, px, py, sw, px, py);
}

} // namespace dlss5
