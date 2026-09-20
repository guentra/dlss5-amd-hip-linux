// GPU replica of the hook's CPU pixel pipeline (src/native_frame_input_check.h and
// src/native_hip_client.h): NativePixelChannel + NativePixelsToRgbaF32 +
// NativeResizeChannels (bilinear) + NativeRgbF32ToPixels.
// Bit-exactness contract: every rounding step must match the CPU reference compiled
// without FMA contraction, so FMA contraction is OFF for the translation unit that
// includes this file (pragma + -ffp-contract=off). The bilinear expression below keeps
// the reference's exact associativity:
//   (1-tx)*(1-ty)*p00 + tx*(1-ty)*p10 + (1-tx)*ty*p01 + tx*ty*p11
// Network geometry is fixed: 1920x1080 RGBA in, 1920x1080 RGB out.
#pragma STDC FP_CONTRACT OFF

#include "dlss5_gpu_interop.hpp"

namespace dlss5 {
namespace gpu {

constexpr int gc_net_w() { return 1920; }
constexpr int gc_net_h() { return 1080; }

__device__ inline float gc_h2f(uint16_t h) {
  // NativePixelHalfToFloat replica: ldexpf is exact, no rounding involved.
  const unsigned e = (h >> 10) & 31u, m = h & 1023u;
  float v = e == 0u
                ? ldexpf(float(m), -24)
                : (e == 31u ? (m ? __uint_as_float(0x7fc00000u) : __uint_as_float(0x7f800000u))
                            : ldexpf(float(1024u + m), int(e) - 25));
  return (h & 0x8000u) ? -v : v;
}

__device__ inline uint16_t gc_f2h(float v) {
  // NativeFloatToHalf replica: round-to-nearest-even with subnormal carry, bit-identical.
  uint32_t b = __float_as_uint(v);
  const uint32_t sign = (b >> 16) & 0x8000u, exp = (b >> 23) & 255u, mant = b & 0x7fffffu;
  if (exp == 255u) return uint16_t(sign | 0x7c00u | (mant ? ((mant >> 13) | 0x200u) : 0u));
  if (exp > 142u) return uint16_t(sign | 0x7c00u);
  if (exp < 102u) return uint16_t(sign);
  const unsigned shift = exp < 113u ? 126u - exp : 13u;
  const uint32_t value = exp < 113u ? mant | 0x800000u : mant;
  uint32_t rounded = value >> shift;
  const uint32_t remainder = value & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
  return uint16_t(sign | ((exp < 113u ? 0u : ((exp - 112u) << 10)) + rounded));
}

template <int FMT>
__device__ inline float gc_channel(const uint8_t* p, unsigned c) {
  // NativePixelChannel replica (p points at the pixel origin; c in 0..3 RGBA order).
  if (FMT == FmtBgra8 && c != 3u) c = 2u - c;
  if (FMT == FmtRgba8 || FMT == FmtBgra8) return float(p[c]) / 255.f;
  const uint16_t h = uint16_t(p[c * 2u]) | uint16_t(uint16_t(p[c * 2u + 1u]) << 8);
  return FMT == FmtRgba16F ? gc_h2f(h) : float(h) / 65535.f;
}

// Fused convert + bilinear resize: raw w x h pixels -> one RGBA output pixel of the
// 1920x1080 grid at (X,Y). Sampling the pre-converted source at the four bilinear
// corners reproduces the two-stage CPU pipeline exactly (the conversion is a pure
// per-channel function, evaluated here at the sample points).
template <int FMT>
__device__ inline void gc_px_to_rgba(const uint8_t* raw, int w, int h, int X, int Y, float* o4) {
  constexpr int BPP = (FMT == FmtRgba8 || FMT == FmtBgra8) ? 4 : 8;
  const int DW = gc_net_w(), DH = gc_net_h();
  float fy = (Y + 0.5f) * float(h) / float(DH) - 0.5f;
  int y0 = int(floorf(fy));
  int y1 = y0 + 1;
  float ty = fy - float(y0);
  if (y0 < 0) {
    y0 = 0;
    y1 = 0;
    ty = 0.f;
  } else if (y0 >= h - 1) {
    y0 = h - 1;
    y1 = h - 1;
    ty = 0.f;
  }
  float fx = (X + 0.5f) * float(w) / float(DW) - 0.5f;
  int x0 = int(floorf(fx));
  int x1 = x0 + 1;
  float tx = fx - float(x0);
  if (x0 < 0) {
    x0 = 0;
    x1 = 0;
    tx = 0.f;
  } else if (x0 >= w - 1) {
    x0 = w - 1;
    x1 = w - 1;
    tx = 0.f;
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float p00 = gc_channel<FMT>(raw + (size_t(y0) * w + x0) * BPP, c);
    const float p10 = gc_channel<FMT>(raw + (size_t(y0) * w + x1) * BPP, c);
    const float p01 = gc_channel<FMT>(raw + (size_t(y1) * w + x0) * BPP, c);
    const float p11 = gc_channel<FMT>(raw + (size_t(y1) * w + x1) * BPP, c);
    o4[c] = (1.f - tx) * (1.f - ty) * p00 + tx * (1.f - ty) * p10 + (1.f - tx) * ty * p01 + tx * ty * p11;
  }
}

// Fused bilinear resize + pack: 1920x1080 RGB floats (src) -> one raw pixel of the
// w x h destination at (X,Y), reading alpha straight from the original packed pixels
// (alpha bytes must remain bit-exact, as on the CPU path). Non-finite or
// non-representable values leave the pixel untouched.
template <int FMT>
__device__ inline void gc_rgb_to_px(const float* rgb, int w, int h, int X, int Y, const uint8_t* alpha_src,
                                    uint8_t* p) {
  constexpr int BPP = (FMT == FmtRgba8 || FMT == FmtBgra8) ? 4 : 8;
  const int DW = gc_net_w(), DH = gc_net_h();
  float fy = (Y + 0.5f) * float(DH) / float(h) - 0.5f;
  int y0 = int(floorf(fy));
  int y1 = y0 + 1;
  float ty = fy - float(y0);
  if (y0 < 0) {
    y0 = 0;
    y1 = 0;
    ty = 0.f;
  } else if (y0 >= DH - 1) {
    y0 = DH - 1;
    y1 = DH - 1;
    ty = 0.f;
  }
  float fx = (X + 0.5f) * float(DW) / float(w) - 0.5f;
  int x0 = int(floorf(fx));
  int x1 = x0 + 1;
  float tx = fx - float(x0);
  if (x0 < 0) {
    x0 = 0;
    x1 = 0;
    tx = 0.f;
  } else if (x0 >= DW - 1) {
    x0 = DW - 1;
    x1 = DW - 1;
    tx = 0.f;
  }
  float v[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float p00 = rgb[(size_t(y0) * DW + x0) * 3 + c];
    const float p10 = rgb[(size_t(y0) * DW + x1) * 3 + c];
    const float p01 = rgb[(size_t(y1) * DW + x0) * 3 + c];
    const float p11 = rgb[(size_t(y1) * DW + x1) * 3 + c];
    v[c] = (1.f - tx) * (1.f - ty) * p00 + tx * (1.f - ty) * p10 + (1.f - tx) * ty * p01 + tx * ty * p11;
  }
  if (!isfinite(v[0]) || !isfinite(v[1]) || !isfinite(v[2])) return;
  p[BPP - 1u] = alpha_src[BPP - 1u];
  if (BPP == 8) p[BPP - 2u] = alpha_src[BPP - 2u];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    float vv = v[c];
    if (FMT == FmtR16Unorm) vv = vv < 0.f ? 0.f : (vv > 1.f ? 1.f : vv);
    if (FMT == FmtRgba8 || FMT == FmtBgra8) {
      float cl = vv < 0.f ? 0.f : (vv > 1.f ? 1.f : vv);
      const unsigned ch = (FMT == FmtBgra8) ? 2u - c : c;
      p[ch] = uint8_t(cl * 255.f + 0.5f);
    } else {
      uint16_t hv = (FMT == FmtRgba16F) ? gc_f2h(vv) : uint16_t(vv * 65535.f + 0.5f);
      if (FMT == FmtRgba16F && (hv & 0x7c00u) == 0x7c00u) return;
      p[c * 2u] = uint8_t(hv);
      p[c * 2u + 1u] = uint8_t(hv >> 8);
    }
  }
}

template <int FMT>
__global__ void k_gc_px_to_rgba(const uint8_t* __restrict__ raw, int w, int h, float* __restrict__ out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= gc_net_w() || y >= gc_net_h()) return;
  float o4[4];
  gc_px_to_rgba<FMT>(raw, w, h, x, y, o4);
  *reinterpret_cast<float4*>(out + (size_t(y) * gc_net_w() + x) * 4) =
      make_float4(o4[0], o4[1], o4[2], o4[3]);
}

template <int FMT>
__global__ void k_gc_rgb_to_pix(const float* __restrict__ rgb, int w, int h,
                                const uint8_t* __restrict__ alpha_src, uint8_t* __restrict__ out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  constexpr int BPP = (FMT == FmtRgba8 || FMT == FmtBgra8) ? 4 : 8;
  gc_rgb_to_px<FMT>(rgb, w, h, x, y, alpha_src + (size_t(y) * w + x) * BPP,
                    out + (size_t(y) * w + x) * BPP);
}

// Whole-frame guards mirroring the CPU pre-passes:
//  - prescan16: rgba16_float input rejected if any half is inf/NaN (all 4 channels).
//  - prescan_f32: output rejected if any float is non-finite.
//  - prescan_rgb16f: rgba16_float output rejected if any value is non-finite or its
//    half rounding is inf/NaN (NativeFloatToHalf result with (h&0x7c00)==0x7c00).
__global__ void k_gc_prescan16(const uint16_t* __restrict__ p, int n, unsigned* __restrict__ bad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && ((p[i] & 0x7c00u) == 0x7c00u)) *bad = 1u;
}

__global__ void k_gc_prescan_f32(const float* __restrict__ p, int n, unsigned* __restrict__ bad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(p[i])) *bad = 1u;
}

__global__ void k_gc_prescan_rgb16f(const float* __restrict__ p, int n, unsigned* __restrict__ bad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && (!isfinite(p[i]) || ((gc_f2h(p[i]) & 0x7c00u) == 0x7c00u))) *bad = 1u;
}

// Temporal blend (shimmer fix for the non-temporal live path): cur = (1-w)*cur + w*prev
// elementwise over n floats. w is the previous-frame weight in [0,1); w=0 is a no-op.
// This is a post-network smoothing stage, intentionally outside the 71-block bit-exact
// contract (it only runs on the in-game raw path, never in the offline bench).
__global__ void k_gc_temporal_blend(float* __restrict__ cur, const float* __restrict__ prev, int n, float w) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float c = cur[i], p = prev[i];
    cur[i] = (1.f - w) * c + w * p;
  }
}

// Gated temporal blend (ghosting reduction): same IIR as k_gc_temporal_blend where
// the frames agree (|c-p| <= lo), no blend where they diverge (|c-p| >= hi), linear
// gate between. Moving content passes the current frame through untouched (no ghost
// trail) while static regions keep the shimmer fix; the residual ghost is bounded
// because the unblended frame becomes the next frame's history.
__global__ void k_gc_temporal_blend_gated(float* __restrict__ cur, const float* __restrict__ prev, int n,
                                          float w, float lo, float hi) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float c = cur[i], p = prev[i];
    const float d = fabsf(c - p);
    const float g = fminf(fmaxf((hi - d) * (1.f / (hi - lo)), 0.f), 1.f);
    const float k = w * g;
    cur[i] = (1.f - k) * c + k * p;
  }
}

}  // namespace gpu
}  // namespace dlss5
