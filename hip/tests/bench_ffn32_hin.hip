// A/B microbenchmark for the production C32 FFN variant:
// k_ffn_f32<32, F8OUT=false, GROUPS=2, HOUT=true, F8IN=false, HIN=true>.
// Reports ms/call and an output hash so source variants can be checked for
// bit-identity without the full graph suite. Tokens default to full-res
// 1920x1152 = 2211840 (the encoder C32 stage size).
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include "../kernels/qkv_norm.hip"
#include "../kernels/ops.hip"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;

// A/B knob: default GROUPS=2 (128 threads, production). GROUPS=4 doubles tokens/CTA.
#ifndef FFN32_GROUPS
#define FFN32_GROUPS 2
#endif

// Variant 1: separate input-staging and hidden LDS (no aliasing). Each group
// then writes its own hidden rows as soon as its expand MACs finish, so the
// second __syncthreads disappears and the four accumulators retire earlier.
// Arithmetic is untouched (same WMMA sequence per output element).
#if !defined(FFN32_VARIANT)
#define FFN32_VARIANT 0
#endif

#if FFN32_VARIANT == 1
template <int C, bool F8OUT, int GROUPS, bool HOUT, bool F8IN, bool HIN>
__global__ void k_ffn_f32_v(const float* input, const u8* weights, const float* scales,
                            float* output, u8* output_f8, uint tokens, bool chain_residual,
                            bool precise_c32, __half* output_h, const u8* input_f8,
                            const __half* input_h) {
    using namespace linalg;
    constexpr uint hidden_c = 4 * C;
    constexpr uint WAVES_PER_GROUP = C / 16;
    const uint first = blockIdx.x * 16 * GROUPS;
    const uint wave = threadIdx.x / 32;
    const uint group = wave / WAVES_PER_GROUP;
    const uint sub = wave % WAVES_PER_GROUP;
    __shared__ u8 hidden[16 * GROUPS * hidden_c];
    __shared__ u8 inb[16 * GROUPS * (C == 32 ? 32 : C)];
    if constexpr (C == 32) {
        const uint count4 = 16 * GROUPS * 8;
        for (uint i4 = threadIdx.x; i4 < count4; i4 += blockDim.x) {
            uint local4 = i4 % (16 * 8);
            uint row = first + (i4 / (16 * 8)) * 16 + local4 / 8;
            uint ch = (local4 % 8) * 4;
            if (F8IN) {
                uint v = row < tokens ? *reinterpret_cast<const u32*>(input_f8 + row * C + ch) : 0u;
                *reinterpret_cast<u32*>(inb + i4 * 4) = v;
            } else if (HIN) {
                const __half2* hp = reinterpret_cast<const __half2*>(input_h + row * C + ch);
                __half2 h0 = row < tokens ? hp[0] : __half2(0.f, 0.f);
                __half2 h1 = row < tokens ? hp[1] : __half2(0.f, 0.f);
                *reinterpret_cast<u32*>(inb + i4 * 4) =
                    u32(e4m3_byte(__half2float(__low2half(h0)))) |
                    u32(e4m3_byte(__half2float(__high2half(h0)))) << 8 |
                    u32(e4m3_byte(__half2float(__low2half(h1)))) << 16 |
                    u32(e4m3_byte(__half2float(__high2half(h1)))) << 24;
            } else {
                const float4 v = row < tokens
                                     ? *reinterpret_cast<const float4*>(input + row * C + ch)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<u32*>(inb + i4 * 4) =
                    u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                    u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
    }
    __syncthreads();
    if (group < GROUPS) {
        const uint t0 = group * 16;
        const uint lane = threadIdx.x & 31u;
        const uint row0 = (lane >= 16u ? 8u : 0u);
        const uint base = lane & ~3u;
        const uint col0 = (lane & 15u) & ~3u;
        for (uint n = 0; n < 4; ++n) {
            C32 ex = C32::Splat(0.f);
            for (uint g = 0; g < C / 32; ++g) {
                A8 a = A8::Load(inb + t0 * C, g * 32, C);
                uint col = (sub * 4 + n) * 16;
                B8 b = B8::LoadRowT(weights, ((col / 16) * (C / 32) + g) * 512, 16);
                ex.MultiplyAccumulate(a, b);
            }
            for (uint i = 0; i < ex.Length(); ++i)
                ex.Set(i, precise_c32 ? ActivatePolyC32(ex.Get(i)) : ActivatePoly(ex.Get(i)));
            const uint hcol = (sub * 4u + n) * 16u + col0;
            for (uint i = 0; i < 8u; ++i) {
                uint b0 = e4m3_byte(ex.Get(i));
                uint b1 = __shfl(b0, base + 1u);
                uint b2 = __shfl(b0, base + 2u);
                uint b3 = __shfl(b0, base + 3u);
                if ((lane & 3u) == 0u)
                    *reinterpret_cast<u32*>(hidden + (t0 + row0 + i) * hidden_c + hcol) =
                        b0 | b1 << 8 | b2 << 16 | b3 << 24;
            }
        }
    }
    // The two subs of a group write each other's hidden columns; one barrier
    // (vs three in the aliased original) makes all tiles visible for the contract.
    __syncthreads();
    if (group < GROUPS) {
        const uint t0 = group * 16;
        C32 acc = C32::Splat(0.f);
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + t0 + rc.x < tokens) {
                float iv = HIN
                               ? __half2float(input_h[(first + t0 + rc.x) * C + sub * 16 + rc.y])
                               : input[(first + t0 + rc.x) * C + sub * 16 + rc.y];
                acc.Set(i, (chain_residual ? from_e4m3(e4m3_byte(iv)) : iv) *
                           scales[sub * 16 + rc.y]);
            }
        }
        for (uint g = 0; g < hidden_c / 32; ++g) {
            A8 a = A8::Load(hidden + t0 * hidden_c, g * 32, hidden_c);
            B8 b = B8::LoadRowT(weights, hidden_c * C + (sub * (hidden_c / 32) + g) * 512, 16);
            acc.MultiplyAccumulate(a, b);
        }
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + t0 + rc.x < tokens)
                output_h[(first + t0 + rc.x) * C + sub * 16 + rc.y] =
                    __float2half_rn(acc.Get(i));
        }
    }
}
#endif

int main(int argc, char** argv) {
    const uint c = 32, tokens = argc > 1 ? std::stoul(argv[1]) : 2211840u;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    std::vector<__half> xh(size_t(tokens) * c);
    for (size_t i = 0; i < xh.size(); ++i) {
        float v = H(float(int((i * 2654435761u) % 1000) - 500) * 0.002f);
        xh[i] = __float2half_rn(v);
    }
    std::vector<float> w(8 * c * c);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int((i * 40503u + i / 37u) % 97) - 48) * 0.01f;
    std::vector<float> sc(c);
    for (uint i = 0; i < c; ++i) sc[i] = H(float(int(i % 13) - 6) * 0.077f);
    std::vector<u8> tiled(w.size());
    pack_tiled_e4m3(tiled.data(), w.data(), 4 * c, c);
    pack_tiled_e4m3(tiled.data() + 4 * c * c, w.data() + 4 * c * c, c, 4 * c);
    DeviceBuf dxh(xh.size() * 2), dw(tiled.size()), ds(sc.size() * 4), dyh(size_t(tokens) * c * 2);
    dxh.upload(xh.data(), xh.size() * 2);
    dw.upload(tiled.data(), tiled.size());
    ds.upload(sc.data(), sc.size() * 4);
    hipStream_t stream;
    hip_check(hipStreamCreate(&stream), "stream");
    auto go = [&] {
        constexpr uint TPR = 16u * FFN32_GROUPS; // tokens per CTA (16 rows x GROUPS)
#if FFN32_VARIANT == 1
        k_ffn_f32_v<32, false, FFN32_GROUPS, true, false, true>
#else
        k_ffn_f32<32, false, FFN32_GROUPS, true, false, true>
#endif
            <<<(tokens + TPR - 1u) / TPR, 2u * 32u * FFN32_GROUPS, 0, stream>>>(
                nullptr, dw.as<u8>(), ds.as<float>(), nullptr, nullptr, tokens, false, true,
                dyh.as<__half>(), nullptr, dxh.as<__half>());
    };
    for (int i = 0; i < 5; ++i) go();
    hip_check(hipStreamSynchronize(stream), "warm");
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) go();
    hipEventRecord(e1, stream);
    hip_check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    ms /= iters;
    std::printf("FFN32_HIN tokens=%u GROUPS=%d V%d iters=%d: %.3f ms/call\n", tokens,
                FFN32_GROUPS, FFN32_VARIANT, iters, ms);
    std::vector<__half> out(size_t(tokens) * c);
    dyh.download(out.data(), out.size() * 2);
    if (const char* dump = std::getenv("DUMP")) {
        FILE* f = std::fopen(dump, "wb");
        std::fwrite(out.data(), 2, out.size(), f);
        std::fclose(f);
    }
    uint64_t hsh = 1469598103934665603ull;
    for (__half v : out) {
        hsh ^= uint64_t(__half_as_ushort(v));
        hsh *= 1099511628211ull;
    }
    float mn = 1e30f, mx = -1e30f;
    for (__half v : out) {
        float f = __half2float(v);
        mn = std::min(mn, f);
        mx = std::max(mx, f);
    }
    std::printf("out_hash=%016llx min=%g max=%g\n", (unsigned long long)hsh, mn, mx);
    return 0;
}
