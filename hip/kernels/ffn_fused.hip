// Native HIP fused FFN (tiled E4M3, HW quantize, FP8 output).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// One group = 16 tokens, WAVES = C/16. Expand  C→4C into LDS, contract 4C→C.
template <int C>
__global__ void k_ffn_fused(const u8* input, const u8* weights, float* output_f32, u8* output_f8,
                            uint width, uint height, int fp8_out) {
    constexpr int HIDDEN = 4 * C;
    constexpr int WAVES = C / 16;
    const uint tid = threadIdx.x;
    const uint wave = tid / 32u;
    using namespace linalg;
    __shared__ u8 hidden[16 * HIDDEN];
    const uint tokens = width * height;
    for (uint gid = blockIdx.x; gid * 16u < tokens; gid += gridDim.x) {
    {
        C32 acc[4];
        for (uint n = 0; n < 4; n++)
            acc[n] = C32::Splat(0.f);
        for (uint g = 0; g < uint(C / 32); g++) {
            A8 a = A8::Load(input, gid * 16u * uint(C) + g * 32u, uint(C));
            for (uint n = 0; n < 4; n++) {
                uint col = (wave * 4u + n) * 16u;
                uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
                B8 b = B8::LoadRowT(weights, boff, 16);
                acc[n].MultiplyAccumulate(a, b);
            }
        }
        for (uint n = 0; n < 4; n++) {
            uint col = (wave * 4u + n) * 16u;
            for (uint i = 0; i < acc[n].Length(); i++)
                acc[n].Set(i, ActivatePoly(acc[n].Get(i)));
            acc[n].StoreF8(hidden, col, uint(HIDDEN));
        }
    }
    __syncthreads();
    {
        C32 acc = C32::Splat(0.f);
        const uint col = wave * 16u;
        for (uint g = 0; g < uint(HIDDEN / 32); g++) {
            A8 a = A8::Load(hidden, g * 32u, uint(HIDDEN));
            uint boff = uint(HIDDEN) * uint(C) + ((col / 16u) * uint(HIDDEN / 32) + g) * 512u;
            B8 b = B8::LoadRowT(weights, boff, 16);
            acc.MultiplyAccumulate(a, b);
        }
        if (!fp8_out) {
            for (uint i = 0; i < acc.Length(); i++)
                acc.Set(i, F(H(acc.Get(i))));
            acc.StoreF32(output_f32, (gid * 16u * uint(C) + col) * 4u, uint(C) * 4u);
        } else {
            acc.StoreF8(output_f8, gid * 16u * uint(C) + col, uint(C));
        }
    }
    __syncthreads();
    }
}

} // namespace dlss5
