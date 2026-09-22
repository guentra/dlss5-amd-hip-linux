// Native HIP projection (FP8 input, tiled B, FP8 store, fast accumulate).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

template <int C, int BLOCK_N>
__global__ void k_project(const u8* input, const u8* weights, const float* scales, u8* output,
                          uint width, uint height, const u8* residual) {
    using namespace linalg;
    const uint gy = blockIdx.y;
    const uint tokens = width * height;
    for (uint gid = blockIdx.x; gid * 16u < tokens; gid += gridDim.x) {
    const uint first = gid * 16u;
    C32 acc[BLOCK_N];
    for (uint n = 0; n < uint(BLOCK_N); n++) {
        acc[n] = C32::Splat(0.f);
        uint col = (gy * uint(BLOCK_N) + n) * 16u;
        for (uint i = 0; i < acc[n].Length(); i++) {
            uint2 rc = acc[n].GetCoordinate(i);
            uint row = col + rc.y;
            float f = from_e4m3(residual[(first + rc.x) * uint(C) + row]);
            acc[n].Set(i, H(f * scales[row]));
        }
    }
    for (uint g = 0; g < uint(C / 32); g++) {
        A8 a = A8::Load(input, first * uint(C) + g * 32u, uint(C));
        for (uint n = 0; n < uint(BLOCK_N); n++) {
            uint col = (gy * uint(BLOCK_N) + n) * 16u;
            uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
            B8 b = B8::LoadRowT(weights, boff, 16);
            acc[n].MultiplyAccumulate(a, b);
        }
    }
    for (uint n = 0; n < uint(BLOCK_N); n++) {
        for (uint i = 0; i < acc[n].Length(); i++)
            acc[n].Set(i, F(H(acc[n].Get(i))));
        acc[n].StoreF8(output, first * uint(C) + (gy * uint(BLOCK_N) + n) * 16u, uint(C));
    }
    }
}

} // namespace dlss5
