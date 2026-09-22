// Native HIP QKV (E4M3 operands, tiled B, H() epilogue).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// Grid: (tokens/16, ceil(3C/64)). Each block writes 4×16 output columns of the 3C QKV matrix.
template <int C>
__global__ void k_qkv(const u8* input, const u8* weights, u8* output, uint width, uint height) {
    using namespace linalg;
    const uint gy = blockIdx.y;
    const uint tokens = width * height;
    for (uint gid = blockIdx.x; gid * 16u < tokens; gid += gridDim.x) {
    C32 acc[4];
    for (uint n = 0; n < 4; n++)
        acc[n] = C32::Splat(0.f);
    for (uint g = 0; g < uint(C / 32); g++) {
        A8 a = A8::Load(input, gid * 16u * uint(C) + g * 32u, uint(C));
        for (uint n = 0; n < 4; n++) {
            uint col = (gy * 4u + n) * 16u;
            if (col >= 3u * uint(C))
                continue;
            uint boff = ((col / 16u) * uint(C / 32) + g) * 512u;
            B8 b = B8::LoadRowT(weights, boff, 16);
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
            output[((gid * 16u + rc.x) * 3u + part) * uint(C) + row + rc.y] =
                e4m3_byte(H(acc[n].Get(i)));
        }
    }
    }
}

} // namespace dlss5
