// Wave-matrix wrapper matching dx::linalg 16x32 A · 32x16 B → 16x16 C.
// gfx1201 native WMMA is 16x16x16 FP8; one D3D12 MMA is two gfx12 ops along K.
#pragma once
#include "dlss5_common.hpp"
#include <rocwmma/rocwmma.hpp>

namespace dlss5 {
namespace linalg {

using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::accumulator;
using rocwmma::row_major;
using rocwmma::float8_t;
using f16 = rocwmma::hfloat16_t;

template <typename DataT, typename Layout>
using FragA = rocwmma::fragment<matrix_a, 16, 16, 16, DataT, Layout>;
template <typename DataT, typename Layout>
using FragB = rocwmma::fragment<matrix_b, 16, 16, 16, DataT, Layout>;
using FragC = rocwmma::fragment<accumulator, 16, 16, 16, float>;

// gfx12 f32 accumulator layout (GPUOpen RDNA 4): 8 elems/thread,
// column = lane%16, rows = (lane>=16 ? 8 : 0) + i.
__device__ inline uint2 acc_coord(uint i) {
    uint lane = threadIdx.x & 31u;
    return uint2{(lane >= 16u ? 8u : 0u) + i, lane & 15u};
}

template <typename DataT>
struct MatrixA {
    FragA<DataT, row_major> k0, k1;
    template <typename Ptr>
    __device__ static MatrixA Load(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixA a;
        rocwmma::load_matrix_sync(a.k0, p, ldm);
        rocwmma::load_matrix_sync(a.k1, p + 16, ldm);
        return a;
    }
};

template <typename DataT>
struct MatrixB {
    FragB<DataT, row_major> k0r, k1r;
    FragB<DataT, col_major> k0c, k1c;
    bool row = false;
    template <typename Ptr>
    __device__ static MatrixB LoadRow(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixB b;
        b.row = true;
        rocwmma::load_matrix_sync(b.k0r, p, ldm);
        rocwmma::load_matrix_sync(b.k1r, p + 16 * ldm, ldm);
        return b;
    }
    // 512-byte B-tile packed [N=16][K=32] row-major (pack_tiled_e4m3). Lane l
    // owns fragment element [e] = B[k = (l>>4)*8+e][n = l&15] for k0r and
    // B[k = (l>>4)*8+16+e][n = l&15] for k1r: two contiguous spans in this
    // layout, so each half-fragment is one vector load (vs the scattered byte
    // loads LoadRow needs for [K][N] tiles).
    template <typename Ptr>
    __device__ static MatrixB LoadRowT(Ptr buf, uint byte_off, uint stride_bytes) {
        (void)stride_bytes;
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        const uint n = threadIdx.x & 15u;
        const uint kb = ((threadIdx.x >> 4) & 1u) * 8u;
        const DataT* base = p + n * 32u + kb;
        MatrixB b;
        b.row = true;
        alignas(16) DataT t0[8], t1[8];
        __builtin_memcpy(t0, base, 8 * sizeof(DataT));
        __builtin_memcpy(t1, base + 16, 8 * sizeof(DataT));
#pragma unroll
        for (uint e = 0; e < 8u; ++e) {
            b.k0r[e] = t0[e];
            b.k1r[e] = t1[e];
        }
        return b;
    }
    template <typename Ptr>
    __device__ static MatrixB LoadRowMajorOut(Ptr buf, uint out_col, uint k0, uint k_stride) {
        const DataT* base = reinterpret_cast<const DataT*>(buf) +
            (size_t)(out_col + (threadIdx.x & 15u)) * k_stride + k0 + ((threadIdx.x >> 4u) & 1u) * 8u;
        MatrixB b;
        b.row = true;
        // One contiguous load per K-half instead of eight scalar gathers.
        // Size follows DataT (8 B for e4m3, 16 B for f16); element e still
        // receives memory[e], so the MMA operands are identical.
        alignas(16) DataT t0[8], t1[8];
        __builtin_memcpy(t0, base, 8 * sizeof(DataT));
        __builtin_memcpy(t1, base + 16, 8 * sizeof(DataT));
#pragma unroll
        for (uint e = 0; e < 8u; ++e) {
            b.k0r[e] = t0[e];
            b.k1r[e] = t1[e];
        }
        return b;
    }
    template <typename Ptr>
    __device__ static MatrixB LoadCol(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixB b;
        b.row = false;
        rocwmma::load_matrix_sync(b.k0c, p, ldm);
        rocwmma::load_matrix_sync(b.k1c, p + 16, ldm);
        return b;
    }
};

struct MatrixC {
    FragC acc{};
    __device__ static MatrixC Splat(float v) {
        MatrixC c;
        rocwmma::fill_fragment(c.acc, v);
        return c;
    }
    __device__ uint Length() const { return FragC::num_elements; }
    __device__ float Get(uint i) const { return acc[i]; }
    __device__ void Set(uint i, float v) { acc[i] = v; }
    __device__ uint2 GetCoordinate(uint i) const { return acc_coord(i); }

    template <typename DataT>
    __device__ void MultiplyAccumulate(const MatrixA<DataT>& a, const MatrixB<DataT>& b) {
        if (b.row) {
            rocwmma::mma_sync(acc, a.k0, b.k0r, acc);
            rocwmma::mma_sync(acc, a.k1, b.k1r, acc);
        } else {
            rocwmma::mma_sync(acc, a.k0, b.k0c, acc);
            rocwmma::mma_sync(acc, a.k1, b.k1c, acc);
        }
    }

    template <typename Ptr>
    __device__ void StoreF32(Ptr buf, uint byte_off, uint stride_bytes) const {
        float* p = reinterpret_cast<float*>(reinterpret_cast<char*>(buf) + byte_off);
        rocwmma::store_matrix_sync(p, acc, stride_bytes / 4u, rocwmma::mem_row_major);
    }

    // Element-wise stores: gfx12 accumulator Cast<> is not a rocWMMA store type.
    template <typename Ptr>
    __device__ void StoreF8(Ptr buf, uint byte_off, uint stride_bytes) const {
        u8* p = reinterpret_cast<u8*>(reinterpret_cast<char*>(buf) + byte_off);
        for (uint i = 0; i < Length(); i++) {
            uint2 rc = GetCoordinate(i);
            p[rc.x * stride_bytes + rc.y] = e4m3_byte(Get(i));
        }
    }

    template <typename Ptr>
    __device__ void StoreF16(Ptr buf, uint byte_off, uint stride_bytes) const {
        f16* p = reinterpret_cast<f16*>(reinterpret_cast<char*>(buf) + byte_off);
        uint ldm = stride_bytes / 2u;
        for (uint i = 0; i < Length(); i++) {
            uint2 rc = GetCoordinate(i);
            p[rc.x * ldm + rc.y] = f16(Get(i));
        }
    }

    template <typename Ptr>
    __device__ static MatrixC LoadF32(Ptr buf, uint byte_off, uint stride_bytes) {
        const float* p = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        MatrixC c;
        rocwmma::load_matrix_sync(c.acc, p, stride_bytes / 4u, rocwmma::mem_row_major);
        return c;
    }
};

template <typename DataT>
__device__ inline MatrixC Multiply(const MatrixA<DataT>& a, const MatrixB<DataT>& b) {
    MatrixC c = MatrixC::Splat(0.f);
    c.MultiplyAccumulate(a, b);
    return c;
}

using A8 = MatrixA<float8_t>;
using B8 = MatrixB<float8_t>;
using A16 = MatrixA<f16>;
using B16 = MatrixB<f16>;
using C32 = MatrixC;

} // namespace linalg
} // namespace dlss5
