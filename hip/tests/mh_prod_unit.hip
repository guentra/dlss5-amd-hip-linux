// Standalone unit test: prod MH FFN+QKV chain vs the existing split chain
// (ffn + p0) on the same f16 window input and the same f32 weight tables.
// Usage: mh_prod_unit <weights_dir> [block_id]
#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace dlss5;

int main(int argc, char** argv) {
    if (argc < 2)
        return 1;
    std::string dir = argv[1];
    int id = argc > 2 ? atoi(argv[2]) : 5;
    const uint c = argc > 3 ? uint(atoi(argv[3])) : 64;
    const uint Wd = 64, Hd = 64;
    const uint sw = Wd, sh = Hd;
    uint tokens = sw * sh;
    const size_t m = size_t(c) * c;

    std::string p = id == 70 ? "post70" : "block" + std::to_string(id);
    std::vector<float> f = read_weights(dir + "/" + p + "-ffn.f32");
    std::vector<float> a = read_weights(dir + "/" + p + "-attention.f32");
    if (f.size() != 9 * m + c || a.size() != 4 * m + (c / 32) * 4096 + c / 32 + c) {
        std::fprintf(stderr, "table sizes %zu %zu\n", f.size(), a.size());
        return 2;
    }

    // Deterministic f16 window input in [-1, 1] (or real input from UNIT_INPUT file).
    std::vector<__half> in16;
    if (const char* infile = getenv("UNIT_INPUT")) {
        FILE* fp = std::fopen(infile, "rb");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", infile); return 3; }
        std::fseek(fp, 0, SEEK_END); long sz = std::ftell(fp); std::fseek(fp, 0, SEEK_SET);
        in16.resize(sz / 2);
        if (std::fread(in16.data(), 1, sz, fp) != (size_t)sz) { std::fprintf(stderr, "short read %s\n", infile); return 4; }
        std::fclose(fp);
        if (in16.size() % c) { std::fprintf(stderr, "input size %zu not multiple of c=%u\n", in16.size(), c); return 5; }
        tokens = uint(in16.size() / c);
    } else {
        in16.resize(size_t(tokens) * c);
        for (size_t t = 0; t < tokens; t++)
            for (uint ch = 0; ch < c; ch++)
                in16[t * c + ch] = __float2half(getenv("BASIS")
                                                     ? (t == 0 && ch == 0 ? 1.0f : 0.0f)
                                                     : getenv("HALF05") ? 0.5f
                                                                        : getenv("CONST1")
                                                                              ? 1.0f
                                                                              : 0.2f *
                                                                                    sinf(0.01f * t +
                                                                                         0.03f * ch));
    }

    DeviceBuf in_d(in16.size() * 2);
    in_d.upload(in16.data(), in16.size() * 2);
    // Optional f8 (stock ByteIn) input path to isolate the custom HalfIn staging.
    const bool bytein = getenv("BYTEIN") != nullptr;
    DeviceBuf in8_d;
    if (bytein) {
        std::vector<u8> in8(in16.size());
        for (size_t i = 0; i < in16.size(); i++)
            in8[i] = e4m3_byte(__half2float(in16[i]));
        in8_d.alloc(in8.size());
        in8_d.upload(in8.data(), in8.size());
    }

    // Split chain: FFN (tiled e4m3, expand at bytes 0 and contract at 4C^2 in one
    // buffer) + p0 (f16 residual * s0, mode 5).
    std::vector<u8> fw_split_b(8 * m);
    pack_tiled_e4m3(fw_split_b.data(), f.data(), 4 * c, c);
    pack_tiled_e4m3(fw_split_b.data() + 4 * m, f.data() + 4 * m, c, 4 * c);
    DeviceBuf fw_split(fw_split_b.size());
    fw_split.upload(fw_split_b.data(), fw_split_b.size());
    DeviceBuf p0_split = upload_tiled_e4m3(f.data() + 8 * m, c, c);
    DeviceBuf s0 = [f, c, m] { DeviceBuf d(c * 4); d.upload(f.data() + 9 * m, c * 4); return d; }();
    DeviceBuf d_f32(size_t(tokens) * c * 4), d8(size_t(tokens) * c);
    DeviceBuf feat8_split(size_t(tokens) * c), feat32_split(size_t(tokens) * c * 4);
    DeviceBuf dbg_ex_split(size_t(tokens) * 4 * c * 4), dbg_ex_prod(size_t(tokens) * 4 * c * 4);
    DeviceBuf dbg_hh_split(size_t(tokens) * 4 * c), dbg_hh_prod(size_t(tokens) * 4 * c);
    DeviceBuf dbg_st(24576);
    if (bytein)
        launch_ffn_f32_f8in_f8out(c, in8_d.as<u8>(), fw_split.as<u8>(), nullptr, d_f32.as<float>(),
                                  d8.as<u8>(), tokens, false, false);
    else
        launch_ffn_f32_hin_f8out(c, in_d.as<__half>(), fw_split.as<u8>(), nullptr, d_f32.as<float>(),
                                 d8.as<u8>(), tokens, false, false, dbg_hh_split.as<u8>(),
                                 dbg_ex_split.as<float>());
    std::fprintf(stderr, "ffn launch: %s sync: %s\n", hipGetErrorName(hipGetLastError()),
                 hipGetErrorName(hipDeviceSynchronize()));
    if (bytein)
        launch_linear_f32_f8in_f8out_f8res(d8.as<u8>(), p0_split.as<u8>(), feat32_split.as<float>(),
                                           feat8_split.as<u8>(), tokens, c, c, in8_d.as<u8>(),
                                           s0.as<float>(), 5, 0, false);
    else
        launch_linear_f32_f8in_f8out_hrid(d8.as<u8>(), p0_split.as<u8>(), feat32_split.as<float>(),
                                          feat8_split.as<u8>(), tokens, c, c, in_d.as<__half>(),
                                          s0.as<float>(), 5, 0, false);
    std::fprintf(stderr, "p0 launch: %s sync: %s\n", hipGetErrorName(hipGetLastError()),
                 hipGetErrorName(hipDeviceSynchronize()));

    // Prod chain: packed tables + fused FFN+QKV kernel.
    // The FFN weight buffer keeps the f32 table size (4*(9C^2+C) bytes). The kernel
    // reads the three matrix regions through a uint* at uint offsets 0 / 4C^2 / 8C^2,
    // i.e. e4m3 byte blocks at byte 0 / 16C^2 / 32C^2, and the scale as f32 at float 9C^2.
    // (Packing the 9C^2 bytes contiguously is wrong: contract/p0 would land in the gaps.)
    std::vector<float> fmh = f, amh = a;
    const size_t CC = m;
    auto pack_ffn = [&CC, c](std::vector<float>& buf, const float* src) {
        auto* b = reinterpret_cast<uint8_t*>(buf.data());
        if (c == 256) {
            // C256 prod FFN kernel reads expand/contract as 16x32 e4m3 tiles; p0 row-major.
            pack_tiled_e4m3(b, src, 4 * c, c);
            pack_tiled_e4m3(b + 16 * CC, src + 4 * CC, c, 4 * c);
            permute_tiled_tiles_to_frag(b, 4 * CC);
            permute_tiled_tiles_to_frag(b + 16 * CC, 4 * CC);
        } else {
            for (size_t i = 0; i < 4 * CC; i++) b[i] = e4m3_byte(src[i]);
            for (size_t i = 0; i < 4 * CC; i++) b[16 * CC + i] = e4m3_byte(src[4 * CC + i]);
        }
        for (size_t i = 0; i < CC; i++) b[32 * CC + i] = e4m3_byte(src[8 * CC + i]);
    };
    pack_ffn(fmh, f.data());
    auto pk8 = [](std::vector<float>& v, size_t count) {
        auto* b = reinterpret_cast<uint8_t*>(v.data());
        for (size_t i = 0; i < count; i++)
            b[i] = e4m3_byte(v[i]);
    };
    pk8(amh, 3 * m); // QKV e4m3 block is dense from byte 0 (FFN+QKV reads aw bytes [0,3C^2))
    { // p1 e4m3 block at byte 12C^2 (project kernel reads aw uint offset 3C^2 -> byte 12C^2)
        auto* b = reinterpret_cast<uint8_t*>(amh.data()) + 12 * m;
        for (size_t i = 0; i < m; i++)
            b[i] = e4m3_byte(a[3 * m + i]);
    }
    if (getenv("CW")) {
        std::vector<float> fcw = f;
        for (size_t i = 0; i < 4 * CC; i++) fcw[i] = 0.01f;
        for (size_t i = 4 * CC; i < 8 * CC; i++) fcw[i] = 0.01f;
        for (size_t i = 8 * CC; i < 9 * CC; i++) fcw[i] = 0.1f;
        pack_ffn(fmh, fcw.data());
    }
    DeviceBuf fw_mh(fmh.size() * 4), aw_mh(amh.size() * 4);
    fw_mh.upload(fmh.data(), fmh.size() * 4);
    aw_mh.upload(amh.data(), amh.size() * 4);
    DeviceBuf feat8_prod(size_t(tokens) * c), qkv8(size_t(tokens) * 3 * c), feat32_prod(size_t(tokens) * c * 4);
    if (bytein)
        launch_mh_prod(nullptr, in8_d.as<u8>(), nullptr, fw_mh.as<float>(), aw_mh.as<float>(),
                       feat8_prod.as<u8>(), qkv8.as<u8>(), nullptr, feat32_prod.as<float>(), c, tokens,
                       sw, sh, Wd, Hd, 0, 0);
    else
        launch_mh_prod(nullptr, nullptr, in_d.as<__half>(), fw_mh.as<float>(), aw_mh.as<float>(),
                       feat8_prod.as<u8>(), qkv8.as<u8>(), nullptr, feat32_prod.as<float>(), c, tokens,
                        sw, sh, Wd, Hd, 0, 0, false, dbg_hh_prod.as<u8>(),
                        ((c == 128 || c == 256) && !getenv("NOSTAGE")) ? dbg_st.as<u8>() : nullptr,
                        dbg_ex_prod.as<float>());

    std::vector<u8> fs(size_t(tokens) * c), fp(size_t(tokens) * c), d8b(size_t(tokens) * c);
    std::fprintf(stderr, "prod launch err: %s sync: %s\n", hipGetErrorName(hipGetLastError()),
                 hipGetErrorName(hipStreamSynchronize(nullptr)));
    {
        std::vector<float> exs(size_t(tokens) * 4 * c), exp(size_t(tokens) * 4 * c);
        hipMemcpy(exs.data(), dbg_ex_split.p, exs.size() * 4, hipMemcpyDeviceToHost);
        hipMemcpy(exp.data(), dbg_ex_prod.p, exp.size() * 4, hipMemcpyDeviceToHost);
        size_t nd = 0;
        for (size_t i = 0; i < exs.size(); i++)
            if (std::memcmp(&exs[i], &exp[i], 4)) {
                if (nd < 40)
                    std::fprintf(stderr, "EXDIF i=%zu tok=%zu h=%zu split=%a prod=%a\n", i, i / (4 * c),
                                 i % (4 * c), exs[i], exp[i]);
                nd++;
            }
        std::fprintf(stderr, "expand_f32 diffs=%zu/%zu split_absmax=%.6g prod_absmax=%.6g\n", nd, exs.size(),
                     *std::max_element(exs.begin(), exs.end(), [](float a, float b) {
                         return std::fabs(a) < std::fabs(b); }),
                     *std::max_element(exp.begin(), exp.end(), [](float a, float b) {
                         return std::fabs(a) < std::fabs(b); }));
    }
    if (c == 128 || c == 256) {
        const size_t st_size = 24576;
        std::vector<u8> st(st_size);
        hipMemcpy(st.data(), dbg_st.p, st.size(), hipMemcpyDeviceToHost);
        char fn[128];
        std::snprintf(fn, sizeof(fn), "/tmp/opencode/c%d_staging.bin", c);
        FILE* fp = std::fopen(fn, "wb");
        if (fp) { std::fwrite(st.data(), 1, st.size(), fp); std::fclose(fp); }
        int nz = 0;
        for (size_t i = 0; i < 16 * (4 * c + 4); i++)
            nz += st[i] != 0;
        std::fprintf(stderr, "staging dump: nonzero=%d/%zu (first 16 rows x (4C+4))\n", nz,
                     size_t(16) * (4 * c + 4));
    }
    {
        std::vector<u8> hs(size_t(tokens) * 4 * c), hp(size_t(tokens) * 4 * c);
        hipMemcpy(hs.data(), dbg_hh_split.p, hs.size(), hipMemcpyDeviceToHost);
        hipMemcpy(hp.data(), dbg_hh_prod.p, hp.size(), hipMemcpyDeviceToHost);
        size_t nd = 0;
        for (size_t i = 0; i < hs.size(); i++)
            if (hs[i] != hp[i]) {
                if (nd < 24)
                    std::fprintf(stderr, "HDIF i=%zu tok=%zu h=%zu split_b=%u(%.5f) prod_b=%u(%.5f)\n", i,
                                 i / (4 * c), i % (4 * c), hs[i], from_e4m3(hs[i]), hp[i],
                                 from_e4m3(hp[i]));
                nd++;
            }
        std::fprintf(stderr, "hidden_byte diffs=%zu/%zu\n", nd, hs.size());
    }
    hipMemcpy(fs.data(), feat8_split.p, fs.size(), hipMemcpyDeviceToHost);
    hipMemcpy(fp.data(), feat8_prod.p, fp.size(), hipMemcpyDeviceToHost);
    hipMemcpy(d8b.data(), d8.p, d8b.size(), hipMemcpyDeviceToHost);
    std::fprintf(stderr, "d8[0:16]");
    for (int i = 0; i < 16; i++)
        std::fprintf(stderr, " %.4f", from_e4m3(d8b[i]));
    std::fprintf(stderr, "\n");
    {
        size_t hist[256] = {};
        for (size_t i = 0; i < d8b.size(); i++)
            hist[d8b[i]]++;
        size_t nnan = hist[0xff] + hist[0x7f];
        std::fprintf(stderr, "d8 nan-bytes=%zu/%zu\n", nnan, d8b.size());
        std::fprintf(stderr, "nan rows (first 32): ");
        size_t shown = 0;
        for (size_t r = 0; r < tokens && shown < 32; r++) {
            bool any = false;
            for (uint ch = 0; ch < c; ch++)
                if (d8b[r * c + ch] == 0xff || d8b[r * c + ch] == 0x7f)
                    any = true;
            if (any) {
                std::fprintf(stderr, "%zu ", r);
                shown++;
            }
        }
        std::fprintf(stderr, "\nnan cols (first 20): ");
        shown = 0;
        for (uint ch = 0; ch < c && shown < 20; ch++) {
            bool any = false;
            for (size_t r = 0; r < tokens; r++)
                if (d8b[r * c + ch] == 0xff || d8b[r * c + ch] == 0x7f)
                    any = true;
            if (any) {
                std::fprintf(stderr, "%u ", ch);
                shown++;
            }
        }
        std::fprintf(stderr, "\n");
    }

    {
        size_t nan_split = 0, nan_prod = 0;
        float mxs = 0, mxp = 0;
        for (size_t i = 0; i < fs.size(); i++) {
            float x = from_e4m3(fs[i]), y = from_e4m3(fp[i]);
            if (std::isnan(x))
                nan_split++;
            else
                mxs = std::max(mxs, std::fabs(x));
            if (std::isnan(y))
                nan_prod++;
            else
                mxp = std::max(mxp, std::fabs(y));
        }
        std::fprintf(stderr, "nan: split=%zu prod=%zu; absmax: split=%.4f prod=%.4f\n", nan_split,
                     nan_prod, mxs, mxp);
        std::fprintf(stderr, "row-nan pattern (rows with any nan): ");
        for (size_t r = 0; r < tokens && r < 64; r++) {
            bool any = false;
            for (uint ch = 0; ch < c; ch++)
                if (std::isnan(from_e4m3(fs[r * c + ch])))
                    any = true;
            if (any)
                std::fprintf(stderr, "%zu ", r);
        }
        std::fprintf(stderr, "\ncol-nan pattern (cols with any nan): ");
        for (uint ch = 0; ch < c; ch++) {
            bool any = false;
            for (size_t r = 0; r < tokens; r++)
                if (std::isnan(from_e4m3(fs[r * c + ch])))
                    any = true;
            if (any)
                std::fprintf(stderr, "%u ", ch);
        }
        std::fprintf(stderr, "\n");
    }
    float maxdiff = 0, meandiff = 0;
    size_t nbad = 0, first = 0;
    size_t nbymism = 0;
    int dhist[9] = {}; // (prod_byte - split_byte) mapped -4..+4 -> 0..8, 4=0
    size_t shown_m = 0;
    for (size_t i = 0; i < fs.size(); i++) {
        float x = from_e4m3(fs[i]), y = from_e4m3(fp[i]);
        float d = std::fabs(x - y);
        if (d > 0.02f * (1.f + std::fabs(x)) && d > 0.02f) {
            if (!nbad)
                first = i;
            nbad++;
        }
        maxdiff = std::max(maxdiff, d);
        meandiff += d;
        if (fs[i] != fp[i]) {
            nbymism++;
            int dd = (int)fp[i] - (int)fs[i];
            if (dd >= -4 && dd <= 4) dhist[dd + 4]++;
            if (shown_m < 24) {
                std::fprintf(stderr, "  MISMATCH i=%zu tok=%zu ch=%u split_b=%u(%.4f) prod_b=%u(%.4f)\n", i,
                             i / c, i % c, fs[i], x, fp[i], y);
                shown_m++;
            }
        }
    }
    meandiff /= (float)fs.size();
    std::fprintf(stderr, "byte_mismatches=%zu/%zu dhist[-4..+4]=", nbymism, fs.size());
    for (int k = 0; k < 9; k++)
        std::fprintf(stderr, "%d ", dhist[k]);
    std::fprintf(stderr, "\n");
    {
        size_t hist[256] = {};
        for (size_t i = 0; i < fs.size(); i++)
            hist[fs[i]]++;
        std::fprintf(stderr, "split raw byte histogram (top): ");
        size_t shown = 0;
        for (int b = 0; b < 256 && shown < 8; b++)
            if (hist[b]) {
                std::fprintf(stderr, "%d:%zu ", b, hist[b]);
                shown++;
            }
        std::fprintf(stderr, "\n");
    }
    {
        uint64_t h = 1469598103934665603ULL;
        for (u8 b : fp) { h ^= b; h *= 1099511628211ULL; }
        std::fprintf(stderr, "prod_feat8_hash=%016llx mode=%s\n", (unsigned long long)h,
                     bytein ? "f8" : "f16");
    }
    std::fprintf(stderr, "feat8: maxdiff=%.6g meandiff=%.6g bad=%zu/%zu first@%zu split=%.4f prod=%.4f\n",
                 maxdiff, meandiff, nbad, fs.size(), first, from_e4m3(fs[first]),
                 from_e4m3(fp[first]));
    {
        std::fprintf(stderr, "check feat_prod ~= in*s0 ?  s0[0:8]");
        std::vector<float> s0h(c);
        s0.download(s0h.data(), c * 4);
        for (int i = 0; i < 8; i++)
            std::fprintf(stderr, " %.5f", s0h[i]);
        std::fprintf(stderr, "\n");
        for (int t = 0; t < 2; t++) {
            std::fprintf(stderr, "t%d: ", t);
            for (int ch = 0; ch < 8; ch++) {
                float in = __half2float(in16[t * c + ch]);
                std::fprintf(stderr, "(in=%.4f in*s0=%.4f split=%.4f prod=%.4f) ", in, in * s0h[ch],
                             from_e4m3(fs[t * c + ch]), from_e4m3(fp[t * c + ch]));
            }
            std::fprintf(stderr, "\n");
        }
    }
    std::fprintf(stderr, "split[0:16]");
    for (int i = 0; i < 16; i++)
        std::fprintf(stderr, " %.4f", from_e4m3(fs[i]));
    std::fprintf(stderr, "\nprod [0:16]");
    for (int i = 0; i < 16; i++)
        std::fprintf(stderr, " %.4f", from_e4m3(fp[i]));
    std::fprintf(stderr, "\n");

    // ---- Stage 2: QKV+normalize, attention, project (split chain from network.hip) ----
    std::vector<u8> qw_b(3 * m);
    pack_tiled_e4m3(qw_b.data(), a.data(), 3 * c, c);
    DeviceBuf qw(qw_b.size());
    qw.upload(qw_b.data(), qw_b.size());
    auto upload_slice = [&](const float* src, size_t n) {
        DeviceBuf d(n * 4);
        std::vector<float> v(n);
        std::memcpy(v.data(), src, n * 4);
        d.upload(v.data(), n * 4);
        return d;
    };
    DeviceBuf qs = upload_slice(a.data() + 4 * m + (c / 32) * 4096, c / 32);
    DeviceBuf bias = upload_slice(a.data() + 4 * m, (c / 32) * 4096);
    DeviceBuf s1 = upload_slice(a.data() + 4 * m + (c / 32) * 4096 + c / 32, c);
    DeviceBuf p1_split = upload_tiled_e4m3(a.data() + 3 * m, c, c);
    DeviceBuf q8_split(size_t(tokens) * 3 * c), att8_split(size_t(tokens) * c);
    DeviceBuf out8_split(size_t(tokens) * c), outf32_split(size_t(tokens) * c * 4);
    launch_qkv_norm_f32_f8in(c, feat8_split.as<u8>(), qw.as<u8>(), q8_split.as<u8>(), tokens,
                             qs.as<float>(), 1.f, false);
    launch_attention(c, q8_split.as<u8>(), bias.as<float>(), att8_split.as<u8>(), sw, sh, 1, true);
    launch_linear_f32_f8in_f8out_raster(att8_split.as<u8>(), p1_split.as<u8>(),
                                        outf32_split.as<float>(), out8_split.as<u8>(), tokens, c, c,
                                        feat32_split.as<float>(), s1.as<float>(), 5, 0, true, sw, sh,
                                        sw, 0, 0);
    // Prod stage 2: relaunch with the out8 buffer so the project f8 store runs too.
    DeviceBuf out8_prod(size_t(tokens) * c);
    if (bytein)
        launch_mh_prod(nullptr, in8_d.as<u8>(), nullptr, fw_mh.as<float>(), aw_mh.as<float>(),
                       feat8_prod.as<u8>(), qkv8.as<u8>(), out8_prod.as<u8>(),
                       feat32_prod.as<float>(), c, tokens, sw, sh, Wd, Hd, 0, 0);
    else
        launch_mh_prod(nullptr, nullptr, in_d.as<__half>(), fw_mh.as<float>(), aw_mh.as<float>(),
                       feat8_prod.as<u8>(), qkv8.as<u8>(), out8_prod.as<u8>(),
                       feat32_prod.as<float>(), c, tokens, sw, sh, Wd, Hd, 0, 0);
    hipStreamSynchronize(nullptr);
    std::vector<u8> q8s(size_t(tokens) * 3 * c), q8p(size_t(tokens) * 3 * c);
    std::vector<u8> o8s(size_t(tokens) * c), o8p(size_t(tokens) * c);
    std::vector<float> of32s(size_t(tokens) * c), of32p(size_t(tokens) * c);
    hipMemcpy(q8s.data(), q8_split.p, q8s.size(), hipMemcpyDeviceToHost);
    hipMemcpy(q8p.data(), qkv8.p, q8p.size(), hipMemcpyDeviceToHost);
    hipMemcpy(o8s.data(), out8_split.p, o8s.size(), hipMemcpyDeviceToHost);
    hipMemcpy(o8p.data(), out8_prod.p, o8p.size(), hipMemcpyDeviceToHost);
    hipMemcpy(of32s.data(), outf32_split.p, of32s.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(of32p.data(), feat32_prod.p, of32p.size() * 4, hipMemcpyDeviceToHost);
    auto byte_diff = [&](const char* name, const std::vector<u8>& s, const std::vector<u8>& p) {
        size_t bad = 0, first_bad = 0;
        for (size_t i = 0; i < s.size(); i++)
            if (s[i] != p[i]) {
                if (!bad) first_bad = i;
                bad++;
            }
        std::fprintf(stderr, "%s: mismatches=%zu/%zu first@%zu split=%u prod=%u\n", name, bad,
                     s.size(), first_bad, s[first_bad], p[first_bad]);
        return bad;
    };
    size_t nq = byte_diff("qkv8", q8s, q8p);
    size_t no = byte_diff("out8", o8s, o8p);
    {
        size_t bad = 0, first_bad = 0;
        float maxd = 0;
        for (size_t i = 0; i < of32s.size(); i++) {
            float d = std::fabs(of32s[i] - of32p[i]);
            maxd = std::max(maxd, d);
            if (of32s[i] != of32p[i]) {
                if (!bad) first_bad = i;
                bad++;
            }
        }
        std::fprintf(stderr, "outf32: mismatches=%zu/%zu maxdiff=%.6g first@%zu split=%.6g prod=%.6g\n",
                     bad, of32s.size(), maxd, first_bad, of32s[first_bad], of32p[first_bad]);
        no += bad;
    }
    return (nbad > 100 || nq || no) ? 3 : 0;
}
