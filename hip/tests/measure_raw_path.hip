// Measures the per-call wall time of dlss5_run_frame_raw_gpu (DEV_PTRS mode) and,
// with DLSS5_FRAME_TIMING=1, the per-stage wall times in /tmp/dlss5_frame_timing.log.
// This is the exact in-game critical-path segment: the vkd3d submit worker is
// blocked for this duration between prefix and suffix.
// usage: measure_raw_path <weights_dir> [frames] [dxgi_format]
#include "dlss5_capi.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s <weights_dir> [frames] [dxgi_format]\n", argv[0]);
    return 2;
  }
  const int frames = argc > 2 ? atoi(argv[2]) : 5;
  const unsigned dxgi = argc > 3 ? atoi(argv[3]) : 87;
  const unsigned W = 1920, H = 1080;
  const size_t bytes = size_t(W) * H * ((dxgi == 28 || dxgi == 87) ? 4u : 8u);
  hipSetDevice(0);
  if (dlss5_init(argv[1], 0) != 0) {
    printf("init failed: %s\n", dlss5_last_error());
    return 2;
  }
  void* dev_in = nullptr, *dev_out = nullptr;
  if (hipMalloc(&dev_in, bytes) != hipSuccess || hipMalloc(&dev_out, bytes) != hipSuccess)
    return 3;
  std::vector<uint8_t> h(bytes);
  for (size_t i = 0; i < bytes; ++i) h[i] = uint8_t((i * 7u + 13u) % 251u);
  hipMemcpy(dev_in, h.data(), bytes, hipMemcpyHostToDevice);
  Dlss5FrameRaw f{};
  f.struct_size = sizeof f;
  f.in_handle = dev_in;
  f.out_handle = dev_out;
  f.width = W;
  f.height = H;
  f.dxgi_format = dxgi;
  f.seed = 7;
  f.flags = DLSS5_HIP_RAW_FLAGS_DEV_PTRS;
  f.paper_white = f.transfer = f.color = 1.f;
  f.handle_gen = 1;
  f.temporal_gen = 1;
  double best = 1e9;
  for (int i = 0; i < frames + 1; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = dlss5_run_frame_raw_gpu(&f);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (i > 0) best = std::min(best, ms);
    printf("frame %d rc=%d wall=%.2f ms\n", i, rc, ms);
  }
  printf("best wall=%.2f ms dxgi=%u bytes=%zu\n", best, dxgi, bytes);
  return 0;
}
