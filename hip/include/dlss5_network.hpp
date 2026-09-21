#pragma once
#include "dlss5_runtime.hpp"
#include <memory>
#include <array>

namespace dlss5 {

struct Network70 {
    Network70();
    ~Network70();
    // Fixed native geometry: input 1920x1080 RGBA F32, output 1920x1152 RGB F32.
    // create is transactional and repeatable; do not race create/destruction with
    // methods on this same object. Runs on one object are serialized internally.
    void create(const std::string& dir);
    // No stream capture: every run consumes the supplied seed. All intermediate
    // activations remain device-resident. DLSS5_HIP_TRACE=1 adds 71 GPU reductions
    // and reports them after completion; it is intentionally expensive.
    // Optional already-warped/reflected 1920x1152 raster RGBA F32 history.
    // nullptr disables history for this call; no history is implicitly retained.
    void run(const float* host_rgb, float* host_out, uint seed, const float* host_history = nullptr);
    // GPU-resident input variant: dev_rgb is a DEVICE pointer to the 1920x1080 RGBA
    // proxy (e.g. the frame_encode output). Skips the H2D input upload and reads the
    // buffer directly. host_out/host_history remain host pointers. Produces
    // bit-identical output to run() for the same input bytes (the upload/download are
    // raw bit-preserving copies).
    void run_gpu(const float* dev_rgb, float* host_out, uint seed, const float* host_history = nullptr);
    // Fully GPU-resident variant: dev_rgb is a DEVICE pointer to the 1920x1080 RGBA
    // proxy and dev_out is a DEVICE pointer that receives the N*12 float output.
    // No H2D input upload and no D2H output download: the network stays on the GPU.
    // Bit-identical to run_gpu() for the same input bytes.
    void run_gpu_gpu(const float* dev_rgb, float* dev_out, uint seed, const float* host_history = nullptr);
    // Non-blocking variant: enqueues the whole network on the internal stream and
    // returns once the launches are queued (no internal hipStreamSynchronize).
    // wait_event (non-null): the network stream first waits on it, ordering this run
    // after whatever produced the input (e.g. frame_encode on the default stream).
    // signal_event (non-null): recorded on the network stream after the last kernel,
    // so the caller's stream can wait on it before consuming dev_out. The caller owns
    // both events and must synchronize the stream(s) itself.
    void run_gpu_gpu_async(const float* dev_rgb, float* dev_out, uint seed, const float* host_history,
                           hipEvent_t wait_event, hipEvent_t signal_event);
    float last_gpu_ms() const;
    struct Trace { uint elements{}, nonfinite{}, hash{}; float minimum{}, maximum{}; double sum{}; bool checked{}; };
    std::array<uint,71> last_block_counts() const;
    std::array<Trace,71> last_trace() const;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace dlss5
