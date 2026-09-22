#include "dlss5_capi.h"
#include "dlss5_network.hpp"
#include "dlss5_launch.hpp"
#include "dlss5_frame.hpp"
#include "dlss5_gpu_interop.hpp"
#include <mutex>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>

static std::mutex g_mu;
static dlss5::Network70* g_net;

namespace {
struct FrameTimer {
    bool on=false;
    std::vector<std::pair<std::string,double>> marks;
    double t0=0;
    FrameTimer() {
        const char* e=getenv("DLSS5_FRAME_TIMING");
        on=e&&std::string(e)!="0";
        if(on)t0=now();
    }
    static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    void mark(const char* name){if(!on)return;marks.emplace_back(name,now());}
    ~FrameTimer(){
        if(!on)return;
        FILE* fp=fopen("/tmp/dlss5_frame_timing.log","a");
        if(!fp)return;
        double prev=t0;
        double total=now()-t0;
        fprintf(fp,"frame total=%.2fms |",total);
        for(auto&m:marks){double dt=m.second-prev;fprintf(fp," %s=%.2f",m.first.c_str(),dt);prev=m.second;}
        if(g_net)fprintf(fp," net_gpu=%.2f",g_net->last_gpu_ms());
        fprintf(fp,"\n");
        fclose(fp);
    }
};
}

static thread_local std::string g_err;
static Dlss5HipBridge g_bridge;
static Dlss5HipFrameBridge g_frame_bridge;
static Dlss5HipDeviceBridge g_device_bridge;
static int g_gpu;
struct FrameBuffers {
    dlss5::DeviceBuf source,proxy,neural,decoded,history,motion,warped,base;
    // HOST_PTRS raw mode: device staging for CPU-mapped D3D12 in/out buffers.
    dlss5::DeviceBuf raw_in,raw_out;
    size_t raw_in_size=0;
    std::vector<float> host_proxy,host_neural,host_warped,host_result;
    bool history_valid=false;
    // Temporal blend (shimmer fix): previous-frame decoded 1920x1080 RGB, plus the
    // continuity tokens that reset it (geometry gen, add-on F6 temporal gen).
    dlss5::DeviceBuf prev_out;
    dlss5_u32 prev_handle_gen=0;
    dlss5_u32 prev_temporal_gen=0;
    bool prev_seeded=false;
    // GPU-resident whole-frame guard flags, one per frame parity (4 device bytes each,
    // nonzero = bypass). The prescan of frame N writes flag[N%2]; the blend and the
    // commit of frame N+1 read flag[N%2]. Alternating parity keeps the two frames'
    // flags apart with no CPU state, so the bypass decision never touches the host.
    dlss5::DeviceBuf guard_a, guard_b;
    int raw_parity=0;
    // Cross-stream ordering: default stream (encode) -> network stream -> default
    // stream (decode/commit). Records on the producer side, waits on the consumer.
    hipEvent_t ev_net_in{}, ev_net_out{};
    dlss5_u32 previous_flags=0;
    float previous_white=0;
    ~FrameBuffers() {
        if (ev_net_in) (void)hipEventDestroy(ev_net_in);
        if (ev_net_out) (void)hipEventDestroy(ev_net_out);
    }
    void create() {
        const size_t n=1920ull*1080, padded=1920ull*1152;
        // neural must hold the network's full 1920x1152 RGB output; only the first
        // 1920x1080 rows are consumed (same as the CPU path's first n*12 bytes).
        source.alloc(n*16);proxy.alloc(n*16);neural.alloc(padded*12);decoded.alloc(n*12);history.alloc(n*12);
        prev_out.alloc(n*12);
        warped.alloc(padded*16);base.alloc(padded*16);
        guard_a.alloc(4);guard_b.alloc(4);
        // 0xff = "bad": the first frame never blends, and the first commit is a
        // no-op if its own prescan has not run (it always does, but a missing
        // prescan must fail safe to the bit-exact input copy, never to garbage).
        dlss5::hip_check(hipMemsetAsync(guard_a.p,0xff,4,nullptr),"guard init a");
        dlss5::hip_check(hipMemsetAsync(guard_b.p,0xff,4,nullptr),"guard init b");
        dlss5::hip_check(hipEventCreate(&ev_net_in),"network in event");
        dlss5::hip_check(hipEventCreate(&ev_net_out),"network out event");
        host_proxy.resize(n*4);host_neural.resize(padded*3);host_warped.resize(padded*4);host_result.resize(n*3);
    }
};
static std::unique_ptr<FrameBuffers> g_frame;

// Temporal blend weight (previous-frame share) for the non-temporal live path. Read
// once from the environment; 0 disables the blend (legacy bit-exact behavior). The
// default is 0.0 so the offline raw pipeline stays bit-exact unless the caller (the
// in-game wrapper) opts in via DLSS5_TEMPORAL_BLEND.
static double temporal_blend_weight() {
    static const double w = [] {
        const char* e = getenv("DLSS5_TEMPORAL_BLEND");
        double v = 0.0;
        if (e && e[0]) {
            char* end = nullptr;
            const double p = strtod(e, &end);
            if (end && end != e) v = p;
        }
        if (v < 0.0) v = 0.0;
        if (v > 0.999999) v = 0.999999;
        return v;
    }();
    return w;
}

// Gated temporal blend thresholds (absolute 0..1 RGB difference). Full blend weight
// for |cur-prev| <= lo, none for >= hi, linear between, so moving content is never
// mixed with a misaligned previous frame (ghosting) while static regions keep the
// shimmer fix. Read once; "off" (or an invalid value) falls back to the plain blend.
struct TemporalBlendGate { bool enabled; float lo; float hi; };
static TemporalBlendGate temporal_blend_gate() {
    static const TemporalBlendGate g = [] {
        const char* e = getenv("DLSS5_TEMPORAL_BLEND_GATE");
        float lo = 4.f / 255.f, hi = 16.f / 255.f;
        if (e && e[0]) {
            if (std::string(e) == "off") return TemporalBlendGate{false, 0.f, 0.f};
            float a = 0.f, b = 0.f;
            if (std::sscanf(e, "%g,%g", &a, &b) == 2 && a >= 0.f && b > a) { lo = a; hi = b; }
        }
        return TemporalBlendGate{true, lo, hi};
    }();
    return g;
}

static void run_log(const char* s) {
    FILE* f = fopen("/tmp/dlss5-hip-run.log", "a");
    if (!f)
        return;
    fprintf(f, "%s\n", s);
    fclose(f);
}

static const char* set_err(const char* s) {
    g_err = s ? s : "";
    return g_err.c_str();
}

extern "C" const char* dlss5_hip_runtime_error(void);

extern "C" int dlss5_init(const char* weights_dir, int gpu) {
    std::lock_guard<std::mutex> lock(g_mu);
    try {
        if (const char* why = dlss5_hip_runtime_error())
            return set_err(why), -1;
        const char* dir = weights_dir && weights_dir[0] ? weights_dir : getenv("DLSS5_HIP_WEIGHTS");
        if (!dir || !dir[0])
            return set_err("no weights dir (DLSS5_HIP_WEIGHTS)"), -1;
        const int candidate_gpu = gpu >= 0 ? gpu : 0;
        dlss5::hip_check(hipSetDevice(candidate_gpu), "hipSetDevice");
        auto candidate = std::make_unique<dlss5::Network70>();
        candidate->create(dir);
        if (g_net) dlss5::hip_check(hipSetDevice(g_gpu), "old device");
        delete g_net;
        g_frame.reset();
        g_net = candidate.release();
        g_gpu = candidate_gpu;
        g_err.clear();
        run_log("init ok");
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

extern "C" int dlss5_run(const float* rgba1080, float* rgb1080, dlss5_u32 seed) {
    std::lock_guard<std::mutex> lock(g_mu);
    // V1 has no motion contract. Even a rejected call breaks frame continuity.
    if (g_frame) g_frame->history_valid=false;
    if (!g_net)
        return set_err("not initialized"), -1;
    if (!rgba1080 || !rgb1080 || rgba1080 == rgb1080)
        return set_err("null buffer"), -1;
    try {
        for (size_t i=0;i<1920ull*1080*4;++i)
            if (!std::isfinite(rgba1080[i])) throw std::runtime_error("nonfinite network input");
        dlss5::hip_check(hipSetDevice(g_gpu), "hipSetDevice");
        std::vector<float> out1152(1920ull * 1152 * 3);
        g_net->run(rgba1080, out1152.data(), seed);
        for (float v:out1152)
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite network output");
        for (unsigned y = 0; y < 1080; y++)
            std::memcpy(rgb1080 + size_t(y) * 1920 * 3, out1152.data() + size_t(y) * 1920 * 3,
                        1920 * 3 * 4);
        static int nlog;
        if (nlog < 4) {
            float md = 0.f, acc = 0.f;
            for (unsigned i = 0; i < 1920u * 1080u; i++) {
                float d0 = rgb1080[i * 3] - rgba1080[i * 4];
                float d1 = rgb1080[i * 3 + 1] - rgba1080[i * 4 + 1];
                float d2 = rgb1080[i * 3 + 2] - rgba1080[i * 4 + 2];
                float a = fabsf(d0) + fabsf(d1) + fabsf(d2);
                if (a > md)
                    md = a;
                acc += a;
            }
            char buf[128];
            snprintf(buf, sizeof buf, "run ok  gpu=%.2fms  max|d|=%.4f  mean|d|=%.4f  rgb0=%.3f %.3f %.3f",
                     g_net->last_gpu_ms(), md, acc / float(1920u * 1080u), rgb1080[0], rgb1080[1],
                     rgb1080[2]);
            run_log(buf);
            nlog++;
        }
        g_err.clear();
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        run_log(e.what());
        return -1;
    }
}

extern "C" void dlss5_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_net && hipSetDevice(g_gpu) != hipSuccess) {
        set_err("cannot select model device for shutdown");
        return;
    }
    delete g_net;
    g_net = nullptr;
    g_frame.reset();
}

extern "C" const char* dlss5_last_error(void) { return g_err.c_str(); }

extern "C" int dlss5_find_device(const char* name) {
    std::lock_guard<std::mutex> lock(g_mu);
    try {
        if (!name || !*name) throw std::runtime_error("graphics adapter name required");
        int count=0, selected=-1;
        dlss5::hip_check(hipGetDeviceCount(&count), "HIP enumeration");
        for (int i=0;i<count;++i) {
            char candidate[256]{};
            dlss5::hip_check(hipDeviceGetName(candidate,sizeof candidate,i), "HIP adapter name");
            if (!std::strcmp(name,candidate)) {
                if (selected!=-1) throw std::runtime_error("ambiguous graphics/HIP adapter name");
                selected=i;
            }
        }
        if (selected<0) throw std::runtime_error("graphics adapter not found in HIP");
        g_err.clear();
        return selected;
    } catch(const std::exception& e) {set_err(e.what());return -1;}
}

extern "C" int dlss5_run_frame(const Dlss5Frame* f) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_net) return set_err("not initialized"), -1;
    if (!f || f->struct_size != sizeof(Dlss5Frame) || (f->flags & ~1u) ||
        !f->rgba || !f->rgb || f->rgb == f->rgba || f->reset > 1) {
        if (g_frame) g_frame->history_valid=false;
        return set_err("invalid frame ABI/buffers"), -1;
    }
    try {
        FrameTimer ft;
        const size_t n=1920ull*1080;
        for (size_t i=0;i<n*4;++i) if (!std::isfinite(f->rgba[i])) throw std::runtime_error("nonfinite frame input");
        if (!std::isfinite(f->motion_uv_scale_x)||!std::isfinite(f->motion_uv_scale_y)) throw std::runtime_error("nonfinite motion scale");
        if (f->motion) {
            if (!f->motion_width||!f->motion_height||f->motion_width>7680||f->motion_height>4320) throw std::runtime_error("motion extent");
            for (size_t i=0;i<size_t(f->motion_width)*f->motion_height*2;++i)
                if (!std::isfinite(f->motion[i])) throw std::runtime_error("nonfinite motion");
        }
        ft.mark("validate");
        dlss5::hip_check(hipSetDevice(g_gpu),"frame device");
        if (!g_frame) {auto next=std::make_unique<FrameBuffers>();next->create();g_frame=std::move(next);}
        auto& b=*g_frame;
        if (f->reset || f->flags!=b.previous_flags || f->paper_white!=b.previous_white) b.history_valid=false;
        b.previous_flags=f->flags;b.previous_white=f->paper_white;
        dlss5::FrameSettings settings{f->paper_white,f->transfer,f->color,(f->flags&1u)!=0};
        b.source.upload(f->rgba,n*16);
        ft.mark("up_src");
        dlss5::frame_encode(b.source.as<float>(),b.proxy.as<float>(),1920,1080,settings,nullptr);
        ft.mark("encode");
        // Proxy stays on the GPU (frame_encode output). No download to host: the
        // network reads the device buffer directly via run_gpu.
        dlss5::hip_check(hipStreamSynchronize(nullptr),"frame encode");
        ft.mark("sync_encode");
        bool temporal=b.history_valid&&f->motion;
        if (temporal) {
            size_t bytes=size_t(f->motion_width)*f->motion_height*8;
            if (b.motion.bytes!=bytes)b.motion.alloc(bytes);
            b.motion.upload(f->motion,bytes);
            dlss5::frame_warp(b.proxy.as<float>(),b.history.as<float>(),b.motion.as<float>(),b.warped.as<float>(),
                1920,1080,1920,1152,f->motion_width,f->motion_height,f->motion_uv_scale_x,f->motion_uv_scale_y,false,nullptr);
            dlss5::frame_warp(b.proxy.as<float>(),nullptr,nullptr,b.base.as<float>(),1920,1080,1920,1152,0,0,0,0,true,nullptr);
            dlss5::frame_guard(b.base.as<float>(),b.warped.as<float>(),1920*1152,3.f/255,12.f/255,nullptr);
            dlss5::hip_check(hipStreamSynchronize(nullptr),"frame history");b.warped.download(b.host_warped.data(),b.host_warped.size()*4);
            ft.mark("warp");
        }
        // Feed the GPU-resident proxy straight to the network (no host round-trip).
        g_net->run_gpu(b.proxy.as<float>(),b.host_neural.data(),f->seed,temporal?b.host_warped.data():nullptr);
        ft.mark("net");
        for (float v:b.host_neural) if (!std::isfinite(v)) throw std::runtime_error("nonfinite neural output");
        b.neural.upload(b.host_neural.data(),n*12);
        ft.mark("up_neural");
        if (temporal) dlss5::frame_smooth(b.warped.as<float>(),b.neural.as<float>(),unsigned(n),10.f/255,.8f,nullptr);
        dlss5::frame_decode(b.source.as<float>(),b.proxy.as<float>(),b.neural.as<float>(),b.decoded.as<float>(),1920,1080,settings,nullptr);
        ft.mark("decode");
        dlss5::hip_check(hipStreamSynchronize(nullptr),"frame decode");b.decoded.download(b.host_result.data(),n*12);
        ft.mark("sync_dl_dec");
        for (float v:b.host_result) if (!std::isfinite(v)) throw std::runtime_error("nonfinite decoded output");
        dlss5::hip_check(hipMemcpy(b.history.p,b.neural.p,n*12,hipMemcpyDeviceToDevice),"save history");b.history_valid=true;
        std::memcpy(f->rgb,b.host_result.data(),n*12);ft.mark("memcpy");g_err.clear();return 0;
    } catch(const std::exception&e) {if(g_frame)g_frame->history_valid=false;set_err(e.what());return -1;}
}

// Fully GPU-resident frame path: dev_in is a DEVICE pointer to the 1920x1080 RGBA
// frame (the D3D12 texture mapped into HIP) and dev_out is a DEVICE pointer that
// receives the 1920x1080 RGB result. No H2D input upload, no D2H output download:
// encode -> network -> decode all run on the GPU. dev_history (optional) is a DEVICE
// pointer to the warped 1920x1152 RGBA history, or null.
extern "C" int dlss5_run_frame_gpu(const float* dev_in, float* dev_out, uint32_t seed,
                                   float paper_white, float transfer, float color, uint32_t flags,
                                   const float* dev_history) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_net) return set_err("not initialized"), -1;
    if (!dev_in || !dev_out || dev_in == dev_out) return set_err("null/aliased gpu buffers"), -1;
    try {
        FrameTimer ft;
        const size_t n=1920ull*1080;
        ft.mark("validate");
        dlss5::hip_check(hipSetDevice(g_gpu),"frame device");
        if (!g_frame) {auto next=std::make_unique<FrameBuffers>();next->create();g_frame=std::move(next);}
        auto& b=*g_frame;
        b.previous_flags=flags;b.previous_white=paper_white;
        dlss5::FrameSettings settings{paper_white,transfer,color,(flags&1u)!=0};
        // Input is already a device buffer (the D3D12 texture). D2D into source —
        // no H2D upload (the CPU path uploads 33MB).
        dlss5::hip_check(hipMemcpyAsync(b.source.p,dev_in,n*16,hipMemcpyDeviceToDevice,nullptr),"gpu input d2d");
        ft.mark("up_src");
        dlss5::frame_encode(b.source.as<float>(),b.proxy.as<float>(),1920,1080,settings,nullptr);
        ft.mark("encode");
        // Fully GPU-resident network: the proxy stays on the GPU and the neural
        // output is written to b.neural on the GPU (no D2H + H2D round-trip).
        g_net->run_gpu_gpu(b.proxy.as<float>(),b.neural.as<float>(),seed,dev_history);
        ft.mark("net");
        if (dev_history) dlss5::frame_smooth(b.warped.as<float>(),b.neural.as<float>(),unsigned(n),10.f/255,.8f,nullptr);
        dlss5::frame_decode(b.source.as<float>(),b.proxy.as<float>(),b.neural.as<float>(),b.decoded.as<float>(),1920,1080,settings,nullptr);
        ft.mark("decode");
        dlss5::hip_check(hipStreamSynchronize(nullptr),"frame decode gpu");
        ft.mark("sync_dl_dec");
        // Output is written to the D3D12 texture (device) — no D2H + CPU memcpy.
        dlss5::hip_check(hipMemcpyAsync(dev_out,b.decoded.p,n*12,hipMemcpyDeviceToDevice,nullptr),"gpu output d2d");
        ft.mark("memcpy");
        g_err.clear();return 0;
    } catch(const std::exception&e) {if(g_frame)g_frame->history_valid=false;set_err(e.what());return -1;}
}

// V3 raw GPU-resident frame: packed game pixels arrive as Win32 D3D12 shared handles
// (or raw device pointers, flags bit1). convert->encode->network->decode->convert all
// run on the GPU; no host round-trip. Output semantics match the CPU path bit-for-bit
// (RGB replaced, alpha preserved from dev_in; bypass = dev_out <- dev_in copy).
extern "C" int dlss5_run_frame_raw_gpu(const Dlss5FrameRaw* f) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_net) return set_err("not initialized"), -1;
    if (!f || f->struct_size != sizeof(Dlss5FrameRaw) || (f->flags & ~0x7u) ||
        (f->flags & DLSS5_HIP_RAW_FLAGS_DEV_PTRS) && (f->flags & DLSS5_HIP_RAW_FLAGS_HOST_PTRS) ||
        !f->in_handle || !f->out_handle ||
        f->width < 1 || f->height < 1 || f->width > 7680 || f->height > 4320 ||
        dlss5::gpu::format_from_dxgi(f->dxgi_format) == dlss5::gpu::FmtUnsupported)
        return set_err("invalid raw frame ABI/buffers"), -1;
    try {
        FrameTimer ft;
        const size_t n=1920ull*1080;
        dlss5::hip_check(hipSetDevice(g_gpu),"raw frame device");
        if (!g_frame) {auto next=std::make_unique<FrameBuffers>();next->create();g_frame=std::move(next);}
        auto& b=*g_frame;
        // Reset the temporal blend when the geometry generation or the add-on's
        // continuity token (F6 bypass generation) changes, or on the first frame.
        // The blend is gated by the previous frame's GPU guard flag, so a reset is
        // "mark the previous flag bad" (the blend kernel skips; prev_out reseeds).
        const bool blend_reset=!b.prev_seeded||f->handle_gen!=b.prev_handle_gen||f->temporal_gen!=b.prev_temporal_gen;
        b.prev_seeded=true;b.prev_handle_gen=f->handle_gen;b.prev_temporal_gen=f->temporal_gen;
        const float blend_w=float(temporal_blend_weight());
        // Alternating parity for the GPU guard flags: frame N owns flag[N%2] (its
        // prescans write it) and reads flag[(N-1)%2] (the previous frame's verdict).
        b.raw_parity^=1;
        const bool pa=b.raw_parity==0;
        unsigned* flag=pa?(unsigned*)b.guard_a.p:(unsigned*)b.guard_b.p;
        const unsigned* flag_prev=pa?(const unsigned*)b.guard_b.p:(const unsigned*)b.guard_a.p;
        if (blend_reset)
            dlss5::hip_check(hipMemsetAsync((void*)flag_prev,0xff,4,nullptr),"guard reset");
        // The prescans only ever write 1 (bad); this frame's flag must start clean
        // (the 0xff create() init is for the blend gate, not for the prescan).
        dlss5::hip_check(hipMemsetAsync((void*)flag,0,4,nullptr),"guard clear");
        const bool dev_ptrs=(f->flags & DLSS5_HIP_RAW_FLAGS_DEV_PTRS)!=0;
        const bool host_ptrs=(f->flags & DLSS5_HIP_RAW_FLAGS_HOST_PTRS)!=0;
        const size_t bytes=size_t(f->width)*f->height*((f->dxgi_format==28||f->dxgi_format==87)?4u:8u);
        void* dev_in=nullptr,* dev_out=nullptr;
        {
            char line[192];
            snprintf(line,sizeof line,"raw_gpu w=%u h=%u dxgi=%u flags=0x%x gen=%u in=%p out=%p bytes=%zu",
                     f->width,f->height,f->dxgi_format,f->flags,f->handle_gen,f->in_handle,f->out_handle,bytes);
            run_log(line);
        }
        if (dev_ptrs) {dev_in=f->in_handle;dev_out=f->out_handle;}
        else if (host_ptrs) {
            // CPU-mapped D3D12 READBACK/UPLOAD buffers: stage packed pixels on the GPU.
            if (b.raw_in_size!=bytes) {b.raw_in.alloc(bytes);b.raw_out.alloc(bytes);b.raw_in_size=bytes;}
            dev_in=b.raw_in.p;dev_out=b.raw_out.p;
            dlss5::hip_check(hipMemcpy(dev_in,f->in_handle,bytes,hipMemcpyHostToDevice),"raw input h2d");
            ft.mark("stage_in");
        }
        else {
            if (!dlss5::gpu::map_device(f->in_handle,bytes,f->handle_gen,&dev_in)) throw std::runtime_error("cannot import input handle");
            if (!dlss5::gpu::map_device(f->out_handle,bytes,f->handle_gen,&dev_out)) throw std::runtime_error("cannot import output handle");
        }
        const auto fmt=dlss5::gpu::format_from_dxgi(f->dxgi_format);
        b.history_valid=false;b.previous_flags=f->flags&1u;b.previous_white=f->paper_white;
        dlss5::FrameSettings settings{f->paper_white,f->transfer,f->color,(f->flags&1u)!=0};
        // Input pre-pass (16-bit only, bit-exact CPU contract): non-finite input sets
        // the guard flag on device. The commit below makes the bypass decision with
        // no CPU readback — the network still runs in this degenerate case (wasteful
        // there only), but out remains a bit-exact copy of in.
        if (fmt==dlss5::gpu::FmtRgba16F)
            dlss5::gpu::launch_guard_prescan((const void*)dev_in,size_t(f->width)*f->height*4,f->dxgi_format,true,flag);
        dlss5::gpu::launch_px_to_rgba((const uint8_t*)dev_in,b.source.as<float>(),f->width,f->height,f->dxgi_format);
        ft.mark("conv_in");
        dlss5::frame_encode(b.source.as<float>(),b.proxy.as<float>(),1920,1080,settings,nullptr);
        ft.mark("encode");
        // Cross-stream ordering with no CPU wait: encode (default stream) ->
        // network (own stream, via ev_net_in) -> decode/commit (default, via
        // ev_net_out). Fixes the latent race where the network could read the
        // encode proxy before the encode stream reached it.
        dlss5::hip_check(hipEventRecord(b.ev_net_in,nullptr),"encode event");
        g_net->run_gpu_gpu_async(b.proxy.as<float>(),b.neural.as<float>(),f->seed,nullptr,b.ev_net_in,b.ev_net_out);
        ft.mark("net");
        dlss5::hip_check(hipStreamWaitEvent(nullptr,b.ev_net_out,0),"net event wait");
        dlss5::frame_decode(b.source.as<float>(),b.proxy.as<float>(),b.neural.as<float>(),b.decoded.as<float>(),1920,1080,settings,nullptr);
        ft.mark("decode");
        // Output guard: scan the decoded RGB and write the same device flag.
        dlss5::gpu::launch_guard_prescan(b.decoded.as<float>(),n*3,f->dxgi_format,false,flag);
        ft.mark("guard");
        // Temporal blend (shimmer fix + ghosting gate): the live path is non-temporal
        // (history reset every frame), so the network amplifies the input's per-frame
        // micro-variation into visible shimmer. Mixing in the previous frame's decoded
        // RGB low-passes that variation. w=blend_w is the previous-frame weight (0
        // disables it). The gate keeps the blend only where the frames agree, so
        // moving content passes through untouched instead of trailing a ghost. The
        // previous frame's flag skips the blend when it bypassed (same semantics as
        // the old CPU prev_valid token).
        if (blend_w>0.f) {
            const size_t bn=n*3;
            const TemporalBlendGate gate=temporal_blend_gate();
            if (gate.enabled)
                dlss5::gpu::launch_temporal_blend_gated(b.decoded.as<float>(),b.prev_out.as<float>(),int(bn),blend_w,gate.lo,gate.hi,flag_prev);
            else
                dlss5::gpu::launch_temporal_blend(b.decoded.as<float>(),b.prev_out.as<float>(),int(bn),blend_w,flag_prev);
            // Unconditional prev update (seeds the next frame even when this blend is
            // gated; a gated prev's content is never consumed because its flag is bad).
            // Kernel copy: same bytes as the old hipMemcpy D2D, stays in-stream.
            dlss5::launch_copy_f32(b.decoded.as<float>(),b.prev_out.as<float>(),bn*4);
        }
        // Final write stage + GPU-resident bypass: flag nonzero copies the input
        // bytes bit-exact (the old CPU bypass), otherwise converts the decoded RGB.
        // Replaces the 4-byte D2H guard readback and the CPU branch: the ABI never
        // conveyed the decision to the caller, so it never needed to be on the host.
        dlss5::gpu::launch_commit(b.decoded.as<float>(),(const uint8_t*)dev_in,(uint8_t*)dev_out,f->width,f->height,f->dxgi_format,flag);
        ft.mark("conv_out");
        // Single completion point for the whole frame. The network stream is joined
        // via ev_net_out above; everything else runs on the default stream.
        dlss5::hip_check(hipStreamSynchronize(nullptr),"raw frame out");
        if (host_ptrs) dlss5::hip_check(hipMemcpy(f->out_handle,dev_out,bytes,hipMemcpyDeviceToHost),"raw output d2h");
        ft.mark("sync_out");
        g_err.clear();return 0;
    } catch(const std::exception&e) {
        // Fail safe: mark both parities "bad" so the next frame skips its blend and
        // its commit degrades to the bit-exact input copy.
        if(g_frame){(void)hipMemsetAsync(g_frame->guard_a.p,0xff,4,nullptr);(void)hipMemsetAsync(g_frame->guard_b.p,0xff,4,nullptr);}
        set_err(e.what());return -1;}
}

__attribute__((constructor)) static void dlss5_hip_publish(void) {
    memset(&g_bridge, 0, sizeof(g_bridge));
    g_bridge.magic = DLSS5_HIP_MAGIC;
    g_bridge.init = dlss5_init;
    g_bridge.run = dlss5_run;
    g_bridge.shutdown = dlss5_shutdown;
    g_bridge.last_error = dlss5_last_error;
    char buf[32];
    snprintf(buf, sizeof(buf), "%p", (void*)&g_bridge);
    if (setenv(DLSS5_HIP_ENV, buf, 1) != 0) {
        g_bridge.magic = 0;
        (void)unsetenv(DLSS5_HIP_ENV);
    }
    g_frame_bridge.magic=DLSS5_HIP_FRAME_MAGIC;g_frame_bridge.run_frame=dlss5_run_frame;
    g_frame_bridge.run_frame_raw_gpu=dlss5_run_frame_raw_gpu;
    snprintf(buf,sizeof(buf),"%p",(void*)&g_frame_bridge);
    if(setenv(DLSS5_HIP_FRAME_ENV,buf,1)!=0){g_frame_bridge.magic=0;(void)unsetenv(DLSS5_HIP_FRAME_ENV);}
    g_device_bridge.magic=DLSS5_HIP_DEVICE_MAGIC;g_device_bridge.find_device=dlss5_find_device;
    snprintf(buf,sizeof(buf),"%p",(void*)&g_device_bridge);
    if(setenv(DLSS5_HIP_DEVICE_ENV,buf,1)!=0){g_device_bridge.magic=0;(void)unsetenv(DLSS5_HIP_DEVICE_ENV);}
}
