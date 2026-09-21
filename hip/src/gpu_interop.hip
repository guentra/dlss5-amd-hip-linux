// GPU-resident frame interop for the in-game path:
//  - device kernels (kernels/gpu_convert.hip) replicating the hook's CPU pixel pipeline;
//  - Win32 D3D12 shared handle -> unix fd -> hipImportExternalMemory import, cached for
//    the process lifetime (same technique as the portable LD_PRELOAD bridge);
//  - host launch helpers used by dlss5_run_frame_raw_gpu in capi.hip.
// FMA contraction must stay OFF here (bit-exact pixel math, see gpu_convert.hip).
#pragma STDC FP_CONTRACT OFF

#include "dlss5_gpu_interop.hpp"
#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include "../kernels/gpu_convert.hip"

#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace dlss5 {
namespace gpu {

Fmt format_from_dxgi(unsigned format) {
  switch (format) {
    case 28:
      return FmtRgba8;
    case 87:
      return FmtBgra8;
    case 11:
      return FmtR16Unorm;
    case 10:
      return FmtRgba16F;
    default:
      return FmtUnsupported;
  }
}

namespace {

static void interop_log(const char* fmt, ...) {
  FILE* f = fopen("/tmp/dlss5-hip-run.log", "a");
  if (!f) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fputc('\n', f);
  fclose(f);
}

// ---------------- Win32 handle -> dma-buf fd (ported from the portable bridge) ----------------

typedef int (*wine_h2fd_t)(void*, unsigned, int*, unsigned*);

wine_h2fd_t resolve_h2fd() {
  static wine_h2fd_t fn = nullptr;
  static bool tried = false;
  if (tried) return fn;
  tried = true;
  fn = (wine_h2fd_t)dlsym(RTLD_DEFAULT, "wine_server_handle_to_fd");
  if (fn) return fn;
  void* h = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
  if (!h) h = dlopen("ntdll.so", RTLD_NOW | RTLD_GLOBAL);
  if (h) fn = (wine_h2fd_t)dlsym(h, "wine_server_handle_to_fd");
  interop_log("resolve_h2fd %s", fn ? "ok" : "missing");
  return fn;
}

// Local (non-dynsym) Wine unix helpers live in win32u.so; scan its .symtab directly.
void* elf_local_sym(void* handle, const char* name) {
  struct link_map* lm = nullptr;
  struct stat st;
  const ElfW(Ehdr)* eh;
  const ElfW(Shdr)* sh;
  const ElfW(Shdr)* symtab = nullptr;
  const ElfW(Shdr)* strtab = nullptr;
  const ElfW(Sym)* sym;
  const char* strs;
  const char* map_bytes;
  void* map;
  void* found = nullptr;
  int fd, i, n;
  size_t file_size;

  if (!handle || !name) return nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || !lm || !lm->l_name || !lm->l_name[0]) return nullptr;
  fd = open(lm->l_name, O_RDONLY);
  if (fd < 0) return nullptr;
  if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(ElfW(Ehdr))) {
    close(fd);
    return nullptr;
  }
  file_size = (size_t)st.st_size;
  map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return nullptr;
  map_bytes = (const char*)map;
  eh = (const ElfW(Ehdr)*)map;
  if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) goto out;
  if (eh->e_shentsize != sizeof(ElfW(Shdr))) goto out;
  if (eh->e_shoff > file_size ||
      (size_t)eh->e_shnum * sizeof(ElfW(Shdr)) > file_size - eh->e_shoff)
    goto out;
  sh = (const ElfW(Shdr)*)(map_bytes + eh->e_shoff);
  for (i = 0; i < eh->e_shnum; i++) {
    if (sh[i].sh_type == SHT_SYMTAB) {
      symtab = &sh[i];
      if (sh[i].sh_link < eh->e_shnum) strtab = &sh[sh[i].sh_link];
      break;
    }
  }
  if (!symtab || !strtab || !symtab->sh_entsize) goto out;
  if (symtab->sh_entsize != sizeof(ElfW(Sym))) goto out;
  if (symtab->sh_offset > file_size || symtab->sh_size > file_size - symtab->sh_offset) goto out;
  if (strtab->sh_offset > file_size || strtab->sh_size > file_size - strtab->sh_offset) goto out;
  strs = map_bytes + strtab->sh_offset;
  // Byte offset, not Elf_Sym index: `(Elf_Sym*)map + sh_offset` walks off the mmap
  // and SIGSEGVs (in-game crash #2: RIP in this .so, RSI near .rodata symbol name).
  sym = (const ElfW(Sym)*)(map_bytes + symtab->sh_offset);
  n = (int)(symtab->sh_size / symtab->sh_entsize);
  for (i = 0; i < n; i++) {
    if (!sym[i].st_name || ELF64_ST_TYPE(sym[i].st_info) != STT_FUNC) continue;
    if (sym[i].st_name >= strtab->sh_size) continue;
    if (strcmp(strs + sym[i].st_name, name) == 0) {
      found = (char*)lm->l_addr + sym[i].st_value;
      break;
    }
  }
out:
  munmap(map, file_size);
  return found;
}

typedef unsigned int d3dkmt_handle_t;
typedef d3dkmt_handle_t (*d3dkmt_open_resource_t)(d3dkmt_handle_t, void*, d3dkmt_handle_t*, d3dkmt_handle_t*);
typedef int (*d3dkmt_object_get_fd_t)(d3dkmt_handle_t);
typedef int (*d3dkmt_destroy_resource_t)(d3dkmt_handle_t);

int d3dkmt_handle_to_fd(void* nt_handle) {
  static int resolved = 0;
  static d3dkmt_open_resource_t open_res = nullptr;
  static d3dkmt_object_get_fd_t get_fd = nullptr;
  static d3dkmt_destroy_resource_t destroy_res = nullptr;
  d3dkmt_handle_t local = 0, mutex = 0, sync = 0;
  int fd = -1;
  void* h;

  if (!resolved) {
    resolved = 1;
    h = dlopen("win32u.so", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("win32u.so", RTLD_NOW);
    if (h) {
      open_res = (d3dkmt_open_resource_t)elf_local_sym(h, "d3dkmt_open_resource");
      get_fd = (d3dkmt_object_get_fd_t)elf_local_sym(h, "d3dkmt_object_get_fd");
      destroy_res = (d3dkmt_destroy_resource_t)elf_local_sym(h, "d3dkmt_destroy_resource");
    }
    interop_log("d3dkmt resolve win32u=%p open=%p get_fd=%p destroy=%p", h, (void*)open_res, (void*)get_fd,
                (void*)destroy_res);
  }
  if (!open_res || !get_fd) return -1;
  local = open_res(0, nt_handle, &mutex, &sync);
  if (!local) return -1;
  fd = get_fd(local);
  if (destroy_res) destroy_res(local);
  return fd;
}

// Wine dxgkrnl IOCTL used by winevulkan get_shared_resource_fd: turn a
// D3DKMT/dxgk NT shared handle (FD_TYPE_INVALID / STATUS_BAD_DEVICE_TYPE)
// into a file-type NT handle that wine_server_handle_to_fd can unwrap.
static constexpr unsigned kDxgkGetSharedResourceFd = 0x23400Cu;

typedef int (*nt_ioctl_fn)(void* handle, void* event, void* apc, void* apc_ctx, void* iosb, unsigned code,
                           void* in_buf, unsigned in_len, void* out_buf, unsigned out_len);
typedef int (*nt_close_fn)(void* handle);

int dxgk_handle_to_fd(void* handle, wine_h2fd_t h2fd) {
  static nt_ioctl_fn ioctl;
  static nt_close_fn nt_close;
  static bool tried;
  struct {
    long status;
    unsigned long information;
  } iosb{};
  int file_handle32 = -1;
  int st, unix_fd = -1;
  void* file_handle;
  if (!handle || !h2fd) return -1;
  if (!tried) {
    tried = true;
    void* ntdll = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD);
    if (!ntdll) ntdll = dlopen("ntdll.so", RTLD_NOW);
    if (ntdll) {
      ioctl = (nt_ioctl_fn)dlsym(ntdll, "NtDeviceIoControlFile");
      nt_close = (nt_close_fn)dlsym(ntdll, "NtClose");
    }
    interop_log("dxgk ioctl=%p NtClose=%p", (void*)ioctl, (void*)nt_close);
  }
  if (!ioctl || !nt_close) return -1;
  st = ioctl(handle, nullptr, nullptr, nullptr, &iosb, kDxgkGetSharedResourceFd, nullptr, 0, &file_handle32, 4);
  interop_log("dxgk ioctl handle=%p st=0x%x out=0x%x iosb=0x%lx", handle, (unsigned)st, (unsigned)file_handle32,
              (unsigned long)iosb.status);
  if (st != 0 || file_handle32 == -1 || file_handle32 == 0) return -1;
  file_handle = (void*)(intptr_t)file_handle32;
  st = h2fd(file_handle, 0x0001u, &unix_fd, nullptr);
  nt_close(file_handle);
  interop_log("dxgk h2fd file=%p st=0x%x fd=%d", file_handle, (unsigned)st, unix_fd);
  if (st != 0 || unix_fd < 0) return -1;
  return unix_fd;
}

int handle_to_fd(void* handle) {
  int unix_fd = -1;
  unsigned options = 0;
  wine_h2fd_t h2fd = resolve_h2fd();
  // winevulkan imports OPAQUE_WIN32 with FILE_READ_DATA (0x1), not GENERIC_READ
  // (0x80000000). GENERIC_* is not granted on D3DKMT NT shared objects, so the
  // old mask failed and fell through into the win32u .symtab scanner.
  static const unsigned accesses[] = {
      0x0001u,                    // FILE_READ_DATA
      0x0003u,                    // FILE_READ_DATA | FILE_WRITE_DATA
      0x00100000u,                // SYNCHRONIZE
      0x00000000u,                // any granted access
      0x80000000u | 0x40000000u,  // GENERIC_READ | GENERIC_WRITE (legacy)
      0x80000000u,                // GENERIC_READ
  };
  unix_fd = dxgk_handle_to_fd(handle, h2fd);
  if (unix_fd >= 0) return unix_fd;
  if (h2fd) {
    for (unsigned a : accesses) {
      unix_fd = -1;
      options = 0;
      int st = h2fd(handle, a, &unix_fd, &options);
      interop_log("h2fd handle=%p access=0x%x st=0x%x fd=%d opt=%u", handle, a, (unsigned)st, unix_fd, options);
      if (st == 0 && unix_fd >= 0) return unix_fd;
    }
  }
  unix_fd = d3dkmt_handle_to_fd(handle);
  interop_log("d3dkmt handle=%p fd=%d", handle, unix_fd);
  return unix_fd;
}

// ---------------- import cache (process lifetime, mirrors the portable bridge) ----------------

struct ImportEntry {
  void* handle;
  size_t size;
  uint32_t gen;
  int fd;
  hipExternalMemory_t ext;
  void* dev;
};
std::vector<ImportEntry> g_imports;
std::mutex g_import_mu;

}  // namespace

bool map_device(void* handle, size_t size, uint32_t generation, void** dev) {
  if (!handle || !size || !dev) return false;
  std::lock_guard<std::mutex> lk(g_import_mu);
  for (const auto& e : g_imports)
    if (e.handle == handle && e.size == size && e.gen == generation) {
      *dev = e.dev;
      return true;
    }
  int fd = handle_to_fd(handle);
  if (fd < 0) {
    interop_log("map_device handle=%p size=%zu gen=%u: no fd", handle, size, generation);
    return false;
  }
  off_t fd_size = lseek(fd, 0, SEEK_END);
  if (fd_size > 0 && (size_t)fd_size < size) {
    interop_log("map_device handle=%p fd=%d dmabuf=%lld < need=%zu", handle, fd, (long long)fd_size, size);
    close(fd);
    return false;
  }
  hipExternalMemoryHandleDesc d{};
  d.type = hipExternalMemoryHandleTypeOpaqueFd;
  d.handle.fd = dup(fd);
  close(fd);
  if (d.handle.fd < 0) return false;
  d.size = (fd_size > 0 && (size_t)fd_size > size) ? (size_t)fd_size : size;
  hipExternalMemory_t ext = nullptr;
  hipError_t err = hipImportExternalMemory(&ext, &d);
  if (err != hipSuccess) {
    interop_log("hipImportExternalMemory fd=%d size=%zu err=%d %s", d.handle.fd, (size_t)d.size, (int)err,
                hipGetErrorString(err));
    close(d.handle.fd);
    return false;
  }
  hipExternalMemoryBufferDesc bd{};
  bd.offset = 0;
  bd.size = size;
  void* ptr = nullptr;
  err = hipExternalMemoryGetMappedBuffer(&ptr, ext, &bd);
  if (err != hipSuccess || !ptr) {
    interop_log("hipExternalMemoryGetMappedBuffer size=%zu err=%d %s", size, (int)err, hipGetErrorString(err));
    (void)hipDestroyExternalMemory(ext);
    close(d.handle.fd);
    return false;
  }
  interop_log("map_device ok handle=%p gen=%u fd=%d map_size=%zu buf=%zu dev=%p", handle, generation, d.handle.fd,
              (size_t)d.size, size, ptr);
  g_imports.push_back({handle, size, generation, d.handle.fd, ext, ptr});
  *dev = ptr;
  return true;
}

// ---------------- launch helpers ----------------

namespace {
constexpr int kW = 1920, kH = 1080;
constexpr int kBlock = 32;
inline dim3 grid_for(int w, int h) {
  return dim3((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock, 1);
}
// The raw-frame pipeline runs on the default stream (like the frame buffers in capi).
hipStream_t raw_stream() { return nullptr; }
}  // namespace

void launch_px_to_rgba(const uint8_t* dev_in, float* out, int w, int h, unsigned dxgi_format) {
  const dim3 blk(kBlock, kBlock, 1);
  switch (format_from_dxgi(dxgi_format)) {
    case FmtRgba8:
      k_gc_px_to_rgba<FmtRgba8><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    case FmtBgra8:
      k_gc_px_to_rgba<FmtBgra8><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    case FmtR16Unorm:
      k_gc_px_to_rgba<FmtR16Unorm><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    case FmtRgba16F:
      k_gc_px_to_rgba<FmtRgba16F><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    default:
      throw std::runtime_error("unsupported dxgi format for gpu conversion");
  }
  hip_check(hipGetLastError(), "launch px_to_rgba");
}

void launch_rgb_to_pix(const float* rgb, const uint8_t* alpha_src, uint8_t* out, int w, int h, unsigned dxgi_format) {
  const dim3 blk(kBlock, kBlock, 1);
  switch (format_from_dxgi(dxgi_format)) {
    case FmtRgba8:
      k_gc_rgb_to_pix<FmtRgba8><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    case FmtBgra8:
      k_gc_rgb_to_pix<FmtBgra8><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    case FmtR16Unorm:
      k_gc_rgb_to_pix<FmtR16Unorm><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    case FmtRgba16F:
      k_gc_rgb_to_pix<FmtRgba16F><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    default:
      throw std::runtime_error("unsupported dxgi format for gpu conversion");
  }
  hip_check(hipGetLastError(), "launch rgb_to_pix");
}

// Temporal blend (shimmer fix): cur = (1-w)*cur + w*prev over n floats, GPU-resident.
// w in [0,1); w=0 is skipped (no launch). skip_if_set = previous frame's guard flag
// (device): nonzero skips the blend (a bypassed frame is never mixed in).
void launch_temporal_blend(float* cur, const float* prev, int n, float w, const unsigned* skip_if_set) {
  if (w <= 0.f) return;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  k_gc_temporal_blend<<<grid, block, 0, raw_stream()>>>(cur, prev, n, w, skip_if_set);
  hip_check(hipGetLastError(), "launch temporal blend");
}

// Gated temporal blend: w/lo/hi contract of k_gc_temporal_blend_gated; requires lo < hi.
// skip_if_set as in launch_temporal_blend.
void launch_temporal_blend_gated(float* cur, const float* prev, int n, float w, float lo, float hi,
                                 const unsigned* skip_if_set) {
  if (w <= 0.f || hi <= lo) return;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  k_gc_temporal_blend_gated<<<grid, block, 0, raw_stream()>>>(cur, prev, n, w, lo, hi, skip_if_set);
  hip_check(hipGetLastError(), "launch gated temporal blend");
}

// Whole-frame guard prescan, GPU-resident: writes nonzero to *flag (device, 4 bytes,
// pre-allocated and caller-owned) when the frame must be bypassed. input_prescan=true
// is the 16-bit input pre-pass (n = number of 16-bit elements, rgba16_float only);
// false is the output guard (n = number of RGB floats: non-finite, plus not
// representable as half for rgba16_float). No readback: the commit kernel consumes
// the flag on device.
void launch_guard_prescan(const void* dev, size_t n, unsigned dxgi_format, bool input_prescan, unsigned* flag) {
  if (input_prescan) {  // only the 16-bit input has a pre-pass (bit-exact CPU contract)
    k_gc_prescan16<<<(int)((n + kBlock - 1) / kBlock), kBlock, 0, raw_stream()>>>((const uint16_t*)dev, int(n), flag);
    hip_check(hipGetLastError(), "guard prescan16 launch");
    return;
  }
  if (format_from_dxgi(dxgi_format) == FmtRgba16F) {
    k_gc_prescan_rgb16f<<<(int)((n + kBlock - 1) / kBlock), kBlock, 0, raw_stream()>>>((const float*)dev, int(n), flag);
  } else {
    k_gc_prescan_f32<<<(int)((n + kBlock - 1) / kBlock), kBlock, 0, raw_stream()>>>((const float*)dev, int(n), flag);
  }
  hip_check(hipGetLastError(), "guard prescan launch");
}

// Final write stage + GPU-resident bypass (k_gc_commit): *bad nonzero copies the input
// pixel bytes bit-exact (the old CPU bypass was a whole-buffer D2D copy), otherwise
// converts the decoded RGB exactly like launch_rgb_to_pix.
void launch_commit(const float* rgb, const uint8_t* pix_in, uint8_t* pix_out, int w, int h, unsigned dxgi_format,
                   const unsigned* bad) {
  const dim3 blk(kBlock, kBlock, 1);
  switch (format_from_dxgi(dxgi_format)) {
    case FmtRgba8:
      k_gc_commit<FmtRgba8><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, pix_in, pix_out, bad);
      break;
    case FmtBgra8:
      k_gc_commit<FmtBgra8><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, pix_in, pix_out, bad);
      break;
    case FmtR16Unorm:
      k_gc_commit<FmtR16Unorm><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, pix_in, pix_out, bad);
      break;
    case FmtRgba16F:
      k_gc_commit<FmtRgba16F><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, pix_in, pix_out, bad);
      break;
    default:
      throw std::runtime_error("unsupported dxgi format for gpu commit");
  }
  hip_check(hipGetLastError(), "launch commit");
}

}  // namespace gpu
}  // namespace dlss5
