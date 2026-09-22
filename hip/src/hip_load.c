/* The game preload is inherited by every process Steam starts, including
 * /usr/bin/env (Proton's shebang). A DT_NEEDED on libamdhip64 makes that
 * process die when a transitive dependency such as libfmt.so.12 is absent.
 * This file is linked only into libdlss5_hip.so. It dlopens the runtime
 * after the process exists and loads each non-system dependency from the
 * library's own RUNPATH or from the usual distro locations. Another fmt
 * SONAME is not a substitute: libfmt.so.11 does not satisfy libfmt.so.12.
 * Fatbin registration calls emitted by the compiler are forwarded. */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { kDirCap = 48, kNeedCap = 64, kVisitCap = 48 };

static void *hip_rt;
static void *last_handle;
static char hip_err[768];
static char visited[kVisitCap][PATH_MAX];
static int nvisited;

static void set_err(const char *text) {
    snprintf(hip_err, sizeof hip_err, "%s", text);
}

static int is_system(const char *soname) {
    static const char *skip[] = {
        "libc.so.6", "libm.so.6", "libdl.so.2", "librt.so.1", "libpthread.so.0",
        "ld-linux-x86-64.so.2", "libgcc_s.so.1", "libstdc++.so.6", NULL};
    for (int i = 0; skip[i]; i++)
        if (strcmp(soname, skip[i]) == 0)
            return 1;
    return 0;
}

static int seen(const char *path) {
    for (int i = 0; i < nvisited; i++)
        if (strcmp(visited[i], path) == 0)
            return 1;
    return 0;
}

static void mark(const char *path) {
    if (nvisited < kVisitCap && !seen(path))
        snprintf(visited[nvisited++], PATH_MAX, "%s", path);
}

static int read_full(int fd, uint64_t off, void *buf, size_t n) {
    if (lseek(fd, (off_t)off, SEEK_SET) < 0)
        return -1;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char *)buf + got, n - got);
        if (r <= 0)
            return -1;
        got += (size_t)r;
    }
    return 0;
}

static int vaddr_offset(const Elf64_Phdr *ph, int nph, uint64_t vaddr, uint64_t *off) {
    for (int i = 0; i < nph; i++) {
        if (ph[i].p_type != PT_LOAD)
            continue;
        if (vaddr >= ph[i].p_vaddr && vaddr < ph[i].p_vaddr + ph[i].p_memsz) {
            *off = ph[i].p_offset + (vaddr - ph[i].p_vaddr);
            return 0;
        }
    }
    return -1;
}

static int read_dynamic(const char *path, char needed[][256], int *nneeded, char runpath[],
                        size_t runpath_len) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    Elf64_Ehdr eh;
    if (read_full(fd, 0, &eh, sizeof eh) != 0 || memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 ||
        eh.e_ident[EI_CLASS] != ELFCLASS64 || eh.e_ident[EI_DATA] != ELFDATA2LSB || eh.e_phnum == 0 ||
        eh.e_phnum > 64 || eh.e_phentsize != sizeof(Elf64_Phdr)) {
        close(fd);
        return -1;
    }
    Elf64_Phdr ph[64];
    if (read_full(fd, eh.e_phoff, ph, eh.e_phnum * sizeof(Elf64_Phdr)) != 0) {
        close(fd);
        return -1;
    }
    uint64_t dyn_off = 0, dyn_sz = 0;
    int found = 0;
    for (int i = 0; i < eh.e_phnum; i++) {
        if (ph[i].p_type == PT_DYNAMIC) {
            dyn_off = ph[i].p_offset;
            dyn_sz = ph[i].p_filesz;
            found = 1;
            break;
        }
    }
    if (!found || dyn_sz < sizeof(Elf64_Dyn) || dyn_sz > 256 * sizeof(Elf64_Dyn)) {
        close(fd);
        return -1;
    }
    Elf64_Dyn dyn[256];
    int ndyn = (int)(dyn_sz / sizeof(Elf64_Dyn));
    if (read_full(fd, dyn_off, dyn, (size_t)ndyn * sizeof(Elf64_Dyn)) != 0) {
        close(fd);
        return -1;
    }
    uint64_t str_addr = 0, strsz = 0, run_off = (uint64_t)-1, rpath_off = (uint64_t)-1;
    for (int i = 0; i < ndyn; i++) {
        if (dyn[i].d_tag == DT_NULL)
            break;
        if (dyn[i].d_tag == DT_STRTAB)
            str_addr = dyn[i].d_un.d_ptr;
        else if (dyn[i].d_tag == DT_STRSZ)
            strsz = dyn[i].d_un.d_val;
        else if (dyn[i].d_tag == DT_RUNPATH)
            run_off = dyn[i].d_un.d_val;
        else if (dyn[i].d_tag == DT_RPATH)
            rpath_off = dyn[i].d_un.d_val;
    }
    if (!str_addr || !strsz || strsz > 1024 * 1024) {
        close(fd);
        return -1;
    }
    uint64_t str_off = 0;
    if (vaddr_offset(ph, eh.e_phnum, str_addr, &str_off) != 0) {
        close(fd);
        return -1;
    }
    char *strtab = malloc(strsz + 1);
    if (!strtab || read_full(fd, str_off, strtab, strsz) != 0) {
        free(strtab);
        close(fd);
        return -1;
    }
    strtab[strsz] = 0;
    close(fd);
    *nneeded = 0;
    for (int i = 0; i < ndyn && *nneeded < kNeedCap; i++) {
        if (dyn[i].d_tag == DT_NULL)
            break;
        if (dyn[i].d_tag == DT_NEEDED && dyn[i].d_un.d_val < strsz)
            snprintf(needed[(*nneeded)++], 256, "%s", strtab + dyn[i].d_un.d_val);
    }
    runpath[0] = 0;
    uint64_t pick = run_off != (uint64_t)-1 ? run_off : rpath_off;
    if (pick != (uint64_t)-1 && pick < strsz)
        snprintf(runpath, runpath_len, "%s", strtab + pick);
    free(strtab);
    return 0;
}

static void add_dir(char dirs[][PATH_MAX], int *n, const char *dir) {
    if (!dir || !dir[0] || *n >= kDirCap)
        return;
    char buf[PATH_MAX];
    if (!realpath(dir, buf))
        return;
    for (int i = 0; i < *n; i++)
        if (strcmp(dirs[i], buf) == 0)
            return;
    snprintf(dirs[(*n)++], PATH_MAX, "%s", buf);
}

static void expand_token(const char *in, const char *origin, char *out, size_t outlen) {
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 1 < outlen;) {
        size_t take = 0;
        const char *rep = NULL;
        size_t rep_len = 0;
        if (strncmp(in + i, "${ORIGIN}", 9) == 0) {
            take = 9;
            rep = origin;
            rep_len = strlen(origin);
        } else if (strncmp(in + i, "$ORIGIN", 7) == 0) {
            take = 7;
            rep = origin;
            rep_len = strlen(origin);
        } else if (strncmp(in + i, "${LIB}", 6) == 0) {
            take = 6;
            rep = "lib64";
            rep_len = 5;
        } else if (strncmp(in + i, "$LIB", 4) == 0) {
            take = 4;
            rep = "lib64";
            rep_len = 5;
        }
        if (rep) {
            if (o + rep_len >= outlen)
                break;
            memcpy(out + o, rep, rep_len);
            o += rep_len;
            i += take;
        } else {
            out[o++] = in[i++];
        }
    }
    out[o] = 0;
}

static void add_runpath(char dirs[][PATH_MAX], int *n, const char *runpath, const char *origin) {
    char copy[1024];
    snprintf(copy, sizeof copy, "%s", runpath);
    for (char *save = NULL, *tok = strtok_r(copy, ":", &save); tok; tok = strtok_r(NULL, ":", &save)) {
        char expanded[PATH_MAX];
        expand_token(tok, origin, expanded, sizeof expanded);
        add_dir(dirs, n, expanded);
    }
}

static void add_search_dirs(char dirs[][PATH_MAX], int *n, const char *origin, const char *runpath) {
    add_runpath(dirs, n, runpath, origin);
    add_dir(dirs, n, origin);
    char sub[PATH_MAX];
    snprintf(sub, sizeof sub, "%s/rocm_sysdeps/lib", origin);
    add_dir(dirs, n, sub);
    snprintf(sub, sizeof sub, "%s/llvm/lib", origin);
    add_dir(dirs, n, sub);
    const char *extra = getenv("DLSS5_HIP_DEP_DIRS");
    if (extra && extra[0]) {
        char copy[4096];
        snprintf(copy, sizeof copy, "%s", extra);
        for (char *save = NULL, *tok = strtok_r(copy, ":", &save); tok; tok = strtok_r(NULL, ":", &save))
            add_dir(dirs, n, tok);
    }
    static const char *fixed[] = {
        "/usr/lib64", "/usr/lib", "/lib64", "/lib", "/usr/local/lib64", "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib64/rocm", "/usr/lib/rocm",
        "/usr/lib/x86_64-linux-gnu/rocm", NULL};
    for (int i = 0; fixed[i]; i++)
        add_dir(dirs, n, fixed[i]);
    DIR *opt = opendir("/opt");
    if (!opt)
        return;
    int found = 0;
    struct dirent *de;
    while ((de = readdir(opt)) && found < 8) {
        if (strncmp(de->d_name, "rocm", 4) != 0)
            continue;
        char base[PATH_MAX];
        snprintf(base, sizeof base, "/opt/%s/lib", de->d_name);
        add_dir(dirs, n, base);
        snprintf(sub, sizeof sub, "%s/rocm_sysdeps/lib", base);
        add_dir(dirs, n, sub);
        snprintf(sub, sizeof sub, "%s/llvm/lib", base);
        add_dir(dirs, n, sub);
        found++;
    }
    closedir(opt);
}

static int find_soname(const char *soname, char dirs[][PATH_MAX], int ndirs, char *out) {
    for (int i = 0; i < ndirs; i++) {
        char candidate[PATH_MAX];
        snprintf(candidate, sizeof candidate, "%s/%s", dirs[i], soname);
        if (access(candidate, R_OK) == 0)
            return realpath(candidate, out) ? 0 : -1;
    }
    return -1;
}

static int load_closure(const char *path, int depth) {
    char resolved[PATH_MAX];
    if (!realpath(path, resolved))
        snprintf(resolved, sizeof resolved, "%s", path);
    if (seen(resolved))
        return 0;
    if (depth > 16) {
        set_err("HIP runtime dependency chain is too deep");
        return -1;
    }
    mark(resolved);
    char needed[kNeedCap][256];
    int nneeded = 0;
    char runpath[1024];
    char origin[PATH_MAX];
    snprintf(origin, sizeof origin, "%s", resolved);
    char *slash = strrchr(origin, '/');
    if (slash)
        *slash = 0;
    else
        snprintf(origin, sizeof origin, ".");
    if (read_dynamic(resolved, needed, &nneeded, runpath, sizeof runpath) == 0) {
        char (*dirs)[PATH_MAX] = calloc(kDirCap, PATH_MAX);
        if (!dirs) {
            set_err("HIP runtime: out of memory");
            return -1;
        }
        int ndirs = 0;
        add_search_dirs(dirs, &ndirs, origin, runpath);
        int failed = 0;
        for (int i = 0; i < nneeded; i++) {
            if (is_system(needed[i]) || strchr(needed[i], '/'))
                continue;
            char found[PATH_MAX];
            if (find_soname(needed[i], dirs, ndirs, found) != 0) {
                char msg[768];
                snprintf(msg, sizeof msg, "HIP runtime dependency not found: %s (needed by %s)", needed[i],
                         resolved);
                set_err(msg);
                failed = 1;
                break;
            }
            if (load_closure(found, depth + 1) != 0) {
                failed = 1;
                break;
            }
        }
        free(dirs);
        if (failed)
            return -1;
    }
    dlerror();
    void *handle = dlopen(resolved, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        char msg[768];
        const char *why = dlerror();
        snprintf(msg, sizeof msg, "HIP runtime: %s", why ? why : "dlopen failed");
        set_err(msg);
        return -1;
    }
    last_handle = handle;
    return 0;
}

static void open_runtime(const char *path) {
    hip_err[0] = 0;
    last_handle = NULL;
    if (load_closure(path, 0) == 0 && last_handle) {
        hip_rt = last_handle;
        return;
    }
    char saved[768];
    snprintf(saved, sizeof saved, "%s", hip_err);
    dlerror();
    hip_rt = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (hip_rt) {
        hip_err[0] = 0;
        return;
    }
    if (saved[0])
        snprintf(hip_err, sizeof hip_err, "%s", saved);
    else {
        const char *why = dlerror();
        snprintf(hip_err, sizeof hip_err, "HIP runtime: %s", why ? why : "dlopen failed");
    }
}

__attribute__((constructor(101))) static void dlss5_load_hip_runtime(void) {
    const char *path = getenv("DLSS5_HIP_LIBRARY");
    if (path && path[0]) {
        open_runtime(path);
        return;
    }
    open_runtime("libamdhip64.so.7");
}

const char *dlss5_hip_runtime_error(void) {
    return hip_rt ? NULL : (hip_err[0] ? hip_err : "HIP runtime did not load");
}

static void *runtime_symbol(const char *name) {
    return hip_rt ? dlsym(hip_rt, name) : NULL;
}

void **__hipRegisterFatBinary(const void *data) {
    void **(*fn)(const void *) = (void **(*)(const void *))runtime_symbol("__hipRegisterFatBinary");
    return fn ? fn(data) : NULL;
}

void __hipRegisterFunction(void **modules, const void *hostFunction, char *deviceFunction,
                           const char *deviceName, unsigned threadLimit, void *tid, void *bid,
                           void *blockDim, void *gridDim, int *wSize) {
    void (*fn)(void **, const void *, char *, const char *, unsigned, void *, void *, void *, void *,
               int *) = (void (*)(void **, const void *, char *, const char *, unsigned, void *, void *,
                                  void *, void *, int *))runtime_symbol("__hipRegisterFunction");
    if (fn)
        fn(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim,
           wSize);
}

void __hipUnregisterFatBinary(void **modules) {
    void (*fn)(void **) = (void (*)(void **))runtime_symbol("__hipUnregisterFatBinary");
    if (fn)
        fn(modules);
}
