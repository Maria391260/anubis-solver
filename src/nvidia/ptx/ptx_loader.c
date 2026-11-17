/*
 * ptx_loader.c -- runtime loader for the hand-written PTX kernels.
 *
 * The kernels in this directory take flat scalar parameters (see
 * sha256_pow.ptx):
 *
 *   mid0..mid7, prefix_len, mode, use_le, difficulty, check_bits,
 *   base, count, results_ptr
 *
 * and report hits through results_ptr, a 304-byte struct:
 *
 *   u32 count; u32 pad[3]; u32 nonces[8]; u32 hashes[8][8];
 *
 * The cubins are built by build_ptx.sh. Every digest is re-verified by
 * src/core/anubis_solver.c before it is accepted.
 *
 * Build (driver API only, no CUDA runtime):
 *   gcc -O2 -shared -fPIC -o libanubis_ptx_loader.so ptx_loader.c -lcuda
 *
 * The HashX interpreter (hashx_interp.ptx) has its own entry points:
 *
 *   anubis_ptx_hashx_available(), anubis_ptx_hashx_info()
 *   anubis_ptx_hashx_set_program(keys, prog, instructions)
 *   anubis_ptx_hashx_solve(difficulty, start, max_nonce,
 *                          out_nonce, out_hex, out_ms)
 *
 * Environment:
 *   ANUBIS_PTX_DIR    directory holding the cubins (default ".")
 *   ANUBIS_PTX_ILP    1, 2 or 4; selects sha256_opt_ilp<N>.cubin (default 2)
 *   ANUBIS_PTX_CHUNK  nonces per launch per device (default 1<<22)
 */

#define _POSIX_C_SOURCE 199309L

#ifdef ANUBIS_MIN_CUDA_H
#include "cuda_min.h"   /* no driver headers on this machine */
#else
#include <cuda.h>
#endif
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(ANUBIS_DLOPEN_CUDA)
/*
 * Some images (Kaggle's, for one) ship the CUDA toolkit but no libcuda.so to
 * link against: the driver library only exists at run time. Resolve the entry
 * points with dlopen()/dlsym() instead of linking -lcuda.
 */
#include <dlfcn.h>

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGetCount)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, void *,
                                    void **, void **);

#define cuInit p_cuInit
#define cuDeviceGetCount p_cuDeviceGetCount
#define cuDeviceGet p_cuDeviceGet
#define cuDevicePrimaryCtxRetain p_cuDevicePrimaryCtxRetain
#define cuCtxSetCurrent p_cuCtxSetCurrent
#define cuCtxSynchronize p_cuCtxSynchronize
#define cuMemAlloc_v2 p_cuMemAlloc_v2
#define cuMemcpyDtoH_v2 p_cuMemcpyDtoH_v2
#define cuMemcpyHtoD_v2 p_cuMemcpyHtoD_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemsetD8_v2 p_cuMemsetD8_v2
#define cuModuleLoadData p_cuModuleLoadData
#define cuModuleGetFunction p_cuModuleGetFunction
#define cuModuleUnload p_cuModuleUnload
#define cuLaunchKernel p_cuLaunchKernel

static int load_libcuda(void)
{
    static const char *const cands[] = {
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so.1"
    };
    static const struct {
        const char *name;
        void **slot;
    } syms[] = {
        {"cuInit", (void **)(void *)&p_cuInit},
        {"cuDeviceGetCount", (void **)(void *)&p_cuDeviceGetCount},
        {"cuDeviceGet", (void **)(void *)&p_cuDeviceGet},
        {"cuDevicePrimaryCtxRetain", (void **)(void *)&p_cuDevicePrimaryCtxRetain},
        {"cuCtxSetCurrent", (void **)(void *)&p_cuCtxSetCurrent},
        {"cuCtxSynchronize", (void **)(void *)&p_cuCtxSynchronize},
        {"cuMemAlloc_v2", (void **)(void *)&p_cuMemAlloc_v2},
        {"cuMemcpyDtoH_v2", (void **)(void *)&p_cuMemcpyDtoH_v2},
        {"cuMemcpyHtoD_v2", (void **)(void *)&p_cuMemcpyHtoD_v2},
        {"cuMemsetD8_v2", (void **)(void *)&p_cuMemsetD8_v2},
        {"cuModuleLoadData", (void **)(void *)&p_cuModuleLoadData},
        {"cuModuleGetFunction", (void **)(void *)&p_cuModuleGetFunction},
        {"cuModuleUnload", (void **)(void *)&p_cuModuleUnload},
        {"cuLaunchKernel", (void **)(void *)&p_cuLaunchKernel}
    };
    size_t i, j;

    if (p_cuInit != NULL) {
        return 0;
    }
    for (i = 0; i < sizeof cands / sizeof cands[0]; i++) {
        void *h = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
        int ok = 1;
        if (h == NULL) {
            continue;
        }
        for (j = 0; j < sizeof syms / sizeof syms[0]; j++) {
            *syms[j].slot = dlsym(h, syms[j].name);
            if (*syms[j].slot == NULL) {
                ok = 0;
            }
        }
        if (ok) {
            return 0;
        }
        dlclose(h);
        p_cuInit = NULL;
    }
    return -1;
}
#endif /* ANUBIS_DLOPEN_CUDA */

#define PTX_MAX_DEV 8
#define PTX_MAX_INSTR 512
#define PTX_PROG_BYTES (PTX_MAX_INSTR * 16)
#define PTX_RESULT_SLOTS 8
#define PTX_BLOCK 256

typedef struct {
    uint32_t count;
    uint32_t pad[3];
    uint32_t nonces[PTX_RESULT_SLOTS];
    uint32_t hashes[PTX_RESULT_SLOTS][8];
} ptx_results;

static struct {
    int inited;
    int available;
    int ndev;
    int have_opt;
    int ilp;
    pthread_mutex_t mu;
    CUcontext ctx[PTX_MAX_DEV];
    CUmodule mod_pow[PTX_MAX_DEV];
    CUmodule mod_opt[PTX_MAX_DEV];
    CUfunction fn_pow[PTX_MAX_DEV];
    CUfunction fn_opt[PTX_MAX_DEV];
    int have_hx;
    CUmodule mod_hx[PTX_MAX_DEV];
    CUfunction fn_hx[PTX_MAX_DEV];
    CUdeviceptr d_prog[PTX_MAX_DEV];
    CUdeviceptr d_res[PTX_MAX_DEV];
    uint64_t hx_keys[4];
    uint32_t hx_code_size;
    ptx_results h_res;
    char dir[512];
    char info[192];
    char reason[192];
} g_ptx = { 0 };

static double ptx_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint32_t env_u32(const char *name, uint32_t def) {
    const char *s = getenv(name);
    if (s != NULL && *s != '\0') {
        return (uint32_t)strtoul(s, NULL, 0);
    }
    return def;
}

static int read_file(const char *path, void **out, size_t *size) {
    FILE *f = fopen(path, "rb");
    long n;
    void *data;
    if (f == NULL) {
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = ftell(f);
    if (n <= 0) { fclose(f); return -1; }
    rewind(f);
    data = malloc((size_t)n);
    if (data == NULL) { fclose(f); return -1; }
    if (fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data); fclose(f); return -1;
    }
    fclose(f);
    *out = data;
    *size = (size_t)n;
    return 0;
}

/* Load one cubin and resolve its entry point. Returns 0 on success. */
static int load_one(const char *name, CUmodule *mod, const char *entry,
                    CUfunction *fn) {
    char path[1024];
    void *data = NULL;
    size_t size = 0;
    CUresult err;

    snprintf(path, sizeof(path), "%s/%s", g_ptx.dir, name);
    if (read_file(path, &data, &size) != 0) {
        fprintf(stderr, "ptx_loader: cannot read %s\n", path);
        return -1;
    }
    err = cuModuleLoadData(mod, data);
    free(data);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "ptx_loader: cuModuleLoadData(%s) failed: %d\n", path, (int)err);
        return -1;
    }
    err = cuModuleGetFunction(fn, *mod, entry);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "ptx_loader: %s has no %s: %d\n", path, entry, (int)err);
        cuModuleUnload(*mod);
        *mod = NULL;
        return -1;
    }
    return 0;
}

static int ptx_init(void) {
    int count = 0;
    int all_opt = 1;
    int have_hx = 1;
    const char *dir;
    char opt_name[64];

    if (g_ptx.inited) {
        return g_ptx.available;
    }
    g_ptx.inited = 1;
    pthread_mutex_init(&g_ptx.mu, NULL);

#if defined(ANUBIS_DLOPEN_CUDA)
    if (load_libcuda() != 0) {
        snprintf(g_ptx.reason, sizeof(g_ptx.reason),
                 "dlopen(libcuda.so.1) failed: no CUDA driver library");
        return 0;
    }
#endif
    if (cuInit(0) != CUDA_SUCCESS) {
        snprintf(g_ptx.reason, sizeof(g_ptx.reason), "cuInit() failed");
        return 0;
    }
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
        snprintf(g_ptx.reason, sizeof(g_ptx.reason),
                 "no CUDA device (cuDeviceGetCount -> %d)", count);
        return 0;
    }
    if (count > PTX_MAX_DEV) {
        count = PTX_MAX_DEV;
    }

    dir = getenv("ANUBIS_PTX_DIR");
    if (dir == NULL || *dir == '\0') {
        dir = ".";
    }
    snprintf(g_ptx.dir, sizeof(g_ptx.dir), "%s", dir);

    g_ptx.ilp = (int)env_u32("ANUBIS_PTX_ILP", 2);
    if (g_ptx.ilp != 1 && g_ptx.ilp != 2 && g_ptx.ilp != 4) {
        g_ptx.ilp = 2;
    }
    snprintf(opt_name, sizeof(opt_name), "sha256_opt_ilp%d.cubin", g_ptx.ilp);

    for (int i = 0; i < count; i++) {
        CUdevice dev;
        CUcontext ctx;
        if (cuDeviceGet(&dev, i) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
            snprintf(g_ptx.reason, sizeof(g_ptx.reason),
                     "device %d: cuDeviceGet/cuDevicePrimaryCtxRetain failed", i);
            return 0;
        }
        g_ptx.ctx[i] = ctx;
        if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
            snprintf(g_ptx.reason, sizeof(g_ptx.reason),
                     "device %d: cuCtxSetCurrent failed", i);
            return 0;
        }
        if (load_one("sha256_pow.cubin", &g_ptx.mod_pow[i],
                     "sha256_pow_kernel", &g_ptx.fn_pow[i]) != 0) {
            snprintf(g_ptx.reason, sizeof(g_ptx.reason),
                     "device %d: could not load sha256_pow.cubin (missing or rejected)",
                     i);
            return 0;
        }
        if (load_one(opt_name, &g_ptx.mod_opt[i],
                     "sha256_opt_kernel", &g_ptx.fn_opt[i]) != 0) {
            g_ptx.fn_opt[i] = NULL;
            all_opt = 0;
        }
        if (cuMemAlloc(&g_ptx.d_res[i], sizeof(ptx_results)) != CUDA_SUCCESS) {
            snprintf(g_ptx.reason, sizeof(g_ptx.reason),
                     "device %d: cuMemAlloc failed", i);
            return 0;
        }
        /* HashX is optional: a build without hashx_interp.cubin still gives
         * the SHA-256 kernels. */
        if (load_one("hashx_interp.cubin", &g_ptx.mod_hx[i], "hashx_kernel",
                     &g_ptx.fn_hx[i]) == 0 &&
            cuMemAlloc(&g_ptx.d_prog[i], PTX_PROG_BYTES) == CUDA_SUCCESS) {
            have_hx = 1;
        } else {
            g_ptx.fn_hx[i] = NULL;
            have_hx = 0;
        }
    }

    g_ptx.ndev = count;
    g_ptx.have_hx = have_hx;
    g_ptx.have_opt = all_opt;
    g_ptx.available = 1;
    snprintf(g_ptx.info, sizeof(g_ptx.info),
             "PTX cubins: %d device(s), %s, ILP=%d%s", count,
             all_opt ? "pow+opt" : "pow only", g_ptx.ilp,
             have_hx ? ", hashx" : "");
    return 1;
}

static void hex_from_words(const uint32_t w[8], char out[65]) {
    static const char hexd[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            unsigned v = (w[i] >> (8 * (3 - j))) & 0xFFu;
            out[(4 * i + j) * 2 + 0] = hexd[v >> 4];
            out[(4 * i + j) * 2 + 1] = hexd[v & 0xFu];
        }
    }
    out[64] = '\0';
}

int anubis_ptx_available(void) {
    return ptx_init();
}

const char *anubis_ptx_info(void) {
    if (!ptx_init()) {
        return g_ptx.reason[0] ? g_ptx.reason : "PTX backend unavailable";
    }
    return g_ptx.info;
}

/*
 * Returns 0 on a hit (the caller re-verifies the digest), 1 if the range was
 * exhausted, -1 on backend failure.
 */
int anubis_ptx_solve(const uint32_t mid[8], uint32_t prefix_len, int mode,
                     int use_le, int difficulty, int check_bits,
                     uint64_t max_nonce, double budget_ms,
                     uint32_t *out_nonce, char out_hex[65], double *out_ms) {
    uint64_t maxn;
    uint64_t base = 0;
    uint32_t chunk;
    double t0;
    int rc = 1;
    int use_opt;

    if (mid == NULL || out_nonce == NULL || out_hex == NULL) {
        return -1;
    }
    if (!ptx_init()) {
        return -1;
    }
    pthread_mutex_lock(&g_ptx.mu);
    t0 = ptx_now();

    maxn = max_nonce > 0xFFFFFFFFull ? 0xFFFFFFFFull : max_nonce;
    chunk = env_u32("ANUBIS_PTX_CHUNK", 1u << 22);
    if (chunk < 4096u) {
        chunk = 4096u;
    }
    if (chunk > (1u << 24)) {
        chunk = 1u << 24;
    }
    use_opt = (mode == 1 && g_ptx.have_opt);

    while (base <= maxn) {
        int launched = 0;
        int found = 0;
        uint32_t best_nonce = 0;
        uint32_t best_hash[8] = { 0 };

        for (int d = 0; d < g_ptx.ndev; d++) {
            uint64_t dev_base = base + (uint64_t)d * (uint64_t)chunk;
            uint64_t remaining;
            uint32_t cnt;
            uint32_t dbase;
            uint32_t m[8];
            uint64_t resptr;
            unsigned blocks;
            void *args[16];

            if (dev_base > maxn) {
                break;
            }
            remaining = maxn - dev_base + 1;
            cnt = (uint32_t)(remaining < (uint64_t)chunk ? remaining : (uint64_t)chunk);
            if (cnt == 0) {
                break;
            }
            dbase = (uint32_t)dev_base;
            memcpy(m, mid, sizeof(m));
            resptr = (uint64_t)g_ptx.d_res[d];

            if (cuCtxSetCurrent(g_ptx.ctx[d]) != CUDA_SUCCESS ||
                cuMemsetD8(g_ptx.d_res[d], 0, sizeof(ptx_results)) != CUDA_SUCCESS) {
                rc = -1;
                goto done;
            }

            args[0] = &m[0];  args[1] = &m[1];  args[2] = &m[2];  args[3] = &m[3];
            args[4] = &m[4];  args[5] = &m[5];  args[6] = &m[6];  args[7] = &m[7];
            args[8] = &prefix_len;
            args[9] = &mode;
            args[10] = &use_le;
            args[11] = &difficulty;
            args[12] = &check_bits;
            args[13] = &dbase;
            args[14] = &cnt;
            args[15] = &resptr;

            if (use_opt) {
                blocks = (cnt + (uint32_t)PTX_BLOCK * (uint32_t)g_ptx.ilp - 1u) /
                         ((uint32_t)PTX_BLOCK * (uint32_t)g_ptx.ilp);
            } else {
                blocks = (cnt + PTX_BLOCK - 1u) / PTX_BLOCK;
            }
            if (blocks == 0) {
                blocks = 1;
            }
            if (cuLaunchKernel(use_opt ? g_ptx.fn_opt[d] : g_ptx.fn_pow[d],
                               blocks, 1, 1, PTX_BLOCK, 1, 1,
                               0, NULL, args, NULL) != CUDA_SUCCESS) {
                rc = -1;
                goto done;
            }
            launched++;
        }
        if (launched == 0) {
            break;
        }

        for (int d = 0; d < launched; d++) {
            if (cuCtxSetCurrent(g_ptx.ctx[d]) != CUDA_SUCCESS ||
                cuCtxSynchronize() != CUDA_SUCCESS ||
                cuMemcpyDtoH(&g_ptx.h_res, g_ptx.d_res[d],
                             sizeof(ptx_results)) != CUDA_SUCCESS) {
                rc = -1;
                goto done;
            }
            /* Hits are stored by whichever thread claimed a slot first, not
             * in nonce order, so pick the smallest of the eight. */
            {
                uint32_t hits = g_ptx.h_res.count;
                if (hits > PTX_RESULT_SLOTS) {
                    hits = PTX_RESULT_SLOTS;
                }
                for (uint32_t s = 0; s < hits; s++) {
                    uint32_t n = g_ptx.h_res.nonces[s];
                    if (!found || n < best_nonce) {
                        found = 1;
                        best_nonce = n;
                        memcpy(best_hash, g_ptx.h_res.hashes[s],
                               sizeof(best_hash));
                    }
                }
            }
        }

        if (found) {
            hex_from_words(best_hash, out_hex);
            *out_nonce = best_nonce;
            rc = 0;
            goto done;
        }

        base += (uint64_t)launched * (uint64_t)chunk;
        if (budget_ms > 0.0 && (ptx_now() - t0) * 1000.0 > budget_ms) {
            rc = 1;
            goto done;
        }
    }

done:
    /* Report elapsed wall time on every return path, including exhaustion, so
     * callers can time a fixed amount of work (throughput). */
    if (out_ms != NULL) {
        *out_ms = (ptx_now() - t0) * 1000.0;
    }
    for (int d = 0; d < g_ptx.ndev; d++) {
        (void)cuCtxSetCurrent(g_ptx.ctx[d]);
    }
    pthread_mutex_unlock(&g_ptx.mu);
    return rc;
}

/* ------------------------------------------------------------------ *
 * HashX
 * ------------------------------------------------------------------ */

/* Upload the program for every device. Returns 0 on success. */
int anubis_ptx_hashx_set_program(const uint64_t keys[4], const uint32_t *prog,
                                 uint32_t instructions) {
    if (keys == NULL || prog == NULL || instructions == 0 ||
        instructions > PTX_MAX_INSTR) {
        return -1;
    }
    if (!ptx_init() || !g_ptx.have_hx) {
        return -1;
    }
    pthread_mutex_lock(&g_ptx.mu);
    for (int d = 0; d < g_ptx.ndev; d++) {
        if (cuCtxSetCurrent(g_ptx.ctx[d]) != CUDA_SUCCESS ||
            cuMemcpyHtoD(g_ptx.d_prog[d], prog,
                         (size_t)instructions * 16u) != CUDA_SUCCESS) {
            pthread_mutex_unlock(&g_ptx.mu);
            return -1;
        }
    }
    memcpy(g_ptx.hx_keys, keys, sizeof(g_ptx.hx_keys));
    g_ptx.hx_code_size = instructions;
    pthread_mutex_unlock(&g_ptx.mu);
    return 0;
}

int anubis_ptx_hashx_available(void) {
    return ptx_init() && g_ptx.have_hx;
}

const char *anubis_ptx_hashx_info(void) {
    if (!anubis_ptx_hashx_available()) {
        return "PTX HashX unavailable (hashx_interp.cubin missing or no device)";
    }
    return g_ptx.info;
}

/*
 * Returns 0 on a hit (the caller re-verifies the digest), 1 if the range was
 * exhausted, -1 on backend failure.
 *
 * The kernel scans [start, max_nonce]. `difficulty` is in leading zero bits.
 */
int anubis_ptx_hashx_solve(int difficulty, uint64_t start,
                           uint64_t max_nonce,
                           uint32_t *out_nonce, char out_hex[65],
                           double *out_ms) {
    uint64_t base;
    uint64_t maxn;
    uint32_t chunk;
    double t0;
    int rc = 1;

    if (out_nonce == NULL || out_hex == NULL) {
        return -1;
    }
    if (!ptx_init() || !g_ptx.have_hx || g_ptx.hx_code_size == 0) {
        return -1;
    }
    pthread_mutex_lock(&g_ptx.mu);
    t0 = ptx_now();

    maxn = max_nonce > 0xFFFFFFFFull ? 0xFFFFFFFFull : max_nonce;
    base = start > maxn ? maxn : start;
    chunk = env_u32("ANUBIS_PTX_CHUNK", 1u << 22);
    if (chunk < 4096u) {
        chunk = 4096u;
    }
    if (chunk > (1u << 24)) {
        chunk = 1u << 24;
    }

    while (base <= maxn) {
        int launched = 0;
        int found = 0;
        uint32_t best_nonce = 0;
        uint32_t best_hash[8] = { 0 };

        for (int d = 0; d < g_ptx.ndev; d++) {
            uint64_t dev_base = base + (uint64_t)d * (uint64_t)chunk;
            uint64_t remaining;
            uint32_t cnt, dbase;
            uint64_t a_k0, a_k1, a_k2, a_k3, a_prog, a_res;
            uint32_t a_code, a_diff, a_base, a_cnt;
            unsigned blocks;
            void *args[10];

            if (dev_base > maxn) {
                break;
            }
            remaining = maxn - dev_base + 1;
            cnt = (uint32_t)(remaining < (uint64_t)chunk ? remaining
                                                         : (uint64_t)chunk);
            if (cnt == 0) {
                break;
            }
            dbase = (uint32_t)dev_base;
            a_k0 = g_ptx.hx_keys[0];
            a_k1 = g_ptx.hx_keys[1];
            a_k2 = g_ptx.hx_keys[2];
            a_k3 = g_ptx.hx_keys[3];
            a_prog = (uint64_t)g_ptx.d_prog[d];
            a_res = (uint64_t)g_ptx.d_res[d];
            a_code = g_ptx.hx_code_size;
            a_diff = (uint32_t)difficulty;
            a_base = dbase;
            a_cnt = cnt;

            if (cuCtxSetCurrent(g_ptx.ctx[d]) != CUDA_SUCCESS ||
                cuMemsetD8(g_ptx.d_res[d], 0, sizeof(ptx_results)) != CUDA_SUCCESS) {
                rc = -1;
                goto hx_done;
            }

            args[0] = &a_k0;
            args[1] = &a_k1;
            args[2] = &a_k2;
            args[3] = &a_k3;
            args[4] = &a_prog;
            args[5] = &a_code;
            args[6] = &a_diff;
            args[7] = &a_base;
            args[8] = &a_cnt;
            args[9] = &a_res;

            blocks = (cnt + PTX_BLOCK - 1u) / PTX_BLOCK;
            if (blocks == 0) {
                blocks = 1;
            }
            if (cuLaunchKernel(g_ptx.fn_hx[d], blocks, 1, 1, PTX_BLOCK, 1, 1,
                               0, NULL, args, NULL) != CUDA_SUCCESS) {
                rc = -1;
                goto hx_done;
            }
            launched++;
        }
        if (launched == 0) {
            break;
        }

        for (int d = 0; d < launched; d++) {
            if (cuCtxSetCurrent(g_ptx.ctx[d]) != CUDA_SUCCESS ||
                cuCtxSynchronize() != CUDA_SUCCESS ||
                cuMemcpyDtoH(&g_ptx.h_res, g_ptx.d_res[d],
                             sizeof(ptx_results)) != CUDA_SUCCESS) {
                rc = -1;
                goto hx_done;
            }
            /* Hits are stored by whichever thread claimed a slot first, not
             * in nonce order, so pick the smallest of the eight. */
            {
                uint32_t hits = g_ptx.h_res.count;
                if (hits > PTX_RESULT_SLOTS) {
                    hits = PTX_RESULT_SLOTS;
                }
                for (uint32_t s = 0; s < hits; s++) {
                    uint32_t n = g_ptx.h_res.nonces[s];
                    if (!found || n < best_nonce) {
                        found = 1;
                        best_nonce = n;
                        memcpy(best_hash, g_ptx.h_res.hashes[s],
                               sizeof(best_hash));
                    }
                }
            }
        }

        if (found) {
            hex_from_words(best_hash, out_hex);
            *out_nonce = best_nonce;
            rc = 0;
            goto hx_done;
        }

        base += (uint64_t)launched * (uint64_t)chunk;
    }
    rc = 1;

hx_done:
    if (out_ms != NULL) {
        *out_ms = (ptx_now() - t0) * 1000.0;
    }
    for (int d = 0; d < g_ptx.ndev; d++) {
        (void)cuCtxSetCurrent(g_ptx.ctx[d]);
    }
    pthread_mutex_unlock(&g_ptx.mu);
    return rc;
}
