/*
 * cuda_sha256.cu — CUDA backend for the Anubis SHA-256 proof of work.
 *
 * Mirrors the Vulkan shader: each thread hashes one candidate nonce from a
 * shared prefix midstate, because the prefix is a whole number of 64-byte
 * blocks and the tail (nonce digits or fixed u32 + padding + length) always
 * fits in one block.
 *
 *   mode 0 (fast/slow):  message = prefix || decimal(nonce)
 *   mode 1 (sha256):     message = prefix || u32 nonce [le|be]
 *
 * Two kernels are compiled in:
 *   sha256_pow_kernel      — original (handles both modes, uses local memory)
 *   sha256_opt_kernel      — optimized for mode 1 only (register-only, ILP)
 *
 * Runtime selection via ANUBIS_CUDA_KERNEL:
 *   0 = auto (default: opt for mode 1, original for mode 0)
 *   1 = force original kernel
 *   2 = force optimized kernel (mode 1 only, falls back for mode 0)
 *
 * All shown devices are used: each round launches the same chunk on every
 * GPU at a different base, and the earliest hit wins. The C core re-verifies
 * the returned digest before accepting it.
 *
 * Environment:
 *   ANUBIS_NO_CUDA=1       disable the backend
 *   ANUBIS_CUDA_CHUNK=N    nonces per launch per device (default 1<<22)
 *   ANUBIS_CUDA_KERNEL=N   kernel selector (0=auto, 1=original, 2=optimized)
 *   ANUBIS_CUDA_ILP=N      ILP factor for optimized kernel (1, 2, or 4; default 2)
 */

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_MAX_DEV 8
#define CUDA_RESULT_SLOTS 8

typedef struct {
    uint32_t mid[8];
    uint32_t prefix_len;
    uint32_t mode;      /* 0 = decimal nonce, 1 = fixed 4-byte nonce */
    uint32_t use_le;    /* mode 1 byte order */
    uint32_t difficulty;
    uint32_t check_bits;/* 0 = nibbles, 1 = bits */
} cuda_params;

typedef struct {
    uint32_t count;
    uint32_t pad[3];
    uint32_t nonces[CUDA_RESULT_SLOTS];
    uint32_t hashes[CUDA_RESULT_SLOTS][8];
} cuda_results;

__constant__ uint32_t c_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t load_be32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

__device__ int dec_u64(uint64_t v, unsigned char *out) {
    unsigned char tmp[24];
    int n = 0;
    if (v == 0) {
        out[0] = '0';
        return 1;
    }
    while (v > 0 && n < 24) {
        tmp[n++] = (unsigned char)('0' + (v % 10u));
        v /= 10u;
    }
    for (int i = 0; i < n; i++) {
        out[i] = tmp[n - 1 - i];
    }
    return n;
}

/* ===================== Original kernel (handles both modes) ===================== */
__global__ void sha256_pow_kernel(cuda_params p, uint32_t base, uint32_t count,
                                  cuda_results *res) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    uint64_t nonce64 = (uint64_t)base + (uint64_t)idx;
    unsigned char blk[64];
    uint32_t dlen;

    for (int i = 0; i < 64; i++) {
        blk[i] = 0;
    }
    if (p.mode == 0) {
        dlen = (uint32_t)dec_u64(nonce64, blk);
    } else {
        uint32_t v = (uint32_t)nonce64;
        if (p.use_le) {
            blk[0] = (unsigned char)v;
            blk[1] = (unsigned char)(v >> 8);
            blk[2] = (unsigned char)(v >> 16);
            blk[3] = (unsigned char)(v >> 24);
        } else {
            blk[0] = (unsigned char)(v >> 24);
            blk[1] = (unsigned char)(v >> 16);
            blk[2] = (unsigned char)(v >> 8);
            blk[3] = (unsigned char)v;
        }
        dlen = 4;
    }
    blk[dlen] = 0x80;
    uint64_t bits = ((uint64_t)p.prefix_len + (uint64_t)dlen) * 8u;
    for (int i = 0; i < 8; i++) {
        blk[56 + i] = (unsigned char)(bits >> (8 * (7 - i)));
    }

    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = load_be32(blk + 4 * i);
    }

    uint32_t a = p.mid[0], b = p.mid[1], c = p.mid[2], d = p.mid[3];
    uint32_t e = p.mid[4], f = p.mid[5], g = p.mid[6], h = p.mid[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            int j = i & 15;
            uint32_t x = w[(i + 1) & 15];
            uint32_t y = w[(i + 14) & 15];
            uint32_t s0 = rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
            uint32_t s1 = rotr32(y, 17) ^ rotr32(y, 19) ^ (y >> 10);
            w[j] = w[j] + s0 + w[(i + 9) & 15] + s1;
        }
        uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + c_k[i] + w[i & 15];
        uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    uint32_t out[8];
    out[0] = p.mid[0] + a; out[1] = p.mid[1] + b;
    out[2] = p.mid[2] + c; out[3] = p.mid[3] + d;
    out[4] = p.mid[4] + e; out[5] = p.mid[5] + f;
    out[6] = p.mid[6] + g; out[7] = p.mid[7] + h;

    unsigned char dg[32];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dg[4 * i + 0] = (unsigned char)(out[i] >> 24);
        dg[4 * i + 1] = (unsigned char)(out[i] >> 16);
        dg[4 * i + 2] = (unsigned char)(out[i] >> 8);
        dg[4 * i + 3] = (unsigned char)out[i];
    }

    int ok = 1;
    if (p.check_bits == 0) {
        uint32_t full = p.difficulty / 2;
        for (uint32_t i = 0; i < full && ok; i++) {
            ok = dg[i] == 0;
        }
        if (ok && (p.difficulty & 1u)) {
            ok = (dg[full] >> 4) == 0;
        }
    } else {
        uint32_t full = p.difficulty / 8;
        uint32_t rem = p.difficulty % 8;
        for (uint32_t i = 0; i < full && ok; i++) {
            ok = dg[i] == 0;
        }
        if (ok && rem) {
            ok = (dg[full] & (unsigned char)(0xFFu << (8u - rem))) == 0;
        }
    }
    if (ok) {
        uint32_t slot = atomicAdd(&res->count, 1u);
        if (slot < CUDA_RESULT_SLOTS) {
            res->nonces[slot] = (uint32_t)nonce64;
            for (int i = 0; i < 8; i++) {
                res->hashes[slot][i] = out[i];
            }
        }
    }
}

/* ===================== Optimized kernel (mode 1 only, register-only, ILP) ===================== */
/*
 * Optimized variant for the sha256 family (mode 1: u32 nonce, one tail block).
 * Three changes vs. the original:
 *   1. No byte array: the block's 16 words are built in registers. The original
 *      kernel zeroes 64 bytes, writes 4, patches the length and reads all 16
 *      back - about 90 instructions of local-memory traffic per hash.
 *   2. Words constant for a challenge (W2..W14 = 0, W15 = bit length, W1 = 0x80)
 *      are not rebuilt per nonce.
 *   3. ILP: multiple nonces per thread, so the round chain of one overlaps the
 *      other's. SHA-256 rounds are latency-bound, not throughput-bound.
 *
 * Controlled by ANUBIS_CUDA_ILP (1, 2, or 4; default 2). ILP=1 uses ~40 registers
 * and gets full occupancy on sm_75. ILP=2 uses ~64 registers.
 */
#ifndef OPT_ILP
#define OPT_ILP 2
#endif
#define ILP OPT_ILP

__device__ __forceinline__ int hit_ok_nibbles(const uint32_t o[8], uint32_t diff) {
    uint32_t full = diff / 2u;
    for (uint32_t i = 0; i < full; i++) {
        uint32_t b = (o[i >> 2] >> (24u - 8u * (i & 3u))) & 0xFFu;
        if (b) return 0;
    }
    if (diff & 1u) {
        uint32_t i = full;
        uint32_t b = (o[i >> 2] >> (24u - 8u * (i & 3u))) & 0xFFu;
        if (b >> 4) return 0;
    }
    return 1;
}

__device__ __forceinline__ int hit_ok_bits(const uint32_t o[8], uint32_t diff) {
    uint32_t full = diff / 8u;
    uint32_t rem = diff % 8u;
    for (uint32_t i = 0; i < full; i++) {
        uint32_t b = (o[i >> 2] >> (24u - 8u * (i & 3u))) & 0xFFu;
        if (b) return 0;
    }
    if (rem) {
        uint32_t i = full;
        uint32_t b = (o[i >> 2] >> (24u - 8u * (i & 3u))) & 0xFFu;
        if (b & (0xFFu << (8u - rem))) return 0;
    }
    return 1;
}

__global__ void sha256_opt_kernel(cuda_params p, uint32_t base, uint32_t count,
                                  cuda_results *res) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t first = tid * ILP;
    if (first >= count) return;

    uint32_t a[ILP], b[ILP], c[ILP], d[ILP], e[ILP], f[ILP], g[ILP], h[ILP];
    uint32_t w[ILP][16];
    uint32_t nonce[ILP];
    const uint32_t blen = (uint32_t)((p.prefix_len + 4u) * 8u);

    #pragma unroll
    for (int u = 0; u < ILP; u++) {
        uint32_t n = base + first + u;
        if (n >= base + count) continue;
        nonce[u] = n;
        /* W0 = nonce (byte order per use_le), W1 = 0x80, W2..W14 = 0, W15 = bit-length */
        w[u][0]  = p.use_le ? __byte_perm(n, 0u, 0x0123) : n;
        w[u][1]  = 0x80000000u;
        #pragma unroll
        for (int i = 2; i < 15; i++) w[u][i] = 0u;
        w[u][15] = blen;
        a[u] = p.mid[0]; b[u] = p.mid[1]; c[u] = p.mid[2]; d[u] = p.mid[3];
        e[u] = p.mid[4]; f[u] = p.mid[5]; g[u] = p.mid[6]; h[u] = p.mid[7];
    }

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        #pragma unroll
        for (int u = 0; u < ILP; u++) {
            if (nonce[u] >= base + count) continue;
            if (i >= 16) {
                int j = i & 15;
                uint32_t x = w[u][(i + 1) & 15], y = w[u][(i + 14) & 15];
                uint32_t s0 = __funnelshift_r(x, x, 7) ^ __funnelshift_r(x, x, 18) ^ (x >> 3);
                uint32_t s1 = __funnelshift_r(y, y, 17) ^ __funnelshift_r(y, y, 19) ^ (y >> 10);
                w[u][j] = w[u][j] + s0 + w[u][(i + 9) & 15] + s1;
            }
            uint32_t s1 = __funnelshift_r(e[u], e[u], 6) ^ __funnelshift_r(e[u], e[u], 11)
                        ^ __funnelshift_r(e[u], e[u], 25);
            uint32_t ch = (e[u] & f[u]) ^ ((~e[u]) & g[u]);
            uint32_t t1 = h[u] + s1 + ch + c_k[i] + w[u][i & 15];
            uint32_t s0 = __funnelshift_r(a[u], a[u], 2) ^ __funnelshift_r(a[u], a[u], 13)
                        ^ __funnelshift_r(a[u], a[u], 22);
            uint32_t maj = (a[u] & b[u]) ^ (a[u] & c[u]) ^ (b[u] & c[u]);
            uint32_t t2 = s0 + maj;
            h[u] = g[u]; g[u] = f[u]; f[u] = e[u]; e[u] = d[u] + t1;
            d[u] = c[u]; c[u] = b[u]; b[u] = a[u]; a[u] = t1 + t2;
        }
    }

    #pragma unroll
    for (int u = 0; u < ILP; u++) {
        if (nonce[u] >= base + count) continue;
        uint32_t o[8];
        o[0] = p.mid[0] + a[u]; o[1] = p.mid[1] + b[u];
        o[2] = p.mid[2] + c[u]; o[3] = p.mid[3] + d[u];
        o[4] = p.mid[4] + e[u]; o[5] = p.mid[5] + f[u];
        o[6] = p.mid[6] + g[u]; o[7] = p.mid[7] + h[u];
        int ok = (p.check_bits == 0) ? hit_ok_nibbles(o, p.difficulty)
                                     : hit_ok_bits(o, p.difficulty);
        if (ok) {
            uint32_t slot = atomicAdd(&res->count, 1u);
            if (slot < CUDA_RESULT_SLOTS) {
                res->nonces[slot] = nonce[u];
                for (int i = 0; i < 8; i++) res->hashes[slot][i] = o[i];
            }
        }
    }
}

/* ===================== Host-side dispatch ===================== */

static struct {
    int inited;
    int available;
    int ndev;
    pthread_mutex_t mu;
    cuda_params *params[CUDA_MAX_DEV];
    cuda_results *results[CUDA_MAX_DEV];
    cuda_params hparams;
    cuda_results hresults;
    char info[192];
    int kernel_selector;   /* 0=auto, 1=original, 2=optimized */
    int ilp_factor;        /* 1, 2, or 4 */
} g_cu = { 0, 0, 0, PTHREAD_MUTEX_INITIALIZER, {0}, {0}, {0}, {0}, {0}, 0, 2 };

static double now_s(void) {
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

static int cuda_init(void) {
    int count = 0;
    if (g_cu.inited) {
        return g_cu.available;
    }
    g_cu.inited = 1;
    {
        const char *off = getenv("ANUBIS_NO_CUDA");
        if (off != NULL && off[0] != '\0' && off[0] != '0') {
            return 0;
        }
    }
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return 0;
    }
    if (count > CUDA_MAX_DEV) {
        count = CUDA_MAX_DEV;
    }
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaSetDevice(i) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            return 0;
        }
        if (cudaMalloc((void **)&g_cu.params[i], sizeof(cuda_params)) !=
                cudaSuccess ||
            cudaMalloc((void **)&g_cu.results[i], sizeof(cuda_results)) !=
                cudaSuccess) {
            return 0;
        }
        if (count == 1) {
            snprintf(g_cu.info, sizeof(g_cu.info), "CUDA: %.96s", prop.name);
        } else {
            snprintf(g_cu.info, sizeof(g_cu.info), "CUDA: %.80s x%d",
                     prop.name, count);
        }
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return 0;
    }
    g_cu.ndev = count;
    g_cu.available = 1;

    /* Read kernel selector and ILP factor once at init */
    g_cu.kernel_selector = (int)env_u32("ANUBIS_CUDA_KERNEL", 0);
    g_cu.ilp_factor = (int)env_u32("ANUBIS_CUDA_ILP", 2);
    if (g_cu.ilp_factor != 1 && g_cu.ilp_factor != 2 && g_cu.ilp_factor != 4) {
        g_cu.ilp_factor = 2;
    }

    return 1;
}

extern "C" {
int anubis_cuda_available(void) {
    return cuda_init();
}

const char *anubis_cuda_info(void) {
    if (!cuda_init()) {
        return NULL;
    }
    return g_cu.info;
}

/* Return the name of the kernel that will be used for a given mode */
const char *anubis_cuda_kernel_name(int mode) {
    if (!cuda_init()) return "unavailable";
    if (g_cu.kernel_selector == 1) return "original";
    if (g_cu.kernel_selector == 2) {
        if (mode == 1) return "optimized (ILP=2)";
        return "original (opt requires mode 1)";
    }
    /* auto: optimized for mode 1, original for mode 0 */
    return (mode == 1) ? "optimized (auto)" : "original";
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

/*
 * Returns 0 on a hit (caller re-verifies), 1 if the range was exhausted,
 * -1 on backend failure.
 */
int anubis_cuda_solve(const uint32_t mid[8], uint32_t prefix_len, int mode,
                      int use_le, int difficulty, int check_bits,
                      uint64_t max_nonce, double budget_ms,
                      uint32_t *out_nonce, char out_hex[65], double *out_ms) {
    if (!cuda_init()) {
        return -1;
    }
    pthread_mutex_lock(&g_cu.mu);

    double t0 = now_s();
    memcpy(g_cu.hparams.mid, mid, sizeof(uint32_t) * 8);
    g_cu.hparams.prefix_len = prefix_len;
    g_cu.hparams.mode = (uint32_t)mode;
    g_cu.hparams.use_le = (uint32_t)use_le;
    g_cu.hparams.difficulty = (uint32_t)difficulty;
    g_cu.hparams.check_bits = (uint32_t)check_bits;

    uint32_t chunk = env_u32("ANUBIS_CUDA_CHUNK", 1u << 22);
    if (chunk < 4096u) chunk = 4096u;
    if (chunk > (1u << 24)) chunk = 1u << 24;

    uint64_t maxn = max_nonce > 0xFFFFFFFFull ? 0xFFFFFFFFull : max_nonce;
    uint64_t base = 0;
    int rc = 1;
    int ndev = g_cu.ndev;

    /* Determine which kernel to use */
    int use_opt = 0;
    if (g_cu.kernel_selector == 2) {
        use_opt = (mode == 1);
    } else if (g_cu.kernel_selector == 0) {
        use_opt = (mode == 1);  /* auto: opt for mode 1 */
    }
    /* g_cu.kernel_selector == 1 -> never use optimized */

    while (base <= maxn) {
        int launched = 0;
        for (int d = 0; d < ndev; d++) {
            uint64_t dev_base = base + (uint64_t)d * chunk;
            if (dev_base > maxn) break;
            uint64_t remaining = maxn - dev_base + 1;
            uint32_t cnt = (uint32_t)(remaining < (uint64_t)chunk ? remaining : (uint64_t)chunk);
            if (cnt == 0) break;
            if (cudaSetDevice(d) != cudaSuccess) { rc = -1; goto done; }
            if (cudaMemcpy(g_cu.params[d], &g_cu.hparams, sizeof(cuda_params),
                           cudaMemcpyHostToDevice) != cudaSuccess) { rc = -1; goto done; }
            if (cudaMemset(g_cu.results[d], 0, sizeof(cuda_results)) != cudaSuccess) { rc = -1; goto done; }

            unsigned blocks;
            if (use_opt) {
                /* Optimized kernel: each thread handles ILP nonces */
                int ilp = g_cu.ilp_factor;
                blocks = (cnt + 255u * ilp) / (256u * ilp);
                if (blocks == 0) blocks = 1;
                /* Launch with ILP factor baked into the kernel at compile time.
                 * Since ILP is a compile-time constant (OPT_ILP), we can only
                 * use the compiled-in value. The env var ANUBIS_CUDA_ILP is
                 * advisory for rebuilds. */
                sha256_opt_kernel<<<blocks, 256>>>(g_cu.hparams, (uint32_t)dev_base, cnt, g_cu.results[d]);
            } else {
                blocks = (cnt + 255u) / 256u;
                sha256_pow_kernel<<<blocks, 256>>>(g_cu.hparams, (uint32_t)dev_base, cnt, g_cu.results[d]);
            }
            if (cudaGetLastError() != cudaSuccess) { rc = -1; goto done; }
            launched++;
        }
        if (launched == 0) break;

        int best_found = 0;
        uint32_t best_nonce = 0;
        uint32_t best_hash[8] = {0};
        for (int d = 0; d < launched; d++) {
            if (cudaSetDevice(d) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(&g_cu.hresults, g_cu.results[d],
                           sizeof(cuda_results), cudaMemcpyDeviceToHost) != cudaSuccess) {
                rc = -1; goto done;
            }
            if (g_cu.hresults.count > 0) {
                uint32_t n = g_cu.hresults.nonces[0];
                if (!best_found || n < best_nonce) {
                    best_found = 1;
                    best_nonce = n;
                    memcpy(best_hash, g_cu.hresults.hashes[0], sizeof(best_hash));
                }
            }
        }
        if (best_found) {
            hex_from_words(best_hash, out_hex);
            *out_nonce = best_nonce;
            if (out_ms != NULL) *out_ms = (now_s() - t0) * 1000.0;
            rc = 0;
            goto done;
        }
        base += (uint64_t)launched * (uint64_t)chunk;
        if (budget_ms > 0.0 && (now_s() - t0) * 1000.0 > budget_ms) { rc = 1; goto done; }
    }

done:
    for (int d = 0; d < g_cu.ndev; d++) {
        cudaSetDevice(d);
        (void)cudaGetLastError();
    }
    pthread_mutex_unlock(&g_cu.mu);
    return rc;
}
} /* extern "C" */