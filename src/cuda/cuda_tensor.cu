/*
 * cuda_tensor.cu — CUDA backend for the `tensor` tier.
 *
 * The proof is one integer GEMM, A (uint8, 0..127) times B (int8) accumulated
 * exactly in int32, so this backend is a small tiled GEMM plus the SHA-256
 * counter-mode expansion that materialises A and B. There is no search: the
 * digest of the result is the answer, which is why the whole thing is two
 * kernels and one copy back.
 *
 *   expand:  SHA256(seed || u32le(block)) for block = 0, 1, ..., concatenated:
 *            the first N*N bytes (masked to 0..127) are A, the next N*N are B.
 *   gemm:    16x16 tiles, 16-deep k, B transposed in shared so that both
 *            operands are read four bytes at a time and folded with one
 *            __dp4a (4 multiply-adds per instruction, sm_61+). Without
 *            __dp4a the same tile runs a scalar int32 path, which is the
 *            same arithmetic in the same order - the digest must not depend
 *            on which instruction the compiler chose.
 *
 * Every device that cudaGetDeviceCount reports is used: the rows of C are
 * split across them and the copies overlap, because a single T4 finishes a
 * 512x512x512 product in well under a millisecond and the launch dominates.
 *
 * Environment:
 *   ANUBIS_NO_CUDA=1        disable this backend entirely
 *   ANUBIS_CUDA_TENSOR_DEV  use at most this many devices (default: all)
 *
 * The C core re-verifies the returned digest against the CPU reference unless
 * GPU verification was disabled (anubis_set_gpu_verify(0)), exactly like the
 * other backends: a wrong GEMM here must fail closed, not mint a challenge.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_TENSOR_MAX_DEV 8
#define TILE 16

/* ==================================================================== */
/* device code                                                          */
/* ==================================================================== */

__constant__ uint32_t t_k[64] = {
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
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__device__ static uint32_t t_rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

__device__ static uint32_t t_ld_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* One 36-byte message, SHA256(seed || u32le(counter)): exactly one padded
 * block, so this is the compression without a message schedule array. */
__device__ static void sha256_36(const uint8_t seed[32], uint32_t counter,
                                 uint8_t out[32]) {
    uint8_t blk[64];
    uint32_t w[16];
    uint32_t a, b, c, d, e, f, g, h;

    for (int i = 0; i < 32; i++) {
        blk[i] = seed[i];
    }
    blk[32] = (uint8_t)counter;
    blk[33] = (uint8_t)(counter >> 8);
    blk[34] = (uint8_t)(counter >> 16);
    blk[35] = (uint8_t)(counter >> 24);
    blk[36] = 0x80;
    for (int i = 37; i < 62; i++) {
        blk[i] = 0;
    }
    blk[62] = 0x01; /* 36 bytes = 288 bits */
    blk[63] = 0x20;

    for (int i = 0; i < 16; i++) {
        w[i] = t_ld_be(blk + 4 * i);
    }

    a = 0x6a09e667u; b = 0xbb67ae85u; c = 0x3c6ef372u; d = 0xa54ff53au;
    e = 0x510e527fu; f = 0x9b05688cu; g = 0x1f83d9abu; h = 0x5be0cd19u;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            int j = i & 15;
            uint32_t x = w[(i + 1) & 15];
            uint32_t y = w[(i + 14) & 15];
            uint32_t s0 = t_rotr(x, 7) ^ t_rotr(x, 18) ^ (x >> 3);
            uint32_t s1 = t_rotr(y, 17) ^ t_rotr(y, 19) ^ (y >> 10);
            w[j] = w[j] + s0 + w[(i + 9) & 15] + s1;
        }
        uint32_t s1 = t_rotr(e, 6) ^ t_rotr(e, 11) ^ t_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + t_k[i] + w[i & 15];
        uint32_t s0 = t_rotr(a, 2) ^ t_rotr(a, 13) ^ t_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    a += 0x6a09e667u; b += 0xbb67ae85u; c += 0x3c6ef372u; d += 0xa54ff53au;
    e += 0x510e527fu; f += 0x9b05688cu; g += 0x1f83d9abu; h += 0x5be0cd19u;

    uint32_t o[8] = {a, b, c, d, e, f, g, h};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[4 * i + 0] = (uint8_t)(o[i] >> 24);
        out[4 * i + 1] = (uint8_t)(o[i] >> 16);
        out[4 * i + 2] = (uint8_t)(o[i] >> 8);
        out[4 * i + 3] = (uint8_t)o[i];
    }
}

typedef struct { uint8_t b[32]; } tensor_seed;

/* One thread per 32-byte block of the stream; blocks are independent. The
 * seed goes in as a kernel argument (constant bank), not a global buffer. */
__global__ void expand_kernel(tensor_seed seed, uint8_t *A, int8_t *B,
                              uint32_t cells) {
    uint32_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nblocks = (2u * cells + 31u) / 32u;
    if (blk >= nblocks) {
        return;
    }
    uint8_t d[32];
    sha256_36(seed.b, blk, d);
    for (uint32_t i = 0; i < 32; i++) {
        uint64_t pos = (uint64_t)blk * 32u + i;
        if (pos < cells) {
            A[pos] = (uint8_t)(d[i] & 0x7f);
        } else if (pos < 2u * (uint64_t)cells) {
            B[pos - cells] = (int8_t)d[i];
        }
    }
}

/* C[row..row+TILE) x C[col..col+TILE), one output per thread, shared tiles
 * with a 4-byte-aligned stride (TILE + 4) so the __dp4a operands can be read
 * as ints without any packing in the inner loop. */
__global__ void gemm_kernel(const uint8_t *__restrict__ A,
                                   const int8_t *__restrict__ B,
                                   int32_t *__restrict__ C, int n, int row0) {
    __shared__ __align__(16) uint8_t As[TILE][TILE + 4];
    __shared__ __align__(16) int8_t Bs[TILE][TILE + 4];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = row0 + blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    int32_t sum = 0;

    for (int k0 = 0; k0 < n; k0 += TILE) {
        As[ty][tx] = A[(size_t)row * n + k0 + tx];
        /* transposed on the way in: Bs[col][k] = B[k][col] */
        Bs[tx][ty] = B[(size_t)(k0 + ty) * n + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k += 4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
            /* A bytes are 0..127, so the signed interpretation of the packed
             * word is the same as the unsigned one and __dp4a (signed) is
             * exact. */
            int pa = *(const int *)(const void *)&As[ty][k];
            int pb = *(const int *)(const void *)&Bs[tx][k];
            sum = __dp4a(pa, pb, sum);
#else
            sum += (int32_t)As[ty][k + 0] * (int32_t)Bs[tx][k + 0];
            sum += (int32_t)As[ty][k + 1] * (int32_t)Bs[tx][k + 1];
            sum += (int32_t)As[ty][k + 2] * (int32_t)Bs[tx][k + 2];
            sum += (int32_t)As[ty][k + 3] * (int32_t)Bs[tx][k + 3];
#endif
        }
        __syncthreads();
    }

    C[(size_t)row * n + col] = sum;
}

/* ==================================================================== */
/* host side                                                            */
/* ==================================================================== */

static struct {
    int inited;
    int available;
    int ndev;
    char info[192];
} g_t = {0, 0, 0, {0}};

static double t_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int t_init(void) {
    int count = 0;

    if (g_t.inited) {
        return g_t.available;
    }
    g_t.inited = 1;
    {
        const char *off = getenv("ANUBIS_NO_CUDA");
        if (off != NULL && off[0] != '\0' && off[0] != '0') {
            return 0;
        }
    }
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return 0;
    }
    {
        const char *cap = getenv("ANUBIS_CUDA_TENSOR_DEV");
        if (cap != NULL && cap[0] != '\0') {
            int want = atoi(cap);
            if (want > 0 && want < count) {
                count = want;
            }
        }
    }
    if (count > CUDA_TENSOR_MAX_DEV) {
        count = CUDA_TENSOR_MAX_DEV;
    }
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaSetDevice(i) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            return 0;
        }
        if (count == 1) {
            snprintf(g_t.info, sizeof(g_t.info), "CUDA: %.96s", prop.name);
        } else {
            snprintf(g_t.info, sizeof(g_t.info), "CUDA: %.80s x%d",
                     prop.name, count);
        }
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return 0;
    }
    g_t.ndev = count;
    g_t.available = 1;
    return 1;
}

extern "C" {

int anubis_cuda_tensor_available(void) {
    return t_init();
}

const char *anubis_cuda_tensor_info(void) {
    if (!t_init()) {
        return NULL;
    }
    return g_t.info;
}

/*
 * C = A*B for the tensor tier, into `c_host` (n*n int32, row-major).
 * Returns 0 on success, -1 on backend failure (the caller then runs the CPU
 * path). The digest is the caller's job: it has a SHA-256 already.
 */
int anubis_cuda_tensor_c(const uint8_t seed[32], int n, int32_t *c_host,
                         double *out_ms) {
    uint8_t *dA[CUDA_TENSOR_MAX_DEV] = {0};
    int8_t *dB[CUDA_TENSOR_MAX_DEV] = {0};
    int32_t *dC[CUDA_TENSOR_MAX_DEV] = {0};
    cudaStream_t streams[CUDA_TENSOR_MAX_DEV] = {0};
    tensor_seed sd;
    size_t cells;
    int ndev, rows_per, rc = -1;
    double t0;

    if (!t_init() || seed == NULL || c_host == NULL || n <= 0) {
        return -1;
    }
    if (n % TILE != 0) {
        return -1; /* the difficulty rule only produces multiples of 64 */
    }
    cells = (size_t)n * (size_t)n;
    ndev = g_t.ndev;
    rows_per = n / ndev;
    rows_per -= rows_per % TILE;
    if (rows_per <= 0) {
        ndev = 1;
        rows_per = n;
    }
    memcpy(sd.b, seed, 32);

    t0 = t_now();
    for (int d = 0; d < ndev; d++) {
        cudaSetDevice(d);
        if (cudaMalloc((void **)&dA[d], cells) != cudaSuccess ||
            cudaMalloc((void **)&dB[d], cells) != cudaSuccess ||
            cudaMalloc((void **)&dC[d], cells * sizeof(int32_t)) != cudaSuccess ||
            cudaStreamCreate(&streams[d]) != cudaSuccess) {
            goto out;
        }
    }

    for (int d = 0; d < ndev; d++) {
        int row0 = d * rows_per;
        int rows = (d == ndev - 1) ? (n - row0) : rows_per;
        uint32_t nb = (uint32_t)((2 * cells + 31) / 32);
        cudaSetDevice(d);
        expand_kernel<<<(nb + 255) / 256, 256, 0, streams[d]>>>(sd, dA[d], dB[d],
                                                               (uint32_t)cells);
        gemm_kernel<<<dim3(n / TILE, rows / TILE), dim3(TILE, TILE), 0,
                      streams[d]>>>(dA[d], dB[d], dC[d], n, row0);
    }

    for (int d = 0; d < ndev; d++) {
        int row0 = d * rows_per;
        int rows = (d == ndev - 1) ? (n - row0) : rows_per;
        cudaSetDevice(d);
        if (cudaMemcpyAsync(c_host + (size_t)row0 * n, dC[d],
                            (size_t)rows * (size_t)n * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, streams[d]) != cudaSuccess) {
            goto out;
        }
        if (cudaStreamSynchronize(streams[d]) != cudaSuccess) {
            goto out;
        }
    }

    rc = 0;

out:
    for (int d = 0; d < ndev; d++) {
        cudaSetDevice(d);
        if (streams[d] != NULL) {
            cudaStreamDestroy(streams[d]);
        }
        if (dA[d] != NULL) cudaFree(dA[d]);
        if (dB[d] != NULL) cudaFree(dB[d]);
        if (dC[d] != NULL) cudaFree(dC[d]);
    }
    cudaSetDevice(0);
    if (rc == 0 && out_ms != NULL) {
        *out_ms = (t_now() - t0) * 1000.0;
    }
    return rc;
}

} /* extern "C" */
