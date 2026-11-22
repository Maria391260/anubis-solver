/*
 * tensor_pjrt.c — the `tensor` tier on a TPU, through the PJRT C API, no Python.
 *
 * The tier's proof is one integer GEMM (A uint8 masked to 0..127 times B int8,
 * accumulated exactly in int32) and the digest of the result. On a TPU that is
 * one s8 x s8 dot with an s32 result, which in HLO is spelled by the
 * instruction's shape: the result element type *is* the preferred element type.
 *
 * This program does the whole loop itself:
 *
 *   1. dlopen a PJRT plugin (libtpu.so), GetPjrtApi(), PJRT_Plugin_Initialize,
 *      create a client, take device 0. That order is the documented lifecycle;
 *      libtpu aborts with "InitGoogle() has not finished yet" if the plugin is
 *      not initialized first, and refuses a second process on the same chip.
 *   2. Derive A and B on the host with the same SHA-256 counter-mode XOF the C
 *      engine and the browser use.
 *   3. Compile one StableHLO module: stablehlo.dot_general on tensor<NxNxi8>
 *      pairs to tensor<NxNxi32>. CompileOptionsProto carries num_replicas=1 /
 *      num_partitions=1, because an empty options message means zero replicas
 *      and the TPU backend will not invent a device assignment for that.
 *   4. Upload A and B as S8 buffers, execute, copy C back as S32, hash it.
 *   5. Compare the digest with a plain-C GEMM (the same reference the engine
 *      uses) unless --no-verify, then time execute-only and end-to-end.
 *
 * Build:  cc -O2 -I. -o tensor_pjrt tensor_pjrt.c -ldl
 * Run:    ./tensor_pjrt --lib /usr/local/lib/.../libtpu/libtpu.so --difficulty 8
 *
 * The digest this prints is the same string the gate compares; if it differs
 * from the CPU reference, the program says so and exits non-zero. A TPU that
 * miscomputes one element must not be reported as fast.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pjrt_c_api.h"

/* ==================================================================== */
/* SHA-256 (host): the XOF that materialises A and B, and the digest     */
/* ==================================================================== */

typedef struct {
    uint32_t h[8];
    uint64_t total;
    uint8_t buf[64];
    size_t used;
} sha256_ctx;

static const uint32_t K256[64] = {
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

static uint32_t rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, hh;
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
               ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

static void sha256_init(sha256_ctx *c) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u};
    memcpy(c->h, iv, sizeof iv);
    c->total = 0;
    c->used = 0;
}

static void sha256_update(sha256_ctx *c, const void *data, size_t len) {
    const uint8_t *p = data;
    c->total += len;
    while (len > 0) {
        size_t take = 64 - c->used;
        if (take > len) {
            take = len;
        }
        memcpy(c->buf + c->used, p, take);
        c->used += take;
        p += take;
        len -= take;
        if (c->used == 64) {
            sha256_block(c->h, c->buf);
            c->used = 0;
        }
    }
}

static void sha256_final(sha256_ctx *c, uint8_t out[32]) {
    uint64_t bits = c->total * 8u;
    uint8_t pad[72];
    size_t padlen;
    c->buf[c->used++] = 0x80;
    if (c->used > 56) {
        padlen = 64 - c->used + 56;
    } else {
        padlen = 56 - c->used;
    }
    memset(pad, 0, sizeof pad);
    sha256_update(c, pad, padlen);
    for (int i = 0; i < 8; i++) {
        pad[i] = (uint8_t)(bits >> (8 * (7 - i)));
    }
    sha256_update(c, pad, 8);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(c->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(c->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(c->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)c->h[i];
    }
}

static void sha256_buf(const void *data, size_t len, uint8_t out[32]) {
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, data, len);
    sha256_final(&c, out);
}

/* ==================================================================== */
/* small helpers                                                        */
/* ==================================================================== */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t hex_decode(const char *hex, uint8_t *out, size_t cap) {
    size_t n = strlen(hex) / 2, i;
    if (n > cap) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        int hi = hexval(hex[2 * i]), lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return 0;
        }
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return n;
}

static void to_hex(const uint8_t *in, size_t len, char *out) {
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = d[in[i] >> 4];
        out[2 * i + 1] = d[in[i] & 15];
    }
    out[2 * len] = '\0';
}

static int tensor_dim(int difficulty) {
    int d = difficulty;
    if (d < 2) d = 2;
    if (d > 16) d = 16;
    return 64 * d;
}

/* A and B, exactly as anubis_tensor_digest builds them. */
static void tensor_expand(const char *challenge_hex, int n, uint8_t *a, int8_t *b) {
    size_t cells = (size_t)n * (size_t)n;
    uint8_t seed[32], block[32];
    uint8_t raw[4096];
    size_t raw_len = hex_decode(challenge_hex, raw, sizeof raw);
    sha256_ctx base, c;
    uint32_t counter = 0;
    size_t pos = 0;

    sha256_buf(raw, raw_len, seed);
    sha256_init(&base);
    sha256_update(&base, seed, sizeof seed);

    while (pos < 2 * cells) {
        uint8_t ctr[4] = {(uint8_t)counter, (uint8_t)(counter >> 8),
                          (uint8_t)(counter >> 16), (uint8_t)(counter >> 24)};
        c = base;
        sha256_update(&c, ctr, sizeof ctr);
        sha256_final(&c, block);
        for (size_t i = 0; i < sizeof block && pos < 2 * cells; i++, pos++) {
            if (pos < cells) {
                a[pos] = (uint8_t)(block[i] & 0x7f);
            } else {
                b[pos - cells] = (int8_t)block[i];
            }
        }
        counter++;
    }
}

/* The reference the gate uses: i-k-j, int32, no overflow possible. */
static void cpu_gemm(const uint8_t *a, const int8_t *b, int32_t *c, int n) {
    for (int i = 0; i < n; i++) {
        int32_t *row = c + (size_t)i * n;
        const uint8_t *arow = a + (size_t)i * n;
        memset(row, 0, (size_t)n * sizeof *row);
        for (int k = 0; k < n; k++) {
            int32_t av = arow[k];
            const int8_t *brow = b + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                row[j] += av * (int32_t)brow[j];
            }
        }
    }
}

static void digest_of(const int32_t *c, int n, char out[65]) {
    size_t cells = (size_t)n * (size_t)n;
    uint8_t *le = malloc(cells * 4);
    uint8_t d[32];
    for (size_t i = 0; i < cells; i++) {
        uint32_t v = (uint32_t)c[i];
        le[4 * i] = (uint8_t)v;
        le[4 * i + 1] = (uint8_t)(v >> 8);
        le[4 * i + 2] = (uint8_t)(v >> 16);
        le[4 * i + 3] = (uint8_t)(v >> 24);
    }
    sha256_buf(le, cells * 4, d);
    to_hex(d, sizeof d, out);
    free(le);
}

/* tensor-v: C[i][j] = sum_k A[i][k] * B[j][k] */
static void cpu_gemm_tv(const uint8_t *a, const int8_t *b, int32_t *c, int n) {
    for (int i = 0; i < n; i++) {
        int32_t *row = c + (size_t)i * n;
        const uint8_t *arow = a + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const int8_t *brow = b + (size_t)j * n;
            int32_t sum = 0;
            for (int k = 0; k < n; k++) {
                sum += (int32_t)arow[k] * (int32_t)brow[k];
            }
            row[j] = sum;
        }
    }
}

static const uint8_t tv_dom_leaf[16] = {
    'A', 'N', 'U', 'B', 'I', 'S', '-', 'T',
    'E', 'N', 'S', 'O', 'R', '-', 'V', 'L'};
static const uint8_t tv_dom_node[16] = {
    'A', 'N', 'U', 'B', 'I', 'S', '-', 'T',
    'E', 'N', 'S', 'O', 'R', '-', 'V', 'N'};
static const uint8_t tv_dom_chal[16] = {
    'A', 'N', 'U', 'B', 'I', 'S', '-', 'T',
    'E', 'N', 'S', 'O', 'R', '-', 'V', 'C'};

static void tv_put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/* The client's commitment: one leaf per 32 int32 of C, then a Merkle tree with
 * the odd last node duplicated. Same bytes as anubis_tensor_v_root(). */
static void tv_root_of(const int32_t *c, int n, const char *challenge_hex,
                       char out[65]) {
    size_t cells = (size_t)n * (size_t)n;
    size_t leaves = cells / 32;
    uint8_t raw[4096];
    size_t raw_len = hex_decode(challenge_hex, raw, sizeof raw);
    uint8_t chal[32];
    uint8_t *level;
    sha256_ctx s;

    sha256_init(&s);
    sha256_update(&s, tv_dom_chal, sizeof tv_dom_chal);
    sha256_update(&s, raw, raw_len);
    sha256_final(&s, chal);

    level = malloc(leaves * 32);
    if (level == NULL) {
        out[0] = '\0';
        return;
    }
    for (size_t b = 0; b < leaves; b++) {
        uint8_t idx[4], bytes[32 * 4];
        for (int i = 0; i < 32; i++) {
            uint32_t v = (uint32_t)c[b * 32 + i];
            bytes[4 * i + 0] = (uint8_t)v;
            bytes[4 * i + 1] = (uint8_t)(v >> 8);
            bytes[4 * i + 2] = (uint8_t)(v >> 16);
            bytes[4 * i + 3] = (uint8_t)(v >> 24);
        }
        tv_put_u32(idx, (uint32_t)b);
        sha256_init(&s);
        sha256_update(&s, tv_dom_leaf, sizeof tv_dom_leaf);
        sha256_update(&s, chal, sizeof chal);
        sha256_update(&s, idx, sizeof idx);
        sha256_update(&s, bytes, sizeof bytes);
        sha256_final(&s, level + b * 32);
    }
    {
        size_t cnt = leaves;
        while (cnt > 1) {
            size_t next = (cnt + 1) / 2;
            for (size_t i = 0; i < next; i++) {
                const uint8_t *l = level + 32 * (2 * i);
                const uint8_t *r = level + 32 * ((2 * i + 1 < cnt) ? 2 * i + 1 : 2 * i);
                sha256_init(&s);
                sha256_update(&s, tv_dom_node, sizeof tv_dom_node);
                sha256_update(&s, l, 32);
                sha256_update(&s, r, 32);
                sha256_final(&s, level + 32 * i);
            }
            cnt = next;
        }
    }
    to_hex(level, 32, out);
    free(level);
}

/* ==================================================================== */
/* PJRT                                                                 */
/* ==================================================================== */

static PJRT_Api *api;
static PJRT_Client *client;
static PJRT_Device *device;

/* -1 = not tried, 0 = plugin refused, 1 = mapped. libtpu on v5e keeps a
 * premapped region (RequiresPremappedBuffers is true for v5e) and bounces any
 * other host destination through it: one DMA plus one host memcpy. DmaMap
 * registers our own buffer so ToHostBuffer can write it directly. */
static int g_dmamap;
static int g_dma_state = -1;

static void *alloc_host(size_t bytes) {
    void *p = NULL;
    int64_t dma_off = (int64_t)offsetof(PJRT_Api, PJRT_Client_DmaMap);

    if (!g_dmamap) {
        return malloc(bytes);
    }
    if (api->PJRT_Client_DmaMap == NULL ||
        (int64_t)api->struct_size < dma_off + (int64_t)sizeof(void *)) {
        fprintf(stderr, "dmamap: plugin does not expose PJRT_Client_DmaMap\n");
        g_dma_state = 0;
        return malloc(bytes);
    }
    if (posix_memalign(&p, 2u << 20, bytes) != 0 || p == NULL) {
        return malloc(bytes);
    }
    PJRT_Client_DmaMap_Args da = {0};
    da.struct_size = sizeof da;
    da.client = client;
    da.data = p;
    da.size = bytes;
    PJRT_Error *e = api->PJRT_Client_DmaMap(&da);
    if (e != NULL) {
        PJRT_Error_Message_Args ma = {0};
        ma.struct_size = sizeof ma;
        ma.error = e;
        api->PJRT_Error_Message(&ma);
        fprintf(stderr, "dmamap: DmaMap refused: %.*s\n", (int)ma.message_size,
                ma.message);
        PJRT_Error_Destroy_Args ed = {0};
        ed.struct_size = sizeof ed;
        ed.error = e;
        api->PJRT_Error_Destroy(&ed);
        free(p);
        g_dma_state = 0;
        return malloc(bytes);
    }
    g_dma_state = 1;
    return p;
}

static void die_error(PJRT_Error *e, const char *what) {
    if (e == NULL) {
        return;
    }
    PJRT_Error_Message_Args ma = {0};
    ma.struct_size = sizeof ma;
    ma.error = e;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "error: %s: %.*s\n", what, (int)ma.message_size, ma.message);
    PJRT_Error_Destroy_Args da = {0};
    da.struct_size = sizeof da;
    da.error = e;
    api->PJRT_Error_Destroy(&da);
    exit(1);
}

static int pjrt_open(const char *path) {
    typedef PJRT_Api *(*get_api_fn)(void);
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        return -1;
    }
    get_api_fn get = (get_api_fn)dlsym(h, "GetPjrtApi");
    if (get == NULL) {
        fprintf(stderr, "dlsym(GetPjrtApi): %s\n", dlerror());
        return -1;
    }
    api = get();
    if (api == NULL || api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        fprintf(stderr, "incompatible PJRT API\n");
        return -1;
    }

    PJRT_Plugin_Initialize_Args ia = {0};
    ia.struct_size = sizeof ia;
    die_error(api->PJRT_Plugin_Initialize(&ia), "plugin initialize");

    PJRT_Client_Create_Args ca = {0};
    ca.struct_size = sizeof ca;
    die_error(api->PJRT_Client_Create(&ca), "client create");
    client = ca.client;

    PJRT_Client_AddressableDevices_Args da = {0};
    da.struct_size = sizeof da;
    da.client = client;
    die_error(api->PJRT_Client_AddressableDevices(&da), "addressable devices");
    if (da.num_addressable_devices < 1) {
        fprintf(stderr, "no addressable devices\n");
        return -1;
    }
    device = da.addressable_devices[0];
    return 0;
}

/* CompileOptionsProto{ executable_build_options{ num_replicas=1,
 * num_partitions=1 } }: field 3 -> fields 4 and 5, both varint 1. */
static const unsigned char compile_options[] = {0x1a, 0x04, 0x20, 0x01, 0x28, 0x01};

/* tensor: C = A.B (contract a's k with b's k-row). tensor-v: BT rows are the
 * contiguous B rows, so C[i][j] = sum_k A[i][k]*BT[j][k], i.e. contract both
 * second dimensions. */
static char *build_hlo(int n, int tv) {
    size_t cap = 2048;
    char *s = malloc(cap);
    snprintf(s, cap,
             "module {\n"
             "  func.func @main(%%a: tensor<%dx%dxi8>, %%b: tensor<%dx%dxi8>)\n"
             "      -> tensor<%dx%dxi32> {\n"
             "    %%0 = stablehlo.dot_general %%a, %%b, contracting_dims = [1] x [%d],\n"
             "        precision = [DEFAULT]\n"
             "        : (tensor<%dx%dxi8>, tensor<%dx%dxi8>) -> tensor<%dx%dxi32>\n"
             "    return %%0 : tensor<%dx%dxi32>\n"
             "  }\n"
             "}\n",
             n, n, n, n, n, n, tv ? 1 : 0,
             n, n, n, n, n, n, n, n);
    return s;
}

static PJRT_LoadedExecutable *compile_dot(int n, int tv, double *compile_ms) {
    char *hlo = build_hlo(n, tv);
    PJRT_Program prog = {0};
    PJRT_Client_Compile_Args cc = {0};
    double t0 = now_s();

    prog.struct_size = sizeof prog;
    prog.code = hlo;
    prog.code_size = strlen(hlo);
    prog.format = "mlir";
    prog.format_size = 4;

    cc.struct_size = sizeof cc;
    cc.client = client;
    cc.program = &prog;
    cc.compile_options = (const char *)compile_options;
    cc.compile_options_size = sizeof compile_options;
    die_error(api->PJRT_Client_Compile(&cc), "compile");
    free(hlo);
    if (compile_ms != NULL) {
        *compile_ms = (now_s() - t0) * 1000.0;
    }
    return cc.executable;
}

static PJRT_Buffer *upload(const void *data, int n, PJRT_Buffer_Type type,
                           double *ms) {
    int64_t dims[2] = {n, n};
    PJRT_Event *done = NULL;
    PJRT_Client_BufferFromHostBuffer_Args ba = {0};
    double t0 = now_s();

    ba.struct_size = sizeof ba;
    ba.client = client;
    ba.data = data;
    ba.type = type;
    ba.dims = dims;
    ba.num_dims = 2;
    ba.byte_strides = NULL;
    ba.num_byte_strides = 0;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    ba.device = device;
    die_error(api->PJRT_Client_BufferFromHostBuffer(&ba), "buffer from host");
    done = ba.done_with_host_buffer;
    if (done != NULL) {
        PJRT_Event_Await_Args ea = {0};
        ea.struct_size = sizeof ea;
        ea.event = done;
        die_error(api->PJRT_Event_Await(&ea), "host transfer");
        PJRT_Event_Destroy_Args ed = {0};
        ed.struct_size = sizeof ed;
        ed.event = done;
        api->PJRT_Event_Destroy(&ed);
    }
    if (ms != NULL) {
        *ms = (now_s() - t0) * 1000.0;
    }
    return ba.buffer;
}

static PJRT_Buffer *execute(PJRT_LoadedExecutable *ex, PJRT_Buffer *a,
                            PJRT_Buffer *b, double *ms) {
    PJRT_Buffer *args0[2] = {a, b};
    PJRT_Buffer *const *arg_lists[1] = {args0};
    PJRT_Buffer *out0[1] = {NULL};
    PJRT_Buffer **out_lists[1] = {out0};
    PJRT_Event *events[1] = {NULL};
    PJRT_ExecuteOptions opts = {0};
    PJRT_LoadedExecutable_Execute_Args ea = {0};

    opts.struct_size = sizeof opts;
    double t0 = now_s();

    ea.struct_size = sizeof ea;
    ea.executable = ex;
    ea.options = &opts;
    ea.argument_lists = arg_lists;
    ea.num_devices = 1;
    ea.num_args = 2;
    ea.output_lists = out_lists;
    ea.device_complete_events = events;
    die_error(api->PJRT_LoadedExecutable_Execute(&ea), "execute");
    if (events[0] != NULL) {
        PJRT_Event_Await_Args wa = {0};
        wa.struct_size = sizeof wa;
        wa.event = events[0];
        die_error(api->PJRT_Event_Await(&wa), "execute wait");
        PJRT_Event_Destroy_Args ed = {0};
        ed.struct_size = sizeof ed;
        ed.event = events[0];
        api->PJRT_Event_Destroy(&ed);
    }
    if (ms != NULL) {
        *ms = (now_s() - t0) * 1000.0;
    }
    return out0[0];
}

static void read_back(PJRT_Buffer *buf, void *dst, size_t bytes, double *ms) {
    PJRT_Event *ev = NULL;
    PJRT_Buffer_ToHostBuffer_Args ta = {0};
    double t0 = now_s();

    ta.struct_size = sizeof ta;
    ta.src = buf;
    ta.dst = dst;
    ta.dst_size = bytes;
    die_error(api->PJRT_Buffer_ToHostBuffer(&ta), "to host");
    ev = ta.event;
    if (ev != NULL) {
        PJRT_Event_Await_Args wa = {0};
        wa.struct_size = sizeof wa;
        wa.event = ev;
        die_error(api->PJRT_Event_Await(&wa), "to host wait");
        PJRT_Event_Destroy_Args ed = {0};
        ed.struct_size = sizeof ed;
        ed.event = ev;
        api->PJRT_Event_Destroy(&ed);
    }
    if (ms != NULL) {
        *ms = (now_s() - t0) * 1000.0;
    }
}

static void buffer_destroy(PJRT_Buffer *b) {
    PJRT_Buffer_Destroy_Args da = {0};
    if (b == NULL) {
        return;
    }
    da.struct_size = sizeof da;
    da.buffer = b;
    api->PJRT_Buffer_Destroy(&da);
}

/* ==================================================================== */
/* benchmark                                                            */
/* ==================================================================== */

static const char DEFAULT_CHALLENGE[] =
    "9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b";

static int cmp_double(const void *pa, const void *pb) {
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return v[n / 2];
}

static int bench_one(int difficulty, int n_override, const char *challenge,
                     int rounds, int verify, int json, int tv) {
    int n = n_override > 0 ? n_override : tensor_dim(difficulty);
    size_t cells = (size_t)n * (size_t)n;
    uint8_t *a = malloc(cells);
    int8_t *b = malloc(cells);
    int32_t *c = alloc_host(cells * sizeof *c);
    int32_t *ref = NULL;
    char gpu_hex[65], cpu_hex[65];
    double compile_ms = 0, upload_ms = 0, exec_ms = 0, read_ms = 0;
    double *execs, *e2es;
    int ok = 1;
    PJRT_LoadedExecutable *ex;
    PJRT_Buffer *dev_a, *dev_b, *dev_c;
    double t_e2e;

    if (a == NULL || b == NULL || c == NULL) {
        return -1;
    }
    tensor_expand(challenge, n, a, b);
    ex = compile_dot(n, tv, &compile_ms);

    dev_a = upload(a, n, PJRT_Buffer_Type_S8, &upload_ms);
    dev_b = upload(b, n, PJRT_Buffer_Type_S8, NULL);

    /* warm-up: first execute pays for whatever the backend defers */
    buffer_destroy(execute(ex, dev_a, dev_b, NULL));

    execs = calloc((size_t)rounds, sizeof *execs);
    e2es = calloc((size_t)rounds, sizeof *e2es);
    for (int r = 0; r < rounds; r++) {
        t_e2e = now_s();
        dev_c = execute(ex, dev_a, dev_b, &exec_ms);
        read_back(dev_c, c, cells * 4, &read_ms);
        buffer_destroy(dev_c);
        execs[r] = exec_ms;
        e2es[r] = (now_s() - t_e2e) * 1000.0;
    }
    exec_ms = median(execs, rounds);
    t_e2e = median(e2es, rounds);
    if (tv) {
        tv_root_of(c, n, challenge, gpu_hex);
    } else {
        digest_of(c, n, gpu_hex);
    }

    if (verify) {
        ref = malloc(cells * sizeof *ref);
        if (ref == NULL) {
            verify = 0;
        }
    }
    if (verify) {
        if (tv) {
            cpu_gemm_tv(a, b, ref, n);
            tv_root_of(ref, n, challenge, cpu_hex);
        } else {
            cpu_gemm(a, b, ref, n);
            digest_of(ref, n, cpu_hex);
        }
        ok = strcmp(gpu_hex, cpu_hex) == 0;
        free(ref);
    } else {
        cpu_hex[0] = '\0';
    }

    {
        double macs = 2.0 * n * n * n;
        if (json) {
            printf("{\"difficulty\":%d,\"n\":%d,\"macs\":%.0f,\"compile_ms\":%.2f,"
                   "\"upload_ms\":%.2f,\"execute_ms\":%.4f,\"e2e_ms\":%.4f,"
                   "\"io_ms\":%.4f,"
                   "\"gmacs_execute\":%.2f,\"gmacs_e2e\":%.2f,\"digest_ok\":%s,"
                   "\"dma_mapped\":%s,"
                   "\"digest\":\"%s\",\"tier\":\"%s\"}\n",
                   difficulty, n, macs, compile_ms, upload_ms, exec_ms, t_e2e,
                   t_e2e - exec_ms,
                   macs / (exec_ms / 1000.0) / 1e9,
                   macs / (t_e2e / 1000.0) / 1e9, ok ? "true" : "false",
                   g_dma_state == 1 ? "true" : (g_dma_state == 0 ? "false" : "null"),
                   gpu_hex,
                   tv ? "tensor-v" : "tensor");
        } else {
            printf("d=%2d N=%4d %8.1f MMAC  execute %9.4f ms (%7.1f GMAC/s)  "
                   "e2e %9.4f ms (%7.1f GMAC/s)  compile %6.0f ms  %s\n",
                   difficulty, n, macs / 1e6, exec_ms,
                   macs / (exec_ms / 1000.0) / 1e9, t_e2e,
                   macs / (t_e2e / 1000.0) / 1e9, compile_ms,
                   verify ? (ok ? (tv ? "root ok" : "digest ok")
                                : (tv ? "ROOT MISMATCH" : "DIGEST MISMATCH"))
                          : "unverified");
        }
        fflush(stdout);
    }

    free(execs);
    free(e2es);
    buffer_destroy(dev_a);
    buffer_destroy(dev_b);
    {
        PJRT_LoadedExecutable_Destroy_Args ed = {0};
        ed.struct_size = sizeof ed;
        ed.executable = ex;
        api->PJRT_LoadedExecutable_Destroy(&ed);
    }
    free(a);
    free(b);
    free(c);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *lib = getenv("LIBTPU");
    const char *challenge = DEFAULT_CHALLENGE;
    int rounds = 5, verify = 1, json = 0, n_override = 0, tv = 0, cpu_root = 0;
    int diffs[16], ndiffs = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc) {
            lib = argv[++i];
        } else if (strcmp(argv[i], "--challenge") == 0 && i + 1 < argc) {
            challenge = argv[++i];
        } else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            rounds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--difficulty") == 0 && i + 1 < argc) {
            char *s = argv[++i];
            while (s != NULL && *s != '\0' && ndiffs < 16) {
                diffs[ndiffs++] = atoi(s);
                s = strchr(s, ',');
                if (s != NULL) s++;
            }
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n_override = atoi(argv[++i]);
            if (n_override % 64 != 0 || n_override < 64) {
                fprintf(stderr, "--n must be a positive multiple of 64\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            verify = 0;
        } else if (strcmp(argv[i], "--tensor-v") == 0) {
            tv = 1;
        } else if (strcmp(argv[i], "--cpu-root") == 0) {
            cpu_root = 1;
        } else if (strcmp(argv[i], "--dmamap") == 0) {
            g_dmamap = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else {
            fprintf(stderr,
                    "usage: %s --lib /path/libtpu.so [--difficulty 4,8,12,16] "
                    "[--n N] [--rounds 5] [--challenge HEX] [--no-verify] [--json]\n"
                    "       [--tensor-v] [--cpu-root] [--dmamap]\n",
                    argv[0]);
            return 2;
        }
    }
    if (ndiffs == 0) {
        diffs[ndiffs++] = 8;
    }
    if (rounds < 1) {
        rounds = 1;
    }
    if (cpu_root) {
        for (int i = 0; i < ndiffs; i++) {
            int n = n_override > 0 ? n_override : tensor_dim(diffs[i]);
            size_t cells = (size_t)n * (size_t)n;
            uint8_t *a = malloc(cells);
            int8_t *b = malloc(cells);
            int32_t *c = alloc_host(cells * sizeof *c);
            char root[65];
            if (a == NULL || b == NULL || c == NULL) {
                return 1;
            }
            tensor_expand(challenge, n, a, b);
            cpu_gemm_tv(a, b, c, n);
            tv_root_of(c, n, challenge, root);
            printf("d=%2d N=%4d root=%s\n", diffs[i], n, root);
            free(a);
            free(b);
            free(c);
        }
        return 0;
    }
    if (lib == NULL) {
        fprintf(stderr, "need --lib or LIBTPU pointing at libtpu.so\n");
        return 2;
    }
    if (pjrt_open(lib) != 0) {
        return 1;
    }
    if (!json) {
        printf("plugin: %s (PJRT 0.%d)\n", lib, api->pjrt_api_version.minor_version);
        fflush(stdout);
    }
    int rc = 0;
    for (int i = 0; i < ndiffs; i++) {
        int r = bench_one(diffs[i], n_override, challenge, rounds, verify, json, tv);
        if (r != 0) {
            rc = r;
        }
    }
    {
        PJRT_Client_Destroy_Args cd = {0};
        cd.struct_size = sizeof cd;
        cd.client = client;
        api->PJRT_Client_Destroy(&cd);
    }
    return rc;
}
