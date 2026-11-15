/*
 * sha256_pjrt.c — the Anubis `sha256` proof of work on a TPU, through the PJRT
 * C API, no Python.
 *
 * The bet this program tests: a sha256 challenge is many *independent* short
 * messages (one per nonce), which is the shape a TPU likes - unlike the tensor
 * tier's single long GEMM chain. The decoded challenge is ~32 bytes, so
 * prefix || u32 nonce || 0x80 || length is exactly one 64-byte block per nonce,
 * and 64 rounds of add/xor/rotate are just vector integer ops. No SHA-256
 * instruction is involved and none is needed.
 *
 * Structure:
 *
 *   1. Host builds the 16 message words for nonce 0, with the nonce field zero
 *      and the padding and length bits already in place. Those words are baked
 *      into the StableHLO module as constants, so the only input is the base
 *      nonce counter: one i32 upload per window, not one block per nonce.
 *   2. The module computes N lanes at once: iota + base gives each lane its
 *      nonce, four byte-splices patch it into the words, 64 rounds run on
 *      tensor<Nxi32>, and the result is tensor<Nx8xi32>.
 *   3. The host reads back N*32 bytes and scans for `difficulty` leading zero
 *      bits, exactly as libanubis_solver does: a hit is only accepted after
 *      pjrt_sha256 reproduces its digest on the CPU. A TPU that miscomputes one
 *      lane fails verification; it cannot mint a challenge.
 *
 * The nonce byte order follows the engine: little-endian when the last decoded
 * challenge byte has its high bit set, big-endian otherwise.
 *
 * Build:  cc -O2 -I. -o sha256_pjrt sha256_pjrt.c -ldl
 * Run:    ./sha256_pjrt --lib libtpu.so --challenge HEX --difficulty 20
 */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"
#include "pjrt_common.h"

/* ==================================================================== */
/* text buffer and HLO generation                                       */
/* ==================================================================== */

typedef struct {
    char *s;
    size_t len, cap;
} buf_t;

static void bprintf(buf_t *b, const char *fmt, ...) {
    va_list ap;
    int need;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (b->len + (size_t)need + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 65536;
        while (cap < b->len + (size_t)need + 1) {
            cap *= 2;
        }
        b->s = realloc(b->s, cap);
        b->cap = cap;
    }
    va_start(ap, fmt);
    vsnprintf(b->s + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)need;
}

static int g_id;
static int g_lanes;

static char *nv(const char *prefix) {
    char *s = malloc(32);
    snprintf(s, 32, "%%%s%d", prefix, g_id++);
    return s;
}

/* functions that need the module buffer */
static char *bc(buf_t *b, uint32_t value) {
    char *scalar = nv("c");
    char *splat = nv("s");
    /* signed decimal: MLIR reads these as signed i32, and 0xffffffff must be
     * -1 rather than an unsigned literal some parsers reject */
    bprintf(b, "    %s = stablehlo.constant dense<%d> : tensor<i32>\n",
            scalar, (int32_t)value);
    bprintf(b, "    %s = stablehlo.broadcast_in_dim %s, dims = [] "
               ": (tensor<i32>) -> tensor<%dxi32>\n", splat, scalar, g_lanes);
    free(scalar);
    return splat;
}

static char *op2(buf_t *b, const char *op, const char *x, const char *y) {
    char *r = nv("v");
    bprintf(b, "    %s = stablehlo.%s %s, %s : tensor<%dxi32>\n", r, op, x, y, g_lanes);
    return r;
}

static char *xor3(buf_t *b, const char *x, const char *y, const char *z) {
    char *t = op2(b, "xor", x, y);
    char *r = op2(b, "xor", t, z);
    free(t);
    return r;
}

static char *add3(buf_t *b, const char *x, const char *y, const char *z) {
    char *t = op2(b, "add", x, y);
    char *r = op2(b, "add", t, z);
    free(t);
    return r;
}

static char *add4(buf_t *b, const char *x, const char *y, const char *z, const char *w) {
    char *t = add3(b, x, y, z);
    char *r = op2(b, "add", t, w);
    free(t);
    return r;
}

static char *rotr(buf_t *b, const char *x, unsigned n) {
    char *sr = op2(b, "shift_right_logical", x, bc(b, n));
    char *sl = op2(b, "shift_left", x, bc(b, 32 - n));
    char *r = op2(b, "or", sr, sl);
    free(sr);
    free(sl);
    return r;
}

/*
 * Build the module. `words` are the 16 big-endian message words with the nonce
 * field zeroed; `nonce_len`, `nonce_off` and `use_le` describe where the four
 * nonce bytes go and in which order they are read.
 */
static char *build_module(int lanes, const uint32_t words[16], int nonce_off,
                          int use_le, int difficulty) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u};
    buf_t b = {0};
    char *vars[8];
    char *w[64];
    char *nb;

    g_lanes = lanes;
    g_id = 0;

    bprintf(&b,
            "module {\n"
            "  func.func @main(%%n: tensor<i32>) -> tensor<%dxi8> {\n"
            "    %%lane = stablehlo.iota dim = 0 : tensor<%dxi32>\n", lanes, lanes);
    {
        char *base = nv("nb");
        bprintf(&b, "    %s = stablehlo.broadcast_in_dim %%n, dims = [] "
                    ": (tensor<i32>) -> tensor<%dxi32>\n", base, lanes);
        nb = op2(&b, "add", "%lane", base);
        free(base);
    }

    /* 16 message words: constants with the four nonce bytes spliced in. */
    for (int j = 0; j < 16; j++) {
        char *cj = nv("c");
        bprintf(&b, "    %s = stablehlo.constant dense<%d> : tensor<i32>\n",
                cj, (int32_t)words[j]);
        w[j] = nv("w");
        bprintf(&b, "    %s = stablehlo.broadcast_in_dim %s, dims = [] "
                    ": (tensor<i32>) -> tensor<%dxi32>\n", w[j], cj, lanes);
        free(cj);
    }
    for (int i = 0; i < 4; i++) {
        int byte = nonce_off + i;
        int word = byte / 4;
        /* the block words are big-endian: byte 0 of a word is its MSB */
        unsigned shift = (unsigned)(8 * (3 - (byte % 4)));
        unsigned nshift = use_le ? (unsigned)(8 * i) : (unsigned)(8 * (3 - i));
        char *shr = op2(&b, "shift_right_logical", nb, bc(&b, nshift));
        char *m = op2(&b, "and", shr, bc(&b, 0xffu));
        char *shl = op2(&b, "shift_left", m, bc(&b, shift));
        char *nw = op2(&b, "add", w[word], shl);
        free(w[word]);
        w[word] = nw;
        free(shr);
        free(m);
        free(shl);
    }

    /* message schedule */
    for (int i = 16; i < 64; i++) {
        char *a1 = rotr(&b, w[i - 15], 7);
        char *a2 = rotr(&b, w[i - 15], 18);
        char *a3 = op2(&b, "shift_right_logical", w[i - 15], bc(&b, 3));
        char *s0 = xor3(&b, a1, a2, a3);
        char *b1 = rotr(&b, w[i - 2], 17);
        char *b2 = rotr(&b, w[i - 2], 19);
        char *b3 = op2(&b, "shift_right_logical", w[i - 2], bc(&b, 10));
        char *s1 = xor3(&b, b1, b2, b3);
        char *t = add3(&b, w[i - 16], s0, w[i - 7]);
        w[i] = op2(&b, "add", t, s1);
        free(a1); free(a2); free(a3); free(s0);
        free(b1); free(b2); free(b3); free(s1); free(t);
    }

    /* state: IV, then 64 rounds. Names are never freed here: the rotation
     * shares them between slots, and the process is about to end anyway. */
    for (int i = 0; i < 8; i++) {
        vars[i] = bc(&b, iv[i]);
    }
    for (int i = 0; i < 64; i++) {
        char *S1 = xor3(&b, rotr(&b, vars[4], 6), rotr(&b, vars[4], 11),
                        rotr(&b, vars[4], 25));
        char *en = op2(&b, "xor", vars[4], bc(&b, 0xffffffffu));
        char *ch = op2(&b, "xor", op2(&b, "and", vars[4], vars[5]),
                       op2(&b, "and", en, vars[6]));
        char *kw = op2(&b, "add", bc(&b, pjrt_k256[i]), w[i]);
        char *t1 = add4(&b, vars[7], S1, ch, kw);
        char *S0 = xor3(&b, rotr(&b, vars[0], 2), rotr(&b, vars[0], 13),
                        rotr(&b, vars[0], 22));
        char *maj = xor3(&b, op2(&b, "and", vars[0], vars[1]),
                         op2(&b, "and", vars[0], vars[2]),
                         op2(&b, "and", vars[1], vars[2]));
        char *t2 = op2(&b, "add", S0, maj);
        char *e_new = op2(&b, "add", vars[3], t1);
        char *a_new = op2(&b, "add", t1, t2);
        vars[7] = vars[6];
        vars[6] = vars[5];
        vars[5] = vars[4];
        vars[4] = e_new;
        vars[3] = vars[2];
        vars[2] = vars[1];
        vars[1] = vars[0];
        vars[0] = a_new;
        free(S1); free(en); free(ch); free(kw); free(t1);
        free(S0); free(maj); free(t2);
    }

    /* The device decides "does this lane meet the target?" and the host gets
     * one byte per lane instead of 32. The digest is re-derived on the CPU for
     * the single lane that matters, so the transfer is B bytes, not B*32 - at
     * these window sizes the copy back was the whole cost. */
    {
        char *h0 = nv("h0");
        char *th = NULL;
        char *cmp = nv("ok");
        char *fl = nv("flag");
        uint32_t threshold = (difficulty >= 32) ? 1u : (1u << (32 - difficulty));
        bprintf(&b, "    %s = stablehlo.add %s, %s : tensor<%dxi32>\n",
                h0, vars[0], bc(&b, iv[0]), lanes);
        (void)th;
        {
            char *tc = nv("tc");
            char *ts = nv("ts");
            bprintf(&b, "    %s = stablehlo.constant dense<%d> : tensor<i32>\n",
                    tc, (int32_t)threshold);
            bprintf(&b, "    %s = stablehlo.broadcast_in_dim %s, dims = [] "
                        ": (tensor<i32>) -> tensor<%dxi32>\n", ts, tc, lanes);
            /* unsigned: the threshold for a 1-bit difficulty is 0x80000000 */
            bprintf(&b, "    %s = \"stablehlo.compare\"(%s, %s) "
                        "{comparison_direction = #stablehlo<comparison_direction LT>, "
                        "compare_type = #stablehlo<comparison_type UNSIGNED>} "
                        ": (tensor<%dxi32>, tensor<%dxi32>) -> tensor<%dxi1>\n",
                    cmp, h0, ts, lanes, lanes, lanes);
            bprintf(&b, "    %s = \"stablehlo.convert\"(%s) "
                        ": (tensor<%dxi1>) -> tensor<%dxi8>\n", fl, cmp, lanes, lanes);
            bprintf(&b, "    return %s : tensor<%dxi8>\n", fl, lanes);
        }
    }
    bprintf(&b, "  }\n}\n");
    return b.s;
}

/* ==================================================================== */
/* PJRT plumbing (same lifecycle rules as tensor_pjrt.c)                */
/* ==================================================================== */

static PJRT_Api *api;
static PJRT_Client *client;
static PJRT_Device *device;

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

static void pjrt_open(const char *path) {
    typedef PJRT_Api *(*get_api_fn)(void);
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    get_api_fn get;
    PJRT_Plugin_Initialize_Args ia = {0};
    PJRT_Client_Create_Args ca = {0};
    PJRT_Client_AddressableDevices_Args da = {0};

    if (h == NULL) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        exit(1);
    }
    get = (get_api_fn)dlsym(h, "GetPjrtApi");
    if (get == NULL) {
        fprintf(stderr, "dlsym(GetPjrtApi): %s\n", dlerror());
        exit(1);
    }
    api = get();
    if (api == NULL || api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        fprintf(stderr, "incompatible PJRT API\n");
        exit(1);
    }
    ia.struct_size = sizeof ia;
    die_error(api->PJRT_Plugin_Initialize(&ia), "plugin initialize");
    ca.struct_size = sizeof ca;
    die_error(api->PJRT_Client_Create(&ca), "client create");
    client = ca.client;
    da.struct_size = sizeof da;
    da.client = client;
    die_error(api->PJRT_Client_AddressableDevices(&da), "addressable devices");
    if (da.num_addressable_devices < 1) {
        fprintf(stderr, "no devices\n");
        exit(1);
    }
    device = da.addressable_devices[0];
}

static const unsigned char compile_options[] = {0x1a, 0x04, 0x20, 0x01, 0x28, 0x01};

static PJRT_LoadedExecutable *compile(const char *code, int *len_out) {
    PJRT_Program prog = {0};
    PJRT_Client_Compile_Args cc = {0};
    double t0 = pjrt_now_s();
    prog.struct_size = sizeof prog;
    prog.code = (char *)code;
    prog.code_size = strlen(code);
    prog.format = "mlir";
    prog.format_size = 4;
    cc.struct_size = sizeof cc;
    cc.client = client;
    cc.program = &prog;
    cc.compile_options = (const char *)compile_options;
    cc.compile_options_size = sizeof compile_options;
    die_error(api->PJRT_Client_Compile(&cc), "compile");
    if (len_out != NULL) {
        *len_out = (int)((pjrt_now_s() - t0) * 1000.0);
    }
    return cc.executable;
}

static PJRT_Buffer *upload_i32(int32_t value) {
    PJRT_Event *done = NULL;
    PJRT_Client_BufferFromHostBuffer_Args ba = {0};
    ba.struct_size = sizeof ba;
    ba.client = client;
    ba.data = &value;
    ba.type = PJRT_Buffer_Type_S32;
    ba.dims = NULL;
    ba.num_dims = 0;
    ba.byte_strides = NULL;
    ba.num_byte_strides = 0;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    ba.device = device;
    die_error(api->PJRT_Client_BufferFromHostBuffer(&ba), "buffer from host");
    done = ba.done_with_host_buffer;
    if (done != NULL) {
        PJRT_Event_Await_Args ea = {0};
        PJRT_Event_Destroy_Args ed = {0};
        ea.struct_size = sizeof ea;
        ea.event = done;
        die_error(api->PJRT_Event_Await(&ea), "host transfer");
        ed.struct_size = sizeof ed;
        ed.event = done;
        api->PJRT_Event_Destroy(&ed);
    }
    return ba.buffer;
}

static PJRT_Buffer *execute(PJRT_LoadedExecutable *ex, PJRT_Buffer *arg, double *ms) {
    PJRT_Buffer *args0[1] = {arg};
    PJRT_Buffer *const *arg_lists[1] = {args0};
    PJRT_Buffer *out0[1] = {NULL};
    PJRT_Buffer **out_lists[1] = {out0};
    PJRT_Event *events[1] = {NULL};
    PJRT_ExecuteOptions opts = {0};
    PJRT_LoadedExecutable_Execute_Args ea = {0};
    double t0 = pjrt_now_s();

    opts.struct_size = sizeof opts;
    ea.struct_size = sizeof ea;
    ea.executable = ex;
    ea.options = &opts;
    ea.argument_lists = arg_lists;
    ea.num_devices = 1;
    ea.num_args = 1;
    ea.output_lists = out_lists;
    ea.device_complete_events = events;
    die_error(api->PJRT_LoadedExecutable_Execute(&ea), "execute");
    if (events[0] != NULL) {
        PJRT_Event_Await_Args wa = {0};
        PJRT_Event_Destroy_Args ed = {0};
        wa.struct_size = sizeof wa;
        wa.event = events[0];
        die_error(api->PJRT_Event_Await(&wa), "execute wait");
        ed.struct_size = sizeof ed;
        ed.event = events[0];
        api->PJRT_Event_Destroy(&ed);
    }
    if (ms != NULL) {
        *ms = (pjrt_now_s() - t0) * 1000.0;
    }
    return out0[0];
}

static void read_back(PJRT_Buffer *buf, void *dst, size_t bytes) {
    PJRT_Event *ev = NULL;
    PJRT_Buffer_ToHostBuffer_Args ta = {0};
    ta.struct_size = sizeof ta;
    ta.src = buf;
    ta.dst = dst;
    ta.dst_size = bytes;
    die_error(api->PJRT_Buffer_ToHostBuffer(&ta), "to host");
    ev = ta.event;
    if (ev != NULL) {
        PJRT_Event_Await_Args wa = {0};
        PJRT_Event_Destroy_Args ed = {0};
        wa.struct_size = sizeof wa;
        wa.event = ev;
        die_error(api->PJRT_Event_Await(&wa), "to host wait");
        ed.struct_size = sizeof ed;
        ed.event = ev;
        api->PJRT_Event_Destroy(&ed);
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
/* host side: block, scan, verify                                       */
/* ==================================================================== */

static const char DEFAULT_CHALLENGE[] =
    "9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b";

static void build_block(const uint8_t *raw, size_t raw_len, uint32_t words[16]) {
    uint8_t block[64] = {0};
    uint64_t bits = (uint64_t)(raw_len + 4) * 8u;
    memcpy(block, raw, raw_len);
    block[raw_len + 4] = 0x80;
    for (int i = 0; i < 8; i++) {
        block[56 + i] = (uint8_t)(bits >> (8 * (7 - i)));
    }
    for (int j = 0; j < 16; j++) {
        words[j] = ((uint32_t)block[4 * j] << 24) | ((uint32_t)block[4 * j + 1] << 16) |
                   ((uint32_t)block[4 * j + 2] << 8) | (uint32_t)block[4 * j + 3];
    }
}

static int digest_bits(const uint8_t *d, int difficulty) {
    int n = 0;
    for (int i = 0; i < 32 && n < difficulty; i++) {
        if (d[i] == 0) {
            n += 8;
            continue;
        }
        n += __builtin_clz((unsigned)d[i]) - 24;
        break;
    }
    return n;
}

/* The reference: SHA-256(raw || nonce) in the engine's byte order. */
static void reference_digest(const uint8_t *raw, size_t raw_len, uint32_t nonce,
                             int use_le, uint8_t out[32]) {
    uint8_t msg[4096 + 4];
    memcpy(msg, raw, raw_len);
    if (use_le) {
        msg[raw_len + 0] = (uint8_t)nonce;
        msg[raw_len + 1] = (uint8_t)(nonce >> 8);
        msg[raw_len + 2] = (uint8_t)(nonce >> 16);
        msg[raw_len + 3] = (uint8_t)(nonce >> 24);
    } else {
        msg[raw_len + 0] = (uint8_t)(nonce >> 24);
        msg[raw_len + 1] = (uint8_t)(nonce >> 16);
        msg[raw_len + 2] = (uint8_t)(nonce >> 8);
        msg[raw_len + 3] = (uint8_t)nonce;
    }
    pjrt_sha256_buf(msg, raw_len + 4, out);
}

int main(int argc, char **argv) {
    const char *lib = getenv("LIBTPU");
    const char *challenge = DEFAULT_CHALLENGE;
    int difficulty = 20, lanes = 32768, json = 0;
    long long bench_hashes = 0;
    uint8_t raw[4096];
    size_t raw_len;
    int use_le, compile_ms = 0;
    uint32_t words[16];
    char *module;
    PJRT_LoadedExecutable *ex;
    uint8_t *out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc) {
            lib = argv[++i];
        } else if (strcmp(argv[i], "--challenge") == 0 && i + 1 < argc) {
            challenge = argv[++i];
        } else if (strcmp(argv[i], "--difficulty") == 0 && i + 1 < argc) {
            difficulty = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            lanes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-hashes") == 0 && i + 1 < argc) {
            bench_hashes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else {
            fprintf(stderr, "usage: %s --lib libtpu.so [--challenge HEX] "
                            "[--difficulty BITS] [--window LANES] "
                            "[--bench-hashes N] [--json]\n", argv[0]);
            return 2;
        }
    }
    if (lib == NULL) {
        fprintf(stderr, "need --lib or LIBTPU\n");
        return 2;
    }
    if (lanes < 1 || lanes > 262144) {
        fprintf(stderr, "--window out of range\n");
        return 2;
    }
    raw_len = pjrt_hex_decode(challenge, raw, sizeof raw);
    if (raw_len == 0 || raw_len + 4 > 55) {
        fprintf(stderr, "challenge must decode to 1..51 bytes (got %zu)\\n", raw_len);
        return 2;
    }
    use_le = raw[raw_len - 1] >= 0x80;
    build_block(raw, raw_len, words);

    pjrt_open(lib);
    if (difficulty < 1 || difficulty > 32) {
        fprintf(stderr, "--difficulty must be 1..32 for the device-side test\n");
        return 2;
    }
    module = build_module(lanes, words, (int)raw_len, use_le, difficulty);
    ex = compile(module, &compile_ms);
    out = malloc((size_t)lanes); /* one flag byte per lane */
    if (out == NULL) {
        return 1;
    }

    {
        double t0 = pjrt_now_s();
        long long done = 0, hits = 0;
        uint32_t base = 0;
        int found = 0;
        uint32_t found_nonce = 0;

        /* one untimed window: the first execute after compile pays for
         * whatever the backend defers, and it is not part of the rate */
        {
            PJRT_Buffer *arg = upload_i32(0);
            PJRT_Buffer *res = execute(ex, arg, NULL);
            read_back(res, out, (size_t)lanes);
            buffer_destroy(arg);
            buffer_destroy(res);
        }
        t0 = pjrt_now_s();

        while (1) {
            PJRT_Buffer *arg = upload_i32((int32_t)base);
            PJRT_Buffer *res = execute(ex, arg, NULL);
            read_back(res, out, (size_t)lanes);
            buffer_destroy(arg);
            buffer_destroy(res);
            for (int lane = 0; lane < lanes; lane++) {
                if (out[lane] != 0) {
                    hits++;
                    if (!found) {
                        found_nonce = base + (uint32_t)lane;
                        found = 1;
                    }
                }
            }
            done += lanes;
            base += (uint32_t)lanes;
            if (bench_hashes > 0) {
                if (done >= bench_hashes) {
                    break;
                }
            } else if (found) {
                break;
            }
            if (base == 0) { /* wrapped: unrealistic, stop cleanly */
                break;
            }
        }

        {
            double ms = (pjrt_now_s() - t0) * 1000.0;
            int verified = 0;
            char hex[65];
            if (found) {
                uint8_t ref[32];
                reference_digest(raw, raw_len, found_nonce, use_le, ref);
                verified = digest_bits(ref, difficulty) >= difficulty;
                pjrt_to_hex(ref, 32, hex);
            } else {
                hex[0] = '\0';
            }
            if (json) {
                printf("{\"hashes\":%lld,\"ms\":%.3f,\"mhash_s\":%.3f,"
                       "\"window\":%d,\"difficulty\":%d,\"compile_ms\":%d,"
                       "\"nonce\":%u,\"digest\":\"%s\",\"verified\":%s,"
                       "\"hits\":%lld}\\n",
                       done, ms, (double)done / (ms / 1000.0) / 1e6, lanes,
                       difficulty, compile_ms, found ? found_nonce : 0, hex,
                       verified ? "true" : "false", hits);
            } else {
                printf("lanes=%d difficulty=%d hashes=%lld %.1f ms = %.2f MHash/s "
                       "compile=%d ms\n", lanes, difficulty, done, ms,
                       (double)done / (ms / 1000.0) / 1e6, compile_ms);
                if (found) {
                    printf("nonce=%u digest=%s %s\n", found_nonce, hex,
                           verified ? "verified" : "VERIFICATION FAILED");

                } else if (bench_hashes > 0) {
                    printf("(benchmark only, hits=%lld)\n", hits);
                }
            }
            fflush(stdout);
            free(out);
            free(module);
            {
                PJRT_LoadedExecutable_Destroy_Args ed = {0};
                PJRT_Client_Destroy_Args cd = {0};
                ed.struct_size = sizeof ed;
                ed.executable = ex;
                api->PJRT_LoadedExecutable_Destroy(&ed);
                cd.struct_size = sizeof cd;
                cd.client = client;
                api->PJRT_Client_Destroy(&cd);
            }
            return found && !verified ? 1 : 0;
        }
    }
}
