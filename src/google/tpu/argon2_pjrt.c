/*
 * argon2_pjrt.c — Argon2id (m=19456 KiB, t=2, p=1) as a StableHLO program,
 * run through the PJRT C API.
 *
 * The gate's argon2id tier hashes with Argon2id(password=challenge,
 * salt=nonce, m=19456, t=2, p=1) and accepts a nonce when the 32-byte tag has
 * enough leading zero bits. Almost all of the work is the memory fill: 19 MiB
 * of 1 KiB blocks, two passes, with Argon2id's data-dependent block addressing
 * in the second half of pass 0 and all of pass 1.
 *
 * This client puts the fill on the TPU:
 *
 *   - the whole matrix is one int32 tensor, [lanes * m * 256]; each 1 KiB
 *     block is 256 int32 (u64 as a lo/hi pair, so the block is already
 *     little-endian on the wire),
 *   - a stablehlo.while walks the 38,912 block computations of a challenge,
 *     gathering the previous and reference block and scattering the new one,
 *   - lanes are *independent nonces*: all of them follow the same schedule, so
 *     one iteration fills one block for every lane at once,
 *   - H0 and the first two blocks come from the host (they are BLAKE2b), and
 *     the final 1 KiB BLAKE2b tag is host-side too (32 bytes per nonce).
 *
 * The graph does not depend on the challenge or the nonce - only on (m, t,
 * lanes) - so one compiled executable serves every challenge on that device,
 * unlike the HashX client, where the program is derived from the challenge.
 *
 * The u64 arithmetic is the same i32-pair construction as hashx_pjrt.c; the
 * plugin's UNSIGNED comparisons are avoided everywhere (see sha256_pjrt.c).
 *
 * Modes:
 *   --hosttest            host Argon2id against fixed vectors (no TPU)
 *   --unit                u64 primitive self-check (no TPU)
 *   --emit                print the StableHLO module
 *   --dump DIR            module + inputs + expected values, for a CPU XLA run
 *   --selftest            TPU: fill vs the host reference (several sizes)
 *   (default)             benchmark nonces, print hashes/s
 */

#include <dlfcn.h>
#include <errno.h>
#include <glob.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "pjrt_c_api.h"
#include "pjrt_common.h"

/* ==================================================================== */
/* HLO text buffer, SSA names, shapes                                   */
/* ==================================================================== */

#define MAX_VALUES 900000

typedef struct {
    char *s;
    size_t len, cap;
} buf_t;

static buf_t g_b;
static int g_lanes;
static int g_m, g_t;
static int g_dump_mem;   /* --dump-all: return the whole matrix (debug) */
static int g_filltest;   /* --filltest: one block fill, no loop (debug) */
static int g_filltest1;  /* --filltest1: one BLAKE2 round, for a cheap check */
static int g_rounds = 8; /* --rounds N: truncate the fill, for a cheap check */
static int g_sched;      /* --sched: the schedule without the fill (debug) */
static int g_host_sched; /* --host-sched: the host reference without the fill */
static char SH_VEC[64], SH_BLK[64], SH_IDX[64], SH_MEM[64], SH_PRED[64],
    SH_PREDB[64];

static int v_known[MAX_VALUES];
static uint32_t v_val[MAX_VALUES];
static int v_n;

static void bprintf(buf_t *b, const char *fmt, ...)
{
    va_list ap;
    int need;

    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) {
        return;
    }
    if (b->len + (size_t)need + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 65536;

        while (cap < b->len + (size_t)need + 1) {
            cap *= 2;
        }
        b->s = realloc(b->s, cap);
        if (b->s == NULL) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        b->cap = cap;
    }
    va_start(ap, fmt);
    vsnprintf(b->s + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)need;
}

static void build_reset(int lanes, int m, int t)
{
    if (g_b.cap == 0) {
        g_b.cap = 65536;
        g_b.s = malloc(g_b.cap);
        if (g_b.s == NULL) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
    }
    g_b.len = 0;
    g_b.s[0] = '\0';
    g_lanes = lanes;
    g_m = m;
    g_t = t;
    v_n = 0;
    snprintf(SH_VEC, sizeof SH_VEC, "tensor<%dxi32>", lanes);
    snprintf(SH_BLK, sizeof SH_BLK, "tensor<%dx256xi32>", lanes);
    snprintf(SH_IDX, sizeof SH_IDX, "tensor<%dx1xi32>", lanes);
    snprintf(SH_MEM, sizeof SH_MEM, "tensor<" "%d" "x256xi32>", lanes * m);
    snprintf(SH_PRED, sizeof SH_PRED, "tensor<%dxi1>", lanes);
    snprintf(SH_PREDB, sizeof SH_PREDB, "tensor<%dx256xi1>", lanes);
}

static int newval(int known, uint32_t value)
{
    int id;

    if (v_n >= MAX_VALUES) {
        fprintf(stderr, "too many SSA values\n");
        exit(1);
    }
    id = v_n++;
    v_known[id] = known;
    v_val[id] = value;
    return id;
}

typedef struct {
    uint32_t v;
    int id;
} cconst;
static cconst g_cc[256];
static int g_ncc;

/* A [lanes] i32 vector holding one broadcast constant. */
static int c32(uint32_t v)
{
    int sid, vid;

    for (int i = 0; i < g_ncc; i++) {
        if (g_cc[i].v == v) {
            return g_cc[i].id;
        }
    }
    sid = newval(1, v);
    bprintf(&g_b, "    %%v%d = stablehlo.constant dense<%d> : tensor<i32>\n",
            sid, (int32_t)v);
    vid = newval(1, v);
    bprintf(&g_b, "    %%v%d = stablehlo.broadcast_in_dim %%v%d, dims = [] "
                  ": (tensor<i32>) -> %s\n", vid, sid, SH_VEC);
    if (g_ncc < (int)(sizeof g_cc / sizeof g_cc[0])) {
        g_cc[g_ncc].v = v;
        g_cc[g_ncc].id = vid;
        g_ncc++;
    }
    return vid;
}

/* A scalar tensor<i32> constant. */
static int sc32(uint32_t v)
{
    int r = newval(1, v);

    bprintf(&g_b,
            "    %%v%d = stablehlo.constant dense<%d> : tensor<i32>\n", r,
            (int32_t)v);
    return r;
}

enum bin_op { OP_ADD, OP_SUB, OP_MUL, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR };

static const char *bin_names[] = {"add",  "subtract", "multiply", "and",
                                  "or",   "xor",      "shift_left",
                                  "shift_right_logical"};

static int bin(enum bin_op op, int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.%s %%v%d, %%v%d : %s\n", r,
            bin_names[op], a, b, SH_VEC);
    if (v_known[a] && v_known[b]) {
        uint32_t x = v_val[a], y = v_val[b], z = 0;

        switch (op) {
        case OP_ADD: z = x + y; break;
        case OP_SUB: z = x - y; break;
        case OP_MUL: z = x * y; break;
        case OP_AND: z = x & y; break;
        case OP_OR: z = x | y; break;
        case OP_XOR: z = x ^ y; break;
        case OP_SHL: z = x << (y & 31u); break;
        case OP_SHR: z = x >> (y & 31u); break;
        }
        v_known[r] = 1;
        v_val[r] = z;
    }
    return r;
}


static int cmp_cc(const char *dir, int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b,
            "    %%v%d = \"stablehlo.compare\"(%%v%d, %%v%d) "
            "{comparison_direction = #stablehlo<comparison_direction %s>, "
            "compare_type = #stablehlo<comparison_type SIGNED>} : "
            "(%s, %s) -> %s\n",
            r, a, b, dir, SH_VEC, SH_VEC, SH_PRED);
    if (v_known[a] && v_known[b]) {
        int32_t x = (int32_t)v_val[a], y = (int32_t)v_val[b], ok = 0;

        if (strcmp(dir, "EQ") == 0) ok = x == y;
        else if (strcmp(dir, "NE") == 0) ok = x != y;
        else if (strcmp(dir, "LT") == 0) ok = x < y;
        else if (strcmp(dir, "LE") == 0) ok = x <= y;
        else if (strcmp(dir, "GT") == 0) ok = x > y;
        else if (strcmp(dir, "GE") == 0) ok = x >= y;
        v_known[r] = 1;
        v_val[r] = ok ? 1u : 0u;
    }
    return r;
}

static int scmp_cc(const char *dir, int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b,
            "    %%v%d = \"stablehlo.compare\"(%%v%d, %%v%d) "
            "{comparison_direction = #stablehlo<comparison_direction %s>, "
            "compare_type = #stablehlo<comparison_type SIGNED>} : "
            "(tensor<i32>, tensor<i32>) -> tensor<i1>\n",
            r, a, b, dir);
    if (v_known[a] && v_known[b]) {
        int32_t x = (int32_t)v_val[a], y = (int32_t)v_val[b], ok = 0;

        if (strcmp(dir, "EQ") == 0) ok = x == y;
        else if (strcmp(dir, "NE") == 0) ok = x != y;
        else if (strcmp(dir, "LT") == 0) ok = x < y;
        else if (strcmp(dir, "LE") == 0) ok = x <= y;
        else if (strcmp(dir, "GT") == 0) ok = x > y;
        else if (strcmp(dir, "GE") == 0) ok = x >= y;
        v_known[r] = 1;
        v_val[r] = ok ? 1u : 0u;
    }
    return r;
}

static int cv(int a, const char *from, const char *to)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = \"stablehlo.convert\"(%%v%d) : "
                  "(tensor<%dx%s>) -> tensor<%dx%s>\n",
            r, a, g_lanes, from, g_lanes, to);
    if (v_known[a]) {
        v_known[r] = 1;
        v_val[r] = v_val[a] ? 1u : 0u;
    }
    return r;
}

static int sel32(int pred, int t, int f)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = \"stablehlo.select\"(%%v%d, %%v%d, %%v%d) : "
                  "(%s, %s, %s) -> %s\n",
            r, pred, t, f, SH_PRED, SH_VEC, SH_VEC, SH_VEC);
    if (v_known[pred] && v_known[t] && v_known[f]) {
        v_known[r] = 1;
        v_val[r] = v_val[pred] ? v_val[t] : v_val[f];
    }
    return r;
}

static int andp(int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.and %%v%d, %%v%d : %s\n", r, a, b,
            SH_PRED);
    if (v_known[a] && v_known[b]) {
        v_known[r] = 1;
        v_val[r] = v_val[a] & v_val[b];
    }
    return r;
}


/* Extract lane 0 of a [lanes] vector as a scalar. */
static int v2s(int id)
{
    int s = newval(0, 0), r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.slice %%v%d[0:1] : (%s) -> "
                  "tensor<1xi32>\n", s, id, SH_VEC);
    bprintf(&g_b, "    %%v%d = stablehlo.reshape %%v%d : (tensor<1xi32>) -> "
                  "tensor<i32>\n", r, s);
    return r;
}

static int reshape_vec_idx(int id)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.reshape %%v%d : (%s) -> %s\n", r, id,
            SH_VEC, SH_IDX);
    return r;
}

static int reshape_vec_col(int id)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.reshape %%v%d : (%s) -> "
                  "tensor<%dx1xi32>\n", r, id, SH_VEC, g_lanes);
    return r;
}

/* ==================================================================== */
/* block-level ops                                                      */
/* ==================================================================== */

static int bxor(int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.xor %%v%d, %%v%d : %s\n", r, a, b,
            SH_BLK);
    return r;
}

static int bcast_pred(int pred)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.broadcast_in_dim %%v%d, dims = [0] : "
                  "(%s) -> %s\n", r, pred, SH_PRED, SH_PREDB);
    return r;
}

static int bsel(int predb, int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = \"stablehlo.select\"(%%v%d, %%v%d, %%v%d) : "
                  "(%s, %s, %s) -> %s\n",
            r, predb, a, b, SH_PREDB, SH_BLK, SH_BLK, SH_BLK);
    return r;
}

/* gather one 256-word block per lane: indices [lanes] -> [lanes,256] */
static int gather_block(int mem, int idx_vec)
{
    int idx = reshape_vec_idx(idx_vec);
    int r = newval(0, 0);

    bprintf(&g_b,
            "    %%v%d = \"stablehlo.gather\"(%%v%d, %%v%d) {\n"
            "        dimension_numbers = #stablehlo.gather<offset_dims = [1], "
            "collapsed_slice_dims = [0], operand_batching_dims = [], "
            "start_indices_batching_dims = [], start_index_map = [0], "
            "index_vector_dim = 1>,\n"
            "        slice_sizes = array<i64: 1, 256>,\n"
            "        indices_are_sorted = false\n"
            "    } : (%s, %s) -> %s\n",
            r, mem, idx, SH_MEM, SH_IDX, SH_BLK);
    return r;
}

static int scatter_block(int mem, int idx_vec, int upd)
{
    int idx = reshape_vec_idx(idx_vec);
    int r = newval(0, 0);
    int a1 = newval(0, 0), a2 = newval(0, 0);

    bprintf(&g_b,
            "    %%v%d = \"stablehlo.scatter\"(%%v%d, %%v%d, %%v%d) ({\n"
            "    ^bb0(%%v%d: tensor<i32>, %%v%d: tensor<i32>):\n"
            "      stablehlo.return %%v%d : tensor<i32>\n"
            "    }) {\n"
            "        scatter_dimension_numbers = "
            "#stablehlo.scatter<update_window_dims = [1], "
            "inserted_window_dims = [0], input_batching_dims = [], "
            "scatter_indices_batching_dims = [], "
            "scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,\n"
            "        indices_are_sorted = false,\n"
            "        unique_indices = true\n"
            "    } : (%s, %s, %s) -> %s\n",
            r, mem, idx, upd, a1, a2, a2, SH_MEM, SH_IDX, SH_BLK, SH_MEM);
    return r;
}

/* one 32-bit column of a [lanes,256] block as a [lanes] vector */
static int slice_col(int blk, int col)
{
    int s = newval(0, 0), r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.slice %%v%d[0:%d, %d:%d] : (%s) -> "
                  "tensor<%dx1xi32>\n", s, blk, g_lanes, col, col + 1, SH_BLK,
            g_lanes);
    bprintf(&g_b, "    %%v%d = stablehlo.reshape %%v%d : (tensor<%dx1xi32>) -> "
                  "%s\n", r, s, g_lanes, SH_VEC);
    return r;
}

static void unpack_block(int blk, int out[256])
{
    for (int k = 0; k < 256; k++) {
        out[k] = slice_col(blk, k);
    }
}

static int pack_block(const int ids[256])
{
    int col[256];
    int r = newval(0, 0);

    for (int k = 0; k < 256; k++) {
        col[k] = reshape_vec_col(ids[k]);
    }
    bprintf(&g_b, "    %%v%d = stablehlo.concatenate ", r);
    for (int k = 0; k < 256; k++) {
        bprintf(&g_b, "%%v%d%s", col[k], k == 255 ? "" : ", ");
    }
    bprintf(&g_b, ", dim = 1 : (");
    for (int k = 0; k < 256; k++) {
        bprintf(&g_b, "tensor<%dx1xi32>%s", g_lanes, k == 255 ? "" : ", ");
    }
    bprintf(&g_b, ") -> %s\n", SH_BLK);
    return r;
}

/* one i32 from a rank-1 table at a scalar index, broadcast to [lanes] */
static int slice_word_bcast(int tab, int idx_scalar)
{
    int sl = newval(0, 0), sc = newval(0, 0), r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.dynamic_slice %%v%d, %%v%d, "
                  "sizes = [1] : (tensor<%dxi32>, tensor<i32>) -> "
                  "tensor<1xi32>\n", sl, tab, idx_scalar, g_m / 2);
    bprintf(&g_b, "    %%v%d = stablehlo.reshape %%v%d : (tensor<1xi32>) -> "
                  "tensor<i32>\n", sc, sl);
    bprintf(&g_b, "    %%v%d = stablehlo.broadcast_in_dim %%v%d, dims = [] : "
                  "(tensor<i32>) -> %s\n", r, sc, SH_VEC);
    return r;
}


/* ==================================================================== */
/* u64 as an i32 pair                                                   */
/* ==================================================================== */

typedef struct {
    int lo, hi;
} u64v;

static u64v pair(int lo, int hi)
{
    u64v r;

    r.lo = lo;
    r.hi = hi;
    return r;
}

static u64v const64(uint64_t v)
{
    return pair(c32((uint32_t)v), c32((uint32_t)(v >> 32)));
}

static u64v xor64(u64v a, u64v b)
{
    return pair(bin(OP_XOR, a.lo, b.lo), bin(OP_XOR, a.hi, b.hi));
}

static int not32(int x)
{
    return bin(OP_XOR, x, c32(0xffffffffu));
}

/* carry-out without an unsigned compare; pure bitwise (see sha256_pjrt.c) */
static u64v add64(u64v a, u64v b)
{
    int lo = bin(OP_ADD, a.lo, b.lo);
    int both = bin(OP_AND, a.lo, b.lo);
    int either = bin(OP_OR, a.lo, b.lo);
    int carry_src = bin(OP_AND, either, not32(lo));
    int carry = bin(OP_SHR, bin(OP_OR, both, carry_src), c32(31));
    int hi = bin(OP_ADD, bin(OP_ADD, a.hi, b.hi), carry);

    return pair(lo, hi);
}

static u64v neg64(u64v a)
{
    int lo = bin(OP_SUB, c32(0), a.lo);
    int iszero = cmp_cc("EQ", a.lo, c32(0));
    int inc = cv(iszero, "i1", "i32");
    int hi = bin(OP_ADD, not32(a.hi), inc);

    return pair(lo, hi);
}

static u64v sub64(u64v a, u64v b)
{
    return add64(a, neg64(b));
}

/* high 32 bits of a 32x32 unsigned product, via 16-bit halves */
static int mulhi32(int x, int y)
{
    int xl = bin(OP_AND, x, c32(0xffffu));
    int xh = bin(OP_SHR, x, c32(16));
    int yl = bin(OP_AND, y, c32(0xffffu));
    int yh = bin(OP_SHR, y, c32(16));
    int ll = bin(OP_MUL, xl, yl);
    int hl = bin(OP_MUL, xh, yl);
    int lh = bin(OP_MUL, xl, yh);
    int hh = bin(OP_MUL, xh, yh);
    int mid = bin(OP_ADD, bin(OP_SHR, ll, c32(16)),
                  bin(OP_ADD, bin(OP_AND, hl, c32(0xffffu)),
                      bin(OP_AND, lh, c32(0xffffu))));
    int hi = bin(OP_ADD, hh,
                 bin(OP_ADD, bin(OP_SHR, hl, c32(16)),
                     bin(OP_SHR, lh, c32(16))));

    return bin(OP_ADD, hi, bin(OP_SHR, mid, c32(16)));
}

static u64v mul64(u64v a, u64v b)
{
    int lo = bin(OP_MUL, a.lo, b.lo);
    int hi = bin(OP_ADD, bin(OP_MUL, a.lo, b.hi), bin(OP_MUL, a.hi, b.lo));

    return pair(lo, bin(OP_ADD, hi, mulhi32(a.lo, b.lo)));
}

static u64v rotl64(u64v x, unsigned n)
{
    int lo, hi;

    n &= 63u;
    if (n == 0) {
        return x;
    }
    if (n == 32) {
        return pair(x.hi, x.lo);
    }
    if (n < 32) {
        lo = bin(OP_OR, bin(OP_SHL, x.lo, c32(n)),
                 bin(OP_SHR, x.hi, c32(32 - n)));
        hi = bin(OP_OR, bin(OP_SHL, x.hi, c32(n)),
                 bin(OP_SHR, x.lo, c32(32 - n)));
        return pair(lo, hi);
    }
    {
        unsigned s = 64 - n;

        lo = bin(OP_OR, bin(OP_SHR, x.lo, c32(s)),
                 bin(OP_SHL, x.hi, c32(32 - s)));
        hi = bin(OP_OR, bin(OP_SHR, x.hi, c32(s)),
                 bin(OP_SHL, x.lo, c32(32 - s)));
    }
    return pair(lo, hi);
}

static u64v rotr64(u64v x, unsigned n)
{
    return rotl64(x, (64u - (n & 63u)) & 63u);
}

/* ==================================================================== */
/* Argon2id block function, on 128 u64 SSA values                       */
/* ==================================================================== */

/* fBlaMka(a,b) = a + b + 2*(u32)a*(u32)b */
static u64v blamka(u64v a, u64v b)
{
    int plo = bin(OP_MUL, a.lo, b.lo);
    int phi = mulhi32(a.lo, b.lo);
    u64v two = pair(bin(OP_SHL, plo, c32(1)),
                    bin(OP_OR, bin(OP_SHL, phi, c32(1)),
                        bin(OP_SHR, plo, c32(31))));

    return add64(add64(a, b), two);
}

static void ag(u64v *a, u64v *b, u64v *c, u64v *d)
{
    *a = blamka(*a, *b);
    *d = rotr64(xor64(*d, *a), 32);
    *c = blamka(*c, *d);
    *b = rotr64(xor64(*b, *c), 24);
    *a = blamka(*a, *b);
    *d = rotr64(xor64(*d, *a), 16);
    *c = blamka(*c, *d);
    *b = rotr64(xor64(*b, *c), 63);
}

static void round16(u64v *v)
{
    ag(&v[0], &v[4], &v[8], &v[12]);
    ag(&v[1], &v[5], &v[9], &v[13]);
    ag(&v[2], &v[6], &v[10], &v[14]);
    ag(&v[3], &v[7], &v[11], &v[15]);
    ag(&v[0], &v[5], &v[10], &v[15]);
    ag(&v[1], &v[6], &v[11], &v[12]);
    ag(&v[2], &v[7], &v[8], &v[13]);
    ag(&v[3], &v[4], &v[9], &v[14]);
}

/*
 * fill_block:  R = ref ^ prev;  Q = R (^ old when with_xor);
 *              16 BLAKE2 rounds over R;  next = Q ^ R.
 * `pred` is a [lanes] i1 that is true on passes > 0.
 */
static void fill_hlo(const u64v prev[128], const u64v ref[128],
                     const u64v curr[128], int pred, u64v out[128])
{
    u64v q[128], v[128];

    for (int k = 0; k < 128; k++) {
        u64v r = xor64(prev[k], ref[k]);
        u64v qq = pair(sel32(pred, bin(OP_XOR, r.lo, curr[k].lo), r.lo),
                       sel32(pred, bin(OP_XOR, r.hi, curr[k].hi), r.hi));

        q[k] = qq;
        v[k] = r;
    }
    for (int r = 0; r < g_rounds; r++) {
        round16(&v[16 * r]);
    }
    for (int r = 0; r < g_rounds; r++) {
        u64v t16[16];

        for (int j = 0; j < 8; j++) {
            t16[2 * j] = v[2 * r + 16 * j];
            t16[2 * j + 1] = v[2 * r + 16 * j + 1];
        }
        round16(t16);
        for (int j = 0; j < 8; j++) {
            v[2 * r + 16 * j] = t16[2 * j];
            v[2 * r + 16 * j + 1] = t16[2 * j + 1];
        }
    }
    for (int k = 0; k < 128; k++) {
        out[k] = xor64(q[k], v[k]);
    }
}

/* ==================================================================== */
/* host BLAKE2b, and Argon2id as a reference                            */
/* ==================================================================== */

static const uint64_t b2b_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

static const uint8_t b2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

typedef struct {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t buf[128];
    size_t buflen;
    size_t outlen;
} b2b_t;

static uint64_t b2b_rotr(uint64_t x, int n)
{
    return (x >> n) | (x << (64 - n));
}

static void b2b_compress(b2b_t *c, const uint8_t *blk, int last)
{
    uint64_t m[16], v[16];

    for (int i = 0; i < 16; i++) {
        uint64_t x = 0;

        for (int j = 7; j >= 0; j--) {
            x = (x << 8) | blk[8 * i + j];
        }
        m[i] = x;
    }
    for (int i = 0; i < 8; i++) {
        v[i] = c->h[i];
        v[i + 8] = b2b_iv[i];
    }
    v[12] ^= c->t[0];
    v[13] ^= c->t[1];
    if (last) {
        v[14] = ~v[14];
    }
#define B2B_G(a, b, cc, d, x, y)                                             \
    do {                                                                     \
        v[a] = v[a] + v[b] + x;                                              \
        v[d] = b2b_rotr(v[d] ^ v[a], 32);                                    \
        v[cc] = v[cc] + v[d];                                                \
        v[b] = b2b_rotr(v[b] ^ v[cc], 24);                                   \
        v[a] = v[a] + v[b] + y;                                              \
        v[d] = b2b_rotr(v[d] ^ v[a], 16);                                    \
        v[cc] = v[cc] + v[d];                                                \
        v[b] = b2b_rotr(v[b] ^ v[cc], 63);                                   \
    } while (0)
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = b2b_sigma[r];

        B2B_G(0, 4, 8, 12, m[s[0]], m[s[1]]);
        B2B_G(1, 5, 9, 13, m[s[2]], m[s[3]]);
        B2B_G(2, 6, 10, 14, m[s[4]], m[s[5]]);
        B2B_G(3, 7, 11, 15, m[s[6]], m[s[7]]);
        B2B_G(0, 5, 10, 15, m[s[8]], m[s[9]]);
        B2B_G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        B2B_G(2, 7, 8, 13, m[s[12]], m[s[13]]);
        B2B_G(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
#undef B2B_G
    for (int i = 0; i < 8; i++) {
        c->h[i] ^= v[i] ^ v[i + 8];
    }
}

static void b2b_init(b2b_t *c, size_t outlen)
{
    memset(c, 0, sizeof *c);
    memcpy(c->h, b2b_iv, sizeof c->h);
    c->h[0] ^= 0x01010000u ^ (uint64_t)outlen;
    c->outlen = outlen;
}

static void b2b_tincr(b2b_t *c, uint64_t inc)
{
    c->t[0] += inc;
    if (c->t[0] < inc) {
        c->t[1]++;
    }
}

static void b2b_update(b2b_t *c, const void *data, size_t len)
{
    const uint8_t *p = data;

    while (len > 0) {
        if (c->buflen == 128) {
            b2b_tincr(c, 128);
            b2b_compress(c, c->buf, 0);
            c->buflen = 0;
        }
        if (c->buflen == 0 && len >= 128) {
            b2b_tincr(c, 128);
            b2b_compress(c, p, 0);
            p += 128;
            len -= 128;
            continue;
        }
        {
            size_t take = 128 - c->buflen;

            if (take > len) {
                take = len;
            }
            memcpy(c->buf + c->buflen, p, take);
            c->buflen += take;
            p += take;
            len -= take;
        }
    }
}

static void b2b_final(b2b_t *c, uint8_t *out)
{
    uint8_t tail[128];

    b2b_tincr(c, (uint64_t)c->buflen);
    memcpy(tail, c->buf, c->buflen);
    memset(tail + c->buflen, 0, sizeof tail - c->buflen);
    b2b_compress(c, tail, 1);
    for (size_t i = 0; i < c->outlen; i++) {
        out[i] = (uint8_t)(c->h[i / 8] >> (8 * (i % 8)));
    }
}

static void b2b_buf(uint8_t *out, size_t outlen, const void *in, size_t inlen)
{
    b2b_t c;

    b2b_init(&c, outlen);
    b2b_update(&c, in, inlen);
    b2b_final(&c, out);
}

/* Argon2's H': BLAKE2b with the output length prefixed, chained past 64. */
static void blake2b_long(uint8_t *out, size_t outlen, const uint8_t *in,
                         size_t inlen)
{
    uint8_t len_le[4];
    b2b_t c;

    len_le[0] = (uint8_t)outlen;
    len_le[1] = (uint8_t)(outlen >> 8);
    len_le[2] = (uint8_t)(outlen >> 16);
    len_le[3] = (uint8_t)(outlen >> 24);
    if (outlen <= 64) {
        b2b_init(&c, outlen);
        b2b_update(&c, len_le, 4);
        b2b_update(&c, in, inlen);
        b2b_final(&c, out);
        return;
    }
    {
        uint8_t obuf[64], ibuf[64];
        size_t togo = outlen - 32;

        b2b_init(&c, 64);
        b2b_update(&c, len_le, 4);
        b2b_update(&c, in, inlen);
        b2b_final(&c, obuf);
        memcpy(out, obuf, 32);
        out += 32;
        while (togo > 64) {
            memcpy(ibuf, obuf, 64);
            b2b_buf(obuf, 64, ibuf, 64);
            memcpy(out, obuf, 32);
            out += 32;
            togo -= 32;
        }
        memcpy(ibuf, obuf, 64);
        b2b_buf(out, togo, ibuf, 64);
    }
}

typedef struct {
    uint64_t v[128];
} a2block;

static void put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void block_to_bytes(const a2block *b, uint8_t out[1024])
{
    for (int i = 0; i < 128; i++) {
        uint64_t w = b->v[i];

        for (int j = 0; j < 8; j++) {
            out[8 * i + j] = (uint8_t)(w >> (8 * j));
        }
    }
}

static void bytes_to_block(const uint8_t in[1024], a2block *b)
{
    for (int i = 0; i < 128; i++) {
        uint64_t w = 0;

        for (int j = 7; j >= 0; j--) {
            w = (w << 8) | in[8 * i + j];
        }
        b->v[i] = w;
    }
}

static uint64_t a2_rotr64(uint64_t x, int n)
{
    return (x >> n) | (x << (64 - n));
}

static uint64_t a2_blamka(uint64_t x, uint64_t y)
{
    uint64_t xy = (x & 0xffffffffULL) * (y & 0xffffffffULL);

    return x + y + 2 * xy;
}

static void a2_g(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d)
{
    *a = a2_blamka(*a, *b);
    *d = a2_rotr64(*d ^ *a, 32);
    *c = a2_blamka(*c, *d);
    *b = a2_rotr64(*b ^ *c, 24);
    *a = a2_blamka(*a, *b);
    *d = a2_rotr64(*d ^ *a, 16);
    *c = a2_blamka(*c, *d);
    *b = a2_rotr64(*b ^ *c, 63);
}

static void a2_round(uint64_t *v)
{
    a2_g(&v[0], &v[4], &v[8], &v[12]);
    a2_g(&v[1], &v[5], &v[9], &v[13]);
    a2_g(&v[2], &v[6], &v[10], &v[14]);
    a2_g(&v[3], &v[7], &v[11], &v[15]);
    a2_g(&v[0], &v[5], &v[10], &v[15]);
    a2_g(&v[1], &v[6], &v[11], &v[12]);
    a2_g(&v[2], &v[7], &v[8], &v[13]);
    a2_g(&v[3], &v[4], &v[9], &v[14]);
}

static void a2_fill_block(const a2block *prev, const a2block *ref,
                          a2block *next, int with_xor)
{
    uint64_t q[128], v[128];

    for (int k = 0; k < 128; k++) {
        uint64_t r = ref->v[k] ^ prev->v[k];

        v[k] = r;
        q[k] = with_xor ? (r ^ next->v[k]) : r;
    }
    for (int k = 0; k < 8; k++) {
        a2_round(&v[16 * k]);
    }
    for (int k = 0; k < 8; k++) {
        uint64_t t[16];

        for (int j = 0; j < 8; j++) {
            t[2 * j] = v[2 * k + 16 * j];
            t[2 * j + 1] = v[2 * k + 16 * j + 1];
        }
        a2_round(t);
        for (int j = 0; j < 8; j++) {
            v[2 * k + 16 * j] = t[2 * j];
            v[2 * k + 16 * j + 1] = t[2 * j + 1];
        }
    }
    for (int k = 0; k < 128; k++) {
        next->v[k] = g_host_sched ? q[k] : (q[k] ^ v[k]);
    }
}

/* pass/slice/index -> reference area and start position (same lane, p=1) */
static void a2_index_alpha(uint32_t pass, uint32_t slice, uint32_t index,
                           uint32_t m, uint32_t seg, uint32_t pseudo,
                           uint32_t *area_out, uint32_t *start_out)
{
    uint32_t area, start;
    uint64_t rel;

    if (pass == 0) {
        area = (slice == 0) ? (index - 1) : (slice * seg + index - 1);
    } else {
        area = m - seg + index - 1;
    }
    rel = pseudo;
    rel = (rel * rel) >> 32;
    rel = area - 1 - ((area * rel) >> 32);
    start = (pass == 0) ? 0 : (((slice + 1) * seg) % m);
    *area_out = area;
    *start_out = (uint32_t)((uint64_t)start + rel) % m;
}

static void a2_next_addresses(a2block *address, a2block *input)
{
    static const a2block zero;

    input->v[6]++;
    a2_fill_block(&zero, input, address, 0);
    a2_fill_block(&zero, address, address, 0);
}

/*
 * Host reference: Argon2id, lanes=1, type=id, version 0x13.
 * Returns the tag and, if out_mem != NULL, the final block; if addr != NULL,
 * the low words of the data-independent address sequence for pass 0.
 */
static int a2_ref_mem(uint32_t m, uint32_t t, const uint8_t *pwd,
                      size_t pwdlen, const uint8_t salt[8], size_t saltlen,
                      uint8_t tag[32], a2block **mem_out)
{
    uint32_t seg = m / 4;
    a2block *mem;
    uint8_t h0[64], seed[72], bytes[1024];
    b2b_t c;

    if (m < 8 || m % 4 != 0 || t < 1) {
        return -1;
    }
    mem = calloc(m, sizeof *mem);
    if (mem == NULL) {
        return -1;
    }

    /* H0 = BLAKE2b-64(p, T, m, t, v, y, |P|, P, |S|, S, 0, 0) */
    b2b_init(&c, 64);
    {
        uint8_t le[4];

        put32(le, 1); b2b_update(&c, le, 4);          /* lanes */
        put32(le, 32); b2b_update(&c, le, 4);         /* tag length */
        put32(le, m); b2b_update(&c, le, 4);
        put32(le, t); b2b_update(&c, le, 4);
        put32(le, 0x13); b2b_update(&c, le, 4);
        put32(le, 2); b2b_update(&c, le, 4);          /* Argon2id */
        put32(le, (uint32_t)pwdlen); b2b_update(&c, le, 4);
        b2b_update(&c, pwd, pwdlen);
        put32(le, (uint32_t)saltlen); b2b_update(&c, le, 4);
        b2b_update(&c, salt, saltlen);
        put32(le, 0); b2b_update(&c, le, 4);          /* secret */
        put32(le, 0); b2b_update(&c, le, 4);          /* associated data */
    }
    b2b_final(&c, h0);

    memcpy(seed, h0, 64);
    for (int i = 0; i < 2; i++) {
        put32(seed + 64, (uint32_t)i);
        put32(seed + 68, 0);                          /* lane */
        blake2b_long(bytes, 1024, seed, 72);
        bytes_to_block(bytes, &mem[i]);
    }

    for (uint32_t pass = 0; pass < t; pass++) {
        for (uint32_t slice = 0; slice < 4; slice++) {
            int di = (pass == 0) && (slice < 2);
            a2block input, address;
            uint32_t start_index = (pass == 0 && slice == 0) ? 2u : 0u;
            uint32_t curr = slice * seg + start_index;
            uint32_t prev = (curr % m == 0) ? (curr + m - 1) : (curr - 1);

            memset(&input, 0, sizeof input);
            memset(&address, 0, sizeof address);
            if (di) {
                input.v[0] = pass;
                input.v[1] = 0;
                input.v[2] = slice;
                input.v[3] = m;
                input.v[4] = t;
                input.v[5] = 2;
                if (start_index == 2) {
                    a2_next_addresses(&address, &input);
                }
            }
            for (uint32_t i = start_index; i < seg; i++, curr++, prev++) {
                uint32_t pseudo32, area, startpos, refidx;
                uint64_t pseudo;

                if (curr % m == 1) {
                    prev = curr - 1;
                }
                if (di) {
                    if (i % 128 == 0) {
                        a2_next_addresses(&address, &input);
                    }
                    pseudo = address.v[i % 128];
                } else {
                    pseudo = mem[prev].v[0];
                }
                pseudo32 = (uint32_t)pseudo;
                a2_index_alpha(pass, slice, i, m, seg, pseudo32, &area,
                               &startpos);
                refidx = startpos;   /* area is implied by the construction */
                (void)area;
                a2_fill_block(&mem[prev], &mem[refidx], &mem[curr],
                              pass > 0);
                if (getenv("A2TRACE") != NULL) {
                    fprintf(stderr, "tr %u %u %u prev=%u ref=%u w=%016llx\n",
                            pass, slice, i, prev, refidx,
                            (unsigned long long)mem[curr].v[0]);
                }
            }
        }
    }

    /* finalize: the last block, then H'(block, 32) */
    block_to_bytes(&mem[m - 1], bytes);
    blake2b_long(tag, 32, bytes, 1024);
    *mem_out = mem;
    return 0;
}

static int a2_ref(uint32_t m, uint32_t t, const uint8_t *pwd, size_t pwdlen,
                  const uint8_t salt[8], size_t saltlen, uint8_t tag[32],
                  a2block *out_last)
{
    a2block *mem = NULL;
    int rc;

    rc = a2_ref_mem(m, t, pwd, pwdlen, salt, saltlen, tag, &mem);
    if (rc != 0) {
        return rc;
    }
    if (out_last != NULL) {
        *out_last = mem[m - 1];
    }
    free(mem);
    return 0;
}

/* ==================================================================== */
/* module builder                                                       */
/* ==================================================================== */

static char *build_module(int lanes, int m, int t)
{
    int seg = m / 4, half = m / 2;
    int a0, a1, a2v, zc, mem0, lane, mub, base, mem1, base1, mem2, i0, p0;
    int res, cb_m, cb_i, cb_p, ps, tcs, cond, bb_m, bb_i, bb_p;

    build_reset(lanes, m, t);
    g_ncc = 0;

    a0 = newval(0, 0);
    a1 = newval(0, 0);
    a2v = newval(0, 0);
    bprintf(&g_b,
            "module {\n"
            "  func.func @main(%%v%d: %s, %%v%d: %s, %%v%d: tensor<%dxi32>) "
            "-> %s {\n",
            a0, SH_BLK, a1, SH_BLK, a2v, half, g_dump_mem ? SH_MEM : SH_BLK);

    if (g_filltest1) {
        g_filltest = 1;
    }
    if (g_filltest) {
        int lo[256], hi[256];
        u64v va[128], vb[128], vc[128], out[128], rcur[128];
        int pred;

        unpack_block(a0, lo);
        unpack_block(a1, hi);
        for (int k = 0; k < 128; k++) {
            va[k] = pair(lo[2 * k], lo[2 * k + 1]);
            vb[k] = pair(hi[2 * k], hi[2 * k + 1]);
        }
        memset(vc, 0, sizeof vc);
        for (int k = 0; k < 128; k++) vc[k] = pair(c32(0), c32(0));
        pred = andp(cmp_cc("GT", c32(0), c32(0)), cmp_cc("GT", c32(0), c32(0)));
        if (g_filltest1) {
            /* one BLAKE2 round only, cheap enough for a CPU XLA check:
             * 1 = column round 0, 2 = row round 0 (strided word selection) */
            u64v r0[16], v0[16];
            int idx[16];

            for (int k = 0; k < 128; k++) out[k] = pair(c32(0), c32(0));
            for (int k = 0; k < 128; k++) rcur[k] = xor64(va[k], vb[k]);
            for (int k = 0; k < 16; k++) idx[k] = k;
            if (g_filltest1 == 2) {
                for (int j = 0; j < 8; j++) {
                    idx[2 * j] = 16 * j;
                    idx[2 * j + 1] = 16 * j + 1;
                }
            }
            for (int k = 0; k < 16; k++) {
                r0[k] = rcur[idx[k]];
                v0[k] = r0[k];
            }
            round16(v0);
            for (int k = 0; k < 16; k++) {
                out[idx[k]] = xor64(r0[k], v0[k]);
            }
        } else {
            fill_hlo(va, vb, vc, pred, out);
        }
        for (int k = 0; k < 256; k++) {
            lo[k] = (k & 1) ? out[k / 2].hi : out[k / 2].lo;
        }
        {
            int blk = pack_block(lo);

            bprintf(&g_b, "    return %%v%d : %s\n  }\n}\n", blk, SH_BLK);
        }
        return g_b.s;
    }

    zc = sc32(0);
    mem0 = newval(0, 0);
    bprintf(&g_b, "    %%v%d = stablehlo.broadcast_in_dim %%v%d, dims = [] "
                  ": (tensor<i32>) -> %s\n", mem0, zc, SH_MEM);

    lane = newval(0, 0);
    bprintf(&g_b, "    %%v%d = stablehlo.iota dim = 0 : %s\n", lane, SH_VEC);
    mub = c32((uint32_t)m);
    base = bin(OP_MUL, lane, mub);
    mem1 = scatter_block(mem0, base, a0);
    base1 = bin(OP_ADD, base, c32(1));
    mem2 = scatter_block(mem1, base1, a1);

    i0 = c32(2);
    p0 = c32(0);

    res = newval(0, 0);
    bprintf(&g_b,
            "    %%v%d:3 = \"stablehlo.while\"(%%v%d, %%v%d, %%v%d) ({\n", res,
            mem2, i0, p0);
    cb_m = newval(0, 0);
    cb_i = newval(0, 0);
    cb_p = newval(0, 0);
    bprintf(&g_b, "    ^bb0(%%v%d: %s, %%v%d: %s, %%v%d: %s):\n", cb_m, SH_MEM,
            cb_i, SH_VEC, cb_p, SH_VEC);
    ps = v2s(cb_p);
    tcs = sc32((uint32_t)t);
    cond = scmp_cc("LT", ps, tcs);
    bprintf(&g_b, "      stablehlo.return %%v%d : tensor<i1>\n", cond);
    bprintf(&g_b, "    }, {\n");
    bb_m = newval(0, 0);
    bb_i = newval(0, 0);
    bb_p = newval(0, 0);
    bprintf(&g_b, "    ^bb0(%%v%d: %s, %%v%d: %s, %%v%d: %s):\n", bb_m, SH_MEM,
            bb_i, SH_VEC, bb_p, SH_VEC);

    {
        int i = bb_i, p = bb_p, mem = bb_m;
        int zero = c32(0), one = c32(1);
        int g1, g2, g3, slice, sslice, is, p0p, refA0, refA1, refA, start1,
            startpos, prev, prevb, pseudop, ltp, dip, clamp, paddrv,
            pseudo, rel, t1, rel2, absv, over, refidx, prevblk, refblk, currblk,
            withxor, tblock, ublock, vblock, newblk, memx, i2, p2, wrap, i1v;
        int tlo[256], plo[256], rlo[256], clo[256];
        u64v rprev[128], rref[128], rcurr[128], rout[128], tpair[128];

        g1 = cmp_cc("GE", i, c32((uint32_t)seg));
        g2 = cmp_cc("GE", i, c32((uint32_t)(2 * seg)));
        g3 = cmp_cc("GE", i, c32((uint32_t)(3 * seg)));
        slice = bin(OP_ADD, bin(OP_ADD, cv(g1, "i1", "i32"), cv(g2, "i1", "i32")),
                    cv(g3, "i1", "i32"));
        sslice = bin(OP_MUL, slice, c32((uint32_t)seg));
        is = bin(OP_SUB, i, sslice);
        p0p = cmp_cc("EQ", p, zero);
        refA0 = bin(OP_SUB, i, one);
        refA1 = bin(OP_SUB, bin(OP_ADD, c32((uint32_t)m), is),
                    c32((uint32_t)(seg + 1)));
        refA = sel32(p0p, refA0, refA1);
        start1 = sel32(cmp_cc("EQ", slice, c32(3)), zero,
                       bin(OP_MUL, bin(OP_ADD, slice, one),
                           c32((uint32_t)seg)));
        startpos = sel32(p0p, zero, start1);
        prev = sel32(cmp_cc("EQ", i, zero), c32((uint32_t)(m - 1)),
                     bin(OP_SUB, i, one));

        prevb = bin(OP_ADD, base, prev);
        prevblk = gather_block(mem, prevb);
        pseudop = slice_col(prevblk, 0);

        ltp = cmp_cc("LT", i, c32((uint32_t)half));
        dip = andp(p0p, ltp);
        clamp = sel32(ltp, i, c32((uint32_t)(half - 1)));
        paddrv = slice_word_bcast(a2v, v2s(clamp));
        pseudo = sel32(dip, paddrv, pseudop);

        /* index_alpha: rel2 = area-1 - ((area * ((p*p)>>32)) >> 32) */
        rel = mulhi32(pseudo, pseudo);
        t1 = mulhi32(refA, rel);
        rel2 = bin(OP_SUB, bin(OP_SUB, refA, one), t1);
        absv = bin(OP_ADD, rel2, startpos);
        over = cmp_cc("GE", absv, c32((uint32_t)m));
        refidx = sel32(over, bin(OP_SUB, absv, c32((uint32_t)m)), absv);

        refblk = gather_block(mem, bin(OP_ADD, base, refidx));
        currblk = gather_block(mem, bin(OP_ADD, base, i));

        withxor = cmp_cc("GT", p, zero);
        tblock = bxor(prevblk, refblk);
        {
            int predb = bcast_pred(withxor);

            ublock = bsel(predb, bxor(tblock, currblk), tblock);
        }

        if (g_sched) {
            memx = scatter_block(mem, bin(OP_ADD, base, i), ublock);
            goto sched_done;
        }
        unpack_block(prevblk, plo);
        unpack_block(refblk, rlo);
        unpack_block(currblk, clo);
        for (int k = 0; k < 128; k++) {
            rprev[k] = pair(plo[2 * k], plo[2 * k + 1]);
            rref[k] = pair(rlo[2 * k], rlo[2 * k + 1]);
            rcurr[k] = pair(clo[2 * k], clo[2 * k + 1]);
        }
        unpack_block(tblock, tlo);
        for (int k = 0; k < 128; k++) {
            tpair[k] = pair(tlo[2 * k], tlo[2 * k + 1]);
        }
        fill_hlo(rprev, rref, rcurr, withxor, rout);
        for (int k = 0; k < 256; k++) {
            tlo[k] = (k & 1) ? rout[k / 2].hi : rout[k / 2].lo;
        }
        vblock = pack_block(tlo);
        newblk = bxor(ublock, vblock);

        memx = scatter_block(mem, bin(OP_ADD, base, i), newblk);

sched_done:
        i1v = bin(OP_ADD, i, one);
        wrap = cmp_cc("EQ", i1v, c32((uint32_t)m));
        i2 = sel32(wrap, zero, i1v);
        p2 = bin(OP_ADD, p, cv(wrap, "i1", "i32"));

        bprintf(&g_b, "      stablehlo.return %%v%d, %%v%d, %%v%d : %s, %s, "
                      "%s\n", memx, i2, p2, SH_MEM, SH_VEC, SH_VEC);
        (void)tpair;
    }
    bprintf(&g_b, "    }) : (%s, %s, %s) -> (%s, %s, %s)\n", SH_MEM, SH_VEC,
            SH_VEC, SH_MEM, SH_VEC, SH_VEC);
    /* constants first seen inside the loop body do not dominate the code
     * after the loop: start a fresh cache for the epilogue */
    g_ncc = 0;

    {
        int final = gather_block(res, base1 /* base + 1 */);

        (void)final;
        if (g_dump_mem) {
            bprintf(&g_b, "    return %%v%d : %s\n  }\n}\n", res, SH_MEM);
        } else {
            int last = bin(OP_ADD, base, c32((uint32_t)(m - 1)));
            int fin = gather_block(res, last);

            bprintf(&g_b, "    return %%v%d : %s\n  }\n}\n", fin, SH_BLK);
        }
    }
    return g_b.s;
}

/* ==================================================================== */
/* PJRT plumbing (same lifecycle rules as sha256_pjrt.c)                */
/* ==================================================================== */

static PJRT_Api *api;
static PJRT_Client *client;
static PJRT_Device *device;

static void die_error(PJRT_Error *e, const char *what)
{
    if (e == NULL) {
        return;
    }
    {
        PJRT_Error_Message_Args ma = {0};
        PJRT_Error_Destroy_Args da = {0};

        ma.struct_size = sizeof ma;
        ma.error = e;
        api->PJRT_Error_Message(&ma);
        fprintf(stderr, "error: %s: %.*s\n", what, (int)ma.message_size,
                ma.message);
        da.struct_size = sizeof da;
        da.error = e;
        api->PJRT_Error_Destroy(&da);
    }
    exit(1);
}

static void pjrt_open(const char *path)
{
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

static const unsigned char compile_options[] = {0x1a, 0x04, 0x20, 0x01, 0x28,
                                                0x01};

static PJRT_LoadedExecutable *compile(const char *code, int *len_out)
{
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

static void upload_into(PJRT_Buffer **out, const void *data, int64_t *dims,
                        size_t ndims)
{
    PJRT_Event *done = NULL;
    PJRT_Client_BufferFromHostBuffer_Args ba = {0};

    ba.struct_size = sizeof ba;
    ba.client = client;
    ba.data = (void *)data;
    ba.type = PJRT_Buffer_Type_S32;
    ba.dims = dims;
    ba.num_dims = ndims;
    ba.byte_strides = NULL;
    ba.num_byte_strides = 0;
    ba.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
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
    *out = ba.buffer;
}

static PJRT_Buffer *execute(PJRT_LoadedExecutable *ex, PJRT_Buffer **args,
                            int nargs, double *ms)
{
    PJRT_Buffer *const *arg_lists[1] = {args};
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
    ea.num_args = nargs;
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

static void read_back(PJRT_Buffer *buf, void *dst, size_t bytes)
{
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

static void buffer_destroy(PJRT_Buffer *b)
{
    PJRT_Buffer_Destroy_Args da = {0};

    if (b == NULL) {
        return;
    }
    da.struct_size = sizeof da;
    da.buffer = b;
    api->PJRT_Buffer_Destroy(&da);
}

static void serialize_executable(PJRT_LoadedExecutable *ex, const char *path)
{
    PJRT_LoadedExecutable_GetExecutable_Args ga = {0};
    PJRT_Executable_Serialize_Args sa = {0};
    PJRT_Executable_Destroy_Args ed = {0};
    FILE *f;

    ga.struct_size = sizeof ga;
    ga.loaded_executable = ex;
    die_error(api->PJRT_LoadedExecutable_GetExecutable(&ga), "get executable");
    if (ga.executable == NULL) {
        fprintf(stderr, "serialize: no executable\n");
        return;
    }
    sa.struct_size = sizeof sa;
    sa.executable = ga.executable;
    die_error(api->PJRT_Executable_Serialize(&sa), "serialize executable");
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "serialize: %s: %s\n", path, strerror(errno));
    } else {
        fwrite(sa.serialized_bytes, 1, sa.serialized_bytes_size, f);
        fclose(f);
        printf("serialized %zu bytes to %s\n", sa.serialized_bytes_size, path);
    }
    if (sa.serialized_executable != NULL &&
        sa.serialized_executable_deleter != NULL) {
        sa.serialized_executable_deleter(sa.serialized_executable);
    }
    ed.struct_size = sizeof ed;
    ed.executable = ga.executable;
    api->PJRT_Executable_Destroy(&ed);
}

/* ==================================================================== */
/* host side: inputs, tags, checks                                      */
/* ==================================================================== */

static int bits_ok(const uint8_t d[32], int difficulty)
{
    int full = difficulty / 8;
    int rem = difficulty % 8;

    for (int i = 0; i < full; i++) {
        if (d[i] != 0) {
            return 0;
        }
    }
    if (rem && (d[full] & (uint8_t)(0xffu << (8 - rem))) != 0) {
        return 0;
    }
    return 1;
}

static void make_salt(uint8_t salt[8], uint32_t nonce)
{
    for (int i = 0; i < 8; i++) {
        salt[i] = (uint8_t)(nonce >> (8 * i));
    }
}

/* H0 = BLAKE2b-64(p, T, m, t, v, y, |P|, P, |S|, S, 0, 0); salt is the nonce */
static void a2_h0(const uint8_t *pwd, size_t pwdlen, const uint8_t *salt,
                  size_t saltlen, uint32_t m, uint32_t t, uint8_t out[64])
{
    b2b_t c;
    uint8_t le[4];

    b2b_init(&c, 64);
    put32(le, 1); b2b_update(&c, le, 4);
    put32(le, 32); b2b_update(&c, le, 4);
    put32(le, m); b2b_update(&c, le, 4);
    put32(le, t); b2b_update(&c, le, 4);
    put32(le, 0x13); b2b_update(&c, le, 4);
    put32(le, 2); b2b_update(&c, le, 4);
    put32(le, (uint32_t)pwdlen); b2b_update(&c, le, 4);
    b2b_update(&c, pwd, pwdlen);
    put32(le, (uint32_t)saltlen); b2b_update(&c, le, 4);
    b2b_update(&c, salt, saltlen);
    put32(le, 0); b2b_update(&c, le, 4);
    put32(le, 0); b2b_update(&c, le, 4);
    b2b_final(&c, out);
}

/* B0 = H'(H0 || 0 || lane), B1 = H'(H0 || 1 || lane); lanes=1 */
static void a2_first_blocks(const uint8_t h0[64], uint8_t b0[1024],
                            uint8_t b1[1024])
{
    uint8_t seed[72];

    memcpy(seed, h0, 64);
    put32(seed + 68, 0);
    put32(seed + 64, 0);
    blake2b_long(b0, 1024, seed, 72);
    put32(seed + 64, 1);
    blake2b_long(b1, 1024, seed, 72);
}

/*
 * The data-independent address words of pass 0, one per block: the sequence
 * only depends on (m, t), so the device uploads it once for every nonce.
 */
static void a2_addr_table(uint32_t m, uint32_t t, uint32_t *addr)
{
    uint32_t seg = m / 4;

    for (uint32_t slice = 0; slice < 2; slice++) {
        a2block input, address;

        memset(&input, 0, sizeof input);
        memset(&address, 0, sizeof address);
        input.v[0] = 0;
        input.v[1] = 0;
        input.v[2] = slice;
        input.v[3] = m;
        input.v[4] = t;
        input.v[5] = 2;
        for (uint32_t i = 0; i < seg; i++) {
            if (i % 128 == 0) {
                a2_next_addresses(&address, &input);
            }
            addr[slice * seg + i] = (uint32_t)address.v[i % 128];
        }
    }
}

/* B0/B1 for one nonce, ready to upload as two 1 KiB blocks */
static int a2_inputs(const uint8_t *raw, size_t raw_len, uint32_t nonce,
                     uint32_t m, uint32_t t, int32_t b0[256], int32_t b1[256])
{
    uint8_t salt[8], h0[64], w0[1024], w1[1024];

    make_salt(salt, nonce);
    a2_h0(raw, raw_len, salt, sizeof salt, m, t, h0);
    a2_first_blocks(h0, w0, w1);
    memcpy(b0, w0, 1024);
    memcpy(b1, w1, 1024);
    return 0;
}

/* ==================================================================== */
/* --unit: u64 pair primitives vs the host's native u64                 */
/* ==================================================================== */

static uint64_t model64(u64v v)
{
    return (uint64_t)v_val[v.lo] | ((uint64_t)v_val[v.hi] << 32);
}

static int unit_tests(void)
{
    uint64_t seed = 0x243f6a8885a308d3ULL;
    int fails = 0, checks = 0;

    build_reset(1, 8, 1);
    g_ncc = 0;

#define CHECK(what, expr, want)                                              \
    do {                                                                     \
        uint64_t got_ = (expr);                                              \
        checks++;                                                            \
        if (got_ != (uint64_t)(want)) {                                      \
            printf("  FAIL %-10s got=%016llx want=%016llx\n", what,          \
                   (unsigned long long)got_, (unsigned long long)(want));    \
            fails++;                                                         \
        }                                                                    \
    } while (0)

    for (int i = 0; i < 64; i++) {
        uint64_t a, b;

        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        a = (seed >> 11);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        b = (seed >> 7);
        if (i & 1) {
            a &= 0xffffffffULL;
        }
        if (i & 2) {
            b &= 0xffffffffULL;
        }
        CHECK("add64", model64(add64(const64(a), const64(b))), a + b);
        CHECK("sub64", model64(sub64(const64(a), const64(b))), a - b);
        CHECK("mul64", model64(mul64(const64(a), const64(b))), a * b);
        CHECK("mulhi32",
              (uint64_t)v_val[mulhi32(c32((uint32_t)a), c32((uint32_t)b))],
              (uint64_t)(((uint64_t)(uint32_t)a * (uint32_t)b) >> 32));
        CHECK("blamka",
              model64(blamka(const64(a), const64(b))),
              a + b + 2 * ((a & 0xffffffffULL) * (b & 0xffffffffULL)));
        for (unsigned n = 0; n < 64; n += 5) {
            uint64_t want = n ? ((a >> n) | (a << (64 - n))) : a;

            CHECK("rotr64", model64(rotr64(const64(a), n)), want);
            want = n ? ((a << n) | (a >> (64 - n))) : a;
            CHECK("rotl64", model64(rotl64(const64(a), n)), want);
        }
    }
#undef CHECK
    printf("unit: %d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}

/* ==================================================================== */
/* --hosttest: the host reference against fixed vectors                 */
/* ==================================================================== */

static int hosttest(void)
{
    static const struct {
        const char *data;
        uint32_t nonce, m, t;
        const char *want;
    } vec[] = {
        {"9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b",
         0, 64, 2,
         "7aa72656793d4af9c955af70c001aa3cc4dc5ab62e74d6c54e128bcc297f3fff"},
        {"9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b",
         1, 64, 2,
         "e5c23c04782fc9075b84b402305ebadd6ca1af39ad13a6e97601e1a985ed483f"},
        {"9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b",
         7, 256, 1,
         "6210d6e0b03c7d9b6caef3ec0888e93c95f1e31ee53df90ebe66c8fb3c6bb593"},
        {"00", 0, 8, 1, "cefd572194b6ca8a737a06bdb283b0d992eaf2fffda141ba220a01116a2f1fea"},
    };
    int fails = 0;

    for (size_t i = 0; i < sizeof vec / sizeof vec[0]; i++) {
        uint8_t raw[64], salt[8], got[32];
        char hex[65];

        size_t n = pjrt_hex_decode(vec[i].data, raw, sizeof raw);

        make_salt(salt, vec[i].nonce);
        if (a2_ref(vec[i].m, vec[i].t, raw, n, salt, 8, got, NULL) != 0) {
            printf("  FAIL vector %zu: a2_ref\n", i);
            fails++;
            continue;
        }
        pjrt_to_hex(got, 32, hex);
        if (strcmp(hex, vec[i].want) != 0) {
            printf("  FAIL vector %zu (m=%u t=%u): got %s want %s\n", i,
                   vec[i].m, vec[i].t, hex, vec[i].want);
            fails++;
        }
    }
    printf("hosttest: %zu vectors, %d failed\n", sizeof vec / sizeof vec[0],
           fails);
    return fails ? 1 : 0;
}

/* ==================================================================== */
/* --dump: write the module and inputs for a CPU XLA run                */
/* ==================================================================== */

static int dump_all(const char *dir, int lanes, uint32_t m, uint32_t t,
                    const uint8_t *raw, size_t raw_len, int difficulty)
{
    char path[4096];
    FILE *f;
    int32_t *b0 = malloc((size_t)lanes * 256 * 4);
    int32_t *b1 = malloc((size_t)lanes * 256 * 4);
    uint32_t *addr = calloc(m / 2, 4);
    uint8_t *tags = malloc((size_t)lanes * 32);
    char *mod;
    char hex[65];

    if (b0 == NULL || b1 == NULL || addr == NULL || tags == NULL) {
        return -1;
    }
    a2_addr_table(m, t, addr);
    for (int l = 0; l < lanes; l++) {
        a2block last;
        uint8_t salt[8];
        static int32_t tb0[256], tb1[256];

        if (a2_inputs(raw, raw_len, (uint32_t)l, m, t, tb0, tb1) != 0) {
            return -1;
        }
        make_salt(salt, (uint32_t)l);
        if (a2_ref(m, t, raw, raw_len, salt, 8, tags + 32 * l, &last) != 0) {
            return -1;
        }
        memcpy(b0 + (size_t)l * 256, tb0, 1024);
        memcpy(b1 + (size_t)l * 256, tb1, 1024);
    }
    mod = build_module(lanes, (int)m, (int)t);

    snprintf(path, sizeof path, "%s/module.mlir", dir);
    f = fopen(path, "wb");
    if (f == NULL || fwrite(mod, 1, strlen(mod), f) != strlen(mod)) {
        fprintf(stderr, "dump: %s: %s\n", path, strerror(errno));
        return -1;
    }
    fclose(f);
    snprintf(path, sizeof path, "%s/b0.i32", dir);
    f = fopen(path, "wb");
    fwrite(b0, 4, (size_t)lanes * 256, f);
    fclose(f);
    snprintf(path, sizeof path, "%s/b1.i32", dir);
    f = fopen(path, "wb");
    fwrite(b1, 4, (size_t)lanes * 256, f);
    fclose(f);
    snprintf(path, sizeof path, "%s/addr.i32", dir);
    f = fopen(path, "wb");
    fwrite(addr, 4, m / 2, f);
    fclose(f);
    snprintf(path, sizeof path, "%s/meta.txt", dir);
    f = fopen(path, "w");
    fprintf(f, "lanes %d\nm %u\nt %u\ndifficulty %d\ntag0 ", lanes, m, t,
            difficulty);
    pjrt_to_hex(tags, 32, hex);
    fprintf(f, "%s\n", hex);
    fclose(f);
    printf("dumped %s: module.mlir (%zu bytes), b0/b1 (%d x 1 KiB), addr "
           "(%u i32), tag0=%s\n", dir, strlen(mod), lanes, m / 2, hex);
    free(b0);
    free(b1);
    free(addr);
    free(tags);
    return 0;
}

/* ==================================================================== */
/* select nonces on the device, verify on the host                      */
/* ==================================================================== */

typedef struct {
    PJRT_LoadedExecutable *ex;
    int lanes;
    uint32_t m, t;
    int32_t *b0, *b1;
    uint32_t *addr;
    int32_t *out;
} runner;

static void runner_free(runner *r)
{
    PJRT_LoadedExecutable_Destroy_Args ed = {0};

    if (r->ex != NULL) {
        ed.struct_size = sizeof ed;
        ed.executable = r->ex;
        api->PJRT_LoadedExecutable_Destroy(&ed);
    }
    free(r->b0);
    free(r->b1);
    free(r->addr);
    free(r->out);
}

/* run one window of `lanes` nonces starting at base; returns blocks in r->out */
static int runner_run(runner *r, const uint8_t *raw, size_t raw_len,
                      uint32_t base, double *ms)
{
    int64_t dblk[2] = {r->lanes, 256};
    int64_t daddr[1] = {(int64_t)(r->m / 2)};
    PJRT_Buffer *args[3] = {NULL, NULL, NULL};
    PJRT_Buffer *res;
    int rc = 0;

    for (int l = 0; l < r->lanes; l++) {
        static int32_t tb0[256], tb1[256];

        if (a2_inputs(raw, raw_len, base + (uint32_t)l, r->m, r->t, tb0,
                      tb1) != 0) {
            return -1;
        }
        memcpy(r->b0 + (size_t)l * 256, tb0, 1024);
        memcpy(r->b1 + (size_t)l * 256, tb1, 1024);
    }
    upload_into(&args[0], r->b0, dblk, 2);
    upload_into(&args[1], r->b1, dblk, 2);
    upload_into(&args[2], r->addr, daddr, 1);
    res = execute(r->ex, args, 3, ms);
    read_back(res, r->out, (size_t)r->lanes * 1024);
    for (int i = 0; i < 3; i++) {
        buffer_destroy(args[i]);
    }
    buffer_destroy(res);
    return rc;
}

/* tag for lane l from the returned block (little-endian i32 words) */
static void lane_tag(const int32_t *out, int lanes, int lane, uint8_t tag[32])
{
    uint8_t bytes[1024];

    for (int i = 0; i < 256; i++) {
        uint32_t w = (uint32_t)out[(size_t)lane * 256 + i];

        bytes[4 * i + 0] = (uint8_t)w;
        bytes[4 * i + 1] = (uint8_t)(w >> 8);
        bytes[4 * i + 2] = (uint8_t)(w >> 16);
        bytes[4 * i + 3] = (uint8_t)(w >> 24);
    }
    blake2b_long(tag, 32, bytes, 1024);
    (void)lanes;
}


/* On a selftest failure: run the same case with the whole matrix as the
 * output and say which block, and which word in it, differs first. */
static void diag_case(int cm, int ct, int cl, const uint8_t *raw, size_t raw_len,
                      const uint32_t *addr, const int32_t *b0,
                      const int32_t *b1)
{
    int old = g_dump_mem;
    char *mod;
    int64_t dblk[2] = {cl, 256};
    int64_t daddr[1] = {cm / 2};
    PJRT_LoadedExecutable *ex;
    int32_t *outmem = malloc((size_t)cl * cm * 1024);
    PJRT_Buffer *args[3] = {NULL, NULL, NULL};
    PJRT_Buffer *res;
    int compile_ms = 0;
    double ms = 0;

    if (outmem == NULL) {
        return;
    }
    g_dump_mem = 1;
    mod = build_module(cl, cm, ct);
    g_dump_mem = old;
    ex = compile(mod, &compile_ms);
    upload_into(&args[0], (void *)b0, dblk, 2);
    upload_into(&args[1], (void *)b1, dblk, 2);
    upload_into(&args[2], (void *)addr, daddr, 1);
    res = execute(ex, args, 3, &ms);
    read_back(res, outmem, (size_t)cl * cm * 1024);
    printf("  diag: m=%d t=%d lanes=%d, whole matrix compiled in %d ms, ran "
           "in %.2f ms\n", cm, ct, cl, compile_ms, ms);
    for (int l = 0; l < cl; l++) {
        a2block *ref = NULL;
        uint8_t tag[32], salt[8];
        int shown = 0;

        make_salt(salt, (uint32_t)l);
        if (a2_ref_mem((uint32_t)cm, (uint32_t)ct, raw, raw_len, salt, 8, tag,
                       &ref) != 0) {
            continue;
        }
        for (int k = 0; k < cm && shown < 3; k++) {
            const int32_t *got = outmem + ((size_t)l * cm + k) * 256;
            int diff = -1;
            uint64_t g = 0;

            for (int w = 0; w < 128; w++) {
                g = (uint32_t)got[2 * w] |
                    ((uint64_t)(uint32_t)got[2 * w + 1] << 32);
                if (g != ref[k].v[w]) {
                    diff = w;
                    break;
                }
            }
            if (diff >= 0) {
                printf("  lane=%d block=%d word=%d got=%016llx want=%016llx\n",
                       l, k, diff, (unsigned long long)g,
                       (unsigned long long)ref[k].v[diff]);
                shown++;
            }
        }
        if (!shown) {
            printf("  lane=%d every block matches (the tag is wrong, not the "
                   "fill)\n", l);
        }
        free(ref);
    }
    free(outmem);
    free(mod);
    for (int i = 0; i < 3; i++) {
        buffer_destroy(args[i]);
    }
    buffer_destroy(res);
    {
        PJRT_LoadedExecutable_Destroy_Args ed = {0};

        ed.struct_size = sizeof ed;
        ed.executable = ex;
        api->PJRT_LoadedExecutable_Destroy(&ed);
    }
}

/* ==================================================================== */
/* main                                                                 */
/* ==================================================================== */

static const char DEFAULT_DATA[] =
    "9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b";

int main(int argc, char **argv)
{
    const char *lib = getenv("LIBTPU");
    const char *data = DEFAULT_DATA;
    const char *dump_dir = NULL;
    const char *serialize_path = NULL;
    int m = 19456, t = 2, lanes = 32, difficulty = 15;
    int json = 0, mode = 0, force = 0, init_timeout = 120;
    long long bench_hashes = 0;
    uint8_t raw[4096];
    size_t raw_len;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc) {
            lib = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data = argv[++i];
        } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t") == 0 && i + 1 < argc) {
            t = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lanes") == 0 && i + 1 < argc) {
            lanes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--difficulty") == 0 && i + 1 < argc) {
            difficulty = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-hashes") == 0 && i + 1 < argc) {
            bench_hashes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--hosttest") == 0) {
            mode = 1;
        } else if (strcmp(argv[i], "--unit") == 0) {
            mode = 2;
        } else if (strcmp(argv[i], "--emit") == 0) {
            mode = 3;
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            mode = 4;
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--selftest") == 0) {
            mode = 5;
        } else if (strcmp(argv[i], "--serialize") == 0 && i + 1 < argc) {
            serialize_path = argv[++i];
        } else if (strcmp(argv[i], "--dump-all") == 0) {
            g_dump_mem = 1;
        } else if (strcmp(argv[i], "--selftest-sched") == 0) {
            mode = 6;
        } else if (strcmp(argv[i], "--sched") == 0) {
            g_sched = 1;
        } else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            g_rounds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--filltest2") == 0) {
            g_filltest1 = 2;
        } else if (strcmp(argv[i], "--filltest1") == 0) {
            g_filltest1 = 1;
        } else if (strcmp(argv[i], "--filltest") == 0) {
            g_filltest = 1;
        } else if (strcmp(argv[i], "--force") == 0) {
            force = 1;
        } else if (strcmp(argv[i], "--init-timeout") == 0 && i + 1 < argc) {
            init_timeout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else {
            fprintf(stderr,
                    "usage: %s --lib libtpu.so [--m KIB] [--t N] [--lanes B]\n"
                    "          [--data HEX] [--difficulty BITS] "
                    "[--bench-hashes N] [--json]\n"
                    "          [--serialize FILE] [--force] "
                    "[--init-timeout S]\n"
                    "       %s --hosttest | --unit | --emit | --dump DIR | "
                    "--selftest\n"
                    "       debug: --dump-all | --filltest | --filltest1 | "
                    "--filltest2\n",
                    argv[0], argv[0]);
            return 2;
        }
    }
    if (difficulty < 1 || difficulty > 32) {
        fprintf(stderr, "--difficulty must be 1..32\n");
        return 2;
    }
    if (lanes < 1 || lanes > 1024) {
        fprintf(stderr, "--lanes must be 1..1024\n");
        return 2;
    }
    if (m < 8 || m % 4 != 0 || m > 65536) {
        fprintf(stderr, "--m must be a multiple of 4 in 8..65536\n");
        return 2;
    }
    if (t < 1 || t > 4) {
        fprintf(stderr, "--t must be 1..4\n");
        return 2;
    }
    raw_len = pjrt_hex_decode(data, raw, sizeof raw);
    if (raw_len == 0) {
        fprintf(stderr, "--data must be non-empty hex\n");
        return 2;
    }

    if (mode == 1) {
        return hosttest();
    }
    if (mode == 2) {
        return unit_tests();
    }
    if (mode == 3) {
        printf("%s", build_module(lanes, m, t));
        return 0;
    }
    if (mode == 4) {
        return dump_all(dump_dir, lanes, (uint32_t)m, (uint32_t)t, raw, raw_len,
                        difficulty) == 0 ? 0 : 1;
    }
    if (mode == 5) {
        /* selftest needs a device; it runs below */
    }

    if (lib == NULL) {
        fprintf(stderr, "need --lib or LIBTPU\n");
        return 2;
    }
    /* Preflight: libtpu's plugin init blocks on a host without a TPU (measured
     * >25 s, and `timeout` could not kill it), so fail fast unless --force. */
    if (!force) {
        glob_t g;
        int have = access("/dev/accel", F_OK) == 0;

        if (!have && glob("/dev/vfio/[0-9]*", 0, NULL, &g) == 0) {
            have = g.gl_pathc > 0;
            globfree(&g);
        }
        if (!have) {
            fprintf(stderr,
                    "no TPU device node (numeric /dev/vfio/<group> or "
                    "/dev/accel) - this host has no TPU; use --force to try "
                    "PJRT_Plugin_Initialize anyway\n");
            return 3;
        }
    }
    if (init_timeout > 0) {
        alarm((unsigned)init_timeout);
    }
    pjrt_open(lib);
    if (init_timeout > 0) {
        alarm(0);
    }

    if (mode == 6) {
        int cm = 8, ct = 2, cl = 4;
        int32_t *b0 = malloc((size_t)cl * 1024);
        int32_t *b1 = malloc((size_t)cl * 1024);
        uint32_t *addr = calloc((size_t)cm / 2, 4);

        if (b0 == NULL || b1 == NULL || addr == NULL) {
            return 1;
        }
        a2_addr_table((uint32_t)cm, (uint32_t)ct, addr);
        for (int l = 0; l < cl; l++) {
            a2_inputs(raw, raw_len, (uint32_t)l, (uint32_t)cm, (uint32_t)ct,
                      b0 + (size_t)l * 256, b1 + (size_t)l * 256);
        }
        g_sched = 1;
        g_host_sched = 1;
        diag_case(cm, ct, cl, raw, raw_len, addr, b0, b1);
        free(b0);
        free(b1);
        free(addr);
        {
            PJRT_Client_Destroy_Args cd = {0};

            cd.struct_size = sizeof cd;
            api->PJRT_Client_Destroy(&cd);
        }
        return 0;
    }

    if (mode == 5) {
        static const struct {
            int m, t, lanes;
        } cases[] = {{8, 1, 1}, {64, 2, 4}};
        int fails = 0;

        for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
            int cm = cases[ci].m, ct = cases[ci].t, cl = cases[ci].lanes;
            char *mod = build_module(cl, cm, ct);
            int64_t dblk[2] = {cl, 256};
            int64_t daddr[1] = {cm / 2};
            PJRT_LoadedExecutable *ex;
            int32_t *b0 = malloc((size_t)cl * 1024);
            int32_t *b1 = malloc((size_t)cl * 1024);
            uint32_t *addr = calloc(cm / 2, 4);
            int32_t *out = malloc((size_t)cl * 1024);
            uint8_t *want = malloc((size_t)cl * 32);
            int compile_ms = 0;
            double ms = 0;

            if (b0 == NULL || b1 == NULL || addr == NULL || out == NULL ||
                want == NULL) {
                return 1;
            }
            a2_addr_table((uint32_t)cm, (uint32_t)ct, addr);
            for (int l = 0; l < cl; l++) {
                a2block last;
                static int32_t tb0[256], tb1[256];
                uint8_t tag[32];

                if (a2_inputs(raw, raw_len, (uint32_t)l, (uint32_t)cm,
                              (uint32_t)ct, tb0, tb1) != 0) {
                    printf("selftest: a2_inputs failed\n");
                    return 1;
                }
                {
                    uint8_t salt[8];

                    make_salt(salt, (uint32_t)l);
                    if (a2_ref((uint32_t)cm, (uint32_t)ct, raw, raw_len, salt,
                               8, tag, &last) != 0) {
                        printf("selftest: a2_ref failed\n");
                        return 1;
                    }
                }
                memcpy(b0 + (size_t)l * 256, tb0, 1024);
                memcpy(b1 + (size_t)l * 256, tb1, 1024);
                memcpy(want + (size_t)l * 32, tag, 32);
            }
            ex = compile(mod, &compile_ms);
            {
                PJRT_Buffer *args[3];
                PJRT_Buffer *res;

                upload_into(&args[0], b0, dblk, 2);
                upload_into(&args[1], b1, dblk, 2);
                upload_into(&args[2], addr, daddr, 1);
                res = execute(ex, args, 3, &ms);
                read_back(res, out, (size_t)cl * 1024);
                buffer_destroy(args[0]);
                buffer_destroy(args[1]);
                buffer_destroy(args[2]);
                buffer_destroy(res);
            }
            {
                int bad = 0;

                for (int l = 0; l < cl; l++) {
                    uint8_t tag[32];
                    char hex[65];

                    lane_tag(out, cl, l, tag);
                    if (memcmp(tag, want + (size_t)l * 32, 32) != 0) {
                        bad++;
                        if (bad == 1) {
                            pjrt_to_hex(want + (size_t)l * 32, 32, hex);
                            printf("  FAIL m=%d t=%d lane=%d want=%s got=", cm,
                                   ct, l, hex);
                            pjrt_to_hex(tag, 32, hex);
                            printf("%s\n", hex);
                        }
                    }
                }
                fails += bad;
                printf("  %s m=%d t=%d lanes=%d: %d/%d tags, compile %d ms, "
                       "execute %.2f ms\n", bad ? "FAIL" : "ok", cm, ct, cl,
                       cl - bad, cl, compile_ms, ms);
                if (bad) {
                    diag_case(cm, ct, cl, raw, raw_len, addr, b0, b1);
                }
            }
            free(b0);
            free(b1);
            free(addr);
            free(out);
            free(want);
            {
                PJRT_LoadedExecutable_Destroy_Args ed = {0};

                ed.struct_size = sizeof ed;
                ed.executable = ex;
                api->PJRT_LoadedExecutable_Destroy(&ed);
            }
        }
        printf("selftest: %d failed\n", fails);
        {
            PJRT_Client_Destroy_Args cd = {0};

            cd.struct_size = sizeof cd;
            api->PJRT_Client_Destroy(&cd);
        }
        return fails ? 1 : 0;
    }

    /* benchmark: each window is `lanes` independent nonces */
    {
        char *mod = build_module(lanes, m, t);
        int compile_ms = 0;
        PJRT_LoadedExecutable *ex = compile(mod, &compile_ms);
        uint8_t *tags = malloc((size_t)lanes * 32);
        runner r;
        double t0;
        long long done = 0, hits = 0;
        uint32_t base = 0, found_nonce = 0;
        int found = 0;

        if (serialize_path != NULL) {
            serialize_executable(ex, serialize_path);
        }
        memset(&r, 0, sizeof r);
        r.ex = ex;
        r.lanes = lanes;
        r.m = (uint32_t)m;
        r.t = (uint32_t)t;
        r.b0 = malloc((size_t)lanes * 1024);
        r.b1 = malloc((size_t)lanes * 1024);
        r.addr = calloc((size_t)m / 2, 4);
        r.out = malloc((size_t)lanes * 1024);
        if (r.addr != NULL) {
            a2_addr_table((uint32_t)m, (uint32_t)t, r.addr);
        }
        if (r.b0 == NULL || r.b1 == NULL || r.addr == NULL || r.out == NULL ||
            tags == NULL) {
            return 1;
        }

        /* one untimed window warms the executable */
        if (runner_run(&r, raw, raw_len, 0, NULL) != 0) {
            return 1;
        }
        t0 = pjrt_now_s();
        while (1) {
            double ms = 0;

            if (runner_run(&r, raw, raw_len, base, &ms) != 0) {
                return 1;
            }
            for (int l = 0; l < lanes; l++) {
                lane_tag(r.out, lanes, l, tags + 32 * l);
                if (bits_ok(tags + 32 * l, difficulty)) {
                    hits++;
                    if (!found) {
                        found = 1;
                        found_nonce = base + (uint32_t)l;
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
            if (base == 0) {
                break;
            }
        }
        {
            double ms = (pjrt_now_s() - t0) * 1000.0;
            char hex[65] = "";
            int verified = 0;

            if (found) {
                uint8_t ref[32];
                uint8_t salt[8];

                make_salt(salt, found_nonce);
                if (a2_ref((uint32_t)m, (uint32_t)t, raw, raw_len, salt, 8,
                           ref, NULL) == 0) {
                    verified = bits_ok(ref, difficulty) &&
                               memcmp(ref, tags + 32 * (found_nonce - (base - (uint32_t)lanes)), 32) == 0;
                    pjrt_to_hex(ref, 32, hex);
                }
            }
            if (json) {
                printf("{\"hashes\":%lld,\"ms\":%.3f,\"h_s\":%.3f,"
                       "\"lanes\":%d,\"m\":%d,\"t\":%d,\"difficulty\":%d,"
                       "\"compile_ms\":%d,\"nonce\":%u,\"tag\":\"%s\","
                       "\"verified\":%s,\"hits\":%lld}\n",
                       done, ms, (double)done / (ms / 1000.0), lanes, m, t,
                       difficulty, compile_ms, found ? found_nonce : 0, hex,
                       verified ? "true" : "false", hits);
            } else {
                printf("lanes=%d m=%d t=%d difficulty=%d hashes=%lld %.1f ms "
                       "= %.2f H/s compile=%d ms\n", lanes, m, t, difficulty,
                       done, ms, (double)done / (ms / 1000.0), compile_ms);
                if (found) {
                    printf("nonce=%u tag=%s %s\n", found_nonce, hex,
                           verified ? "verified" : "VERIFICATION FAILED");
                } else if (bench_hashes > 0) {
                    printf("(benchmark only, hits=%lld)\n", hits);
                }
            }
            runner_free(&r);
            free(tags);
            free(mod);
            {
                PJRT_Client_Destroy_Args cd = {0};

                cd.struct_size = sizeof cd;
                api->PJRT_Client_Destroy(&cd);
            }
            return found && !verified ? 1 : 0;
        }
    }
}
