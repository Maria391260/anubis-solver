/*
 * hashx_pjrt.c — HashX on a TPU, through the PJRT C API.
 *
 * This is the HashX counterpart of sha256_pjrt.c: it turns one challenge's
 * HashX program into a single StableHLO module and runs it over a window of
 * nonces, all lanes at once. The device only answers "does this lane meet the
 * target?"; the host re-derives the 32-byte digest of the reported lane with
 * hashx_hash() before believing it, exactly like the SHA-256 tier.
 *
 * Two things make the port possible, both structural:
 *
 * 1. The program is fixed per challenge. Every nonce seeds the same 8-register
 *    program, so the interpreter loop becomes straight-line code: the dst/src
 *    register indices, the immediates and the opcodes are compile-time
 *    constants. There is no indexed register file at runtime - which is the
 *    thing that forces the CUDA and Vulkan HashX interpreters into local
 *    memory - and the 8 u64 registers live as SSA values.
 *
 * 2. The control flow is block-local. A generated program is 512 instructions
 *    in 16 blocks of 32; block k is
 *
 *        [32k]      one op
 *        [32k+1]    INSTR_TARGET
 *        [32k+2..16] body
 *        [32k+17]   INSTR_BRANCH   (jump back to the target, once per lane)
 *        [32k+18..31] tail
 *
 *    The interpreter jumps back at most once per lane (`branch_enable`), and
 *    the branch only ever jumps to the target of its own block. So each block
 *    executes its body either once or twice, and a second pass only changes
 *    the state that enters the tail of the same block. Emitting the body a
 *    second time and blending per lane with `select` is exact, and costs 16
 *    extra instructions per block, not a duplicated program tail. That is why
 *    this is ~1.5x the straight-line work rather than ~16x.
 *
 * The TPU lane is 32-bit, so each u64 register is an i32 pair and the 64-bit
 * mul / mulhi / rot / shift are emulated with i32 ops - the same arithmetic
 * the Vulkan kernel does with `uvec2` in src/vulkan/hashx.comp, and checked
 * here by --selftest against hashx_hash() before any TPU is involved.
 *
 * Build: cc -O2 -I. -I../../hashx -o hashx_pjrt hashx_pjrt.c \
 *                     ../../hashx/hashx.c -ldl
 * Run:   ./hashx_pjrt --lib libtpu.so --data HEX --difficulty 15 --window 16384
 *        ./hashx_pjrt --selftest --data HEX --difficulty 15
 */
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <glob.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"
#include "pjrt_common.h"
#include "hashx.h"

/* ==================================================================== */
/* SSA values: each carries the emitted name and, for --selftest, the    */
/* concrete value the same op would produce. Everything is emitted once  */
/* and evaluated by the same code path.                                  */
/* ==================================================================== */

#define MAX_VALUES 800000

typedef struct {
    char *s;
    size_t len, cap;
} buf_t;

static buf_t g_b;
static int g_lanes;
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

static void build_reset(int lanes)
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
    v_n = 0;
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

/* Constant i32 vector, cached: an unrolled program emits the same handful of
 * masks and shift counts thousands of times, and each should be one SSA name. */
typedef struct {
    uint32_t v;
    int id;
} cconst;
static cconst g_cc[256];
static int g_ncc;

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
                  ": (tensor<i32>) -> tensor<%dxi32>\n", vid, sid, g_lanes);
    if (g_ncc < (int)(sizeof g_cc / sizeof g_cc[0])) {
        g_cc[g_ncc].v = v;
        g_cc[g_ncc].id = vid;
        g_ncc++;
    }
    return vid;
}

enum bin_op { OP_ADD, OP_SUB, OP_MUL, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR };

static int bin(enum bin_op op, int a, int b)
{
    static const char *names[] = {"add", "subtract", "multiply", "and",
                                  "or", "xor", "shift_left",
                                  "shift_right_logical"};
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = stablehlo.%s %%v%d, %%v%d : tensor<%dxi32>\n",
            r, names[op], a, b, g_lanes);
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

static int cmp_eq(int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b,
            "    %%v%d = \"stablehlo.compare\"(%%v%d, %%v%d) "
            "{comparison_direction = #stablehlo<comparison_direction EQ>, "
            "compare_type = #stablehlo<comparison_type SIGNED>} : "
            "(tensor<%dxi32>, tensor<%dxi32>) -> tensor<%dxi1>\n",
            r, a, b, g_lanes, g_lanes, g_lanes);
    if (v_known[a] && v_known[b]) {
        v_known[r] = 1;
        v_val[r] = (v_val[a] == v_val[b]) ? 1u : 0u;
    }
    return r;
}

static int cmp_ne(int a, int b)
{
    int r = newval(0, 0);

    bprintf(&g_b,
            "    %%v%d = \"stablehlo.compare\"(%%v%d, %%v%d) "
            "{comparison_direction = #stablehlo<comparison_direction NE>, "
            "compare_type = #stablehlo<comparison_type SIGNED>} : "
            "(tensor<%dxi32>, tensor<%dxi32>) -> tensor<%dxi1>\n",
            r, a, b, g_lanes, g_lanes, g_lanes);
    if (v_known[a] && v_known[b]) {
        v_known[r] = 1;
        v_val[r] = (v_val[a] != v_val[b]) ? 1u : 0u;
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
        v_val[r] = v_val[a];
    }
    return r;
}

static int sel32(int pred, int t, int f)
{
    int r = newval(0, 0);

    bprintf(&g_b, "    %%v%d = \"stablehlo.select\"(%%v%d, %%v%d, %%v%d) : "
                  "(tensor<%dxi1>, tensor<%dxi32>, tensor<%dxi32>) -> "
                  "tensor<%dxi32>\n",
            r, pred, t, f, g_lanes, g_lanes, g_lanes, g_lanes);
    if (v_known[pred] && v_known[t] && v_known[f]) {
        v_known[r] = 1;
        v_val[r] = v_val[pred] ? v_val[t] : v_val[f];
    }
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

/* Carry-out of a 32-bit add without an unsigned compare: the TPU plugin
 * mis-handled UNSIGNED comparisons (see sha256_pjrt.c), and this formula is
 * pure bitwise. */
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
    int iszero = cmp_eq(a.lo, c32(0));
    int inc = cv(iszero, "i1", "i32");
    int hi = bin(OP_ADD, not32(a.hi), inc);

    return pair(lo, hi);
}

static u64v sub64(u64v a, u64v b)
{
    return add64(a, neg64(b));
}

static u64v sel64(int pred, u64v t, u64v f)
{
    return pair(sel32(pred, t.lo, f.lo), sel32(pred, t.hi, f.hi));
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

/* 64x64 -> high 64, the umulh/smulh the HashX program uses. This is the
 * uint128 shift written out in 32-bit pieces (hashx.c's non-int128 fallback):
 *
 *     p0 = al*bl, p1 = al*bh, p2 = ah*bl, p3 = ah*bh
 *     mid = hi(p0) + lo(p1) + lo(p2)
 *     hi64 = p3 + hi(p1) + hi(p2) + hi(mid)
 *
 * The plain 32-bit products in the Vulkan kernel lose those high words. */
static u64v umulh64(u64v a, u64v b)
{
    int p0_hi = mulhi32(a.lo, b.lo);
    int p1_lo = bin(OP_MUL, a.lo, b.hi);
    int p1_hi = mulhi32(a.lo, b.hi);
    int p2_lo = bin(OP_MUL, a.hi, b.lo);
    int p2_hi = mulhi32(a.hi, b.lo);
    int p3_lo = bin(OP_MUL, a.hi, b.hi);
    int p3_hi = mulhi32(a.hi, b.hi);
    u64v mid = add64(pair(p0_hi, c32(0)), pair(p1_lo, c32(0)));
    u64v r = pair(p3_lo, p3_hi);

    mid = add64(mid, pair(p2_lo, c32(0)));
    r = add64(r, pair(p1_hi, c32(0)));
    r = add64(r, pair(p2_hi, c32(0)));
    r = add64(r, pair(mid.hi, c32(0)));
    return r;
}

static u64v smulh64(u64v a, u64v b)
{
    u64v h = umulh64(a, b);
    int sa = bin(OP_SHR, a.hi, c32(31));
    int sb = bin(OP_SHR, b.hi, c32(31));

    h = sel64(cmp_ne(sa, c32(0)), sub64(h, b), h);
    return sel64(cmp_ne(sb, c32(0)), sub64(h, a), h);
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

static u64v shiftl64(u64v x, unsigned n)
{
    n &= 63u;
    if (n == 0) {
        return x;
    }
    if (n >= 32) {
        return pair(c32(0), bin(OP_SHL, x.lo, c32(n - 32)));
    }
    return pair(bin(OP_SHL, x.lo, c32(n)),
                bin(OP_OR, bin(OP_SHL, x.hi, c32(n)),
                    bin(OP_SHR, x.lo, c32(32 - n))));
}

static u64v sign_const64(uint32_t imm32)
{
    return const64((uint64_t)(int64_t)(int32_t)imm32);
}

/* ==================================================================== */
/* HashX on 8 u64 SSA values                                            */
/* ==================================================================== */

static void sipround(u64v *v)
{
    v[0] = add64(v[0], v[1]);
    v[2] = add64(v[2], v[3]);
    v[1] = rotl64(v[1], 13);
    v[3] = rotl64(v[3], 16);
    v[1] = xor64(v[1], v[0]);
    v[3] = xor64(v[3], v[2]);
    v[0] = rotl64(v[0], 32);
    v[2] = add64(v[2], v[1]);
    v[0] = add64(v[0], v[3]);
    v[1] = rotl64(v[1], 17);
    v[3] = rotl64(v[3], 21);
    v[1] = xor64(v[1], v[2]);
    v[3] = xor64(v[3], v[0]);
    v[2] = rotl64(v[2], 32);
}

/* The register key + nonce expansion from hashx.c, hashx_siphash24_ctr_state512 */
static void init_state(const uint64_t keys[4], u64v nonce, u64v r[8])
{
    u64v v[4];

    v[0] = const64(keys[0]);
    v[1] = const64(keys[1]);
    v[2] = const64(keys[2]);
    v[3] = const64(keys[3]);

    v[1] = xor64(v[1], const64(0xee));
    v[3] = xor64(v[3], nonce);
    sipround(v);
    sipround(v);
    v[0] = xor64(v[0], nonce);
    v[2] = xor64(v[2], const64(0xee));
    sipround(v);
    sipround(v);
    sipround(v);
    sipround(v);
    r[0] = v[0];
    r[1] = v[1];
    r[2] = v[2];
    r[3] = v[3];

    v[1] = xor64(v[1], const64(0xdd));
    sipround(v);
    sipround(v);
    sipround(v);
    sipround(v);
    r[4] = v[0];
    r[5] = v[1];
    r[6] = v[2];
    r[7] = v[3];
}

/* one non-branch instruction; `result` is the branch-condition byte HashX keeps
 * as a 32-bit side value, updated only by the wide multiplies */
static void exec_op(const uint32_t *in, u64v r[8], int *result)
{
    uint32_t op = in[0];
    uint32_t dst = in[1];
    uint32_t src = in[2];
    uint32_t imm = in[3];

    switch (op) {
    case 0:
        r[dst] = umulh64(r[dst], r[src]);
        *result = r[dst].lo;
        break;
    case 1:
        r[dst] = smulh64(r[dst], r[src]);
        *result = r[dst].lo;
        break;
    case 2:
        r[dst] = mul64(r[dst], r[src]);
        break;
    case 3:
        r[dst] = sub64(r[dst], r[src]);
        break;
    case 4:
        r[dst] = xor64(r[dst], r[src]);
        break;
    case 5:
        r[dst] = add64(r[dst], shiftl64(r[src], imm & 63u));
        break;
    case 6:
        r[dst] = rotr64(r[dst], imm & 63u);
        break;
    case 7:
        r[dst] = add64(r[dst], sign_const64(imm));
        break;
    case 8:
        r[dst] = xor64(r[dst], sign_const64(imm));
        break;
    case 9: /* INSTR_TARGET */
        break;
    default:
        fprintf(stderr, "hashx_pjrt: unsupported opcode %u\n", op);
        exit(1);
    }
}

/*
 * Build the module for `lanes` nonces starting at `base`. When `selftest` is
 * set, the single lane's nonce is pinned to `base` in the value tracker, so
 * the same op sequence also yields the concrete digest.
 *
 * Returns the module text; out_w[4] and out_flag receive the SSA ids of the
 * four output words and the difficulty flag.
 */
static char *build_module(int lanes, const uint64_t keys[4],
                          const uint32_t *prog, size_t n, int difficulty,
                          uint32_t base, int selftest, u64v out_w[4],
                          int *out_flag)
{
    u64v r[8];
    (void)n;
    int result, enable;
    int lane_id, base_id, nonce_id;

    build_reset(lanes);
    g_ncc = 0;
    bprintf(&g_b,
            "module {\n"
            "  func.func @main(%%n: tensor<i32>) -> tensor<%dxi8> {\n",
            lanes);
    lane_id = newval(0, 0);
    bprintf(&g_b, "    %%v%d = stablehlo.iota dim = 0 : tensor<%dxi32>\n",
            lane_id, lanes);
    base_id = newval(0, 0);
    bprintf(&g_b, "    %%v%d = stablehlo.broadcast_in_dim %%n, dims = [] "
                  ": (tensor<i32>) -> tensor<%dxi32>\n", base_id, lanes);
    nonce_id = bin(OP_ADD, lane_id, base_id);
    if (selftest) {
        v_known[lane_id] = 1;
        v_val[lane_id] = 0;
        v_known[base_id] = 1;
        v_val[base_id] = base;
        v_known[nonce_id] = 1;
        v_val[nonce_id] = base;
    }

    init_state(keys, pair(nonce_id, c32(0)), r);

    result = c32(0);
    enable = c32(1);

    for (int k = 0; k < 16; k++) {
        size_t b0 = (size_t)k * 32;
        u64v r2[8];
        int result2;

        exec_op(&prog[4 * (b0 + 0)], r, &result);

        /* first pass over the loop body */
        for (size_t i = b0 + 2; i <= b0 + 16; i++) {
            exec_op(&prog[4 * i], r, &result);
        }

        /* the branch: (enable && (result & imm) == 0) */
        {
            uint32_t imm = prog[4 * (b0 + 17) + 3];
            int masked = bin(OP_AND, result, c32(imm));
            int iszero = cv(cmp_eq(masked, c32(0)), "i1", "i32");
            int take = bin(OP_AND, enable, iszero);
            int take_pred = cmp_ne(take, c32(0));

            memcpy(r2, r, sizeof r2);
            result2 = result;
            for (size_t i = b0 + 2; i <= b0 + 16; i++) {
                exec_op(&prog[4 * i], r2, &result2);
            }
            for (int j = 0; j < 8; j++) {
                r[j] = sel64(take_pred, r2[j], r[j]);
            }
            result = sel32(take_pred, result2, result);
            enable = sel32(take_pred, c32(0), enable);
        }

        /* the rest of the block */
        for (size_t i = b0 + 18; i <= b0 + 31; i++) {
            exec_op(&prog[4 * i], r, &result);
        }
    }

    /* finalization: mix the key back in and run one SipRound per half */
    r[0] = add64(r[0], const64(keys[0]));
    r[1] = add64(r[1], const64(keys[1]));
    r[6] = add64(r[6], const64(keys[2]));
    r[7] = add64(r[7], const64(keys[3]));
    sipround(&r[0]);
    sipround(&r[4]);

    out_w[0] = xor64(r[0], r[4]);
    out_w[1] = xor64(r[1], r[5]);
    out_w[2] = xor64(r[2], r[6]);
    out_w[3] = xor64(r[3], r[7]);

    /* the digest is w0..w3 little-endian; the leading bits are the top bits of
     * bswap32(w0), which is byte 0 first */
    {
        int w = out_w[0].lo;
        int b0 = bin(OP_AND, w, c32(0x000000ffu));
        int b1 = bin(OP_AND, w, c32(0x0000ff00u));
        int b2 = bin(OP_AND, w, c32(0x00ff0000u));
        int b3 = bin(OP_AND, w, c32(0xff000000u));
        int bs = bin(OP_OR, bin(OP_SHL, b0, c32(24)),
                     bin(OP_SHL, b1, c32(8)));
        int shifted, ok, flag;

        bs = bin(OP_OR, bs, bin(OP_SHR, b2, c32(8)));
        bs = bin(OP_OR, bs, bin(OP_SHR, b3, c32(24)));
        shifted = bin(OP_SHR, bs, c32(32 - difficulty));
        ok = cmp_eq(shifted, c32(0));
        flag = cv(ok, "i1", "i8");
        bprintf(&g_b, "    return %%v%d : tensor<%dxi8>\n  }\n}\n", flag,
                lanes);
        *out_flag = flag;
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

static PJRT_Buffer *upload_i32(int32_t value)
{
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
    return ba.buffer;
}

static PJRT_Buffer *execute(PJRT_LoadedExecutable *ex, PJRT_Buffer *arg,
                            double *ms)
{
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

/* Dump the platform-specific serialized executable. This is the compiler's
 * own output (HLO + whatever the TPU backend embeds), obtained through the
 * plugin API instead of reverse-engineering libtpu. */
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
/* host side                                                            */
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

/* the generated program shape this builder relies on */
static int check_shape(const uint32_t *prog, size_t n)
{
    if (n != 512) {
        fprintf(stderr, "hashx_pjrt: expected 512 instructions, got %zu\n", n);
        return 0;
    }
    for (int k = 0; k < 16; k++) {
        size_t t = (size_t)k * 32 + 1;
        size_t b = (size_t)k * 32 + 17;

        if (prog[4 * t] != 9 || prog[4 * b] != 10) {
            fprintf(stderr,
                    "hashx_pjrt: block %d layout is not target@+1/branch@+17 "
                    "(got op %u at +1, %u at +17)\n",
                    k, prog[4 * t], prog[4 * b]);
            return 0;
        }
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t op = prog[4 * i];

        if (op == 9 && i % 32 != 1) {
            fprintf(stderr, "hashx_pjrt: stray target at %zu\n", i);
            return 0;
        }
        if (op == 10 && i % 32 != 17) {
            fprintf(stderr, "hashx_pjrt: stray branch at %zu\n", i);
            return 0;
        }
    }
    return 1;
}

static int run_selftest(const uint64_t keys[4], const uint32_t *prog,
                        size_t n, int difficulty, const uint8_t *raw,
                        size_t raw_len, int verbose)
{
    void *state = NULL;
    size_t state_size = 0;
    int fails = 0, checks = 0;
    char hex[65];

    if (hashx_init(raw, raw_len, &state, &state_size) != 0) {
        fprintf(stderr, "hashx_init failed\n");
        return 1;
    }
    for (uint32_t nonce = 0; nonce < 8; nonce++) {
        u64v w[4];
        int flag;
        uint64_t words[4];
        uint8_t want[32], got[32];
        char *module;

        module = build_module(1, keys, prog, n, difficulty, nonce, 1, w,
                              &flag);
        (void)module;
        for (int i = 0; i < 4; i++) {
            if (!v_known[w[i].lo] || !v_known[w[i].hi] || !v_known[flag]) {
                fprintf(stderr, "selftest: value tracker lost lane %u\n",
                        nonce);
                hashx_free(state);
                return 1;
            }
            words[i] = (uint64_t)v_val[w[i].lo] |
                       ((uint64_t)v_val[w[i].hi] << 32);
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                got[8 * i + j] = (uint8_t)(words[i] >> (8 * j));
            }
        }
        hashx_hash(state, state_size, nonce, want);
        checks++;
        if (memcmp(got, want, 32) != 0) {
            pjrt_to_hex(want, 32, hex);
            printf("  FAIL nonce=%u want=%s\n", nonce, hex);
            pjrt_to_hex(got, 32, hex);
            printf("            got =%s\n", hex);
            fails++;
        } else if (verbose) {
            pjrt_to_hex(got, 32, hex);
            printf("  ok   nonce=%u digest=%s\n", nonce, hex);
        }
        if (!!v_val[flag] != bits_ok(want, difficulty)) {
            printf("  FAIL nonce=%u flag=%u wanted %d\n", nonce,
                   (unsigned)v_val[flag], bits_ok(want, difficulty));
            fails++;
        }
        checks++;
    }
    hashx_free(state);
    printf("selftest: %d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}


/* ------------------------------------------------------------------ */
/* --unit: check every 64-bit pair primitive against the native one.   */
/* ------------------------------------------------------------------ */
static uint64_t model64(u64v v)
{
    return (uint64_t)v_val[v.lo] | ((uint64_t)v_val[v.hi] << 32);
}

static int unit_tests(void)
{
    uint64_t seed = 0x243f6a8885a308d3ULL;
    int fails = 0, checks = 0;

    build_reset(1);
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
        CHECK("umulh64", model64(umulh64(const64(a), const64(b))),
              (uint64_t)(((__uint128_t)a * b) >> 64));
        CHECK("smulh64",
              model64(smulh64(const64(a), const64(b))),
              (uint64_t)(((__int128)(int64_t)a * (int64_t)b) >> 64));
        CHECK("mulhi32",
              (uint64_t)v_val[mulhi32(c32((uint32_t)a), c32((uint32_t)b))],
              (uint64_t)(((uint64_t)(uint32_t)a * (uint32_t)b) >> 32));
        for (unsigned n = 0; n < 64; n += 3) {
            uint64_t want = n ? ((a << n) | (a >> (64 - n))) : a;

            CHECK("rotl64", model64(rotl64(const64(a), n)), want);
            want = n ? ((a >> n) | (a << (64 - n))) : a;
            CHECK("rotr64", model64(rotr64(const64(a), n)), want);
            want = (n >= 64) ? 0 : (n ? (a << n) : a);
            CHECK("shiftl64", model64(shiftl64(const64(a), n)), want);
        }
    }
#undef CHECK
    printf("unit: %d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}

static const char DEFAULT_DATA[] =
    "9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b";

int main(int argc, char **argv)
{
    const char *lib = getenv("LIBTPU");
    const char *data = DEFAULT_DATA;
    int difficulty = 15, lanes = 16384, json = 0, selftest = 0, emit = 0;
    const char *serialize_path = NULL;
    int force = 0, init_timeout = 120;
    long long bench_hashes = 0;
    uint8_t raw[4096];
    size_t raw_len;
    uint64_t keys[4];
    static uint32_t prog[4 * 512];
    size_t n = 0;
    void *state = NULL;
    size_t state_size = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc) {
            lib = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data = argv[++i];
        } else if (strcmp(argv[i], "--difficulty") == 0 && i + 1 < argc) {
            difficulty = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            lanes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-hashes") == 0 && i + 1 < argc) {
            bench_hashes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--selftest") == 0) {
            selftest = 2;
        } else if (strcmp(argv[i], "--unit") == 0) {
            selftest = 3;
        } else if (strcmp(argv[i], "--emit") == 0) {
            emit = 1;
        } else if (strcmp(argv[i], "--serialize") == 0 && i + 1 < argc) {
            serialize_path = argv[++i];
        } else if (strcmp(argv[i], "--force") == 0) {
            force = 1;
        } else if (strcmp(argv[i], "--init-timeout") == 0 && i + 1 < argc) {
            init_timeout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else {
            fprintf(stderr,
                    "usage: %s --lib libtpu.so [--data HEX] "
                    "[--difficulty BITS] [--window LANES] "
                    "[--bench-hashes N] [--json] [--emit] "
                    "[--serialize FILE] [--force] [--init-timeout S]\n"
                    "       %s --selftest [--data HEX] [--difficulty BITS]\n",
                    argv[0], argv[0]);
            return 2;
        }
    }
    if (difficulty < 1 || difficulty > 32) {
        fprintf(stderr, "--difficulty must be 1..32\n");
        return 2;
    }
    raw_len = pjrt_hex_decode(data, raw, sizeof raw);
    if (raw_len == 0) {
        fprintf(stderr, "--data must be non-empty hex\n");
        return 2;
    }
    if (hashx_init(raw, raw_len, &state, &state_size) != 0) {
        fprintf(stderr, "hashx_init failed\n");
        return 1;
    }
    hashx_export(state, state_size, keys, prog, &n);
    hashx_free(state);
    if (!check_shape(prog, n)) {
        return 1;
    }

    if (selftest == 3) {
        return unit_tests();
    }
    if (emit) {
        u64v w[4];
        int flag;

        if (lanes < 1 || lanes > 16777216) {
            fprintf(stderr, "--window out of range\n");
            return 2;
        }
        printf("%s", build_module(lanes, keys, prog, n, difficulty, 0, 0, w,
                                  &flag));
        return 0;
    }
    if (selftest == 2) {
        return run_selftest(keys, prog, n, difficulty, raw, raw_len, 1);
    }

    if (lib == NULL) {
        fprintf(stderr, "need --lib or LIBTPU\n");
        return 2;
    }
    if (lanes < 1 || lanes > 16777216) {
        fprintf(stderr, "--window out of range\n");
        return 2;
    }

    /* Preflight: libtpu's plugin init blocks on a host with no TPU instead of
     * returning an error (measured: >25 s, and `timeout` could not kill it),
     * so check for a VFIO group or an accel node and fail fast, then keep a
     * hard alarm around the initialization anyway. --force skips the check. */
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
    {
        u64v w[4];
        int flag;
        char *module = build_module(lanes, keys, prog, n, difficulty, 0, 0, w,
                                    &flag);
        PJRT_LoadedExecutable *ex;
        int compile_ms = 0;
        uint8_t *out;

        (void)w;
        (void)flag;
        ex = compile(module, &compile_ms);
        if (serialize_path != NULL) {
            serialize_executable(ex, serialize_path);
        }
        out = malloc((size_t)lanes);
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
             * whatever the backend defers */
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
                if (base == 0) {
                    break;
                }
            }

            {
                double ms = (pjrt_now_s() - t0) * 1000.0;
                int verified = 0;
                char hex[65] = "";

                if (found) {
                    uint8_t ref[32];
                    void *st2 = NULL;
                    size_t ss2 = 0;

                    if (hashx_init(raw, raw_len, &st2, &ss2) == 0) {
                        hashx_hash(st2, ss2, found_nonce, ref);
                        hashx_free(st2);
                        verified = bits_ok(ref, difficulty);
                        pjrt_to_hex(ref, 32, hex);
                    }
                }
                if (json) {
                    printf("{\"hashes\":%lld,\"ms\":%.3f,\"mhash_s\":%.3f,"
                           "\"window\":%d,\"difficulty\":%d,"
                           "\"compile_ms\":%d,\"nonce\":%u,"
                           "\"digest\":\"%s\",\"verified\":%s,\"hits\":%lld}\n",
                           done, ms, (double)done / (ms / 1000.0) / 1e6, lanes,
                           difficulty, compile_ms, found ? found_nonce : 0, hex,
                           verified ? "true" : "false", hits);
                } else {
                    printf("lanes=%d difficulty=%d hashes=%lld %.1f ms = "
                           "%.2f MHash/s compile=%d ms\n",
                           lanes, difficulty, done, ms,
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
}
