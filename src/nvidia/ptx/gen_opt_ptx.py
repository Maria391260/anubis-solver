#!/usr/bin/env python3
"""Generate sha256_opt.ptx -- the register-only, ILP variant of the Anubis
SHA-256 proof-of-work kernel (mode 1: fixed u32 nonce only).

Why a generator: a register-only message schedule needs the 16 schedule words
in named registers, and a runtime loop cannot index registers. The 64 rounds
are therefore fully unrolled, and ILP is emitted by interleaving N independent
nonce lanes. Both of those are mechanical, so the PTX is generated here and the
result is checked in (regenerate with `python3 gen_opt_ptx.py --ilp 2`).

`--loop` adds an outer grid-stride loop: a thread keeps hashing
(blockDim*gridDim*ilp) nonces further on until it runs past `count`, instead of
dying after one lane group. The loop count is not baked in -- the host sizes the
grid from ANUBIS_PTX_LOOP -- so one cubin serves every loop depth. Two reasons
it is faster than one hash per thread:

  * a block lives for many nonces, so block ramp-up/drain and the prologue are
    amortised;
  * everything in the message schedule that does not depend on the nonce is
    loop-invariant (only w0 varies; w1..w15 are constants, so w17, w19 ... are
    warp-uniform), and ptxas can hoist it out of the loop.

Usage: python3 gen_opt_ptx.py [--ilp 1|2|4] [--loop] > sha256_opt.ptx
"""

import argparse

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

STATE = ["a", "b", "c", "d", "e", "f", "g", "h"]


# When --imad-add is on, every register+register add is emitted as
# `mad.lo.u32 d, a, %one, b` with `one` a kernel parameter that is always 1.
# ptxas cannot fold a runtime multiplier, so the add issues as IMAD on the FMA
# pipe instead of IADD3 on the ALU pipe. SHA-256 is ALU-bound (shifts and LOP3
# can only run there), so moving the adds off it is the only way to raise IPC
# above the 2 warp-instructions/clock the ALU pipe sustains.
IMAD_ADD = False


def add(w, dst, a, b):
    if IMAD_ADD:
        w(f"    mad.lo.u32 {dst}, {a}, %one, {b};")
    else:
        w(f"    add.u32 {dst}, {a}, {b};")


def generate(ilp: int, loop: bool = False) -> str:
    out = []
    w = out.append

    w("/*")
    w(" * sha256_opt.ptx -- register-only Anubis SHA-256 PoW kernel (mode 1).")
    w(" *")
    w(" * GENERATED FILE -- do not edit. Regenerate with:")
    if loop:
        w(f" *     python3 gen_opt_ptx.py --ilp {ilp} --loop > sha256_opt_loop.ptx")
    else:
        w(f" *     python3 gen_opt_ptx.py --ilp {ilp} > sha256_opt.ptx")
    w(" *")
    w(f" * One thread hashes {ilp} independent nonce(s) (ILP={ilp}). The 16-word")
    w(" * message schedule is held in registers and the 64 rounds are fully")
    w(" * unrolled; no local memory is used. Hits are reported through")
    w(" * results_ptr in the same layout as sha256_pow.ptx.")
    if loop:
        w(" *")
        w(" * This variant wraps the lane group in a grid-stride loop: the host")
        w(" * sizes the grid to count/(block*ilp*ANUBIS_PTX_LOOP), so a thread")
        w(" * hashes about ANUBIS_PTX_LOOP lane groups before it falls off the")
        w(" * end of the range. Loop-invariant message-schedule words stay")
        w(" * hoisted out of the loop.")
    w(" */")
    w("")
    w(".version 7.5")
    w(".target sm_75")
    w(".address_size 64")
    w("")
    w(".visible .entry sha256_opt_kernel(")
    w("    .param .u32 mid0, .param .u32 mid1, .param .u32 mid2, .param .u32 mid3,")
    w("    .param .u32 mid4, .param .u32 mid5, .param .u32 mid6, .param .u32 mid7,")
    w("    .param .u32 prefix_len,")
    w("    .param .u32 mode,")
    w("    .param .u32 use_le,")
    w("    .param .u32 difficulty,")
    w("    .param .u32 check_bits,")
    w("    .param .u32 base,")
    w("    .param .u32 count,")
    if IMAD_ADD:
        w("    .param .u32 one,")
    w("    .param .u64 results_ptr")
    w(") {")
    w("    .reg .pred %p, %pd;")
    w("    .reg .u32 %mid0, %mid1, %mid2, %mid3, %mid4, %mid5, %mid6, %mid7;")
    w("    .reg .u32 %prefix_len, %use_le, %difficulty, %check_bits, %base, %count;")
    w("    .reg .u64 %res;")
    w("    .reg .u32 %nt, %ct, %tx, %idx, %first, %blen;")
    if loop:
        w("    .reg .u32 %nc, %stride;")
    w("    .reg .u32 %nbits, %fullw, %remb, %amt, %ok, %slot, %off;")
    w("    .reg .u32 %s0, %s1, %x, %ch, %maj, %t1, %t2;")
    if IMAD_ADD:
        w("    .reg .u32 %one;")
    w("    .reg .u64 %rd;")
    for u in range(ilp):
        regs = [f"%{s}{u}" for s in STATE]
        regs += [f"%w{u}_{k}" for k in range(16)]
        regs += [f"%nonce{u}", f"%fi{u}"]
        regs += [f"%o{u}_{k}" for k in range(8)]
        w("    .reg .u32 " + ", ".join(regs) + ";")
        w(f"    .reg .pred %pact{u};")
    w("")

    for name in ["mid0", "mid1", "mid2", "mid3", "mid4", "mid5", "mid6", "mid7",
                 "prefix_len", "use_le", "difficulty", "check_bits", "base", "count"]:
        w(f"    ld.param.u32 %{name}, [{name}];")
    if IMAD_ADD:
        w("    ld.param.u32 %one, [one];")
    w("    ld.param.u64 %res, [results_ptr];")
    w("")
    w("    mov.u32 %nt, %ntid.x;")
    w("    mov.u32 %ct, %ctaid.x;")
    w("    mov.u32 %tx, %tid.x;")
    w("    mad.lo.u32 %idx, %ct, %nt, %tx;")
    w(f"    mul.lo.u32 %first, %idx, {ilp};")
    if loop:
        # stride = nctaid.x * ntid.x * ilp, i.e. every nonce group this grid
        # covers in one pass; the loop repeats until `first` runs past count.
        w("    mov.u32 %nc, %nctaid.x;")
        w("    mul.lo.u32 %stride, %nc, %nt;")
        w(f"    mul.lo.u32 %stride, %stride, {ilp};")
        w("")
        w("    add.u32 %x, %prefix_len, 4;")
        w("    shl.b32 %blen, %x, 3;")
        w("")
        w("    /* difficulty -> leading zero bits (a nibble is 4 bits) */")
        w("    mov.u32 %nbits, %difficulty;")
        w("    setp.ne.u32 %p, %check_bits, 0;")
        w("    @%p bra BB_NBITS;")
        w("    mul.lo.u32 %nbits, %difficulty, 4;")
        w("BB_NBITS:")
        w("    shr.u32 %fullw, %nbits, 5;")
        w("    and.b32 %remb, %nbits, 31;")
        w("    mov.u32 %amt, 32;")
        w("    sub.u32 %amt, %amt, %remb;")
        w("")
        w("L_LOOP:")
    w("    setp.ge.u32 %p, %first, %count;")
    w("    @%p bra BB_DONE;")
    w("")
    if not loop:
        w("    add.u32 %x, %prefix_len, 4;")
        w("    shl.b32 %blen, %x, 3;")
        w("")
        w("    /* difficulty -> leading zero bits (a nibble is 4 bits) */")
        w("    mov.u32 %nbits, %difficulty;")
        w("    setp.ne.u32 %p, %check_bits, 0;")
        w("    @%p bra BB_NBITS;")
        w("    mul.lo.u32 %nbits, %difficulty, 4;")
        w("BB_NBITS:")
        w("    shr.u32 %fullw, %nbits, 5;")
        w("    and.b32 %remb, %nbits, 31;")
        w("    mov.u32 %amt, 32;")
        w("    sub.u32 %amt, %amt, %remb;")
        w("")

    # ---- lane init: schedule words and state ----
    for u in range(ilp):
        w(f"    /* ---- lane {u} init ---- */")
        w(f"    add.u32 %nonce{u}, %base, %first;")
        w(f"    add.u32 %nonce{u}, %nonce{u}, {u};")
        w(f"    add.u32 %fi{u}, %first, {u};")
        w(f"    setp.lt.u32 %pact{u}, %fi{u}, %count;")
        w("    setp.eq.u32 %pd, %use_le, 1;")
        w(f"    @%pd bra L{u}_LE;")
        w(f"    mov.u32 %w{u}_0, %nonce{u};")
        w(f"    bra L{u}_W0D;")
        w(f"L{u}_LE:")
        w(f"    and.b32 %x, %nonce{u}, 0xff;")
        w("    shl.b32 %x, %x, 24;")
        w(f"    and.b32 %t1, %nonce{u}, 0xff00;")
        w("    shl.b32 %t1, %t1, 8;")
        w("    or.b32 %x, %x, %t1;")
        w(f"    shr.b32 %t1, %nonce{u}, 8;")
        w("    and.b32 %t1, %t1, 0xff00;")
        w("    or.b32 %x, %x, %t1;")
        w(f"    shr.b32 %t1, %nonce{u}, 24;")
        w("    or.b32 %x, %x, %t1;")
        w(f"    mov.u32 %w{u}_0, %x;")
        w(f"L{u}_W0D:")
        w(f"    mov.u32 %w{u}_1, 0x80000000;")
        for k in range(2, 15):
            w(f"    mov.u32 %w{u}_{k}, 0;")
        w(f"    mov.u32 %w{u}_15, %blen;")
        for k, s in enumerate(STATE):
            w(f"    mov.u32 %{s}{u}, %mid{k};")
        w("")

    # ---- 64 rounds, interleaved across lanes for ILP ----
    w(f"    /* ---- 64 rounds, {ilp} lane(s) interleaved ---- */")
    for i in range(64):
        for u in range(ilp):
            emit_round(w, u, i)
    w("")

    # ---- finalize each lane ----
    for u in range(ilp):
        w(f"    /* ---- lane {u} finalize ---- */")
        for k, s in enumerate(STATE):
            w(f"    add.u32 %o{u}_{k}, %mid{k}, %{s}{u};")
        w(f"    @!%pact{u} bra L{u}_DONE;")
        w("    mov.u32 %ok, 1;")
        for j in range(8):
            w(f"    setp.ge.u32 %p, %fullw, {j + 1};")
            w(f"    @!%p bra L{u}_SK{j};")
            w(f"    setp.ne.u32 %pd, %o{u}_{j}, 0;")
            w(f"    @%pd bra L{u}_FAIL;")
            w(f"L{u}_SK{j}:")
            w(f"    setp.eq.u32 %p, %fullw, {j};")
            w(f"    @!%p bra L{u}_SP{j};")
            w("    setp.eq.u32 %pd, %remb, 0;")
            w(f"    @%pd bra L{u}_SP{j};")
            w(f"    shr.u32 %t1, %o{u}_{j}, %amt;")
            w("    setp.ne.u32 %pd, %t1, 0;")
            w(f"    @%pd bra L{u}_FAIL;")
            w(f"L{u}_SP{j}:")
        w(f"    bra L{u}_CHK_DONE;")
        w(f"L{u}_FAIL:")
        w("    mov.u32 %ok, 0;")
        w(f"L{u}_CHK_DONE:")
        w("    setp.eq.u32 %p, %ok, 0;")
        w(f"    @%p bra L{u}_DONE;")
        w("    atom.global.add.u32 %slot, [%res], 1;")
        w("    setp.ge.u32 %p, %slot, 8;")
        w(f"    @%p bra L{u}_DONE;")
        w("    mul.lo.u32 %off, %slot, 4;")
        w("    add.u32 %off, %off, 16;")
        w("    cvt.u64.u32 %rd, %off;")
        w("    add.u64 %rd, %res, %rd;")
        w(f"    st.global.u32 [%rd], %nonce{u};")
        w("    mul.lo.u32 %off, %slot, 32;")
        w("    add.u32 %off, %off, 48;")
        w("    cvt.u64.u32 %rd, %off;")
        w("    add.u64 %rd, %res, %rd;")
        for k in range(8):
            w(f"    st.global.u32 [%rd+{4 * k}], %o{u}_{k};")
        w(f"L{u}_DONE:")
        w("")

    if loop:
        w("    add.u32 %first, %first, %stride;")
        w("    bra L_LOOP;")
        w("")
    w("BB_DONE:")
    w("    ret;")
    w("}")
    w("")
    return "\n".join(out)


def emit_round(w, u, i):
    j = i & 15
    if i >= 16:
        j1 = (i + 1) & 15
        j14 = (i + 14) & 15
        j9 = (i + 9) & 15
        w(f"    shf.r.wrap.b32 %s0, %w{u}_{j1}, %w{u}_{j1}, 7;")
        w(f"    shf.r.wrap.b32 %x, %w{u}_{j1}, %w{u}_{j1}, 18;")
        w("    xor.b32 %s0, %s0, %x;")
        w(f"    shr.b32 %x, %w{u}_{j1}, 3;")
        w("    xor.b32 %s0, %s0, %x;")
        w(f"    shf.r.wrap.b32 %s1, %w{u}_{j14}, %w{u}_{j14}, 17;")
        w(f"    shf.r.wrap.b32 %x, %w{u}_{j14}, %w{u}_{j14}, 19;")
        w("    xor.b32 %s1, %s1, %x;")
        w(f"    shr.b32 %x, %w{u}_{j14}, 10;")
        w("    xor.b32 %s1, %s1, %x;")
        add(w, f"%w{u}_{j}", f"%w{u}_{j}", "%s0")
        add(w, f"%w{u}_{j}", f"%w{u}_{j}", f"%w{u}_{j9}")
        add(w, f"%w{u}_{j}", f"%w{u}_{j}", "%s1")
    w(f"    shf.r.wrap.b32 %s1, %e{u}, %e{u}, 6;")
    w(f"    shf.r.wrap.b32 %x, %e{u}, %e{u}, 11;")
    w("    xor.b32 %s1, %s1, %x;")
    w(f"    shf.r.wrap.b32 %x, %e{u}, %e{u}, 25;")
    w("    xor.b32 %s1, %s1, %x;")
    w(f"    and.b32 %ch, %e{u}, %f{u};")
    w(f"    not.b32 %x, %e{u};")
    w(f"    and.b32 %x, %x, %g{u};")
    w("    xor.b32 %ch, %ch, %x;")
    add(w, "%t1", f"%h{u}", "%s1")
    add(w, "%t1", "%t1", "%ch")
    w(f"    add.u32 %t1, %t1, 0x{K[i]:08x};")
    add(w, "%t1", "%t1", f"%w{u}_{j}")
    w(f"    shf.r.wrap.b32 %s0, %a{u}, %a{u}, 2;")
    w(f"    shf.r.wrap.b32 %x, %a{u}, %a{u}, 13;")
    w("    xor.b32 %s0, %s0, %x;")
    w(f"    shf.r.wrap.b32 %x, %a{u}, %a{u}, 22;")
    w("    xor.b32 %s0, %s0, %x;")
    w(f"    and.b32 %maj, %a{u}, %b{u};")
    w(f"    and.b32 %x, %a{u}, %c{u};")
    w("    xor.b32 %maj, %maj, %x;")
    w(f"    and.b32 %x, %b{u}, %c{u};")
    w("    xor.b32 %maj, %maj, %x;")
    add(w, "%t2", "%s0", "%maj")
    w(f"    mov.u32 %h{u}, %g{u};")
    w(f"    mov.u32 %g{u}, %f{u};")
    w(f"    mov.u32 %f{u}, %e{u};")
    add(w, f"%e{u}", f"%d{u}", "%t1")
    w(f"    mov.u32 %d{u}, %c{u};")
    w(f"    mov.u32 %c{u}, %b{u};")
    w(f"    mov.u32 %b{u}, %a{u};")
    add(w, f"%a{u}", "%t1", "%t2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ilp", type=int, default=2, choices=(1, 2, 4))
    ap.add_argument("--loop", action="store_true",
                    help="grid-stride loop: one thread hashes many nonce groups")
    ap.add_argument("--imad-add", action="store_true",
                    help="emit adds as mad.lo.u32 with a runtime 1 so they "
                         "issue on the FMA pipe (adds a `one` kernel parameter)")
    args = ap.parse_args()
    global IMAD_ADD
    IMAD_ADD = bool(args.imad_add)
    print(generate(args.ilp, args.loop))


if __name__ == "__main__":
    main()
