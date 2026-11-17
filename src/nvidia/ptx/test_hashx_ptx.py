#!/usr/bin/env python3
"""Check hashx_interp.ptx against the CPU HashX reference, with no GPU.

The PTX is executed by ptxsim.py, one thread at a time, and every digest it
reports is compared with hashx_hash() from src/hashx/hashx.c (through the tiny
libhashx_shim.so wrapper built here).

    python3 test_hashx_ptx.py

What is covered:

  - the SipHash register expansion for arbitrary nonces (including the
    upper end of the u32 range), via the difficulty-0 path;
  - interpretation of real generated programs, including the branch
    instruction, by searching for low difficulties and re-checking every
    stored hit;
  - the leading-zero-bit test itself: every hit must have exactly the
    requested number of zero bits, and non-hits must not;
  - the ILP-free launch geometry (more than one block) and the result slots.
"""

import ctypes
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ptxsim  # noqa: E402

SHIM = os.path.join(HERE, "libhashx_shim.so")
PROG_BASE = 1 << 20          # where the program lives inside simulated global
GM_SIZE = PROG_BASE + 512 * 16 + 64
MAX_INSTR = 512


def build_shim():
    src = os.path.join(HERE, "hashx_shim.c")
    ref = os.path.join(HERE, "..", "..", "hashx", "hashx.c")
    r = subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", SHIM, ref, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("shim build failed:\n" + r.stderr)


class Ref:
    def __init__(self, data):
        build_shim()
        self.lib = ctypes.CDLL(SHIM)
        self.lib.hx_shim_init.restype = ctypes.c_int
        self.lib.hx_shim_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_size_t)]
        self.lib.hx_shim_export.restype = ctypes.c_int
        self.lib.hx_shim_export.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32)]
        self.lib.hx_shim_hash.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_uint64, ctypes.c_char_p]

        st = ctypes.c_void_p()
        sz = ctypes.c_size_t()
        if self.lib.hx_shim_init(data, len(data), ctypes.byref(st),
                                 ctypes.byref(sz)) != 0:
            raise SystemExit("hashx_init failed")
        self.state, self.size = st, sz

        keys = (ctypes.c_uint64 * 4)()
        prog = (ctypes.c_uint32 * (4 * MAX_INSTR))()
        cnt = ctypes.c_uint32(0)
        if self.lib.hx_shim_export(self.state, self.size, keys, prog,
                                   ctypes.byref(cnt)) != 0:
            raise SystemExit("hashx_export failed")
        self.keys = list(keys)
        self.count = cnt.value
        self.prog = list(prog)[:4 * self.count]

    def hash(self, nonce):
        out = ctypes.create_string_buffer(32)
        self.lib.hx_shim_hash(self.state, self.size, nonce, out)
        return out.raw

    def close(self):
        self.lib.hx_shim_free(self.state)


def kernel_params(ref):
    return {
        'k0': ref.keys[0], 'k1': ref.keys[1],
        'k2': ref.keys[2], 'k3': ref.keys[3],
        'prog_ptr': ptxsim.GLOBAL_BASE + PROG_BASE,
        'code_size': ref.count,
        'results_ptr': ptxsim.GLOBAL_BASE,
    }


def make_global(ref):
    gm = bytearray(GM_SIZE)
    for i, v in enumerate(ref.prog):
        off = PROG_BASE + 4 * i
        gm[off:off + 4] = struct.pack('<I', v)
    return gm


def run(k, ref, base, count, difficulty, blockdim=128):
    params = kernel_params(ref)
    params['difficulty'] = difficulty
    params['base'] = base
    params['count'] = count
    gm = make_global(ref)
    blocks = (count + blockdim - 1) // blockdim
    for b in range(blocks):
        for tx in range(blockdim):
            k.run_thread(params, gm, blockdim, b, tx)
    return gm


def leading_zero_bits(digest):
    n = 0
    for byte in digest:
        if byte == 0:
            n += 8
            continue
        for bit in range(7, -1, -1):
            if (byte >> bit) & 1:
                return n
            n += 1
        break
    return n


def main():
    print("== parsing hashx_interp.ptx ==")
    k = ptxsim.Kernel(os.path.join(HERE, "hashx_interp.ptx"))
    print("  %d PTX instructions, %d local bytes" % (len(k.instrs), k.local_size))

    data = bytes(range(32))          # any seed the generator accepts
    ref = Ref(data)
    print("  program: %d instructions" % ref.count)
    ops = {}
    for i in range(ref.count):
        op = ref.prog[4 * i]
        ops[op] = ops.get(op, 0) + 1
    print("  opcodes: %s" % " ".join("%d:%d" % kv for kv in sorted(ops.items())))

    ok = 0
    failed = 0

    print("== difficulty 0: the digest must match the CPU exactly ==")
    for base, count in ((0, 1), (1, 1), (12345, 3), (0xFFFFFFF0, 4),
                        (1000, 8)):
        gm = run(k, ref, base, count, 0)
        n, nonces, hashes = ptxsim.results(gm)
        if n != count:
            print("  FAIL base=%d count=%d: %d hits" % (base, count, n))
            failed += 1
            continue
        bad = 0
        for i in range(count):
            nonce = nonces[i]
            if nonce != base + i or hashes[i] != ref.hash(nonce):
                bad += 1
        if bad:
            print("  FAIL base=%d count=%d: %d wrong digests" % (base, count, bad))
            failed += 1
        else:
            print("  ok   base=%-10d count=%-4d" % (base, count))
            ok += 1

    print("== bit difficulty: hits are correct and really have the zeros ==")
    for bits in (4, 8):
        base, count = 5000, 400
        gm = run(k, ref, base, count, bits)
        n, nonces, hashes = ptxsim.results(gm)
        expected = sum(1 for nonce in range(base, base + count)
                       if leading_zero_bits(ref.hash(nonce)) >= bits)
        bad = 0
        for i in range(min(n, 8)):
            nonce = nonces[i]
            if hashes[i] != ref.hash(nonce) or leading_zero_bits(hashes[i]) < bits:
                bad += 1
        if n != expected:
            bad += 1                  # hits found != hits that exist
        if bad:
            print("  FAIL %d bits: %d bad (found %d, expected %d)" %
                  (bits, bad, n, expected))
            failed += 1
        else:
            print("  ok   %d bits: %d hit(s) == %d real solutions, all correct" %
                  (bits, n, expected))
            ok += 1

    print("== multiple blocks: 200 nonces over 2 blocks of 128 threads ==")
    gm = run(k, ref, 1000, 200, 0)
    n, nonces, hashes = ptxsim.results(gm)
    if n == 200 and nonces[:8] == list(range(1000, 1008)) and \
            hashes[0] == ref.hash(1000) and hashes[7] == ref.hash(1007):
        print("  ok   count=%d, stored slots are nonces 1000..1007" % n)
        ok += 1
    else:
        print("  FAIL count=%d nonces=%s" % (n, nonces[:8]))
        failed += 1

    print("== result slots: at most 8, count keeps incrementing ==")
    gm = run(k, ref, 0, 64, 0)
    n, nonces, hashes = ptxsim.results(gm)
    if n == 64 and nonces[:8] == list(range(8)):
        print("  ok   count=%d, first 8 nonces stored" % n)
        ok += 1
    else:
        print("  FAIL count=%d nonces=%s" % (n, nonces[:8]))
        failed += 1

    ref.close()
    print()
    if failed:
        print("%d CHECK(S) FAILED" % failed)
        return 1
    print("ALL %d HASHX PTX CHECKS PASSED" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
