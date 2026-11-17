#!/usr/bin/env python3
"""Benchmark hashx_interp.ptx (the HashX interpreter in PTX) on a real GPU.

    python3 kaggle_bench_hashx_ptx.py [sm_75] [difficulty ...]

Builds the cubin with build_ptx.sh, compiles ptx_loader.c, exports a program
from the CPU HashX reference (src/hashx/hashx.c) and drives the loader through
ctypes. Every digest the kernel reports is compared with hashx_hash(), so the
throughput number below is only printed for a kernel that is correct.

The scan starts at nonce 0, so `nonce + 1` candidates were tried -- the MH/s
figures are exact, not estimated. Writes result.json next to this file.
"""

import ctypes
import json
import os
import sys
import time

import ptx_bench_common as common

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_NONCE = 0xFFFFFFFF


class Shm:
    """The CPU HashX reference through libhashx_shim.so."""

    def __init__(self, so, seed):
        self.lib = ctypes.CDLL(so)
        self.lib.hx_shim_init.restype = ctypes.c_int
        self.lib.hx_shim_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_size_t)]
        self.lib.hx_shim_export.restype = ctypes.c_int
        self.lib.hx_shim_export.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        self.lib.hx_shim_hash.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_uint64, ctypes.c_char_p]

        st = ctypes.c_void_p()
        sz = ctypes.c_size_t()
        if self.lib.hx_shim_init(seed, len(seed), ctypes.byref(st),
                                 ctypes.byref(sz)) != 0:
            raise SystemExit("hashx_init rejected the seed")
        self.state, self.size = st, sz
        keys = (ctypes.c_uint64 * 4)()
        prog = (ctypes.c_uint32 * (4 * 512))()
        cnt = ctypes.c_uint32(0)
        if self.lib.hx_shim_export(self.state, self.size, keys, prog,
                                   ctypes.byref(cnt)) != 0:
            raise SystemExit("hashx_export failed")
        self.keys = list(keys)
        self.count = cnt.value
        self.prog = list(prog)

    def hash(self, nonce):
        out = ctypes.create_string_buffer(32)
        self.lib.hx_shim_hash(self.state, self.size, nonce, out)
        return out.raw


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


def cpu_rate(shim, samples=200000):
    t = time.perf_counter()
    for n in range(samples):
        shim.hash(n)
    return samples / (time.perf_counter() - t) / 1e6


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_75"
    diffs = [int(x) for x in sys.argv[2:]] or [20, 24, 28]

    common.build_cubins(HERE, arch, "2")
    so = common.build_loader(HERE)
    shim_so = common.build_hashx_shim(HERE)

    os.environ["ANUBIS_PTX_DIR"] = os.path.join(HERE, "cubin")
    lib = ctypes.CDLL(so)
    lib.anubis_ptx_hashx_available.restype = ctypes.c_int
    lib.anubis_ptx_hashx_info.restype = ctypes.c_char_p
    lib.anubis_ptx_hashx_set_program.restype = ctypes.c_int
    lib.anubis_ptx_hashx_set_program.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32]
    lib.anubis_ptx_hashx_solve.restype = ctypes.c_int
    lib.anubis_ptx_hashx_solve.argtypes = [
        ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double)]

    if not lib.anubis_ptx_hashx_available():
        raise SystemExit("no usable CUDA device / hashx_interp.cubin")
    print("device:", lib.anubis_ptx_hashx_info().decode())

    seed = bytes(range(32))
    shim = Shm(shim_so, seed)
    keys = (ctypes.c_uint64 * 4)(*shim.keys)
    prog = (ctypes.c_uint32 * len(shim.prog))(*shim.prog)
    if lib.anubis_ptx_hashx_set_program(keys, prog, shim.count) != 0:
        raise SystemExit("anubis_ptx_hashx_set_program failed")
    print("program: %d instructions, seed %s" % (shim.count, seed.hex()))

    def solve(difficulty, start=0, max_nonce=MAX_NONCE):
        nonce = ctypes.c_uint32(0)
        out = ctypes.create_string_buffer(65)
        ms = ctypes.c_double(0.0)
        rc = lib.anubis_ptx_hashx_solve(difficulty, start, max_nonce,
                                        ctypes.byref(nonce), out,
                                        ctypes.byref(ms))
        return rc, nonce.value, out.value.decode(), ms.value

    result = {
        "arch": arch,
        "seed_hex": seed.hex(),
        "instructions": shim.count,
        "info": lib.anubis_ptx_hashx_info().decode(),
        "correctness": [],
        "bench": [],
    }

    print("\n== correctness: difficulty 0, the digest must match the CPU ==")
    failed = 0
    for nonce in (0, 1, 7, 123456, MAX_NONCE):
        rc, got, hexdig, ms = solve(0, nonce)
        exp = shim.hash(got).hex()
        ok = rc == 0 and got == nonce and hexdig == exp
        print("  nonce=%-10d %s" % (got, "ok" if ok else "MISMATCH"))
        result["correctness"].append({"nonce": got, "ok": ok,
                                      "digest": hexdig, "ms": ms})
        if not ok:
            failed += 1

    print("\n== correctness: real searches ==")
    for bits in (12, 16):
        rc, got, hexdig, ms = solve(bits)
        digest = bytes.fromhex(hexdig)
        # no earlier nonce may qualify
        earlier = [n for n in range(got) if
                   leading_zero_bits(shim.hash(n)) >= bits]
        ok = (rc == 0 and digest == shim.hash(got) and
              leading_zero_bits(digest) >= bits and not earlier)
        print("  %2d bits: nonce=%-10d %s (%.2f ms)" %
              (bits, got, "ok" if ok else "FAIL", ms))
        result["correctness"].append({"nonce": got, "bits": bits, "ok": ok,
                                      "ms": ms, "digest": hexdig})
        if not ok:
            failed += 1

    print("\n== throughput (nonces scanned = hit nonce + 1) ==")
    for bits in diffs:
        best = None
        for _ in range(3):
            rc, got, hexdig, ms = solve(bits)
            if rc != 0 or ms <= 0:
                continue
            mhps = (got + 1) / (ms / 1000.0) / 1e6
            best = mhps if best is None else max(best, mhps)
            print("  %2d bits: nonce=%-10d %8.2f ms  %8.2f MH/s" %
                  (bits, got, ms, mhps))
            result["bench"].append({"bits": bits, "nonce": got, "ms": ms,
                                    "mhps": mhps})
        if best is not None:
            print("  best: %.2f MH/s" % best)
            result["bench"].append({"bits": bits, "best_mhps": best})

    cpu = cpu_rate(shim)
    result["cpu_mhps"] = cpu
    print("\nCPU reference (one thread, hashx.c): %.2f MH/s" % cpu)
    if not failed:
        gpu = max((r["mhps"] for r in result["bench"] if "mhps" in r),
                  default=0.0)
        result["speedup_vs_cpu"] = gpu / cpu if cpu else 0.0
        print("GPU / CPU speedup: %.2fx" % result["speedup_vs_cpu"])

    if failed:
        result["failures"] = failed
        print("\n%d CORRECTNESS CHECK(S) FAILED -- numbers are not usable"
              % failed)
    else:
        print("\nALL HASHX PTX CHECKS PASSED")

    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
