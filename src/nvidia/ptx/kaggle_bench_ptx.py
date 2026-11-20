#!/usr/bin/env python3
"""Benchmark the real hand-written PTX kernels on a GPU (e.g. a Kaggle T4).

Unlike the previous version, this does not embed stub PTX. It builds the
checked-in kernels with build_ptx.sh, compiles ptx_loader.c, and drives it
through ctypes.

    python3 src/nvidia/ptx/kaggle_bench_ptx.py [sm_75] [ilp] [loop] [block]

`loop` selects sha256_opt_loop_ilp<N>.cubin and the grid-stride depth
(ANUBIS_PTX_LOOP); `block` is the threads per block (ANUBIS_PTX_BLOCK). Both are
measurement knobs, not correctness knobs.

Two throughput numbers are reported. "kernel" comes from the CUDA events that
bracket each launch, so it excludes the results memset, the context sync and
the copy-back -- that is the number comparable to hashcat's. "wall" is what the
caller sees. They differ by the per-chunk host round trip.

Every digest is checked against hashlib.
"""

import ctypes
import glob
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def sh(cmd, cwd=None):
    print("$", " ".join(cmd) if isinstance(cmd, list) else cmd)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
        raise SystemExit("command failed")


def sm_clock():
    """Current SM clock in MHz, or None. Hashcat timing is meaningless without
    it: a throttled T4 is ~20% slower at the same kernel."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return [int(x) for x in out.stdout.split()] if out.returncode == 0 else None
    except Exception:
        return None


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_75"
    ilp = sys.argv[2] if len(sys.argv) > 2 else "2"
    loop = sys.argv[3] if len(sys.argv) > 3 else "1"
    block = sys.argv[4] if len(sys.argv) > 4 else "256"

    sh(["bash", "build_ptx.sh", arch, ilp], cwd=HERE)
    so = os.path.join(HERE, "libanubis_ptx_loader.so")
    # Images such as Kaggle's ship ptxas but not the driver headers, and gcc
    # does not search /usr/local/cuda/include by default. Try a real cuda.h
    # first, then fall back to the bundled minimal declarations.
    base = ["gcc", "-O2", "-shared", "-fPIC", "-o", so, "ptx_loader.c"]
    incs = [d for pat in ("/usr/local/cuda*/include",
                          "/usr/local/cuda*/targets/*/include",
                          "/usr/include")
            for d in glob.glob(pat) if os.path.exists(os.path.join(d, "cuda.h"))]
    libs = ["-L" + d for d in glob.glob("/usr/local/cuda*/lib64/stubs")]
    hdr = [["-I" + d] for d in incs[:1]] + [[]]
    tries = []
    for h in hdr:                      # no link-time driver library needed
        tries.append(h + ["-DANUBIS_DLOPEN_CUDA"])
        tries.append(h + ["-DANUBIS_MIN_CUDA_H", "-DANUBIS_DLOPEN_CUDA"])
    for h in hdr:
        tries.append(h + libs + ["-lcuda"])
        tries.append(h + libs + ["-DANUBIS_MIN_CUDA_H", "-lcuda"])
    for extra in tries:
        r = subprocess.run(base + extra, cwd=HERE, capture_output=True, text=True)
        if r.returncode == 0:
            print("$", " ".join(base + extra))
            break
    else:
        print(r.stderr.strip())
        raise SystemExit("loader build failed")

    os.environ["ANUBIS_PTX_DIR"] = os.path.join(HERE, "cubin")
    os.environ["ANUBIS_PTX_ILP"] = ilp
    os.environ["ANUBIS_PTX_LOOP"] = loop
    os.environ["ANUBIS_PTX_BLOCK"] = block
    os.environ.setdefault("ANUBIS_PTX_CHUNK", str(1 << 24))

    lib = ctypes.CDLL(so)
    lib.anubis_ptx_available.restype = ctypes.c_int
    lib.anubis_ptx_info.restype = ctypes.c_char_p
    lib.anubis_ptx_last_kernel_ms.restype = ctypes.c_double
    lib.anubis_ptx_solve.restype = ctypes.c_int
    lib.anubis_ptx_solve.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64, ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
    ]

    if not lib.anubis_ptx_available():
        raise SystemExit("no usable CUDA device / cubins")
    print("device:", lib.anubis_ptx_info().decode())

    mid = (ctypes.c_uint32 * 8)(*IV)

    def solve(mode, use_le, difficulty, bits):
        nonce = ctypes.c_uint32(0)
        out = ctypes.create_string_buffer(65)
        ms = ctypes.c_double(0.0)
        rc = lib.anubis_ptx_solve(mid, 0, mode, use_le, difficulty, bits,
                                  1 << 63, 0.0, ctypes.byref(nonce),
                                  out, ctypes.byref(ms))
        km = lib.anubis_ptx_last_kernel_ms()
        return rc, nonce.value, out.value.decode(), ms.value, km

    # correctness: prefix is empty, so the message is just the nonce
    print("\n== correctness ==")
    for mode, use_le, name in ((1, 1, "sha256 le"), (1, 0, "sha256 be"),
                               (0, 1, "decimal")):
        rc, nonce, hexdig, ms, km = solve(mode, use_le, 0, 0)
        raw = nonce.to_bytes(4, "little" if use_le else "big") if mode == 1 \
            else str(nonce).encode()
        exp = hashlib.sha256(raw).hexdigest()
        ok = rc == 0 and hexdig == exp
        print("  %-10s nonce=%-6d %s" % (name, nonce, "ok" if ok else "MISMATCH"))

    # throughput: nonces scanned = returned nonce + 1
    print("\n== throughput (mode 1, ILP=%s loop=%s block=%s chunk=2^%d) ==" %
          (ilp, loop, block, int(os.environ["ANUBIS_PTX_CHUNK"]).bit_length() - 1))
    clk = sm_clock()
    print("  SM clock before: %s MHz" % (clk if clk else "unknown"))
    for bits in (24, 28):
        for _ in range(3):
            rc, nonce, hexdig, ms, km = solve(1, 1, bits, 1)
            if rc != 0 and ms <= 0:
                continue
            n = nonce + 1
            wall = n / (ms / 1000.0) / 1e6
            kern = n / (km / 1000.0) / 1e6 if km > 0 else 0.0
            print("  %2d bits: nonce=%-9d %7.2f ms  wall %9.1f MH/s  kernel %9.1f MH/s"
                  % (bits, nonce, ms, wall, kern))
    clk = sm_clock()
    print("  SM clock after:  %s MHz" % (clk if clk else "unknown"))


if __name__ == "__main__":
    main()
