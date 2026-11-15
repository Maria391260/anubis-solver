#!/usr/bin/env python3
"""bench_tensor_cuda.py — the `tensor` tier on CUDA, measured.

Three numbers per difficulty, because they answer different questions:

  gpu        the GEMM and nothing else (anubis_set_gpu_verify(0)), which is
             what the hardware does
  gpu+verify the same solve with the CPU reference re-check that ships enabled:
             the digest is computed by the GPU and independently by the CPU, and
             they must match. This is the default and the honest end-to-end
             number for a client.
  cpu        anubis_tensor_digest, the reference, for the speedup column

    python3 tools/bench_tensor_cuda.py            # text table
    python3 tools/bench_tensor_cuda.py --json     # machine-readable

On Kaggle the JSON lands in /kaggle/working/result.json when --out is given.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import anubis_solver as A  # noqa: E402

DIFFICULTIES = [2, 4, 8, 12, 16]
CHALLENGE = "9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--out", help="also write the JSON here")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--difficulty", type=int, action="append",
                    help="override the difficulty list (repeatable)")
    args = ap.parse_args()

    native = A._Native()
    lib = native.lib
    cuda_available = hasattr(lib, "anubis_cuda_tensor_available") and \
        lib.anubis_cuda_tensor_available()
    info = None
    if hasattr(lib, "anubis_cuda_tensor_info"):
        lib.anubis_cuda_tensor_info.restype = ctypes.c_char_p
        raw = lib.anubis_cuda_tensor_info()
        info = raw.decode() if raw else None

    difficulties = args.difficulty or DIFFICULTIES
    rows = []
    for d in difficulties:
        n = 64 * max(2, min(16, d))
        gpu_ms, verify_ms, cpu_ms = [], [], []

        # the GPU path, verification off, so this is the GEMM
        lib.anubis_set_gpu_verify(0)
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            gpu_nonce, gpu_digest, _ = native.solve_tensor(CHALLENGE, d, 1, 0)
            gpu_ms.append((time.perf_counter() - t0) * 1000.0)

        # the default path: GPU digest re-checked against the CPU reference
        lib.anubis_set_gpu_verify(1)
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            ver_nonce, ver_digest, _ = native.solve_tensor(CHALLENGE, d, 1, 0)
            verify_ms.append((time.perf_counter() - t0) * 1000.0)
        assert ver_digest == gpu_digest, "GPU digest changed with verification on"
        assert ver_nonce == 0

        for _ in range(max(2, args.rounds // 2)):
            t0 = time.perf_counter()
            native.verify_tensor(CHALLENGE, 0, d)
            cpu_ms.append((time.perf_counter() - t0) * 1000.0)

        # The solve above includes everything the client pays: the GPU work, the
        # 4 MiB copy back and the host SHA-256 of C. Those three have very
        # different costs, and a single "total" number hides that the GEMM is
        # a fraction of it, so time the CUDA call on its own as well.
        gemm_ms = []
        seed = hashlib.sha256(bytes.fromhex(CHALLENGE)).digest()
        c_buf = (ctypes.c_int32 * (n * n))()
        if cuda_available:
            fn = lib.anubis_cuda_tensor_c
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_char_p, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)]
            seed_buf = ctypes.create_string_buffer(seed, 32)
            for _ in range(args.rounds):
                ms = ctypes.c_double(0.0)
                rc = fn(seed_buf, n, c_buf, ctypes.byref(ms))
                if rc != 0:
                    gemm_ms.append(float("nan"))
                else:
                    gemm_ms.append(ms.value)

        macs = 2 * n ** 3
        row = {
            "difficulty": d,
            "n": n,
            "macs": macs,
            "gpu_ms": statistics.median(gpu_ms),
            "gpu_gmacs": macs / (statistics.median(gpu_ms) / 1000.0) / 1e9,
            "gpu_verify_ms": statistics.median(verify_ms),
            "cpu_ms": statistics.median(cpu_ms),
            "cpu_gmacs": macs / (statistics.median(cpu_ms) / 1000.0) / 1e9,
            "cuda_call_ms": statistics.median(gemm_ms) if gemm_ms else None,
            "cuda_call_gmacs": (macs / (statistics.median(gemm_ms) / 1000.0) / 1e9)
            if gemm_ms and not any(m != m for m in gemm_ms) else None,
            "digest": gpu_digest,
        }
        row["speedup_vs_cpu"] = row["cpu_ms"] / row["gpu_ms"] if row["gpu_ms"] else None
        rows.append(row)
        if not args.json:
            print(f"d={d:2d} N={n:4d} {macs/1e6:8.1f} MMAC  "
                  f"gpu {row['gpu_ms']:8.3f} ms ({row['gpu_gmacs']:7.1f} GMAC/s)  "
                  f"+verify {row['gpu_verify_ms']:8.2f} ms  "
                  f"cpu {row['cpu_ms']:8.2f} ms  "
                  f"speedup {row['speedup_vs_cpu']:6.1f}x  "
                  f"cuda-call {row['cuda_call_ms'] if row['cuda_call_ms'] is None else round(row['cuda_call_ms'], 3)} ms", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "host": platform.node(),
        "gpu": info,
        "cuda_tensor": bool(cuda_available),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0],
        "challenge": CHALLENGE,
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
