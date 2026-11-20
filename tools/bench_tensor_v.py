#!/usr/bin/env python3
"""tools/bench_tensor_v.py — the client/server split of the `tensor-v` tier.

The point of the tier is the asymmetry: the client runs the whole N^3 integer
GEMM and builds a Merkle tree; the server samples 96 blocks, checks their proofs
and recomputes 96 elements. This measures both sides with the shipped
`libanubis_solver.so`, at the same difficulties the gate uses.

    python3 tools/bench_tensor_v.py
    python3 tools/bench_tensor_v.py --json results/tensor-v.json
    python3 tools/bench_tensor_v.py --samples 64   # what a weaker k would cost

The witness numbers include rebuilding the tree, because this benchmark is
stateless; a browser keeps the tree between the two requests, so its real
step 2 is the hashing of the revealed paths only. The server column is the
honest one: anubis_tensor_v_verify() over a fresh witness.
"""
import argparse
import ctypes
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "libanubis_solver.so")
CHALLENGE = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
SECRET = "tensor-v-benchmark-secret"


def load():
    lib = ctypes.CDLL(LIB)
    lib.anubis_tensor_v_root.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    lib.anubis_tensor_v_root.restype = ctypes.c_int
    lib.anubis_tensor_v_sample.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32)]
    lib.anubis_tensor_v_sample.restype = ctypes.c_int
    lib.anubis_tensor_v_witness.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_char_p,
        ctypes.c_size_t]
    lib.anubis_tensor_v_witness.restype = ctypes.c_int
    lib.anubis_tensor_v_verify.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.anubis_tensor_v_verify.restype = ctypes.c_int
    return lib


def timed(fn, *args):
    t0 = time.perf_counter()
    rc = fn(*args)
    return rc, (time.perf_counter() - t0) * 1000.0


def row(lib, difficulty, k):
    n = 64 * difficulty

    root = ctypes.create_string_buffer(65)
    rc, root_ms = timed(lib.anubis_tensor_v_root, CHALLENGE.encode(), difficulty, root)
    assert rc == 0, f"root failed at d={difficulty}"
    root = root.value

    blocks = (ctypes.c_uint32 * k)()
    positions = (ctypes.c_uint32 * k)()
    rc, sample_ms = timed(lib.anubis_tensor_v_sample, CHALLENGE.encode(), difficulty,
                          root, SECRET.encode(), k, blocks, positions)
    assert rc == 0, f"sample failed at d={difficulty}"

    cap = 1 << 22
    witness = ctypes.create_string_buffer(cap)
    rc, witness_ms = timed(lib.anubis_tensor_v_witness, CHALLENGE.encode(), difficulty,
                           blocks, positions, k, witness, cap)
    assert rc == 0, f"witness failed at d={difficulty}"

    err = ctypes.create_string_buffer(256)
    rc, verify_ms = timed(lib.anubis_tensor_v_verify, CHALLENGE.encode(), difficulty,
                          root, witness.value, err, len(err))
    assert rc == 0, f"verify failed at d={difficulty}: {err.value.decode()}"

    return {
        "difficulty": difficulty,
        "n": n,
        "samples": k,
        "client_root_ms": round(root_ms, 2),
        "client_witness_ms": round(witness_ms, 2),
        "server_sample_ms": round(sample_ms, 4),
        "server_verify_ms": round(verify_ms, 3),
        "ratio": round((root_ms + witness_ms) / max(verify_ms, 1e-9), 1),
        "witness_bytes": len(witness.value) // 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=96,
                    help="sampled blocks per challenge (the gate uses 96)")
    ap.add_argument("--difficulties", default="2,4,8,12,16")
    ap.add_argument("--json", help="write the rows and the machine to this file")
    args = ap.parse_args()

    if not os.path.exists(LIB):
        print(f"{LIB} is not built; run make first", file=sys.stderr)
        return 1
    lib = load()
    difficulties = [int(x) for x in args.difficulties.split(",") if x]
    rows = [row(lib, d, args.samples) for d in difficulties]

    print(f"{'d':>3} {'N':>5} {'client ms':>10} {'witness ms':>11} "
          f"{'server ms':>10} {'ratio':>7} {'proof':>9}")
    for r in rows:
        print(f"{r['difficulty']:>3} {r['n']:>5} {r['client_root_ms']:>10.1f} "
              f"{r['client_witness_ms']:>11.1f} {r['server_verify_ms']:>10.3f} "
              f"{r['ratio']:>6.0f}x {r['witness_bytes']:>8} B")
    print("\nthe client number is the C solver; in wasm it is 3-8x slower, which "
          "is the ratio that matters for a browser\n"
          "the witness rebuilds the tree here; the page keeps it between requests")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "date": time.strftime("%Y-%m-%d"),
                "samples": args.samples,
                "lib": os.path.basename(LIB),
                "rows": rows,
            }, f, indent=2)
            f.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
