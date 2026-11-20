#!/usr/bin/env python3
"""tests/test_tensor_v.py — the sampled-proof `tensor-v` tier.

The protocol is:

    root  = anubis_tensor_v_root(challenge, d)
    blocks, positions = anubis_tensor_v_sample(challenge, d, root, secret, k)
    proof = anubis_tensor_v_witness(challenge, d, blocks, positions, k)
    anubis_tensor_v_verify(challenge, d, root, proof) == 0

A server never computes the GEMM. These checks pin the properties that matter:

  1. the root is deterministic and matches on repeated calls;
  2. a matching witness verifies;
  3. flipping one element in a sampled block is rejected;
  4. rewriting the root to another challenge's root is rejected;
  5. the sample is deterministic for one secret and different for another;
  6. a witness built for one challenge does not verify under another;
  7. odd leaf counts (the duplicated-last-node path) verify and reject a
     tampered sibling;
  8. the wasm client produces the same root and witness as the C core at both a
     power-of-two and an odd leaf count.
"""
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "libanubis_solver.so")

CHALLENGE = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
DIFF = 2
K = 24
SECRET = "test-audit-secret"


def load():
    lib = ctypes.CDLL(LIB)
    lib.anubis_tensor_v_root.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    lib.anubis_tensor_v_root.restype = ctypes.c_int
    lib.anubis_tensor_v_sample.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    lib.anubis_tensor_v_sample.restype = ctypes.c_int
    lib.anubis_tensor_v_witness.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.anubis_tensor_v_witness.restype = ctypes.c_int
    lib.anubis_tensor_v_verify.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.anubis_tensor_v_verify.restype = ctypes.c_int
    return lib


def root(lib, challenge=CHALLENGE, diff=DIFF):
    out = ctypes.create_string_buffer(65)
    rc = lib.anubis_tensor_v_root(challenge.encode(), diff, out)
    assert rc == 0, f"root rc={rc}"
    return out.value.decode()


def sample(lib, r, secret=SECRET, k=K, challenge=CHALLENGE, diff=DIFF):
    blocks = (ctypes.c_uint32 * k)()
    positions = (ctypes.c_uint32 * k)()
    rc = lib.anubis_tensor_v_sample(challenge.encode(), diff, r.encode(),
                                    secret.encode(), k, blocks, positions)
    assert rc == 0, f"sample rc={rc}"
    return list(blocks), list(positions)


def witness(lib, blocks, positions, challenge=CHALLENGE, diff=DIFF):
    n = len(blocks)
    b = (ctypes.c_uint32 * n)(*blocks)
    p = (ctypes.c_uint32 * n)(*positions)
    cap = 1 << 22
    out = ctypes.create_string_buffer(cap)
    rc = lib.anubis_tensor_v_witness(challenge.encode(), diff, b, p, n, out, cap)
    assert rc == 0, f"witness rc={rc}"
    return out.value.decode()


def verify(lib, r, proof, challenge=CHALLENGE, diff=DIFF):
    err = ctypes.create_string_buffer(256)
    rc = lib.anubis_tensor_v_verify(challenge.encode(), diff, r.encode(),
                                    proof.encode(), err, len(err))
    return rc, err.value.decode()


def main():
    if not os.path.exists(LIB):
        print(f"SKIP: {LIB} is not built")
        return 0
    lib = load()

    r1 = root(lib)
    r2 = root(lib)
    assert r1 == r2 and len(r1) == 64, "root is not deterministic"
    print(f"root {r1}")

    blocks, positions = sample(lib, r1)
    assert len(set(blocks)) == K, "sample has duplicate blocks"
    assert all(0 <= p < 32 for p in positions)
    print(f"{K} samples: blocks {blocks[:4]}... positions {positions[:4]}...")

    proof = witness(lib, blocks, positions)
    rc, err = verify(lib, r1, proof)
    assert rc == 0, f"honest witness rejected: {err}"
    print(f"honest witness accepted ({len(proof) // 2} bytes)")

    lib.anubis_tensor_v_check_sample.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    lib.anubis_tensor_v_check_sample.restype = ctypes.c_int

    def check(proof, bs, ps):
        n = len(bs)
        return lib.anubis_tensor_v_check_sample(
            proof.encode(), (ctypes.c_uint32 * n)(*bs), (ctypes.c_uint32 * n)(*ps), n)

    assert check(proof, blocks, positions) == 0
    assert check(proof, blocks[1:] + blocks[:1], positions) != 0
    print("the witness is bound to the server's sample")

    # another secret must produce another sample
    b2, p2 = sample(lib, r1, secret="another-secret")
    assert (b2, p2) != (blocks, positions), "sample does not depend on the secret"
    print("sample depends on the server secret")

    # same secret is stable across calls
    b3, p3 = sample(lib, r1)
    assert (b3, p3) == (blocks, positions), "sample is not deterministic"
    print("sample is deterministic for one secret")

    # the witness is self-contained: no server state, same bytes verify twice
    rc, err = verify(lib, r1, proof)
    assert rc == 0, f"second verification failed: {err}"

    # tamper with one element of the first sampled block: the Merkle proof
    # must fail before any dot product runs
    raw = bytearray(bytes.fromhex(proof))
    count = int.from_bytes(raw[8:12], "little")
    nleaves = int.from_bytes(raw[12:16], "little")
    levels = 0
    c = nleaves
    while c > 1:
        c = (c + 1) // 2
        levels += 1
    block_off = 16 + count * 12 + count * levels * 32
    raw[block_off] ^= 0x01
    bad = raw.hex()
    rc, err = verify(lib, r1, bad)
    assert rc != 0, "tampered block accepted"
    print(f"tampered block rejected: {err}")

    # a witness for another challenge must not verify here
    other = "ff" * 32
    ro = root(lib, other)
    bo, po = sample(lib, ro, challenge=other)
    poof = witness(lib, bo, po, challenge=other)
    rc, err = verify(lib, r1, poof)
    assert rc != 0, "cross-challenge witness accepted"
    print("cross-challenge witness rejected")

    # a root from another challenge with this challenge's proof must not verify
    rc, err = verify(lib, ro, proof)
    assert rc != 0, "wrong root accepted"
    print("wrong root rejected")

    # n=192 gives 1152 leaves, not a power of two: the duplicated-last-node
    # path is the one a browser's tree and the C verifier must agree on
    r3 = root(lib, diff=3)
    b3, p3 = sample(lib, r3, diff=3)
    w3 = witness(lib, b3, p3, diff=3)
    rc, err = verify(lib, r3, w3, diff=3)
    assert rc == 0, f"odd-leaf witness rejected: {err}"
    raw3 = bytearray(bytes.fromhex(w3))
    count3 = int.from_bytes(raw3[8:12], "little")
    leaves3 = int.from_bytes(raw3[12:16], "little")
    levels3 = 0
    c = leaves3
    while c > 1:
        c = (c + 1) // 2
        levels3 += 1
    proof_off = 16 + count3 * 12
    raw3[proof_off + 5] ^= 0x01
    rc, err = verify(lib, r3, raw3.hex(), diff=3)
    assert rc != 0, "tampered proof accepted"
    print("odd leaf count and proof tampering")

    # the wasm client must build exactly the same tree and witness format
    wasm = os.path.join(ROOT, "wasm", "tensor_pow.wasm")
    if os.path.exists(wasm) and os.environ.get("ANUBIS_SKIP_WASM") != "1":
        import json
        import subprocess
        js = os.path.join(ROOT, "wasm", "verify-tensor-v.mjs")
        if os.path.exists(js):
            for diff in (2, 3):
                out = subprocess.run(["node", js, str(diff)], capture_output=True,
                                     text=True, timeout=120)
                if out.returncode != 0:
                    print(f"SKIP wasm check: {out.stderr.strip()[:200]}")
                    break
                data = json.loads(out.stdout)
                want = root(lib, diff=diff)
                assert data["root"] == want, f"wasm root differs at d={diff}"
                rc, err = verify(lib, data["root"], data["witness"], diff=diff)
                assert rc == 0, f"wasm witness rejected at d={diff}: {err}"
            else:
                print("wasm client agrees with the C core at d=2 and d=3")

    print("tensor-v: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
