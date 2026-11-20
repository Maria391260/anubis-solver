#!/usr/bin/env python3
"""Offline correctness test for the hand-written PTX kernels.

Executes sha256_pow.ptx and sha256_opt.ptx with ptxsim.py (a small PTX
interpreter) and compares every reported digest against hashlib. This needs
neither a GPU nor ptxas; it checks the kernel logic, not driver integration.

    python3 src/nvidia/ptx/test_ptx.py
"""
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptxsim import (Kernel, midstate, sha_from_midstate, launch, results,
                    params_for, tail_mode1, tail_mode0)

PTX = os.path.dirname(os.path.abspath(__file__))
prefix = (b"anubis-ptx-self-test: " * 8)[:128]
assert len(prefix) == 128
st = midstate(prefix)
PL = len(prefix)

fails = 0
def check(cond, msg):
    global fails
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails += 1

def exp0(n):
    return hashlib.sha256(prefix + str(n).encode()).digest()

def exp1(n, le):
    return hashlib.sha256(prefix + n.to_bytes(4, 'little' if le else 'big')).digest()

# sanity: the midstate + padded-tail path equals hashlib on the raw message
for n in (0, 1, 99, 100):
    check(sha_from_midstate(st, tail_mode0(n, PL)) == exp0(n),
          "midstate path == hashlib (mode0 %d)" % n)
for le in (1, 0):
    for n in (0, 1, 12345):
        check(sha_from_midstate(st, tail_mode1(n, PL, le)) == exp1(n, le),
              "midstate path == hashlib (mode1 le=%d %d)" % (le, n))

pow_k = Kernel(PTX + '/sha256_pow.ptx')
opt_k = Kernel(PTX + '/sha256_opt.ptx')

print("== sha256_pow.ptx: mode 1, difficulty 0, one nonce ==")
for le in (1, 0):
    for base in (0, 1, 12345, 0xFFFFFFFE):
        p = params_for(st, PL, 1, le, 0, 0, base, 1)
        gm = launch(pow_k, p, 1, blockdim=256)
        cnt, nonces, hashes = results(gm)
        check(cnt >= 1 and nonces[0] == base and hashes[0] == exp1(base, le),
              "le=%d base=%d nonce/hash" % (le, base))

print("== sha256_pow.ptx: mode 0 (decimal), difficulty 0, one nonce ==")
for base in (0, 7, 10, 99, 100, 123456, 4294967295):
    p = params_for(st, PL, 0, 1, 0, 0, base, 1)
    gm = launch(pow_k, p, 1, blockdim=256)
    cnt, nonces, hashes = results(gm)
    check(cnt >= 1 and nonces[0] == base and hashes[0] == exp0(base),
          "base=%d decimal nonce/hash" % base)

print("== sha256_pow.ptx: bit difficulty filters (4 bits, 512 nonces) ==")
p = params_for(st, PL, 1, 1, 4, 1, 0, 512)
gm = launch(pow_k, p, 512, blockdim=256)
cnt, nonces, hashes = results(gm)
check(cnt >= 1, "found %d hit(s)" % cnt)
ok = True
for i in range(min(cnt, 8)):
    e = exp1(nonces[i], 1)
    if hashes[i] != e or (e[0] >> 4) != 0:
        ok = False
check(ok, "every stored hit is correct and has 4 leading zero bits")

print("== sha256_pow.ptx: nibble difficulty filters (1 nibble, 512 nonces) ==")
p = params_for(st, PL, 1, 0, 1, 0, 0, 512)
gm = launch(pow_k, p, 512, blockdim=256)
cnt, nonces, hashes = results(gm)
ok = cnt >= 1
for i in range(min(cnt, 8)):
    e = exp1(nonces[i], 0)
    if hashes[i] != e or (e[0] >> 4) != 0:
        ok = False
check(ok, "found %d hit(s), nibble check correct" % cnt)

print("== sha256_opt.ptx (ILP=2): difficulty 0 ==")
for le in (1, 0):
    for base in (0, 1, 100, 0xFFFFFFF0):
        for count in (1, 2):
            p = params_for(st, PL, 1, le, 0, 0, base, count)
            gm = launch(opt_k, p, count, blockdim=256, ilp=2)
            cnt, nonces, hashes = results(gm)
            good = cnt >= count
            for u in range(count):
                if nonces[u] != base + u or hashes[u] != exp1(base + u, le):
                    good = False
            check(good, "le=%d base=%d count=%d lanes" % (le, base, count))

print("== sha256_opt.ptx (ILP=2): bit difficulty filters (4 bits, 512 nonces) ==")
p = params_for(st, PL, 1, 1, 4, 1, 0, 512)
gm = launch(opt_k, p, 512, blockdim=256, ilp=2)
cnt, nonces, hashes = results(gm)
ok = cnt >= 1
for i in range(min(cnt, 8)):
    e = exp1(nonces[i], 1)
    if hashes[i] != e or (e[0] >> 4) != 0:
        ok = False
check(ok, "found %d hit(s), digests correct" % cnt)

print("== sha256_opt_loop.ptx: grid-stride loop covers the range exactly ==")
loop_k = Kernel(PTX + '/sha256_opt_loop.ptx')
# count <= 8 so every nonce fits in the 8 result slots and can be checked
for le in (1, 0):
    for count, ilp, lp in ((8, 2, 1), (8, 2, 2), (8, 1, 3), (5, 2, 4), (1, 2, 8)):
        p = params_for(st, PL, 1, le, 0, 0, 0, count)
        gm = launch(loop_k, p, count, blockdim=8, ilp=ilp, loop=lp)
        cnt, nonces, hashes = results(gm)
        good = cnt == count and sorted(nonces[:min(cnt, 8)]) == list(range(count))
        for i in range(min(cnt, 8)):
            if hashes[i] != exp1(nonces[i], le):
                good = False
        check(good, "le=%d count=%d ilp=%d loop=%d -> %d nonce(s)"
              % (le, count, ilp, lp, cnt))

print("== sha256_opt_loop.ptx: bit difficulty filters (4 bits, 64 nonces) ==")
p = params_for(st, PL, 1, 1, 4, 1, 0, 64)
gm = launch(loop_k, p, 64, blockdim=8, ilp=2, loop=2)
cnt, nonces, hashes = results(gm)
ok = cnt >= 1
for i in range(min(cnt, 8)):
    e = exp1(nonces[i], 1)
    if hashes[i] != e or (e[0] >> 4) != 0:
        ok = False
check(ok, "found %d hit(s), digests correct" % cnt)

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("ALL PTX CHECKS PASSED")
