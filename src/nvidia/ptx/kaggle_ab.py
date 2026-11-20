#!/usr/bin/env python3
"""A/B the PTX SHA-256 kernel configurations on one GPU and write JSON.

The loader reads its knobs from the environment once, at first use, so one
process can only exercise one configuration. This driver therefore rebuilds the
cubins into ``build/<tag>/`` and re-runs itself as a child per configuration,
with ``ANUBIS_PTX_DIR`` pointing at that directory.

    python3 kaggle_ab.py                 # the matrix, as JSON on stdout
    AB_IMAD=1 python3 kaggle_ab.py --one DIR ILP LOOP BLOCK

What it measures, and why:

  * ``kernel_ms`` comes from the CUDA events around each launch, so it excludes
    the results memset, the sync and the copy-back. Comparing wall clock numbers
    between runs measures the host as much as the GPU.
  * ``clocks.sm`` is logged before and after. On a T4 at 1.59 GHz vs 1.20 GHz the
    same kernel is 30% apart, so a number without a clock is not evidence.
  * The throughput run uses ``difficulty=255``, which cannot be hit, so the whole
    range is scanned and every reported number is N nonces over N nonces' time.

The matrix is chosen to answer four questions: does the grid-stride loop help
(``loop``), does occupancy matter (``maxrreg`` and ``block``), is ILP worth the
registers (``ilp``), and is the answer stable across repeats.
"""
import ctypes
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PTX = Path(__file__).resolve().parent
ARCH = os.environ.get("AB_ARCH", "sm_75")
TOTAL = int(os.environ.get("AB_TOTAL", 1 << 29))   # nonces per throughput run
REPS = int(os.environ.get("AB_REPS", 3))

IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def sh(cmd, cwd=None, env=None, check=True):
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=e,
                           capture_output=True, text=True)
    except FileNotFoundError as exc:

        class R:
            returncode = 127
            stdout = ""
            stderr = str(exc)
        r = R()
    if check and r.returncode != 0:
        raise RuntimeError("%s failed:\n%s\n%s" %
                           (cmd if isinstance(cmd, str) else " ".join(cmd),
                            r.stdout[-2000:], r.stderr[-2000:]))
    return r


def sm_clock():
    try:
        r = sh(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm",
                "--format=csv,noheader,nounits"], check=False)
        if r.returncode != 0:
            return None
        return [int(x) for x in r.stdout.split(",")[0].split()]
    except Exception:
        return None


def find_ptxas():
    p = os.environ.get("PTXAS") or None
    if p and Path(p).exists():
        return p
    w = sh(["bash", "-lc", "command -v ptxas"], check=False).stdout.strip()
    if w:
        return w
    for cand in sorted(glob.glob("/usr/local/cuda*/bin/ptxas")):
        return cand
    raise RuntimeError("no ptxas found")


def build(ilp, maxrreg):
    """Build every cubin this configuration needs into build/<tag>/."""
    tag = "ilp%d_r%d" % (ilp, maxrreg)
    out = PTX / "build" / tag
    out.mkdir(parents=True, exist_ok=True)
    r = sh(["bash", "build_ptx.sh", ARCH, str(ilp), str(maxrreg)], cwd=PTX,
           env={"PTXAS": find_ptxas(), "PTX_OUT_DIR": str(out)}, check=False)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-1500:]
    return str(out), None


def compile_loader():
    """Same fallback ladder as the other Kaggle harnesses: real cuda.h first,
    then the vendored minimal header, dlopen before -lcuda."""
    so = str(PTX / "libanubis_ptx_loader.so")
    base = ["gcc", "-O2", "-shared", "-fPIC", "-o", so, "ptx_loader.c"]
    incs = []
    for pat in ("/usr/local/cuda*/include", "/usr/local/cuda*/targets/*/include",
                "/usr/include"):
        incs += [d for d in glob.glob(pat) if os.path.exists(os.path.join(d, "cuda.h"))]
    libs = ["-L" + d for d in glob.glob("/usr/local/cuda*/lib64/stubs")]
    cands = []
    for h in ([["-I" + incs[0]]] if incs else []) + [[]]:
        cands.append(h + ["-DANUBIS_DLOPEN_CUDA"])
        cands.append(h + ["-DANUBIS_MIN_CUDA_H", "-DANUBIS_DLOPEN_CUDA"])
        cands.append(h + libs + ["-lcuda"])
        cands.append(h + libs + ["-DANUBIS_MIN_CUDA_H", "-lcuda"])
    last = None
    for extra in cands:
        last = sh(base + extra, cwd=PTX, check=False)
        if last.returncode == 0:
            return so
    raise RuntimeError("could not build the loader:\n" + last.stderr[-2000:])


class Lib:
    def __init__(self, so):
        self.lib = ctypes.CDLL(so)
        self.lib.anubis_ptx_available.restype = ctypes.c_int
        self.lib.anubis_ptx_info.restype = ctypes.c_char_p
        self.lib.anubis_ptx_last_kernel_ms.restype = ctypes.c_double
        self.lib.anubis_ptx_solve.restype = ctypes.c_int
        self.lib.anubis_ptx_solve.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint64, ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double)]
        self.mid = (ctypes.c_uint32 * 8)(*IV)
        self.avail = bool(self.lib.anubis_ptx_available())
        raw = self.lib.anubis_ptx_info()
        self.info = raw.decode() if raw else None

    def solve(self, difficulty, bits, max_nonce):
        nonce = ctypes.c_uint32(0)
        out = ctypes.create_string_buffer(65)
        ms = ctypes.c_double(0.0)
        rc = self.lib.anubis_ptx_solve(self.mid, 0, 1, 1, difficulty, bits,
                                       max_nonce, 0.0, ctypes.byref(nonce),
                                       out, ctypes.byref(ms))
        km = self.lib.anubis_ptx_last_kernel_ms()
        return rc, nonce.value, out.value.decode(), ms.value, km


def child(cubin_dir, ilp, loop, block):
    # one GPU per process unless the caller asked otherwise, so a row means a
    # GPU and not "whatever this box has"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("AB_DEVICES", "0")
    os.environ["ANUBIS_PTX_DIR"] = cubin_dir
    os.environ["ANUBIS_PTX_ILP"] = str(ilp)
    os.environ["ANUBIS_PTX_LOOP"] = str(loop)
    os.environ["ANUBIS_PTX_BLOCK"] = str(block)
    os.environ["ANUBIS_PTX_CHUNK"] = str(1 << 24)
    imad = int(os.environ.get("AB_IMAD", "0"))
    os.environ["ANUBIS_PTX_IMAD"] = str(imad)
    out = {"ilp": ilp, "loop": loop, "block": block, "imad": imad,
           "cubin_dir": Path(cubin_dir).name, "bench": [], "correctness": []}
    clock0 = sm_clock()
    L = Lib(str(PTX / "libanubis_ptx_loader.so"))
    out["info"] = L.info
    if not L.avail:
        out["error"] = L.info or "no device"
        print("RESULT " + json.dumps(out))
        return

    # a real hit: 20 leading zero bits, digest checked against hashlib
    rc, nonce, hexd, ms, km = L.solve(20, 1, 1 << 32)
    exp = hashlib.sha256(nonce.to_bytes(4, "little")).hexdigest()
    ok = rc == 0 and hexd == exp and exp.startswith("0" * 5)
    out["correctness"].append({"name": "20-bit hit", "nonce": nonce, "ok": ok})
    # and the trivial one, which must be exact
    rc, nonce, hexd, ms, km = L.solve(0, 0, 1 << 16)
    exp = hashlib.sha256(nonce.to_bytes(4, "little")).hexdigest()
    out["correctness"].append({"name": "first nonce", "nonce": nonce,
                               "ok": rc == 0 and hexd == exp})

    # A T4 parks at 300 MHz between kernels and takes ~1 s of load to boost, so
    # a 0.2 s run measures the ramp, not the kernel. AB_WARMUP burns that second
    # first; every rep is then timed at the boosted clock and stamped with it.
    warmup = float(os.environ.get("AB_WARMUP", "0"))
    if warmup > 0:
        t0 = time.time()
        while time.time() - t0 < warmup:
            L.solve(255, 1, 1 << 27)
        out["clock_warm"] = sm_clock()

    for rep in range(REPS):
        rc, nonce, hexd, ms, km = L.solve(255, 1, TOTAL - 1)
        wall = TOTAL / (ms / 1000.0) / 1e6 if ms > 0 else 0.0
        kern = TOTAL / (km / 1000.0) / 1e6 if km > 0 else 0.0
        out["bench"].append({"rep": rep, "ms": ms, "kernel_ms": km,
                             "mhps": wall, "kernel_mhps": kern,
                             "clock": sm_clock()})
    out["clock_before"] = clock0
    out["clock_after"] = sm_clock()
    print("RESULT " + json.dumps(out))


def driver():
    compile_loader()
    matrix = []
    for ilp in (1, 2):
        for loop in (1, 2, 8, 32):
            for block in (256, 512):
                matrix.append((ilp, loop, block, 0))
    matrix += [(1, 8, 128, 0), (1, 8, 1024, 0)]
    matrix += [(1, 8, 256, 32), (1, 8, 256, 40), (2, 8, 256, 32)]

    results = {"arch": ARCH, "total": TOTAL, "reps": REPS,
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "nvidia_smi": sh(["nvidia-smi", "-L"], check=False).stdout.strip(),
               "rows": []}
    built = {}
    for ilp, loop, block, mr in matrix:
        if (ilp, mr) not in built:
            built[(ilp, mr)] = build(ilp, mr)
        d, err = built[(ilp, mr)]
        if d is None:
            results["rows"].append({"ilp": ilp, "loop": loop, "block": block,
                                    "maxrreg": mr, "error": err})
            continue
        r = sh([sys.executable, __file__, "--one", d, str(ilp), str(loop),
                str(block)], cwd=PTX, check=False)
        row = {"ilp": ilp, "loop": loop, "block": block, "maxrreg": mr}
        line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        if r.returncode != 0 or not line:
            row["error"] = (r.stdout + r.stderr)[-1500:]
        else:
            row.update(json.loads(line[-1][7:]))
            row["maxrreg"] = mr
            best = max((b["kernel_mhps"] for b in row["bench"]), default=0.0)
            row["best_kernel_mhps"] = best
        results["rows"].append(row)
        print("  ilp=%d loop=%-2d block=%-4d r=%d  %s" %
              (ilp, loop, block, mr, row.get("error", "%.1f GH/s" %
                                             (row.get("best_kernel_mhps", 0) / 1000.0))),
              file=sys.stderr, flush=True)

    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (PTX / "ab.json").write_text(json.dumps(results, indent=1))
    rows = [r for r in results["rows"] if "best_kernel_mhps" in r]
    rows.sort(key=lambda r: -r["best_kernel_mhps"])
    print("\n%-4s %-5s %-6s %-8s %-7s %10s" %
          ("ilp", "loop", "block", "maxrreg", "clock", "kernel GH/s"))
    for r in rows[:20]:
        clk = (r.get("clock_after") or [0])[0]
        print("%-4d %-5d %-6d %-8d %-7s %10.2f" %
              (r["ilp"], r["loop"], r["block"], r["maxrreg"], clk,
               r["best_kernel_mhps"] / 1000.0))
    print("\nwrote", PTX / "ab.json")


if __name__ == "__main__":
    # The tarball ships sources, not binaries, so build the loader before
    # anything needs it -- the child does the same check, which makes --one
    # usable on its own.
    if not (PTX / "libanubis_ptx_loader.so").exists():
        compile_loader()
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        driver()
