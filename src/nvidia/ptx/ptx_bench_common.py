#!/usr/bin/env python3
"""Build helpers shared by the PTX benchmark scripts.

Kaggle images ship `ptxas` and `gcc` but neither the driver headers nor a
linkable `libcuda.so`, so the loader is built with a list of fallbacks and
the first one that compiles wins.
"""

import glob
import os
import subprocess


def sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit("command failed: %s\n%s%s" %
                         (" ".join(cmd) if isinstance(cmd, list) else cmd,
                          r.stdout, r.stderr))
    return r


def build_cubins(here, arch="sm_75", ilp="2"):
    """Run build_ptx.sh (needs ptxas on PATH or in PTXAS)."""
    r = sh(["bash", "build_ptx.sh", arch, ilp], cwd=here, check=False)
    if r.returncode != 0:
        raise SystemExit("build_ptx.sh failed:\n" + r.stdout + r.stderr)
    print(r.stdout.strip())


def build_loader(here):
    """Compile ptx_loader.c into libanubis_ptx_loader.so."""
    so = os.path.join(here, "libanubis_ptx_loader.so")
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
    err = ""
    for extra in tries:
        r = sh(base + extra, cwd=here, check=False)
        if r.returncode == 0:
            print("$ " + " ".join(base + extra))
            return so
        err = r.stderr
    raise SystemExit("loader build failed:\n" + err)


def build_hashx_shim(here):
    """Compile the CPU HashX reference + the ctypes wrapper."""
    so = os.path.join(here, "libhashx_shim.so")
    shim = os.path.join(here, "hashx_shim.c")
    ref = os.path.normpath(os.path.join(here, "..", "..", "hashx", "hashx.c"))
    r = sh(["gcc", "-O2", "-shared", "-fPIC", "-o", so, ref, shim],
           cwd=here, check=False)
    if r.returncode != 0:
        raise SystemExit("hashx shim build failed:\n" + r.stderr)
    return so
