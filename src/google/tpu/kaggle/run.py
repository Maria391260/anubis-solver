# First contact with a Kaggle TPU: clone, build the C probe, find libtpu, and
# ask it whether it will give us a PJRT client. No Python in the measure path;
# Python is only used to locate the libtpu wheel and to write result.json.
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working")
REPO = OUT / "anubis-solver"
RESULT = {}


def sh(cmd, timeout=900, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or str(OUT))
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-4000:], flush=True)
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "(timed out)"


def main():
    RESULT["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _, out = sh("env | grep -i -E 'tpu|pjrt' | sort")
    RESULT["env"] = out.strip()
    rc, out = sh("python3 -c 'import libtpu, os; print(os.path.dirname(libtpu.__file__))'")
    RESULT["libtpu_dir"] = out.strip()
    candidates = sorted(set(glob.glob("/usr/local/lib/python3*/dist-packages/libtpu/libtpu.so") +
                            glob.glob("/usr/local/lib/python3*/site-packages/libtpu/libtpu.so") +
                            glob.glob("/opt/**/libtpu/libtpu.so", recursive=True)))
    # the image may not ship the wheel at all: ask the filesystem, the package
    # manager and the kernel driver, and say what each answered
    rc, jaxdev = sh("python3 -c \"import jax; print(jax.default_backend()); print(jax.devices())\" 2>&1")
    RESULT["jax"] = jaxdev.strip()
    rc, found = sh("find / -name 'libtpu*' 2>/dev/null | head -20", timeout=240)
    RESULT["find_libtpu"] = found.strip()
    rc, pip = sh("python3 -m pip list 2>/dev/null | grep -i -E 'tpu|jax' || true")
    RESULT["pip"] = pip.strip()
    for line in found.splitlines():
        line = line.strip()
        if line.endswith(".so") and line not in candidates:
            candidates.append(line)
    rc, devs = sh("ls -la /dev | grep -i -E 'tpu|vfio' || true; ls -la /dev/vfio 2>/dev/null || true")
    RESULT["dev"] = devs.strip()
    if os.path.exists(os.path.join(RESULT["libtpu_dir"], "libtpu.so")):
        candidates.insert(0, os.path.join(RESULT["libtpu_dir"], "libtpu.so"))
    RESULT["candidates"] = candidates
    if not REPO.exists():
        sh("git clone --depth 1 https://github.com/celikfana-lab/anubis-solver.git " + str(REPO))
    rc, out = sh("git -C %s log --oneline -1" % REPO, cwd=str(OUT))
    RESULT["commit"] = out.strip()

    if not (REPO / "src/google/tpu/pjrt_probe.c").exists():
        RESULT["build"] = {"rc": None, "why": "repo clone has no src/google/tpu; commit and push first"}
        finish()
    rc, out = sh("cc -O2 -I. -o /tmp/pjrt_probe pjrt_probe.c -ldl",
                 cwd=str(REPO / "src/google/tpu"))
    RESULT["build"] = {"rc": rc, "tail": out[-1200:]}
    if rc == 0 and candidates:
        rc, out = sh("/tmp/pjrt_probe " + candidates[0])
        RESULT["probe"] = {"rc": rc, "lib": candidates[0], "out": out[-3000:]}
    else:
        RESULT["probe"] = {"rc": None, "why": "no build or no libtpu candidate"}

    finish()


def finish():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "result.json").write_text(json.dumps(RESULT, indent=2))
    print("\n::RESULT::", json.dumps(RESULT, indent=2)[:4000], flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
