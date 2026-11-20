# `tensor-v` on a Kaggle box: clone the unstable branch, build the library,
# run the three tensor-v suites, then measure the client/server split and one
# end-to-end challenge through the gate. Everything lands in
# /kaggle/working/result.json, so a failure is still evidence.
#
#   kaggle kernels push -p src/core/kaggle_tensor_v
#   kaggle kernels status monktelv31/anubis-tensor-v
#   kaggle kernels output monktelv31/anubis-tensor-v -p /tmp/tv \
#       --file-pattern 'result\.json'
#
# This tier is CPU work (the client's GEMM and Merkle tree, the server's
# sampled check), so the kernel is CPU-only on purpose: a GPU kernel gets two
# vCPUs, a CPU kernel gets four.
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working")
REPO = OUT / "anubis-solver"
BRANCH = "unstable"
CLONE = "https://github.com/celikfana-lab/anubis-solver.git"
PORT = 8091
BACKEND_PORT = 8092
RESULT = {}


def sh(cmd, timeout=2400, cwd=None, env=None):
    print(f"\n$ {cmd}", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        workdir = cwd or (str(REPO) if REPO.exists() else str(OUT))
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=e, cwd=workdir)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-4000:], flush=True)
        return r.returncode, out
    except subprocess.TimeoutExpired:
        print("(timed out)", flush=True)
        return -1, "(timed out)"


def main():
    RESULT["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for key, cmd in (
        ("nproc", "nproc"),
        ("cpu", "lscpu | grep -E 'Model name|Socket|Core|Thread|MHz' | head -8"),
        ("memory", "free -g | head -2"),
        ("python", "python3 -V"),
        ("node", "node -v 2>&1 || true"),
        ("os", "uname -sr"),
    ):
        _, out = sh(cmd)
        RESULT[key] = out.strip()

    if not REPO.exists():
        rc, out = sh(f"git clone --depth 1 -b {BRANCH} {CLONE} {REPO}",
                     cwd=str(OUT))
        RESULT["clone"] = {"rc": rc, "tail": out[-800:]}
        if rc != 0:
            finish()
    _, out = sh("git -C %s log --oneline -1" % REPO, cwd=str(OUT))
    RESULT["commit"] = out.strip()

    rc, out = sh("make -j$(nproc)")
    RESULT["build"] = {"rc": rc, "tail": out[-1500:]}
    if rc != 0:
        finish()

    _, out = sh("nm -D --defined-only libanubis_solver.so | "
                "grep -c 'anubis_tensor_v_'")
    RESULT["exports_tensor_v"] = out.strip()

    # The gate must be built here, from this source tree: the checked-in
    # binary was linked against the author's glibc and cannot run on Kaggle.
    rc, out = sh("make -B -C betterAnusbis")
    RESULT["gate_build"] = {"rc": rc, "tail": out[-1500:]}
    if rc != 0:
        finish()

    for name in ("test_tensor_v", "test_tensor_v_gate"):
        rc, out = sh(f"python3 tests/{name}.py")
        RESULT[name] = {"rc": rc, "tail": out[-2500:]}

    # The wasm kernel is checked in; node is only needed for the cross-check.
    if "not found" not in RESULT.get("node", "") and RESULT.get("node"):
        rc, out = sh("node wasm/verify-tensor-v.mjs")
        RESULT["verify_tensor_v_wasm"] = {"rc": rc, "tail": out[-2000:]}
    else:
        RESULT["verify_tensor_v_wasm"] = {"rc": None, "tail": "no node"}

    rc, out = sh("python3 tools/bench_tensor_v.py "
                 "--json /kaggle/working/tensor_v_bench.json")
    bench = {}
    try:
        bench = json.load(open("/kaggle/working/tensor_v_bench.json"))
    except OSError:
        bench = {"error": "no json"}
    RESULT["bench"] = {"rc": rc, "rows": bench.get("rows", []),
                       "tail": out[-1500:] if rc else ""}

    # One end-to-end challenge: real backend, real gate, the Python client.
    Path("/tmp/ba-site").mkdir(exist_ok=True)
    Path("/tmp/ba-site/index.html").write_text(
        "<html><body><h1>PAST THE WALL</h1></body></html>")
    sh("(cd /tmp/ba-site && nohup python3 -m http.server %d --bind 127.0.0.1 "
       ">/dev/null 2>&1 &)" % BACKEND_PORT, cwd="/tmp")
    time.sleep(1)
    sh("./betterAnusbis --listen 127.0.0.1:%d --target "
       "http://127.0.0.1:%d --algorithm tensor-v --difficulty 4 "
       ">/tmp/gate.log 2>&1 &" % (PORT, BACKEND_PORT),
       cwd=str(REPO / "betterAnusbis"))
    time.sleep(1)
    rc, out = sh("python3 anubis_solver.py http://127.0.0.1:%d/" % PORT,
                 timeout=300)
    RESULT["gate_tensor_v"] = {"rc": rc,
                               "passed": "PAST THE WALL" in out,
                               "tail": out[-1000:]}
    _, log = sh("tail -20 /tmp/gate.log", cwd="/tmp")
    RESULT["gate_log"] = log[-1200:]
    sh("pkill -f 'betterAnusbis --listen' || true", cwd="/tmp")

    finish()


def finish():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = OUT / "result.json"
    path.write_text(json.dumps(RESULT, indent=2, default=str))
    print("\nresult.json written to", path, flush=True)
    print(json.dumps({k: v for k, v in RESULT.items() if k != "bench"},
                     indent=2, default=str)[:4000], flush=True)
    print("bench", json.dumps(RESULT.get("bench", {}).get("rows", []),
                              indent=1)[:2500], flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
