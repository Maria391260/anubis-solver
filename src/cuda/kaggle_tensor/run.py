# Tensor tier on Kaggle's 2x Tesla T4: build with CUDA, run the correctness
# test, then measure the GEMM on one T4 and on both. Everything that happens is
# copied into /kaggle/working/result.json, so a failure is still evidence.
#
#   kaggle kernels push -p src/cuda/kaggle_tensor
#   kaggle kernels status monktelv31/anubis-t4-tensor
#   kaggle kernels output monktelv31/anubis-t4-tensor -p /tmp/t4 --file-pattern 'result\.json'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working")
REPO = OUT / "anubis-solver"
PORT = 8091
BACKEND_PORT = 8092
RESULT = {}


def sh(cmd, timeout=2400, env=None, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    e = dict(os.environ)
    e["PATH"] = "/usr/local/cuda/bin:" + e.get("PATH", "")
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
    _, out = sh("nvidia-smi -L")
    RESULT["nvidia_smi"] = out.strip()
    _, out = sh("nvcc --version | tail -2")
    RESULT["nvcc"] = out.strip()

    if not REPO.exists():
        rc, out = sh("git clone --depth 1 https://github.com/celikfana-lab/anubis-solver.git " + str(REPO),
                     cwd=str(OUT))
        RESULT["clone"] = {"rc": rc, "tail": out[-800:]}
    _, out = sh("git -C %s log --oneline -1" % REPO, cwd=str(OUT))
    RESULT["commit"] = out.strip()

    rc, out = sh("make -j4 CUDA_ARCH=sm_75")
    RESULT["build"] = {"rc": rc, "tail": out[-1500:]}
    if rc != 0:
        finish()

    rc, out = sh("python3 tests/test_tensor_cuda.py")
    RESULT["test_tensor_cuda"] = {"rc": rc, "tail": out[-2500:]}

    for label, visible in (("one_t4", "0"), ("two_t4", "0,1")):
        rc, out = sh(
            "python3 tools/bench_tensor_cuda.py --rounds 5 --json "
            f"--out /tmp/{label}.json",
            env={"CUDA_VISIBLE_DEVICES": visible},
        )
        bench = {}
        try:
            bench = json.load(open(f"/tmp/{label}.json"))
        except OSError:
            bench = {"error": "no json"}
        RESULT[label] = {"rc": rc, "bench": bench, "tail": out[-1200:] if rc else ""}

    # end to end through the gate: tensor d=8, solved by the Python client
    Path("/tmp/ba-site").mkdir(exist_ok=True)
    Path("/tmp/ba-site/index.html").write_text("<html><body><h1>PAST THE WALL</h1></body></html>")
    sh("(cd /tmp/ba-site && nohup python3 -m http.server %d --bind 127.0.0.1 >/dev/null 2>&1 &)" % BACKEND_PORT,
       cwd="/tmp")
    time.sleep(1)
    rc, out = sh("make -C betterAnusbis")
    RESULT["gate_build"] = rc
    if rc == 0:
        sh("./betterAnusbis --listen 127.0.0.1:%d --target http://127.0.0.1:%d "
           "--algorithm tensor --difficulty 8 >/tmp/gate.log 2>&1 &" % (PORT, BACKEND_PORT),
           cwd=str(REPO / "betterAnusbis"))
        time.sleep(1)
        rc, out = sh("cd %s && python3 anubis_solver.py http://127.0.0.1:%d/" % (REPO, PORT), timeout=300)
        RESULT["gate_tensor"] = {"rc": rc, "passed": "PAST THE WALL" in out, "tail": out[-800:]}
        sh("pkill -f 'betterAnusbis --listen' || true", cwd="/tmp")

    finish()


def finish():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = OUT / "result.json"
    path.write_text(json.dumps(RESULT, indent=2, default=str))
    print("\nresult.json written to", path, flush=True)
    print(json.dumps({k: v for k, v in RESULT.items() if k not in ("one_t4", "two_t4")},
                     indent=2, default=str)[:4000], flush=True)
    for k in ("one_t4", "two_t4"):
        if k in RESULT and isinstance(RESULT[k].get("bench"), dict):
            print(k, json.dumps(RESULT[k]["bench"].get("rows", []), indent=1)[:2500], flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
