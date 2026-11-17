# HashX in PTX on a Kaggle T4: clone the branch with the new kernel, build it,
# check every digest against the CPU reference, then benchmark.
#
# Run from a notebook:
#   !kaggle kernels push -p src/nvidia/ptx/kaggle_hashx
#   !kaggle kernels status monktelv31/anubis-hashx-ptx
#   !kaggle kernels output monktelv31/anubis-hashx-ptx -p /tmp/hx \
#       --file-pattern 'result\.json'
#
# The kernel writes /kaggle/working/result.json (or error.json) so a failure is
# still evidence.
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working")
SRC = Path("/kaggle/src")
REPO = SRC / "anubis-solver"
BRANCH = "ptx-hashx"
URL = "https://gitlab.com/celikfana/anubis-solver.git"
PTX = REPO / "src" / "nvidia" / "ptx"


def sh(cmd, cwd=None, timeout=1800, check=True):
    print("$", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if r.returncode != 0 and r.stderr:
        print(r.stderr[-4000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError("failed: %s" % (cmd,))
    return r


def stage(name, **kw):
    print("STAGE: %s %s" % (name, json.dumps(kw)), flush=True)
    (OUT / "status.json").write_text(json.dumps({"stage": name, **kw}))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    result = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        stage("gpu")
        sh("nvidia-smi -L", check=False)
        sh("nvidia-smi --query-gpu=name,compute_cap,memory.total "
           "--format=csv,noheader", check=False)

        stage("clone")
        if REPO.exists():
            shutil.rmtree(REPO)
        sh(["git", "clone", "--depth", "1", "-b", BRANCH, URL, str(REPO)])

        stage("toolkit")
        sh("which ptxas && ptxas --version | tail -2", check=False)
        sh("python3 -V", check=False)

        ptxas = shutil.which("ptxas")
        if ptxas is None:
            # Kaggle images keep the toolkit in a fixed place but not on PATH
            for cand in ("/usr/local/cuda/bin/ptxas",
                         "/usr/local/cuda-13.0/bin/ptxas"):
                if Path(cand).exists():
                    os.environ["PTXAS"] = cand
                    os.environ["PATH"] = str(Path(cand).parent) + ":" + \
                        os.environ["PATH"]
                    print("using", cand)
                    break

        stage("offline")
        # the PTX simulator test: no GPU, checks the kernel against hashx.c
        r = sh([sys.executable, "test_hashx_ptx.py"], cwd=PTX, check=False)
        result["offline_test_rc"] = r.returncode
        result["offline_test_pass"] = r.returncode == 0
        result["offline_test_tail"] = (r.stdout or "")[-2000:]
        (OUT / "offline.log").write_text(r.stdout + r.stderr)

        stage("bench")
        r = sh([sys.executable, "kaggle_bench_hashx_ptx.py"],
               cwd=PTX, check=False)
        (OUT / "bench.log").write_text(r.stdout + r.stderr)
        result["bench_rc"] = r.returncode

        if (PTX / "result.json").exists():
            result.update(json.loads((PTX / "result.json").read_text()))
        else:
            result["bench_output_tail"] = (r.stdout or "")[-3000:]

        stage("done")
    except Exception as exc:  # noqa: BLE001 - the log is the evidence
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        stage("error", error=str(exc))

    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print("WROTE", OUT / "result.json")


if __name__ == "__main__":
    main()
