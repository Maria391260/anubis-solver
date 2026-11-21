# Kaggle TPU check with the fast-fail client: build, self-test, then ask
# hashx_pjrt to initialise libtpu. With no /dev/vfio/<group> and no /dev/accel
# the preflight exits 3 in milliseconds instead of hanging in InitGoogle.
import glob, json, os, subprocess, sys, time
from pathlib import Path
OUT = Path("/kaggle/working")
R = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
DATA = "9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b"

def sh(cmd, timeout=300):
    print(f"\n$ {cmd}", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, "(timed out)"
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-3000:], flush=True)
    return p.returncode, out

def finish():
    R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "result.json").write_text(json.dumps(R, indent=2))
    print("\n::RESULT::", json.dumps(R, indent=2)[:5000], flush=True)
    sys.exit(0)

src = None
for c in sorted(glob.glob("/kaggle/input/**/hashx_pjrt.c", recursive=True)):
    src = Path(c).parent
    break
R["src"] = str(src) if src else None
if src is None:
    R["error"] = "dataset not mounted"
    finish()
rc, out = sh(f"cc -O2 -I{src} -o /tmp/hashx_pjrt {src}/hashx_pjrt.c {src}/hashx.c -ldl")
R["build_rc"] = rc
if rc:
    finish()
for name, cmd in (("unit", "/tmp/hashx_pjrt --unit"),
                  ("selftest", f"/tmp/hashx_pjrt --selftest --data {DATA} --difficulty 15"),
                  ("tpu_env", "env | grep -iE 'tpu|pjrt' | sort || true"),
                  ("tpu_dev", "ls -la /dev | grep -iE 'vfio|accel|tpu' || true")):
    rc, out = sh(cmd)
    R[name] = {"rc": rc, "tail": out.strip()[-1200:]}
rc, out = sh(f"timeout 30 /tmp/hashx_pjrt --lib /nonexistent --data {DATA} "
             f"--difficulty 8 --window 1024", timeout=60)
R["fast_fail"] = {"rc": rc, "out": out.strip()[-1200:]}
finish()
