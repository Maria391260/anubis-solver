#!/usr/bin/env python3
"""tests/test_tensor_v_gate.py — the two-step `tensor-v` protocol end to end.

Starts a backend and betterAnusbis with `--algorithm tensor-v`, then drives the
exact requests the browser makes:

    1. GET the challenge page                       -> id, randomData
    2. GET pass-challenge?id&response=<root>&nonce=0 -> 96 (block, position) pairs
    3. POST audit-tensor {id, root, audit}           -> Set-Cookie + redirect

and checks that the audit endpoint refuses a tampered block, a witness for some
other sample, and a replay of a spent challenge. The heavy lifting (root,
sample, witness) goes through libanubis_solver.so with the gate's own
`--secret-hex`, so this is the same arithmetic the server and the client run.

    python3 tests/test_tensor_v_gate.py
"""
import ctypes
import http.cookiejar
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "libanubis_solver.so")
GATE = os.path.join(ROOT, "betterAnusbis", "betterAnusbis")
SECRET = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
DIFF = 2
K = 96


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Backend(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"backend-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.anubis_tensor_v_root.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    lib.anubis_tensor_v_root.restype = ctypes.c_int
    lib.anubis_tensor_v_witness.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.anubis_tensor_v_witness.restype = ctypes.c_int
    return lib


def challenge(opener, base, ua):
    req = urllib.request.Request(base + "/", headers={"User-Agent": ua})
    page = opener.open(req, timeout=10).read().decode("utf-8", "replace")
    rd = re.search(r'"randomData":"([0-9a-f]+)"', page)
    cid = re.search(r'"id":"([^"]+)"', page)
    if not rd or not cid:
        raise AssertionError("the challenge page did not carry id/randomData")
    return cid.group(1), rd.group(1)


def make_root(lib, rd):
    out = ctypes.create_string_buffer(65)
    assert lib.anubis_tensor_v_root(rd.encode(), DIFF, out) == 0
    return out.value.decode()


def make_witness(lib, rd, blocks, positions):
    n = len(blocks)
    b = (ctypes.c_uint32 * n)(*blocks)
    p = (ctypes.c_uint32 * n)(*positions)
    out = ctypes.create_string_buffer(1 << 22)
    assert lib.anubis_tensor_v_witness(rd.encode(), DIFF, b, p, n, out, len(out)) == 0
    return out.value.decode()


def get_sample(opener, base, cid, root):
    q = urllib.parse.urlencode({"id": cid, "response": root, "nonce": "0"})
    r = opener.open(base + "/.within.website/x/cmd/anubis/api/pass-challenge?" + q,
                    timeout=10)
    assert r.status == 200
    return json.load(r)["audit"]


def post_audit(opener, base, cid, root, audit, redir="/"):
    body = json.dumps({"id": cid, "root": root, "audit": audit, "redir": redir}).encode()
    req = urllib.request.Request(
        base + "/.within.website/x/cmd/anubis/api/audit-tensor", data=body,
        headers={"Content-Type": "application/json"})
    return opener.open(req, timeout=30)


def expect_reject(opener, base, cid, root, audit, what):
    try:
        post_audit(opener, base, cid, root, audit)
    except urllib.error.HTTPError as e:
        assert e.code in (400, 403), f"{what}: unexpected status {e.code}"
        return
    raise AssertionError(f"{what}: the gate accepted it")


def main():
    for path in (LIB, GATE):
        if not os.path.exists(path):
            print(f"SKIP: {path} is not built")
            return 0
    lib = load_lib()
    backend_port = free_port()
    gate_port = free_port()
    backend = http.server.ThreadingHTTPServer(("127.0.0.1", backend_port), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    gate = subprocess.Popen(
        [GATE, "--listen", f"127.0.0.1:{gate_port}",
         "--target", f"http://127.0.0.1:{backend_port}",
         "--algorithm", "tensor-v", "--difficulty", str(DIFF),
         "--secret-hex", SECRET,
         "--web", os.path.join(ROOT, "betterAnusbis", "web")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{gate_port}"
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", gate_port), 0.1).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("the gate did not start")

        jar = http.cookiejar.CookieJar()
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

        cid, rd = challenge(op, base, "tensor-v-test/1")
        root = make_root(lib, rd)
        sample = get_sample(op, base, cid, root)
        assert sample["count"] == K and len(sample["blocks"]) == K
        audit = make_witness(lib, rd, sample["blocks"], sample["positions"])
        r = post_audit(op, base, cid, root, audit)
        j = json.load(r)
        assert j["ok"] is True and r.headers.get("Set-Cookie")
        print("honest audit accepted; cookie issued")

        # the cookie must open the door, and the backend must answer
        door = op.open(base + "/", timeout=10)
        assert door.read() == b"backend-ok", "the pass cookie did not reach the backend"
        print("pass cookie grants access to the backend")

        # a replay of the same audit must fail: --stateful is the default
        expect_reject(op, base, cid, root, audit, "replayed audit")
        print("replayed audit rejected")

        jar.clear()

        # one flipped byte in a revealed block
        cid, rd = challenge(op, base, "tensor-v-test/2")
        root = make_root(lib, rd)
        sample = get_sample(op, base, cid, root)
        audit = make_witness(lib, rd, sample["blocks"], sample["positions"])
        raw = bytearray(bytes.fromhex(audit))
        count = int.from_bytes(raw[8:12], "little")
        leaves = int.from_bytes(raw[12:16], "little")
        levels = 0
        c = leaves
        while c > 1:
            c = (c + 1) // 2
            levels += 1
        block_off = 16 + count * 12 + count * levels * 32
        raw[block_off] ^= 1
        expect_reject(op, base, cid, root, raw.hex(), "tampered block")
        print("tampered block rejected")

        # a correct witness for a *different* set of blocks (the first K) must
        # fail the sample binding even though every proof is valid
        cid, rd = challenge(op, base, "tensor-v-test/3")
        root = make_root(lib, rd)
        get_sample(op, base, cid, root)
        fake = make_witness(lib, rd, list(range(K)), [0] * K)
        expect_reject(op, base, cid, root, fake, "witness for another sample")
        print("witness for a different sample rejected")

        # the Python client must drive the same two requests end to end
        sys.path.insert(0, ROOT)
        import anubis_solver
        client = anubis_solver.AnubisClient(base, lib_path=LIB, timeout=30)
        door = client.get("/")
        assert door.status_code == 200 and door.text == "backend-ok"
        assert client.solutions and client.solutions[0].algorithm == "tensor-v"
        print("AnubisClient solves tensor-v")
    finally:
        gate.terminate()
        try:
            gate.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gate.kill()
        backend.shutdown()

    print("tensor-v gate: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
