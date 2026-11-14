#!/usr/bin/env python3
"""Valkey/Redis store-backend tests for betterAnusbis.

The gate is expected to grow a ``--store valkey:URL`` backend (``redis://``,
``rediss://``, ``unix://`` or a bare host:port) that keeps challenges under
``betteranusbis:challenge:<id>`` and honeypot counters under
``betteranusbis:honeypot:<networkkey>`` as JSON values with a TTL, and looks
them up with GET on every instance so two gates can share one store.

By default this file starts a small threaded RESP2 server in-process, so the
tests need neither Valkey nor Redis.  Every command the mock receives is kept
in a log, which is how the tests prove PING at startup, SET ... EX, and the
cross-instance GET.  If a real ``valkey-server`` or ``redis-server`` is on
PATH (or a URL is supplied), pass ``--real-server`` / ``--real-url``; the
store assertions then go through a tiny RESP client instead, and the few
observations that need the command log are noted (or skipped) accordingly.

Run:

    cd betterAnusbis
    python3 tests_valkey.py                 # mock RESP2 server (default)
    python3 tests_valkey.py --real-server   # start valkey-server/redis-server
    python3 tests_valkey.py --real-url redis://127.0.0.1:6379/0

If ``--store valkey`` is not advertised by ``./betterAnusbis --help`` the whole
suite prints one SKIP line per case and exits 0 (the state before the C backend
lands).  Any FAIL exits 1.
"""

from __future__ import annotations

import argparse
import fnmatch
import http.client
import json
import os
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlencode, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GATE = os.path.join(HERE, "betterAnusbis")
sys.path.insert(0, REPO)

API = "/.within.website/x/cmd/anubis/api/"
CHALLENGE_PREFIX = "betteranusbis:challenge:"
HONEYPOT_PREFIX = "betteranusbis:honeypot:"
DIFFICULTY = "5"  # small enough to solve instantly, big enough to be real
BACKEND_PORT = 0

PASS = 0
FAIL = 0
SKIP = 0

SOLVER = None  # created lazily so --help-only runs need no native library


# ---------------------------------------------------------------------------
# tiny RES P2 wire helpers
# ---------------------------------------------------------------------------

def _enc(text) -> bytes:
    return text.encode("utf-8") if isinstance(text, str) else bytes(text)


def resp_simple(text: str) -> bytes:
    return b"+" + _enc(text) + b"\r\n"


def resp_error(text: str) -> bytes:
    return b"-" + _enc(text) + b"\r\n"


def resp_int(value: int) -> bytes:
    return b":%d\r\n" % int(value)


def resp_bulk(data) -> bytes:
    if data is None:
        return b"$-1\r\n"
    data = _enc(data)
    return b"$%d\r\n" % len(data) + data + b"\r\n"


def resp_array(items) -> bytes:
    return b"*%d\r\n" % len(items) + b"".join(items)


# ---------------------------------------------------------------------------
# mock valkey
# ---------------------------------------------------------------------------

class MockStore:
    """RESP2 server state with monotonic TTLs and a command log."""

    has_command_log = True

    def __init__(self, password: str | None = None):
        self.password = password
        self._data: dict[str, tuple[bytes, float | None]] = {}
        self._lock = threading.Lock()
        self._log: list[dict] = []
        self._server = None
        self._thread = None
        self.port = 0
        self.url = ""

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> MockStore:
        self._server = MockValkey(("127.0.0.1", 0), self)
        self.port = self._server.server_address[1]
        self.url = f"redis://127.0.0.1:{self.port}/0"
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="mock-valkey", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(2)
            self._thread = None

    # -- observation ------------------------------------------------------
    @property
    def log(self) -> list[dict]:
        with self._lock:
            return list(self._log)

    def mark(self) -> int:
        with self._lock:
            return len(self._log)

    def commands(self, since: int = 0) -> list[list[str]]:
        with self._lock:
            return [entry["args"] for entry in self._log if entry["n"] >= since]

    def keys(self, prefix: str = "") -> list[str]:
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            return [k for k in self._data if k.startswith(prefix)]

    def value(self, key: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            item = self._data.get(key)
            return item[0] if item else None

    def ttl(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            item = self._data.get(key)
            if item is None:
                return -2
            if item[1] is None:
                return -1
            return max(0, int(item[1] - now))

    # -- internals --------------------------------------------------------
    def _purge_locked(self, now: float) -> None:
        for key in [k for k, (_, exp) in self._data.items() if exp is not None and exp <= now]:
            del self._data[key]

    def execute(self, args: list[str]) -> bytes:
        name = args[0].upper() if args else ""
        with self._lock:
            now = time.monotonic()
            self._purge_locked(now)
            reply = self._dispatch_locked(name, args, now)
            self._log.append(
                {
                    "n": len(self._log),
                    "args": list(args),
                    "reply": reply[:96].decode("utf-8", "replace"),
                }
            )
            return reply

    def _dispatch_locked(self, name: str, args: list[str], now: float) -> bytes:
        if name == "PING":
            if len(args) > 1:
                return resp_bulk(args[1])
            return resp_simple("PONG")

        if name == "AUTH":
            if len(args) < 2:
                return resp_error("ERR wrong number of arguments for 'auth' command")
            if self.password is None or args[-1] == self.password:
                return resp_simple("OK")
            return resp_error("WRONGPASS invalid username-password pair")

        if name == "SELECT":
            try:
                db = int(args[1])
            except (IndexError, ValueError):
                return resp_error("ERR invalid DB index")
            if db < 0 or db > 15:
                return resp_error("ERR DB index is out of range")
            return resp_simple("OK")

        if name == "GET":
            if len(args) != 2:
                return resp_error("ERR wrong number of arguments for 'get' command")
            item = self._data.get(args[1])
            return resp_bulk(item[0] if item else None)

        if name == "SET":
            return self._set_locked(args, now)

        if name == "DEL":
            removed = 0
            for key in args[1:]:
                if self._data.pop(key, None) is not None:
                    removed += 1
            return resp_int(removed)

        if name == "TTL":
            if len(args) != 2:
                return resp_error("ERR wrong number of arguments for 'ttl' command")
            item = self._data.get(args[1])
            if item is None:
                return resp_int(-2)
            if item[1] is None:
                return resp_int(-1)
            return resp_int(max(0, int(item[1] - now)))

        if name == "FLUSHDB":
            self._data.clear()
            return resp_simple("OK")

        if name == "SCAN":
            return self._scan_locked(args)

        if name == "QUIT":
            return resp_simple("OK")

        return resp_error(f"ERR unknown command '{args[0] if args else ''}'")

    def _set_locked(self, args: list[str], now: float) -> bytes:
        if len(args) < 3:
            return resp_error("ERR wrong number of arguments for 'set' command")
        key, value = args[1], args[2]
        expire_at = None
        keep_ttl = False
        nx = False
        xx = False
        i = 3
        while i < len(args):
            opt = args[i].upper()
            if opt == "EX" and i + 1 < len(args):
                try:
                    expire_at = now + float(args[i + 1])
                except ValueError:
                    return resp_error("ERR value is not an integer or out of range")
                i += 2
            elif opt == "PX" and i + 1 < len(args):
                try:
                    expire_at = now + float(args[i + 1]) / 1000.0
                except ValueError:
                    return resp_error("ERR value is not an integer or out of range")
                i += 2
            elif opt == "KEEPTTL":
                keep_ttl = True
                i += 1
            elif opt == "NX":
                nx = True
                i += 1
            elif opt == "XX":
                xx = True
                i += 1
            else:
                return resp_error("ERR syntax error")
        exists = key in self._data
        if nx and exists:
            return resp_bulk(None)
        if xx and not exists:
            return resp_bulk(None)
        if keep_ttl and exists:
            expire_at = self._data[key][1]
        self._data[key] = (_enc(value), expire_at)
        return resp_simple("OK")

    def _scan_locked(self, args: list[str]) -> bytes:
        if len(args) < 2:
            return resp_error("ERR wrong number of arguments for 'scan' command")
        try:
            int(args[1])
        except ValueError:
            return resp_error("ERR invalid cursor")
        match = "*"
        i = 2
        while i < len(args):
            opt = args[i].upper()
            if opt == "MATCH" and i + 1 < len(args):
                match = args[i + 1]
                i += 2
            elif opt == "COUNT" and i + 1 < len(args):
                try:
                    int(args[i + 1])
                except ValueError:
                    return resp_error("ERR value is not an integer or out of range")
                i += 2
            else:
                return resp_error("ERR syntax error")
        # A cursor of 0 with the full matching set is a legal, terminating
        # iteration; the gate loops until it sees cursor 0.
        matched = [key for key in self._data if fnmatch.fnmatchcase(key, match)]
        return resp_array([resp_bulk("0"), resp_array([resp_bulk(k) for k in matched])])


class MockValkey(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, address, store: MockStore):
        self.store = store
        super().__init__(address, _MockHandler)

    def handle_error(self, request, client_address):  # keep test output clean
        pass


class _MockHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        store: MockStore = self.server.store
        while True:
            try:
                args = self._read_command()
            except (EOFError, OSError, ValueError):
                return
            if args is None or not args:
                return
            try:
                reply = store.execute(args)
            except Exception as exc:  # never kill the listener on one bad input
                reply = resp_error(f"ERR mock failure: {exc}")
            try:
                self.wfile.write(reply)
                self.wfile.flush()
            except OSError:
                return

    def _read_command(self):
        first = self.rfile.read(1)
        if not first:
            raise EOFError
        if first == b"*":
            count_line = self.rfile.readline()
            if not count_line:
                raise EOFError
            count = int(count_line.strip())
            if count <= 0:
                return []
            args = []
            for _ in range(count):
                header = self.rfile.readline()
                if not header:
                    raise EOFError
                if not header.startswith(b"$"):
                    raise ValueError("expected a bulk string")
                length = int(header[1:].strip())
                data = self.rfile.read(length)
                if len(data) != length:
                    raise EOFError
                self.rfile.read(2)
                args.append(data.decode("utf-8", "replace"))
            return args
        # inline command (handy if a client sends "PING\r\n")
        line = first + self.rfile.readline()
        text = line.decode("utf-8", "replace").strip()
        return text.split() if text else []


# ---------------------------------------------------------------------------
# real server support (optional)
# ---------------------------------------------------------------------------

class RespError(Exception):
    pass


class RespClient:
    """Minimal RESP2 client used to observe a real valkey/redis server."""

    def __init__(self, url: str):
        parts = urlsplit(url)
        if parts.scheme not in ("redis", "tcp", "valkey"):
            raise RuntimeError(
                f"{url!r}: this test harness only speaks plain redis:// to a real server"
            )
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 6379
        self.username = parts.username
        self.password = parts.password
        raw_db = (parts.path or "/0").strip("/") or "0"
        self.db = int(raw_db)

    def command(self, *args):
        sock = socket.create_connection((self.host, self.port), timeout=5)
        sock.settimeout(5)
        try:
            stream = sock.makefile("rb")
            if self.password is not None:
                if self.username:
                    self._send(sock, "AUTH", self.username, self.password)
                else:
                    self._send(sock, "AUTH", self.password)
                self._read(stream)
            if self.db:
                self._send(sock, "SELECT", str(self.db))
                self._read(stream)
            self._send(sock, *args)
            return self._read(stream)
        finally:
            sock.close()

    @staticmethod
    def _send(sock, *args):
        parts = [_enc(a) for a in args]
        payload = b"*%d\r\n" % len(parts)
        payload += b"".join(b"$%d\r\n%s\r\n" % (len(p), p) for p in parts)
        sock.sendall(payload)

    def _read(self, stream):
        line = stream.readline()
        if not line:
            raise RespError("connection closed")
        kind, rest = line[:1], line[1:-2]
        if kind == b"+":
            return rest.decode("utf-8", "replace")
        if kind == b"-":
            raise RespError(rest.decode("utf-8", "replace"))
        if kind == b":":
            return int(rest)
        if kind == b"$":
            length = int(rest)
            if length == -1:
                return None
            data = stream.read(length)
            stream.read(2)
            return data
        if kind == b"*":
            count = int(rest)
            if count == -1:
                return None
            return [self._read(stream) for _ in range(count)]
        raise RespError(f"unexpected RESP reply {line[:40]!r}")


class RealStore:
    has_command_log = False

    def __init__(self, url: str):
        self.url = url
        self._client = RespClient(url)

    def keys(self, prefix: str = "") -> list[str]:
        found: list[str] = []
        cursor = 0
        while True:
            reply = self._client.command(
                "SCAN", str(cursor), "MATCH", prefix + "*", "COUNT", "1000"
            )
            cursor = int(reply[0])
            found.extend(k.decode("utf-8", "replace") for k in reply[1])
            if cursor == 0:
                break
        return sorted(k for k in found if k.startswith(prefix))

    def value(self, key: str) -> bytes | None:
        return self._client.command("GET", key)

    def ttl(self, key: str) -> int:
        return int(self._client.command("TTL", key))

    def mark(self) -> int:
        return 0

    def commands(self, since: int = 0) -> list[list[str]]:
        return []


class RealProvider:
    def __init__(self, url: str, proc=None, tmpdir: str | None = None):
        self.store = RealStore(url)
        self.proc = proc
        self.tmpdir = tmpdir

    def reset(self) -> None:
        self.store._client.command("FLUSHDB")

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.store._client.command("SHUTDOWN", "NOSAVE")
            except Exception:
                pass
            try:
                self.proc.wait(3)
            except Exception:
                self.proc.kill()
                self.proc.wait(2)
            self.proc = None
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None


def find_real_server() -> str | None:
    for name in ("valkey-server", "redis-server"):
        path = shutil.which(name)
        if path:
            return path
    return None


def start_real_server() -> RealProvider:
    binary = find_real_server()
    if binary is None:
        raise RuntimeError("no valkey-server or redis-server found on PATH")
    port = free_port()
    tmpdir = tempfile.mkdtemp(prefix="valkeytests-")
    proc = subprocess.Popen(
        [
            binary,
            "--bind", "127.0.0.1",
            "--port", str(port),
            "--save", "",
            "--appendonly", "no",
            "--daemonize", "no",
            "--dir", tmpdir,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"redis://127.0.0.1:{port}/0"
    deadline = time.monotonic() + 10
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"{binary} exited during startup: {stderr[-400:]}")
        try:
            client = RespClient(url)
            if client.command("PING") == "PONG":
                return RealProvider(url, proc, tmpdir)
        except (OSError, RespError) as exc:
            last_error = str(exc)
        time.sleep(0.05)
    proc.kill()
    proc.wait(2)
    shutil.rmtree(tmpdir, ignore_errors=True)
    raise RuntimeError(f"{binary} did not answer PING: {last_error}")


@contextmanager
def open_store(provider):
    if provider is None:
        with MockStore() as store:
            yield store
    else:
        provider.reset()
        yield provider.store


# ---------------------------------------------------------------------------
# gate / backend / http helpers
# ---------------------------------------------------------------------------

def store_arg(url: str) -> str:
    """The CLI wants the backend scheme in front of the URL."""
    return url if url.startswith("valkey:") else "valkey:" + url


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def can_connect(port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout):
            return True
    except OSError:
        return False


class Gate:
    def __init__(self, store_url: str, *extra: str):
        self.store_url = store_url
        self.extra = list(extra)
        self.port = free_port()
        self.proc = None
        self._lines: list[str] = []
        self._reader = None

    def __enter__(self) -> Gate:
        args = [
            GATE,
            "--listen", f"127.0.0.1:{self.port}",
            "--target", f"http://127.0.0.1:{BACKEND_PORT}",
            "--difficulty", DIFFICULTY,
            "--store", self.store_url,
            *self.extra,
        ]
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._drain, name="gate-log", daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                time.sleep(0.05)
                raise RuntimeError(
                    f"gate exited {self.proc.returncode} during startup: {self.log()[-800:]}"
                )
            if can_connect(self.port):
                return self
            time.sleep(0.02)
        raise RuntimeError(f"gate did not listen within 10s: {self.log()[-800:]}")

    def __exit__(self, *exc) -> None:
        self.stop()

    def _drain(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.append(line)
        except (OSError, ValueError):
            pass

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(2)
                except subprocess.TimeoutExpired:
                    pass
        if self._reader is not None:
            self._reader.join(1)
            self._reader = None
        self.proc = None

    def log(self) -> str:
        return "".join(self._lines)


class _ProtectedBackend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        body = b"<html><body><h1>PAST THE WALL</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def request(port: int, method: str, path: str, headers=None, timeout: float = 8):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        body = response.read()
        return response.status, response.getheaders(), body
    finally:
        conn.close()


def header(headers, name: str):
    for key, value in headers:
        if key.lower() == name.lower():
            return value
    return None


def wait_for(predicate, timeout: float = 3.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate() or None


def run_process(args, timeout: float):
    started = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return None, output, time.monotonic() - started


# ---------------------------------------------------------------------------
# challenge helpers
# ---------------------------------------------------------------------------

def get_challenge(port: int):
    status, headers, body = request(port, "GET", "/")
    if status != 200:
        raise AssertionError(f"challenge page returned {status}: {body[:160]!r}")
    import anubis_solver

    challenge = anubis_solver.parse_challenge(body.decode("utf-8", "replace"))
    if challenge is None:
        raise AssertionError(f"no challenge in page: {body[:200]!r}")
    return challenge, body


def solve(challenge):
    assert SOLVER is not None
    return SOLVER.solve(challenge)


def pass_challenge(port: int, challenge, solution, redir: str | None = None):
    query = {
        "id": challenge.id,
        "response": solution.response_hex,
        "nonce": str(solution.nonce),
    }
    if redir is not None:
        query["redir"] = redir
    return request(port, "GET", API + "pass-challenge?" + urlencode(query))


# ---------------------------------------------------------------------------
# assertions and result printing
# ---------------------------------------------------------------------------

class CaseSkip(Exception):
    pass


def require(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _set_commands(store, key: str, since: int = 0):
    if not getattr(store, "has_command_log", False):
        return []
    return [
        cmd
        for cmd in store.commands(since)
        if len(cmd) >= 3 and cmd[0].upper() == "SET" and cmd[1] == key
    ]


def _parse_ex(command: list[str]) -> int | None:
    for index, token in enumerate(command):
        if token.upper() == "EX" and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                return None
    return None


def _json_of(command: list[str]):
    if len(command) < 3:
        return None
    try:
        return json.loads(command[2])
    except (ValueError, TypeError):
        return None


def _is_spent(command: list[str]) -> bool:
    data = _json_of(command)
    if not isinstance(data, dict):
        return False
    return data.get("spent") in (True, 1, "1", "true")


def _challenge_key(store, challenge) -> str | None:
    for key in store.keys(CHALLENGE_PREFIX):
        raw = store.value(key)
        if raw is None:
            continue
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("randomData") == challenge.random_data:
            return key
        if challenge.id and key == CHALLENGE_PREFIX + challenge.id:
            return key
    return None


def _commands_named(store, name: str, since: int = 0):
    if not getattr(store, "has_command_log", False):
        return None
    return [
        cmd for cmd in store.commands(since) if cmd and cmd[0].upper() == name
    ]


def _await_startup_scans(store, gates: int, timeout: float = 2.5) -> None:
    """Give each gate's asynchronous startup SCAN a chance to finish.

    Without this, a gate B started just before a challenge is issued could
    still be loading the store when the test expects a remote GET, which would
    make the sharing tests racy rather than wrong.
    """
    if not getattr(store, "has_command_log", False):
        return
    wait_for(
        lambda: len(
            [cmd for cmd in store.commands() if cmd and cmd[0].upper() == "SCAN"]
        )
        >= gates,
        timeout,
    )


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def case_startup(provider):
    with open_store(provider) as store, Gate(store_arg(store.url)) as gate:
        pings = wait_for(lambda: _commands_named(store, "PING"), 3)
        if pings is None:
            raise CaseSkip("real server: the command log cannot show PING")
        require(pings, "gate never sent PING to the store")
        return f"{len(pings)} PING(s), gate on 127.0.0.1:{gate.port}"


def case_issue(provider):
    with open_store(provider) as store, Gate(store_arg(store.url)) as gate:
        challenge, _ = get_challenge(gate.port)
        key = wait_for(lambda: _challenge_key(store, challenge), 2)
        require(
            key is not None,
            "no betteranusbis:challenge:* record matched randomData "
            f"{challenge.random_data[:16]}...",
        )
        raw = store.value(key)
        require(raw is not None, f"challenge key {key} has no value")
        try:
            record = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as exc:
            raise AssertionError(
                f"{key}: value is not JSON: {raw[:120]!r} ({exc})"
            ) from exc
        for field in ("randomData", "algorithm", "difficulty"):
            require(field in record, f"{key}: JSON lacks {field!r}: {record}")

        if store.has_command_log:
            sets = wait_for(lambda: _set_commands(store, key), 2) or []
            require(sets, f"no SET was issued for {key}")
            ex = _parse_ex(sets[-1])
            require(
                ex is not None and ex > 0,
                f"SET for {key} carried no positive EX: {sets[-1]}",
            )
            detail = f"{key}; SET EX {ex}"
        else:
            ttl = store.ttl(key)
            require(ttl > 0, f"{key}: stored challenge has no TTL ({ttl})")
            detail = f"{key}; TTL {ttl}s (real server)"
        return detail


def case_restart_impl(provider):
    with open_store(provider) as store:
        with Gate(store_arg(store.url)) as first:
            challenge, _ = get_challenge(first.port)
        solution = solve(challenge)

        mark = store.mark()
        with Gate(store_arg(store.url)) as second:
            if store.has_command_log:
                scans = wait_for(
                    lambda: [
                        cmd
                        for cmd in store.commands(mark)
                        if cmd and cmd[0].upper() == "SCAN"
                    ],
                    3,
                )
                require(scans, "restarted gate did not SCAN the store")
            status, _, body = pass_challenge(second.port, challenge, solution)
            require(
                status == 302,
                f"pass-challenge after restart returned {status}: {body[:180]!r}",
            )
        return "challenge issued before the restart still verifies (302)"


def case_sharing(provider):
    with open_store(provider) as store:
        with Gate(store_arg(store.url)) as gate_a, Gate(store_arg(store.url)) as gate_b:
            _await_startup_scans(store, 2)
            challenge, _ = get_challenge(gate_a.port)
            solution = solve(challenge)
            mark = store.mark()
            status, _, body = pass_challenge(gate_b.port, challenge, solution)
            require(
                status == 302,
                f"gate B rejected gate A's challenge with {status}: {body[:180]!r}",
            )
            detail = "B accepted A's challenge"
            if store.has_command_log:
                gets = wait_for(
                    lambda: [
                        cmd
                        for cmd in store.commands(mark)
                        if len(cmd) >= 2
                        and cmd[0].upper() == "GET"
                        and cmd[1].startswith(CHALLENGE_PREFIX)
                    ],
                    2,
                )
                require(gets, "gate B never GET the challenge key from the shared store")
                detail += f"; B fetched {gets[-1][1]} with GET"
            else:
                detail += "; GET not observable (real server)"
            return detail


def case_replay(provider):
    with open_store(provider) as store, Gate(store_arg(store.url)) as gate:
        challenge, _ = get_challenge(gate.port)
        solution = solve(challenge)
        key = _challenge_key(store, challenge)
        require(key is not None, "challenge record missing before the replay test")

        mark = store.mark()
        first, _, body1 = pass_challenge(gate.port, challenge, solution)
        require(first == 302, f"the first submission returned {first}: {body1[:180]!r}")
        second, headers2, body2 = pass_challenge(gate.port, challenge, solution)
        require(
            second == 403,
            f"the replay returned {second} instead of 403 "
            f"(Location={header(headers2, 'Location')!r}): {body2[:180]!r}",
        )
        detail = "first 302, replay 403"
        if store.has_command_log:
            spent = [cmd for cmd in _set_commands(store, key, mark) if _is_spent(cmd)]
            if spent:
                detail += "; spent=true SET persisted in the store"
            else:
                require(
                    store.value(key) is None,
                    f"{key} was neither marked spent nor deleted after the replay",
                )
                detail += "; record deleted after spend"
        return detail


def case_honeypot(provider):
    with open_store(provider) as store:
        with Gate(store_arg(store.url)) as gate_a, Gate(store_arg(store.url)) as gate_b:
            _await_startup_scans(store, 2)
            status, _, body = request(gate_a.port, "GET", "/")
            require(status == 200, f"challenge page returned {status}")
            match = re.search(rb"honeypot/([^/\"'\s<>]+)/init", body)
            require(match is not None, "challenge page has no honeypot trap link")
            trap = match.group(1).decode("utf-8", "replace")

            status, _, _ = request(
                gate_a.port, "GET", API + f"honeypot/{trap}/init", timeout=10
            )
            require(status in (200, 302, 303), f"honeypot init returned {status}")

            keys = wait_for(lambda: store.keys(HONEYPOT_PREFIX), 3) or []
            require(keys, "no betteranusbis:honeypot:* key in the store")
            ttl = store.ttl(keys[0])
            require(ttl > 0, f"{keys[0]}: honeypot record has no TTL ({ttl})")
            detail = f"{keys[0]}; TTL {ttl}s"

            if store.has_command_log:
                mark = store.mark()
                request(gate_b.port, "GET", "/")
                gets = wait_for(
                    lambda: [
                        cmd
                        for cmd in store.commands(mark)
                        if len(cmd) >= 2
                        and cmd[0].upper() == "GET"
                        and cmd[1].startswith(HONEYPOT_PREFIX)
                    ],
                    3,
                )
                require(gets, "gate B never GET the honeypot key from the shared store")
                detail += f"; B fetched {gets[-1][1]} with GET"
            else:
                detail += "; GET not observable (real server)"
            return detail


def case_unreachable(provider):
    args = [
        GATE,
        "--listen", f"127.0.0.1:{free_port()}",
        "--target", f"http://127.0.0.1:{BACKEND_PORT}",
        "--difficulty", DIFFICULTY,
        "--store", "valkey:redis://127.0.0.1:1/0",
    ]
    returncode, output, elapsed = run_process(args, timeout=7)
    require(
        returncode is not None,
        "gate was still running after 7s with an unreachable store",
    )
    require(
        returncode != 0,
        f"an unreachable store did not fail startup (exit {returncode})",
    )
    require(
        elapsed <= 6.0,
        f"rejecting an unreachable store took {elapsed:.1f}s (want <= ~5s)",
    )
    require(
        re.search(r"valkey|redis|connect|refused|unreachable", output, re.I),
        f"error does not mention valkey/connect: {output.strip()[-220:]!r}",
    )
    return f"exit {returncode} in {elapsed:.2f}s: {output.strip().splitlines()[-1][:120]}"


def case_bad_url(provider):
    details = []
    for bad in ("valkey:", "valkey:http:bad"):
        args = [
            GATE,
            "--listen", f"127.0.0.1:{free_port()}",
            "--target", f"http://127.0.0.1:{BACKEND_PORT}",
            "--difficulty", DIFFICULTY,
            "--store", bad,
        ]
        returncode, output, _ = run_process(args, timeout=4)
        require(returncode is not None, f"{bad!r}: gate hung instead of rejecting the URL")
        require(returncode != 0, f"{bad!r}: accepted (exit {returncode})")
        require(output.strip(), f"{bad!r}: exited silently")
        require(
            re.search(r"valkey|store|url|redis|invalid|bad|want", output, re.I),
            f"{bad!r}: message is not clear: {output.strip()[-200:]!r}",
        )
        details.append(f"{bad!r} -> exit {returncode}")
    return "; ".join(details)


def case_spend_ttl(provider):
    with open_store(provider) as store, Gate(store_arg(store.url)) as gate:
        challenge, _ = get_challenge(gate.port)
        key = _challenge_key(store, challenge)
        require(key is not None, "no stored challenge record after issuing")

        if store.has_command_log:
            issued = [cmd for cmd in _set_commands(store, key) if not _is_spent(cmd)]
            require(issued, f"no unspent SET was issued for {key}")
            original_ex = _parse_ex(issued[-1])
            require(
                original_ex is not None and original_ex > 0,
                f"issued SET has no positive EX: {issued[-1]}",
            )

            # Spend a moment later so a backend that writes the *remaining*
            # lifetime is exercised; the contract only requires <=, so the
            # assertion below still accepts an unchanged TTL.
            time.sleep(1.2)
            solution = solve(challenge)
            status, _, body = pass_challenge(gate.port, challenge, solution)
            require(status == 302, f"solving returned {status}: {body[:180]!r}")

            spent = [cmd for cmd in _set_commands(store, key) if _is_spent(cmd)]
            require(spent, f"spending did not SET a spent record for {key}")
            spent_ex = _parse_ex(spent[-1])
            require(
                spent_ex is not None,
                f"spent SET has no EX: {spent[-1]}",
            )
            require(
                spent_ex <= original_ex,
                f"spent SET EX {spent_ex} exceeds the original TTL {original_ex}",
            )
            return f"issued EX {original_ex}, spent EX {spent_ex}"

        # real server: the TTL command is the best available proxy, and only
        # says the stored lifetime did not grow.
        before = store.ttl(key)
        require(before > 0, f"{key}: issued record has no TTL ({before})")
        solution = solve(challenge)
        status, _, body = pass_challenge(gate.port, challenge, solution)
        require(status == 302, f"solving returned {status}: {body[:180]!r}")
        after = store.ttl(key)
        require(
            after == -2 or after <= before + 1,
            f"spent record TTL {after} grew past the original {before}",
        )
        return f"TTL {before}s -> {after}s (real server; EX not observable)"


CASES = [
    ("startup connects to the store and PINGs", case_startup),
    ("issuing a challenge writes JSON with SET ... EX", case_issue),
    ("a challenge survives a gate restart", case_restart_impl),
    ("two gates share one store (B GETs A's challenge)", case_sharing),
    ("replaying a solved challenge is rejected", case_replay),
    ("honeypot record is stored with a TTL and read by B", case_honeypot),
    ("an unreachable store fails startup quickly", case_unreachable),
    ("empty / malformed valkey URLs are rejected", case_bad_url),
    ("spending keeps the challenge TTL (spent SET EX <= issued EX)", case_spend_ttl),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def help_text() -> str:
    try:
        proc = subprocess.run(
            [GATE, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return (proc.stdout or "") + "\n"
    except (OSError, subprocess.TimeoutExpired):
        return ""


def valkey_advertised() -> bool:
    text = help_text()
    if not text:
        return False
    flat = " ".join(text.split())
    if re.search(r"valkey[^.]{0,48}not implemented", flat, re.I):
        return False
    if re.search(r"--store\s+valkey[: ]", flat, re.I):
        return True
    if re.search(r"store[^;]{0,400}valkey:", flat, re.I):
        return True
    if re.search(r"valkey:\s*(?:redis|rediss|unix|//)", text, re.I):
        return True
    return False


def run_case(number: int, title: str, function, provider) -> None:
    global PASS, FAIL, SKIP
    try:
        detail = function(provider) or ""
        PASS += 1
        status = "PASS"
    except CaseSkip as exc:
        SKIP += 1
        status = "SKIP"
        detail = str(exc)
    except Exception as exc:  # noqa: BLE001 - a failed case must not stop the run
        FAIL += 1
        status = "FAIL"
        detail = f"{type(exc).__name__}: {exc}"
    line = f"{status:<4} {number}. {title}"
    if detail:
        line += f"  [{detail}]"
    print(line, flush=True)


def main(argv=None) -> int:
    global BACKEND_PORT, SOLVER

    parser = argparse.ArgumentParser(
        description="Valkey store-backend tests for betterAnusbis"
    )
    parser.add_argument(
        "--real-server",
        action="store_true",
        help="start a real valkey-server/redis-server from PATH instead of the mock",
    )
    parser.add_argument(
        "--real-url",
        metavar="URL",
        help="use an already-running real server (redis://host:port/db)",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(GATE):
        print(f"FAIL 0. {GATE} is missing; run make first")
        return 1

    if not valkey_advertised():
        reason = "--store valkey is not advertised in ./betterAnusbis --help"
        print(f"Valkey store backend not implemented yet: {reason}\n")
        for number, (title, _fn) in enumerate(CASES, 1):
            print(f"SKIP {number}. {title}  [{reason}]", flush=True)
        print(f"\n0 passed, 0 failed, {len(CASES)} skipped")
        return 0

    import anubis_solver

    SOLVER = anubis_solver.AnubisSolver(threads=4)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), _ProtectedBackend)
    backend.daemon_threads = True
    BACKEND_PORT = backend.server_address[1]
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    provider = None
    try:
        if args.real_url:
            provider = RealProvider(args.real_url)
            provider.store._client.command("PING")
            print(f"store: real server at {args.real_url}\n")
        elif args.real_server:
            provider = start_real_server()
            print(f"store: real server at {provider.store.url} (from PATH)\n")
        else:
            print("store: in-process RESP2 mock (default)\n")

        for number, (title, function) in enumerate(CASES, 1):
            run_case(number, title, function, provider)
    finally:
        if provider is not None:
            provider.close()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(2)

    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
