#!/usr/bin/env python3
"""HTTP/2 and HTTP/3 integration checks for betterAnusbis.

Run:  python3 tests_http23.py -v [name-filter ...]

Each case prints exactly one PASS / FAIL / SKIP line and starts every server it
needs on a dynamic port, so the cases do not depend on each other.  A missing
dependency or a binary that does not advertise the flag under test is a SKIP,
not a FAIL; the exit status is nonzero only when a case actually fails.

The flags under test are the agreed surface; a case skips when the built gate
does not advertise the flag it needs, so the same file is useful before and
after the implementation lands:

    --http2 / --no-http2       HTTP/2 on the listener (default on; ALPN h2 for
                               TLS, h2c prior knowledge on the plain listener)
    --http3 / --no-http3       HTTP/3 on the listener (default off; QUIC on the
                               same UDP port, needs --tls-cert/--tls-key)
    --target-http2             HTTP/2 towards the target (default auto)
    --target-http3             HTTP/3 towards the target (default off)

Optional python dependencies are imported lazily: httpx (multiplexing), h2 (the
HTTP/2 backend) and aioquic (the HTTP/3 client and backend).  Name filters match
case names as substrings, e.g. `python3 tests_http23.py -v 03 08`.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GATE = os.path.join(HERE, "betterAnusbis")

# The gate's own API prefix.  /api/ping is accepted as an alias because the task
# description names it that way, but the canonical route is the anubis one.
API_PREFIX = "/.within.website/x/cmd/anubis/api/"
PING_PATHS = (API_PREFIX + "ping", "/api/ping")
CHALLENGE_MARK = b"anubis_challenge"

H2_OK_BODY = b"BACKEND-H2-OK\n"
H3_OK_BODY = b"BACKEND-H3-OK\n"
H1_PAGE = b"<html><body><h1>PAST THE WALL</h1></body></html>"

# Filled in by case 1.
WORK: str | None = None
GLOBAL: dict[str, object] = {
    "cert": None,
    "key": None,
    "gate_usable": False,
    "gate_reason": "setup did not complete",
    "help": "",
}

VERBOSE = False


# --------------------------------------------------------------------- harness
class Fail(Exception):
    """A case failed; the message is the reason."""


class Skip(Exception):
    """A case could not run here; the message is the reason."""


def note(msg: str) -> None:
    if VERBOSE:
        print(f"      · {msg}")


def check(cond: object, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def run_async(coro, timeout: float = 10.0):
    """Run one coroutine to completion with a hard deadline.

    The loop is built by hand so that aioquic's cancelled-handshake future does
    not dump "Future exception was never retrieved" on stderr: for QUIC a failed
    or timed-out handshake is the answer we are looking for, not noise.
    """
    loop = asyncio.new_event_loop()

    def quiet_handler(loop_, context):
        if "Future exception was never retrieved" in context.get("message", ""):
            return
        loop_.default_exception_handler(context)

    loop.set_exception_handler(quiet_handler)
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout))
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


# ------------------------------------------------------------------- processes
class Process:
    """A child process with captured output and a bounded shutdown."""

    def __init__(self, cmd: list[str], name: str):
        self.cmd = cmd
        self.name = name
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._reader: threading.Thread | None = None
        self._log = threading.Lock()

    def start(self) -> Process:
        note(f"start {self.name}: {' '.join(self.cmd)}")
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        return self

    def _drain(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                with self._log:
                    self.lines.append(line.rstrip("\n"))
        except (ValueError, OSError):
            pass

    def tail(self, n: int = 40) -> str:
        with self._log:
            chunk = self.lines[-n:]
        return "\n".join(chunk) if chunk else "(no output)"

    def poll(self):
        return self.proc.poll() if self.proc else None

    def wait_listening(self, port: int, timeout: float = 8.0) -> None:
        assert self.proc is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise Fail(
                    f"{self.name} exited during startup (rc={self.proc.returncode}):\n"
                    f"{self.tail()}"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), 0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise Fail(f"{self.name} did not listen on 127.0.0.1:{port}:\n{self.tail()}")

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        if self.proc.stdout:
            try:
                self.proc.stdout.close()
            except OSError:
                pass
        if self._reader:
            self._reader.join(timeout=1)


class Gate(Process):
    """A betterAnusbis process on a dynamic listener port."""

    def __init__(self, port: int, target: str, extra: list[str] | None = None):
        super().__init__(
            [
                GATE,
                "--listen",
                f"127.0.0.1:{port}",
                "--target",
                target,
                *(extra or []),
            ],
            "gate",
        )
        self.port = port

    def __enter__(self) -> Gate:
        self.start()
        self.wait_listening(self.port)
        return self

    def __exit__(self, etype, evalue, tb) -> bool:
        self.stop()
        if isinstance(evalue, (Fail, Skip)):
            evalue.args = (
                f"{evalue.args[0]}\n    gate output:\n    " + self.tail().replace("\n", "\n    "),
            )
        return False


def run_to_exit(cmd: list[str], timeout: float = 5.0) -> tuple[int | None, str, bool]:
    """Run a short-lived process; return (rc, output, timed_out)."""
    proc = subprocess.Popen(
        cmd,
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", False
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        return proc.returncode, out or "", True


# ------------------------------------------------------------------- ports/fs
def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_tcp_udp_port() -> int:
    """A port number that is currently free for both TCP and UDP."""
    last = None
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.bind(("127.0.0.1", 0))
            port = tcp.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
                try:
                    udp.bind(("127.0.0.1", port))
                except OSError as exc:  # UDP side taken, try another number
                    last = str(exc)
                    continue
        return port
    raise Fail(f"could not find a free TCP+UDP port pair (last error: {last})")


def tls_args() -> list[str]:
    assert GLOBAL["cert"] and GLOBAL["key"]
    return ["--tls-cert", str(GLOBAL["cert"]), "--tls-key", str(GLOBAL["key"])]


def write_policy(text: str) -> str:
    assert WORK
    path = os.path.join(WORK, "policy.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def require_gate() -> None:
    if not GLOBAL["gate_usable"]:
        raise Skip(str(GLOBAL["gate_reason"]))


def help_text() -> str:
    return str(GLOBAL["help"])


def has_flag(name: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", help_text()) is not None


def require_flag(*names: str) -> None:
    require_gate()
    if not any(has_flag(n) for n in names):
        raise Skip(f"binary does not advertise {' or '.join(names)}")


def require_curl() -> str:
    curl = shutil.which("curl")
    if not curl:
        raise Skip("curl is not installed")
    return curl


def curl_supports_h2() -> bool:
    version = subprocess.run(
        [shutil.which("curl") or "curl", "--version"], capture_output=True, text=True
    ).stdout
    return "HTTP2" in version


def curl_help_all() -> str:
    cached = GLOBAL.get("curl_help")
    if not cached:
        cached = subprocess.run(
            [shutil.which("curl") or "curl", "--help", "all"],
            capture_output=True,
            text=True,
        ).stdout
        GLOBAL["curl_help"] = cached
    return str(cached)


def parse_header_dump(text: str) -> tuple[str | None, list[tuple[str, str]]]:
    """Parse curl -D output, keeping only the last response's headers."""
    status: str | None = None
    headers: list[tuple[str, str]] = []
    for line in text.splitlines():
        if line.upper().startswith("HTTP/"):
            status = line.strip()
            headers = []
        elif ":" in line and line.strip():
            name, value = line.split(":", 1)
            headers.append((name.strip().lower(), value.strip()))
    return status, headers


def header(headers: list[tuple[str, str]], name: str) -> str | None:
    for key, value in headers:
        if key == name.lower():
            return value
    return None


def curl_fetch(
    url: str,
    *,
    extra: list[str] | None = None,
    timeout: float = 6.0,
    tls: bool = True,
) -> dict:
    assert WORK
    curl = require_curl()
    header_path = os.path.join(WORK, "curl.headers")
    body_path = os.path.join(WORK, "curl.body")
    cmd = [
        curl,
        "-sS",
        "--max-time",
        str(timeout),
        "-o",
        body_path,
        "-D",
        header_path,
        "-w",
        "%{http_version} %{http_code}",
    ]
    if tls:
        cmd.append("-k")
    cmd.extend(extra or [])
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 3)
    if proc.returncode != 0:
        raise Fail(f"curl failed (rc={proc.returncode}): {proc.stderr.strip()}")
    version, _, code = proc.stdout.strip().partition(" ")
    try:
        with open(body_path, "rb") as fh:
            body = fh.read()
        with open(header_path, encoding="latin-1") as fh:
            header_text = fh.read()
    except OSError as exc:
        raise Fail(f"could not read curl output: {exc}") from exc
    status_line, headers = parse_header_dump(header_text)
    return {
        "version": version,
        "code": int(code) if code.isdigit() else None,
        "status_line": status_line,
        "headers": headers,
        "body": body,
    }


# ------------------------------------------------------------------- backends
class _H1Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/ping" or path == API_PREFIX + "ping":
            body, ctype = b"pong\n", "text/plain; charset=utf-8"
        else:
            body, ctype = H1_PAGE, "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class H1Backend:
    """A tiny plain-HTTP backend; the gate's challenge never needs it."""

    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _H1Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> H1Backend:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        return False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def start_h2_tls_backend() -> dict:
    """A TLS HTTP/2 backend built on the h2 library, in a thread."""
    try:
        import h2.config
        import h2.connection
        import h2.events
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Skip(f"python 'h2' package is missing: {exc}") from exc

    assert GLOBAL["cert"] and GLOBAL["key"]
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(GLOBAL["cert"]), str(GLOBAL["key"]))
    context.set_alpn_protocols(["h2"])

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.25)
    state: dict = {"protocol": None, "requests": 0, "error": None, "port": listener.getsockname()[1]}
    stop = threading.Event()

    def serve_connection(conn: socket.socket) -> None:
        try:
            tls = context.wrap_socket(conn, server_side=True)
        except (ssl.SSLError, OSError) as exc:
            state["error"] = f"tls handshake: {exc}"
            conn.close()
            return
        try:
            state["protocol"] = tls.selected_alpn_protocol()
            conn_h2 = h2.connection.H2Connection(
                config=h2.config.H2Configuration(
                    client_side=False, header_encoding="utf-8"
                )
            )
            conn_h2.initiate_connection()
            tls.sendall(conn_h2.data_to_send())
            tls.settimeout(0.5)
            while not stop.is_set():
                try:
                    data = tls.recv(65535)
                except TimeoutError:
                    continue
                except (ssl.SSLError, OSError):
                    break
                if not data:
                    break
                terminated = False
                for event in conn_h2.receive_data(data):
                    if isinstance(event, h2.events.RequestReceived):
                        state["requests"] += 1
                        conn_h2.send_headers(
                            event.stream_id,
                            [
                                (":status", "200"),
                                ("content-type", "text/plain"),
                                ("content-length", str(len(H2_OK_BODY))),
                            ],
                        )
                        conn_h2.send_data(event.stream_id, H2_OK_BODY, end_stream=True)
                    elif isinstance(event, h2.events.ConnectionTerminated):
                        terminated = True
                out = conn_h2.data_to_send()
                if out:
                    tls.sendall(out)
                if terminated:
                    break
        except (ssl.SSLError, OSError, h2.exceptions.H2Error) as exc:  # type: ignore[attr-defined]
            state["error"] = str(exc)
        finally:
            try:
                tls.close()
            except OSError:
                pass

    def accept_loop() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=serve_connection, args=(conn,), daemon=True).start()

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    state["stop"] = stop
    state["listener"] = listener
    state["thread"] = thread
    return state


def stop_h2_tls_backend(state: dict) -> None:
    stop = state.get("stop")
    listener = state.get("listener")
    if stop:
        stop.set()
    if listener:
        try:
            listener.close()
        except OSError:
            pass
    thread = state.get("thread")
    if thread:
        thread.join(timeout=2)


def make_h3_client_class():
    try:
        from aioquic.asyncio.protocol import QuicConnectionProtocol
        from aioquic.h3.connection import H3Connection
        from aioquic.h3.events import DataReceived, HeadersReceived
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Skip(f"python 'aioquic' package is missing: {exc}") from exc

    class H3Client(QuicConnectionProtocol):
        def __init__(self, quic, stream_handler=None):
            super().__init__(quic, stream_handler=stream_handler)
            self.http = H3Connection(quic)
            self.responses: dict[int, dict] = {}

        def request(self, authority: str, path: str) -> int:
            stream_id = self._quic.get_next_available_stream_id()
            record = {
                "headers": [],
                "body": bytearray(),
                "done": asyncio.Event(),
            }
            self.responses[stream_id] = record
            self.http.send_headers(
                stream_id=stream_id,
                headers=[
                    (b":method", b"GET"),
                    (b":scheme", b"https"),
                    (b":authority", authority.encode()),
                    (b":path", path.encode()),
                ],
                end_stream=True,
            )
            self.transmit()
            return stream_id

        def quic_event_received(self, event):
            for http_event in self.http.handle_event(event):
                record = self.responses.get(http_event.stream_id)
                if record is None:
                    continue
                if isinstance(http_event, HeadersReceived):
                    record["headers"].extend(http_event.headers)
                elif isinstance(http_event, DataReceived):
                    record["body"].extend(http_event.data)
                    if http_event.stream_ended:
                        record["done"].set()

    return H3Client


def start_h3_upstream(state: dict) -> dict:
    """A minimal HTTP/3 backend on a dynamic UDP port, run in a thread."""
    try:
        from aioquic.asyncio import serve
        from aioquic.asyncio.protocol import QuicConnectionProtocol
        from aioquic.h3.connection import H3_ALPN, H3Connection
        from aioquic.h3.events import HeadersReceived
        from aioquic.quic.configuration import QuicConfiguration
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Skip(f"python 'aioquic' package is missing: {exc}") from exc

    assert GLOBAL["cert"] and GLOBAL["key"]
    port = free_tcp_udp_port()
    state.update({"port": port, "seen": None, "error": None, "ready": threading.Event()})

    class H3Upstream(QuicConnectionProtocol):
        def __init__(self, quic, stream_handler=None):
            super().__init__(quic, stream_handler=stream_handler)
            self.http = H3Connection(quic)

        def quic_event_received(self, event):
            for http_event in self.http.handle_event(event):
                if isinstance(http_event, HeadersReceived):
                    state["seen"] = "h3"
                    self.http.send_headers(
                        stream_id=http_event.stream_id,
                        headers=[
                            (b":status", b"200"),
                            (b"content-type", b"text/plain"),
                            (b"content-length", str(len(H3_OK_BODY)).encode()),
                        ],
                    )
                    self.http.send_data(
                        http_event.stream_id, H3_OK_BODY, end_stream=True
                    )
                    self.transmit()

    async def main() -> None:
        config = QuicConfiguration(is_client=False, alpn_protocols=H3_ALPN)
        config.load_cert_chain(str(GLOBAL["cert"]), str(GLOBAL["key"]))
        server = await serve(
            "127.0.0.1",
            port,
            configuration=config,
            create_protocol=lambda quic, stream_handler=None: H3Upstream(
                quic, stream_handler=stream_handler
            ),
        )
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        state["shutdown"] = lambda: loop.call_soon_threadsafe(stop_event.set)
        state["server"] = server
        state["ready"].set()
        await stop_event.wait()
        server.close()

    def runner() -> None:
        try:
            asyncio.run(main())
        except BaseException as exc:  # noqa: BLE001 - reported to the case
            state["error"] = exc
            state["ready"].set()

    thread = threading.Thread(target=runner, daemon=True)
    state["thread"] = thread
    thread.start()
    if not state["ready"].wait(timeout=5):
        raise Fail("HTTP/3 upstream did not start within 5s")
    if state["error"] is not None:
        raise Fail(f"HTTP/3 upstream failed to start: {state['error']}")
    return state


def stop_h3_upstream(state: dict) -> None:
    shutdown = state.get("shutdown")
    if shutdown is not None:
        try:
            # The aioquic server runs in its own thread; hop onto its loop.
            shutdown()
        except Exception:
            pass
    thread = state.get("thread")
    if thread:
        thread.join(timeout=3)


async def _h3_connect_once(port: int):
    from aioquic.asyncio import connect
    from aioquic.h3.connection import H3_ALPN
    from aioquic.quic.configuration import QuicConfiguration

    config = QuicConfiguration(
        is_client=True,
        alpn_protocols=H3_ALPN,
        verify_mode=ssl.CERT_NONE,
        server_name="localhost",
    )
    async with connect("127.0.0.1", port, configuration=config):
        return True


def h3_reachable(port: int, timeout: float = 2.0) -> bool:
    try:
        run_async(_h3_connect_once(port), timeout)
        return True
    except (TimeoutError, ConnectionError, OSError, ssl.SSLError):
        return False
    except Exception:  # noqa: BLE001 - any failure means no usable h3 endpoint
        return False


# ---------------------------------------------------------------------- cases
def setup_environment() -> None:
    """Generate the shared certificate and prove the binary is usable."""
    global WORK
    assert WORK
    openssl = shutil.which("openssl")
    check(openssl is not None, "openssl is not installed; cannot make a certificate")

    cert = os.path.join(WORK, "localhost.crt")
    key = os.path.join(WORK, "localhost.key")
    cmd = [
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-sha256",
        "-nodes",
        "-days",
        "2",
        "-keyout",
        key,
        "-out",
        cert,
        "-subj",
        "/CN=localhost",
        "-addext",
        "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        # Very old openssl releases do not know -addext; retry with a config.
        conf = os.path.join(WORK, "openssl.cnf")
        with open(conf, "w", encoding="utf-8") as fh:
            fh.write(
                "[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n"
                "[dn]\nCN=localhost\n"
                "[ext]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n"
            )
        cmd = [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-nodes",
            "-days",
            "2",
            "-keyout",
            key,
            "-out",
            cert,
            "-config",
            conf,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        check(
            proc.returncode == 0,
            f"openssl req failed: {(proc.stderr or proc.stdout).strip()[:400]}",
        )
    check(os.path.getsize(cert) > 0 and os.path.getsize(key) > 0, "empty cert/key")
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)
    except ssl.SSLError as exc:
        raise Fail(f"generated certificate does not load: {exc}") from exc
    GLOBAL["cert"], GLOBAL["key"] = cert, key
    note(f"certificate: {cert}")

    port = free_tcp_udp_port()
    check(port > 0, "no free TCP+UDP port pair found")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        tcp.bind(("127.0.0.1", port))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.bind(("127.0.0.1", port))
    note(f"free TCP+UDP port pair: {port}")

    if not os.path.exists(GATE):
        raise Skip(f"{GATE} is missing (run make in {HERE} first)")
    if not os.access(GATE, os.X_OK):
        raise Skip(f"{GATE} is not executable")

    rc, out, timed_out = run_to_exit([GATE, "--help"], timeout=5)
    GLOBAL["help"] = out
    if "usage:" not in out.lower() and "betteranusbis" not in out.lower():
        reason = (out or "").strip().splitlines()
        reason_text = reason[0] if reason else f"rc={rc}"
        raise Skip(f"{GATE} is present but not runnable here: {reason_text}")
    check(not timed_out, "`betterAnusbis --help` hung")

    # Actually start and stop a gate once, so the process plumbing is covered.
    probe_port = free_tcp_udp_port()
    probe = Gate(probe_port, f"http://127.0.0.1:{free_tcp_port()}")
    with probe:
        pass
    check(probe.poll() is not None, "the gate did not stop on SIGTERM")

    GLOBAL["gate_usable"] = True
    GLOBAL["gate_reason"] = ""


def case_01_setup() -> None:
    setup_environment()


def case_02_http1_regression() -> None:
    require_gate()
    with H1Backend() as backend:
        port = free_tcp_udp_port()
        with Gate(port, backend.url) as _gate:
            page = curl_fetch(f"http://127.0.0.1:{port}/", tls=False)
            check(page["code"] == 200, f"GET / returned {page['code']}")
            check(
                CHALLENGE_MARK in page["body"],
                f"GET / body does not contain anubis_challenge: {page['body'][:120]!r}",
            )
            check(
                page["version"] == "1.1",
                f"GET / spoke HTTP/{page['version']}, expected 1.1",
            )

            ping = None
            for path in PING_PATHS:
                result = curl_fetch(f"http://127.0.0.1:{port}{path}", tls=False)
                if result["code"] == 200 and b"pong" in result["body"]:
                    ping = result
                    note(f"ping answered on {path}")
                    break
            check(ping is not None, f"no pong from any of {', '.join(PING_PATHS)}")


def case_03_http2_tls_frontend() -> None:
    require_flag("--http2", "--no-http2")
    if not curl_supports_h2():
        raise Skip("this curl has no HTTP/2 support")
    port = free_tcp_udp_port()
    with Gate(port, f"http://127.0.0.1:{free_tcp_port()}", tls_args()) as _gate:
        result = curl_fetch(
            f"https://127.0.0.1:{port}/", extra=["--http2"], timeout=6
        )
        check(
            result["version"] == "2",
            f"curl negotiated HTTP/{result['version']}, expected 2 "
            f"(status line {result['status_line']!r})",
        )
        check(result["code"] == 200, f"GET / over h2 returned {result['code']}")
        check(
            CHALLENGE_MARK in result["body"],
            f"GET / over h2 lacks anubis_challenge: {result['body'][:120]!r}",
        )
        ctype = header(result["headers"], "content-type") or ""
        check("text/html" in ctype, f"challenge Content-Type is {ctype!r}")
        check(
            header(result["headers"], "connection") is None,
            "HTTP/2 response leaked a Connection header "
            f"({header(result['headers'], 'connection')!r})",
        )
        check(
            header(result["headers"], "transfer-encoding") is None,
            "HTTP/2 response leaked Transfer-Encoding "
            f"({header(result['headers'], 'transfer-encoding')!r})",
        )
        length = header(result["headers"], "content-length")
        if length is not None:
            check(
                length.isdigit() and int(length) == len(result["body"]),
                f"Content-Length {length!r} does not match the {len(result['body'])} byte body",
            )
        else:
            check(len(result["body"]) > 0, "response has neither Content-Length nor a body")

        ping = curl_fetch(
            f"https://127.0.0.1:{port}{PING_PATHS[0]}",
            extra=["--http2"],
            timeout=6,
        )
        check(ping["version"] == "2", f"ping over h2 spoke HTTP/{ping['version']}")
        check(ping["code"] == 200, f"ping over h2 returned {ping['code']}")
        check(b"pong" in ping["body"], f"ping over h2 body: {ping['body'][:60]!r}")


def case_04_h2c_prior_knowledge() -> None:
    require_flag("--http2", "--no-http2")
    if "http2-prior-knowledge" not in curl_help_all():
        raise Skip("this curl has no --http2-prior-knowledge option")
    port = free_tcp_udp_port()
    with Gate(port, f"http://127.0.0.1:{free_tcp_port()}") as _gate:
        result = curl_fetch(
            f"http://127.0.0.1:{port}/",
            extra=["--http2-prior-knowledge"],
            timeout=6,
            tls=False,
        )
        check(
            result["version"] == "2",
            f"h2c negotiated HTTP/{result['version']}, expected 2",
        )
        check(result["code"] == 200, f"h2c GET / returned {result['code']}")
        check(
            CHALLENGE_MARK in result["body"],
            f"h2c challenge body: {result['body'][:120]!r}",
        )
        ping = curl_fetch(
            f"http://127.0.0.1:{port}{PING_PATHS[0]}",
            extra=["--http2-prior-knowledge"],
            timeout=6,
            tls=False,
        )
        check(ping["version"] == "2", f"h2c ping spoke HTTP/{ping['version']}")
        check(b"pong" in ping["body"], f"h2c ping body: {ping['body'][:60]!r}")


def case_05_http2_multiplexing() -> None:
    require_flag("--http2", "--no-http2")
    try:
        import httpx
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Skip(f"python 'httpx' package is missing: {exc}") from exc
    try:
        import h2  # noqa: F401 - httpx needs it for http2=True
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Skip(f"python 'h2' package is missing: {exc}") from exc

    warnings.filterwarnings("ignore", message=".*Unverified HTTPS.*")
    port = free_tcp_udp_port()
    with Gate(port, f"http://127.0.0.1:{free_tcp_port()}", tls_args()) as _gate:
        try:
            client = httpx.Client(
                http2=True, verify=False, trust_env=False, timeout=6.0
            )
        except ImportError as exc:
            raise Skip(f"httpx cannot do http2 here: {exc}") from exc
        with client:
            versions = set()
            for i in range(10):
                response = client.get(f"https://127.0.0.1:{port}{PING_PATHS[0]}")
                versions.add(response.http_version)
                check(
                    response.status_code == 200,
                    f"request {i + 1}/10 returned {response.status_code}",
                )
                check(
                    b"pong" in response.content,
                    f"request {i + 1}/10 body: {response.content[:60]!r}",
                )
            check(
                versions == {"HTTP/2"},
                f"ten requests over one client spoke {sorted(versions)}, expected HTTP/2",
            )


def case_06_http3_frontend() -> None:
    require_flag("--http3", "--no-http3")
    H3Client = make_h3_client_class()
    try:
        from aioquic.asyncio import connect
        from aioquic.h3.connection import H3_ALPN
        from aioquic.quic.configuration import QuicConfiguration
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Skip(f"python 'aioquic' package is missing: {exc}") from exc

    port = free_tcp_udp_port()
    with Gate(port, f"http://127.0.0.1:{free_tcp_port()}", tls_args() + ["--http3"]) as _gate:

        async def fetch() -> dict:
            config = QuicConfiguration(
                is_client=True,
                alpn_protocols=H3_ALPN,
                verify_mode=ssl.CERT_NONE,
                server_name="localhost",
            )
            results: dict[str, dict] = {}
            async with connect(
                "127.0.0.1", port, configuration=config, create_protocol=H3Client
            ) as protocol:
                streams = {
                    path: protocol.request(f"127.0.0.1:{port}", path)
                    for path in ("/", PING_PATHS[0])
                }
                for path, stream_id in streams.items():
                    try:
                        await asyncio.wait_for(
                            protocol.responses[stream_id]["done"].wait(), 5.0
                        )
                    except asyncio.TimeoutError as exc:
                        raise Fail(f"no HTTP/3 response for {path}") from exc
                    record = protocol.responses[stream_id]
                    headers = {
                        name.decode(): value.decode()
                        for name, value in record["headers"]
                    }
                    results[path] = {
                        "status": headers.get(":status"),
                        "body": bytes(record["body"]),
                    }
            return results

        try:
            results = run_async(fetch(), timeout=6.0)
        except TimeoutError as exc:
            raise Fail(
                "timed out waiting for the HTTP/3 handshake/response on UDP "
                f"{port} (h3 listener not answering?)"
            ) from exc
        except (ConnectionError, OSError, ssl.SSLError) as exc:
            raise Fail(f"QUIC connection failed on UDP {port}: {exc}") from exc
        root = results["/"]
        check(root["status"] == "200", f"HTTP/3 GET / status {root['status']!r}")
        check(
            CHALLENGE_MARK in root["body"],
            f"HTTP/3 GET / body lacks anubis_challenge: {root['body'][:120]!r}",
        )
        ping = results[PING_PATHS[0]]
        check(ping["status"] == "200", f"HTTP/3 ping status {ping['status']!r}")
        check(b"pong" in ping["body"], f"HTTP/3 ping body: {ping['body'][:60]!r}")


def case_07_alt_svc() -> None:
    require_flag("--http3", "--no-http3")
    port = free_tcp_udp_port()
    with Gate(port, f"http://127.0.0.1:{free_tcp_port()}", tls_args() + ["--http3"]) as _gate:
        result = curl_fetch(
            f"https://127.0.0.1:{port}/", extra=["--http1.1"], timeout=6
        )
        check(result["code"] == 200, f"HTTP/1.1 GET / returned {result['code']}")
        alt_svc = header(result["headers"], "alt-svc")
        check(alt_svc is not None, "no Alt-Svc header on a plain TLS response")
        match = re.search(r'h3(?:-[0-9]+)?\s*=\s*"(?P<authority>[^"]*)"', alt_svc or "")
        check(match is not None, f"Alt-Svc does not name h3: {alt_svc!r}")
        authority = match.group("authority")
        adv_port = re.search(r":(\d+)$", authority or "")
        if adv_port:
            check(
                int(adv_port.group(1)) == port,
                f"Alt-Svc advertises UDP port {adv_port.group(1)}, listener is {port}",
            )
        else:
            note(f"Alt-Svc h3 authority has no explicit port: {authority!r}")


def case_08_http2_upstream() -> None:
    require_flag("--target-http2")
    backend = start_h2_tls_backend()
    try:
        policy = write_policy("allow path prefix /\n")
        port = free_tcp_udp_port()
        with Gate(
            port,
            f"https://127.0.0.1:{backend['port']}",
            ["--target-insecure", "--target-http2", "--policy", policy],
        ) as _gate:
            result = curl_fetch(f"http://127.0.0.1:{port}/", tls=False)
            check(result["code"] == 200, f"proxied GET / returned {result['code']}")
            check(
                H2_OK_BODY.strip() in result["body"],
                f"backend page did not come back: {result['body'][:160]!r}",
            )
            check(
                backend["protocol"] == "h2",
                f"backend saw ALPN {backend['protocol']!r}, expected 'h2'",
            )
            check(
                backend["requests"] >= 1,
                "backend never received the request (0 h2 requests)",
            )
            check(
                not (backend["error"] and backend["requests"] == 0),
                f"h2 backend failed before serving a request: {backend['error']}",
            )
    finally:
        stop_h2_tls_backend(backend)


def case_09_http3_upstream() -> None:
    require_flag("--target-http3")
    state = start_h3_upstream({})
    try:
        policy = write_policy("allow path prefix /\n")
        port = free_tcp_udp_port()
        with Gate(
            port,
            f"https://127.0.0.1:{state['port']}",
            ["--target-insecure", "--target-http3", "--policy", policy],
        ) as _gate:
            result = curl_fetch(f"http://127.0.0.1:{port}/", tls=False)
            check(result["code"] == 200, f"proxied GET / returned {result['code']}")
            check(
                H3_OK_BODY.strip() in result["body"],
                f"h3 backend page did not come back: {result['body'][:160]!r}",
            )
            check(
                state["seen"] == "h3",
                f"backend was not reached over h3 (seen={state['seen']!r})",
            )
    finally:
        stop_h3_upstream(state)


def case_10_negative() -> None:
    require_flag("--http3", "--no-http3")

    # (a) --http3 is off by default: the same UDP port must not speak QUIC.
    port = free_tcp_udp_port()
    with Gate(port, f"http://127.0.0.1:{free_tcp_port()}", tls_args()) as _gate:
        check(
            not h3_reachable(port, timeout=2.0),
            f"HTTP/3 handshake unexpectedly succeeded on UDP {port} with --http3 off",
        )

    # (b) --http3 without --tls-cert/--tls-key must be refused at startup.
    reject_port = free_tcp_udp_port()
    rc, out, timed_out = run_to_exit(
        [
            GATE,
            "--listen",
            f"127.0.0.1:{reject_port}",
            "--target",
            "http://127.0.0.1:9",
            "--http3",
        ],
        timeout=5,
    )
    check(
        not timed_out,
        "--http3 without TLS was accepted (the gate kept running instead of exiting)",
    )
    check(
        rc not in (0, None),
        f"--http3 without TLS exited with rc={rc}, expected a startup error; output: {out[:300]}",
    )
    check(
        re.search(r"(?i)http3|http/3|quic", out) is not None
        and re.search(r"(?i)tls|cert|secure", out) is not None,
        f"startup error does not clearly mention HTTP/3 and TLS/cert: {out.strip()[:400]!r}",
    )


CASES: list[tuple[str, object]] = [
    ("01 setup: cert, dynamic ports, clean start/stop", case_01_setup),
    ("02 HTTP/1.1 regression: challenge and ping", case_02_http1_regression),
    ("03 HTTP/2 over TLS: ALPN h2, challenge, headers, ping", case_03_http2_tls_frontend),
    ("04 h2c prior knowledge on the plain listener", case_04_h2c_prior_knowledge),
    ("05 HTTP/2 multiplexing: 10 pings on one httpx client", case_05_http2_multiplexing),
    ("06 HTTP/3 frontend: QUIC h3 challenge and ping", case_06_http3_frontend),
    ("07 Alt-Svc advertises h3", case_07_alt_svc),
    ("08 HTTP/2 upstream: gate speaks h2 to the backend", case_08_http2_upstream),
    ("09 HTTP/3 upstream: gate speaks h3 to the backend", case_09_http3_upstream),
    ("10 negative: h3 off refuses QUIC, --http3 without TLS rejected", case_10_negative),
]


# ----------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    global WORK, VERBOSE
    filters: list[str] = []
    for arg in argv:
        if arg in ("-v", "--verbose"):
            VERBOSE = True
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        elif arg.startswith("-"):
            print(f"unknown option: {arg}", file=sys.stderr)
            return 2
        else:
            filters.append(arg)

    if shutil.which("curl") is None:
        print("note: curl is not installed; curl-based cases will SKIP")

    WORK = tempfile.mkdtemp(prefix="betteranusbis-http23-")
    passed = failed = skipped = 0
    selected = 0
    started = time.monotonic()
    try:
        if filters and not any(f.lower() in "01 setup" for f in filters):
            # A filtered run still needs the shared certificate/binary check.
            try:
                setup_environment()
            except Skip as exc:
                GLOBAL["gate_usable"] = False
                GLOBAL["gate_reason"] = f"setup skipped: {exc}"
            except Fail as exc:
                GLOBAL["gate_usable"] = False
                GLOBAL["gate_reason"] = f"setup failed: {exc}"
        for name, case in CASES:
            if filters and not any(f.lower() in name.lower() for f in filters):
                continue
            selected += 1
            t0 = time.monotonic()
            try:
                case()  # type: ignore[operator]
            except Skip as exc:
                skipped += 1
                print(f"SKIP {name}: {exc}")
            except Fail as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - a harness bug still fails the case
                failed += 1
                print(f"FAIL {name}: unexpected {type(exc).__name__}: {exc}")
                if VERBOSE:
                    traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {name} ({time.monotonic() - t0:.2f}s)")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)

    elapsed = time.monotonic() - started
    print(
        f"\n{passed} passed, {failed} failed, {skipped} skipped"
        f"{' (of ' + str(selected) + ' selected)' if filters else ''} in {elapsed:.2f}s"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
