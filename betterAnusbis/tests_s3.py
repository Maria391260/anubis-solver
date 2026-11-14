#!/usr/bin/env python3
"""S3 store-backend tests for betterAnusbis.

The gate is expected to grow a ``--store s3api:BUCKET`` backend that keeps the
challenge ledger and honeypot counters in an S3 bucket, using the same logical
keys as the Valkey/bbolt stores (``betteranusbis:challenge:<id>`` and
``betteranusbis:honeypot:<network>``, normalized to S3 keys by replacing ``:``
with ``/``).  Each object carries ``x-amz-meta-x-anubis-expiry-ms`` (expiry
epoch milliseconds); a Get whose metadata says the object has expired must be
reported as not found (and the object deleted best-effort), so a challenge
never resurrects.

This file starts a small threaded in-process S3 mock.  It **independently
verifies AWS Signature V4** with hashlib/hmac: it parses ``Authorization``,
rebuilds the canonical request from the method, canonical URI, canonical query,
the signed headers and their values, and the payload hash, derives the signing
key and compares signatures.  Bad signatures are rejected with an S3 XML 403
``SignatureDoesNotMatch``; missing credentials give ``AccessDenied``.  The mock
stores objects and metadata, serves path-style (and, defensively, virtual-host)
requests, and implements PUT/GET/HEAD/DELETE plus HEAD-bucket and ListObjectsV2
(the shapes a startup probe tends to use).  Every request is logged, which is
how the cases prove what the gate asked for.

Run:

    cd betterAnusbis
    python3 tests_s3.py                  # mock S3 on a dynamic port
    python3 tests_s3.py --selftest       # additionally check the mock verifier
                                         # against AWS's published SigV4 vectors

Required gate environment (the harness sets these):
``AWS_REGION=us-east-1``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
``AWS_ENDPOINT_URL_S3=http://127.0.0.1:PORT``, ``AWS_S3_PATH_STYLE=1``.

If ``--store s3api`` is not advertised by ``./betterAnusbis --help`` (or the
help still says S3 is not implemented) the whole suite prints one SKIP line per
case and exits 0.  Any FAIL exits 1.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import (
    quote_from_bytes,
    unquote_to_bytes,
    urlencode,
    urlsplit,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GATE = os.path.join(HERE, "betterAnusbis")
sys.path.insert(0, REPO)

API = "/.within.website/x/cmd/anubis/api/"
CH_PREFIX = "betteranusbis/challenge/"
HP_PREFIX = "betteranusbis/honeypot/"
COOKIE_TTL_MS = 604800 * 1000
HP_WINDOW = 3600
DIFFICULTY = "5"
BACKEND_PORT = 0

DEFAULT_BUCKET = "anubis-s3-test"
DEFAULT_REGION = "us-east-1"
DEFAULT_ACCESS_KEY = "AKIDBOLTFIXTURES3TEST"
DEFAULT_SECRET_KEY = "boltfixtures/s3/secret+key/0123456789"

PASS = 0
FAIL = 0
SKIP = 0

SOLVER = None  # created lazily so --help-only runs need no native library


# ---------------------------------------------------------------------------
# SigV4 (independent implementation, stdlib only)
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str | bytes) -> bytes:
    if isinstance(msg, str):
        msg = msg.encode()
    return hmac.new(key, msg, hashlib.sha256).digest()


def _norm_header_value(value: str) -> str:
    # AWS: trim leading/trailing space and collapse sequential spaces.
    return " ".join(value.strip().split())


def _canonical_uri(path: str) -> str:
    if not path:
        path = "/"
    # Normalise percent-encoding, then encode all bytes except unreserved + "/".
    return quote_from_bytes(unquote_to_bytes(path), safe="/-_.~")


def _canonical_query(query: str) -> str:
    if not query:
        return ""
    pairs = []
    for part in query.split("&"):
        if not part:
            continue
        name, _, value = part.partition("=")
        pairs.append(
            (
                quote_from_bytes(unquote_to_bytes(name), safe="-_.~"),
                quote_from_bytes(unquote_to_bytes(value), safe="-_.~"),
            )
        )
    pairs.sort()
    return "&".join(f"{name}={value}" for name, value in pairs)


def _parse_authorization(value: str):
    """Return (access_key, scope_parts, signed_headers, signature) or None."""
    if not value:
        return None
    value = value.strip()
    if not value.startswith("AWS4-HMAC-SHA256"):
        return None
    rest = value[len("AWS4-HMAC-SHA256") :].strip()
    fields = {}
    for chunk in rest.split(","):
        chunk = chunk.strip()
        name, sep, val = chunk.partition("=")
        if not sep:
            return None
        fields[name.strip()] = val.strip()
    credential = fields.get("Credential", "")
    signed = fields.get("SignedHeaders", "")
    signature = fields.get("Signature", "")
    if not credential or not signed or not signature:
        return None
    parts = credential.split("/")
    if len(parts) != 5:
        return None
    return parts[0], parts[1:], signed, signature


def verify_sigv4(
    method: str,
    raw_path: str,
    raw_query: str,
    header_items,
    body: bytes,
    *,
    region: str,
    access_key: str,
    secret_key: str,
    service: str = "s3",
):
    """Independently check one AWS SigV4 request.

    Returns ``(True, None, None)`` or ``(False, code, message)`` where ``code``
    is an S3 XML error code.  No boto/awssdk code is involved.
    """
    groups: dict[str, list[str]] = {}
    for name, value in header_items:
        lname = name.strip().lower()
        groups.setdefault(lname, []).append(_norm_header_value(value))

    auth_values = groups.get("authorization", [])
    if not auth_values:
        return False, "AccessDenied", "Anonymous requests are not allowed"
    parsed = _parse_authorization(auth_values[0])
    if parsed is None:
        return False, "AuthorizationHeaderMalformed", "Malformed Authorization header"
    akid, scope, signed_headers_text, signature = parsed
    if akid != access_key:
        return False, "InvalidAccessKeyId", "The access key id does not exist"
    if len(scope) != 4:
        return False, "AuthorizationHeaderMalformed", "Malformed credential scope"
    scope_date, scope_region, scope_service, terminator = scope
    if terminator != "aws4_request":
        return False, "AuthorizationHeaderMalformed", "Credential scope must end in aws4_request"
    if scope_region != region:
        return False, "AuthorizationHeaderMalformed", f"Region should be {region}"
    if scope_service != service:
        return False, "AuthorizationHeaderMalformed", f"Service should be {service}"

    amz_date_values = groups.get("x-amz-date", [])
    if not amz_date_values:
        return False, "AccessDenied", "Missing x-amz-date"
    amz_date = amz_date_values[0]
    if not re.fullmatch(r"\d{8}T\d{6}Z", amz_date):
        return False, "AccessDenied", "Malformed x-amz-date"
    if not amz_date.startswith(scope_date):
        return False, "AuthorizationHeaderMalformed", "Credential date does not match x-amz-date"

    signed_names = signed_headers_text.split(";") if signed_headers_text else []
    if not signed_names or signed_names != sorted(signed_names):
        return False, "AuthorizationHeaderMalformed", "SignedHeaders must be lowercase and sorted"
    if "host" not in signed_names:
        return False, "AuthorizationHeaderMalformed", "host must be signed"
    for name in signed_names:
        if name not in groups:
            return False, "AuthorizationHeaderMalformed", f"Signed header {name!r} is absent"

    payload_header = groups.get("x-amz-content-sha256", [None])[0]
    body_hash = _sha256(body)
    if payload_header:
        if payload_header == "UNSIGNED-PAYLOAD":
            payload_hash = payload_header
        elif payload_header.startswith("STREAMING-"):
            return False, "XAmzContentSHA256Mismatch", "Chunked SigV4 payloads are not supported"
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", payload_header):
                return False, "XAmzContentSHA256Mismatch", "Malformed x-amz-content-sha256"
            if payload_header != body_hash:
                return (
                    False,
                    "XAmzContentSHA256Mismatch",
                    "The provided x-amz-content-sha256 does not match the body",
                )
            payload_hash = payload_header
    else:
        payload_hash = body_hash

    canonical_headers = "".join(f"{name}:{','.join(groups[name])}\n" for name in signed_names)
    canonical_request = "\n".join(
        [
            method.upper(),
            _canonical_uri(raw_path),
            _canonical_query(raw_query),
            canonical_headers,
            signed_headers_text,
            payload_hash,
        ]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            f"{scope_date}/{scope_region}/{scope_service}/aws4_request",
            _sha256(canonical_request.encode()),
        ]
    )
    k_date = _hmac(("AWS4" + secret_key).encode(), scope_date)
    k_region = _hmac(k_date, scope_region)
    k_service = _hmac(k_region, scope_service)
    k_signing = _hmac(k_service, "aws4_request")
    expected = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature.lower()):
        return False, "SignatureDoesNotMatch", "The request signature we calculated does not match"
    return True, None, None


# ---------------------------------------------------------------------------
# XML / HTTP helpers
# ---------------------------------------------------------------------------


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def xml_error(code: str, message: str, key: str | None = None, extra: str = "") -> bytes:
    key_xml = f"<Key>{_xml_escape(key)}</Key>" if key is not None else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Error><Code>{_xml_escape(code)}</Code>"
        f"<Message>{_xml_escape(message)}</Message>{key_xml}{extra}"
        f"<RequestId>{secrets.token_hex(8).upper()}</RequestId>"
        f"<HostId>{base64.b64encode(secrets.token_bytes(12)).decode()}</HostId>"
        "</Error>"
    ).encode()


def _read_request_body(handler) -> bytes:
    encoding = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in encoding:
        chunks = []
        while True:
            line = handler.rfile.readline()
            if not line:
                break
            size_text = line.split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError:
                break
            if size == 0:
                handler.rfile.readline()
                break
            chunks.append(handler.rfile.read(size))
            handler.rfile.read(2)
        return b"".join(chunks)
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        length = 0
    if length <= 0:
        return b""
    if length > 32 * 1024 * 1024:
        return b""
    return handler.rfile.read(length)


# ---------------------------------------------------------------------------
# mock S3
# ---------------------------------------------------------------------------


@dataclass
class StoredObject:
    key: str
    body: bytes
    metadata: dict[str, str] = field(default_factory=dict)  # names without x-amz-meta-
    content_type: str = "application/octet-stream"
    etag: str = ""
    last_modified: float = field(default_factory=time.time)


@dataclass
class S3Request:
    method: str
    path: str
    bucket: str
    key: str
    query: dict[str, list[str]]
    status: int = 0
    code: str = ""
    sig_ok: bool = False
    auth_error: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    headers: list[tuple[str, str]] = field(default_factory=list)
    when: float = field(default_factory=time.time)

    @property
    def body_json(self):
        try:
            return json.loads(self.body.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return None


class MockS3:
    """Threaded path-style S3 mock with independent SigV4 verification."""

    def __init__(
        self,
        bucket: str = DEFAULT_BUCKET,
        *,
        region: str = DEFAULT_REGION,
        access_key: str = DEFAULT_ACCESS_KEY,
        secret_key: str = DEFAULT_SECRET_KEY,
        probe: bool = True,
    ):
        self.bucket = bucket
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.probe = probe
        self._objects: dict[str, StoredObject] = {}
        self._buckets: set[str] = {bucket}
        self._lock = threading.Lock()
        self._requests: list[S3Request] = []
        self._server = None
        self._thread = None
        self.port = 0

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> MockS3:
        mock = self

        class _Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

            def handle_error(self, request, client_address):
                pass

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _do(self):
                mock.handle(self)

            do_GET = _do
            do_HEAD = _do
            do_PUT = _do
            do_DELETE = _do
            do_POST = _do

        self._server = _Server(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="mock-s3", daemon=True
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

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def gate_env(self, *, secret_key: str | None = None, **overrides) -> dict:
        """Environment for a gate that talks to this mock."""
        env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
        # Keep the process away from real credentials/config/metadata.
        home = tempfile.mkdtemp(prefix="s3tests-home-")
        env.update(
            {
                "HOME": home,
                "AWS_CONFIG_FILE": os.path.join(home, "config"),
                "AWS_SHARED_CREDENTIALS_FILE": os.path.join(home, "credentials"),
                "AWS_EC2_METADATA_DISABLED": "true",
                "AWS_REGION": self.region,
                "AWS_DEFAULT_REGION": self.region,
                "AWS_ACCESS_KEY_ID": self.access_key,
                "AWS_SECRET_ACCESS_KEY": secret_key if secret_key is not None else self.secret_key,
                "AWS_ENDPOINT_URL_S3": self.endpoint,
                "AWS_S3_PATH_STYLE": "1",
            }
        )
        env.update({k: str(v) for k, v in overrides.items() if v is not None})
        return env

    # -- observation ------------------------------------------------------
    def mark(self) -> int:
        with self._lock:
            return len(self._requests)

    def requests(
        self, since: int = 0, method: str | None = None, prefix: str | None = None
    ) -> list[S3Request]:
        with self._lock:
            found = list(self._requests[since:])
        if method is not None:
            found = [r for r in found if r.method == method]
        if prefix is not None:
            found = [r for r in found if r.key.startswith(prefix)]
        return found

    def puts(self, prefix: str = "", since: int = 0) -> list[S3Request]:
        return self.requests(since, "PUT", prefix)

    def gets(self, prefix: str = "", since: int = 0) -> list[S3Request]:
        return self.requests(since, "GET", prefix)

    def objects(self, prefix: str = "") -> list[StoredObject]:
        with self._lock:
            return [obj for name, obj in sorted(self._objects.items()) if name.startswith(prefix)]

    def object(self, key: str) -> StoredObject | None:
        with self._lock:
            return self._objects.get(key)

    def put_object(
        self, key: str, body: bytes, metadata: dict[str, str] | None = None
    ) -> StoredObject:
        obj = StoredObject(
            key=key,
            body=bytes(body),
            metadata=dict(metadata or {}),
            etag=hashlib.md5(body).hexdigest(),
        )
        with self._lock:
            self._objects[key] = obj
        return obj

    def set_metadata(self, key: str, metadata: dict[str, str]) -> bool:
        with self._lock:
            obj = self._objects.get(key)
            if obj is None:
                return False
            obj.metadata.update(metadata)
            return True

    def delete_object(self, key: str) -> None:
        with self._lock:
            self._objects.pop(key, None)

    def add_bucket(self, name: str) -> None:
        with self._lock:
            self._buckets.add(name)

    # -- request handling -------------------------------------------------
    def _route(self, path: str, host: str) -> tuple[str, str]:
        host = (host or "").split(":", 1)[0].strip().lower()
        segments = [segment for segment in path.split("/") if segment != ""]
        # Path-style: /bucket/key...
        if segments and segments[0] in self._buckets:
            return segments[0], "/".join(segments[1:])
        # Virtual-host style: bucket.host/key...
        if host and "." in host:
            first, remainder = host.split(".", 1)
            if first in self._buckets and remainder:
                return first, "/".join(segments)
        if segments:
            return segments[0], "/".join(segments[1:])
        return "", ""

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        parts = urlsplit(handler.path)
        raw_path = parts.path
        raw_query = parts.query
        try:
            body = _read_request_body(handler)
        except OSError:
            body = b""
        headers = list(handler.headers.items())
        bucket, key = self._route(raw_path, handler.headers.get("Host", ""))
        query: dict[str, list[str]] = {}
        for part in raw_query.split("&") if raw_query else []:
            if not part:
                continue
            name, _, value = part.partition("=")
            query.setdefault(unquote_to_bytes(name).decode("utf-8", "replace"), []).append(
                unquote_to_bytes(value).decode("utf-8", "replace")
            )

        record = S3Request(
            method=handler.command.upper(),
            path=raw_path,
            bucket=bucket,
            key=key,
            query=query,
            metadata={},
            body=body[: 2 * 1024 * 1024],
            headers=headers,
        )
        status, response_headers, response_body = self._dispatch(handler, record, raw_query)
        record.status = status
        with self._lock:
            self._requests.append(record)

        handler.send_response(status)
        handler.send_header("Server", "AmazonS3")
        handler.send_header("x-amz-request-id", secrets.token_hex(8).upper())
        handler.send_header("x-amz-id-2", base64.b64encode(secrets.token_bytes(12)).decode())
        for name, value in response_headers.items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        if handler.command.upper() != "HEAD" and response_body:
            try:
                handler.wfile.write(response_body)
            except OSError:
                pass

    def _dispatch(self, handler, record: S3Request, raw_query: str):
        method = record.method
        ok, code, message = verify_sigv4(
            method,
            record.path,
            raw_query,
            record.headers,
            b"" if method == "HEAD" else record.body,
            region=self.region,
            access_key=self.access_key,
            secret_key=self.secret_key,
            service="s3",
        )
        record.sig_ok = ok
        if not ok:
            record.code = code
            record.auth_error = message
            status = (
                400
                if code
                in (
                    "AuthorizationHeaderMalformed",
                    "XAmzContentSHA256Mismatch",
                    "InvalidRequest",
                )
                else 403
            )
            return (
                status,
                {"Content-Type": "application/xml"},
                xml_error(code, message, record.key or None),
            )

        if not record.bucket or record.bucket not in self._buckets:
            record.code = "NoSuchBucket"
            return (
                404,
                {"Content-Type": "application/xml"},
                xml_error(
                    "NoSuchBucket",
                    "The specified bucket does not exist",
                    extra=f"<BucketName>{_xml_escape(record.bucket)}</BucketName>",
                ),
            )

        if "location" in record.query:
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<LocationConstraint xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"{self.region}</LocationConstraint>"
            ).encode()
            return 200, {"Content-Type": "application/xml"}, body

        if not record.key:
            return self._bucket_request(record)
        return self._object_request(record)

    def _bucket_request(self, record: S3Request):
        if record.method == "HEAD":
            return 200, {"x-amz-bucket-region": self.region}, b""
        if record.method == "GET":
            prefix = (record.query.get("prefix") or [""])[0]
            objects = self.objects(prefix)
            contents = []
            for obj in objects[:1000]:
                modified = datetime.datetime.fromtimestamp(
                    obj.last_modified, datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                contents.append(
                    "<Contents>"
                    f"<Key>{_xml_escape(obj.key)}</Key>"
                    f"<LastModified>{modified}</LastModified>"
                    f'<ETag>"{obj.etag}"</ETag>'
                    f"<Size>{len(obj.body)}</Size>"
                    "<StorageClass>STANDARD</StorageClass>"
                    "</Contents>"
                )
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<Name>{_xml_escape(record.bucket)}</Name>"
                f"<Prefix>{_xml_escape(prefix)}</Prefix>"
                f"<KeyCount>{len(objects)}</KeyCount>"
                f"<MaxKeys>1000</MaxKeys>"
                "<IsTruncated>false</IsTruncated>" + "".join(contents) + "</ListBucketResult>"
            ).encode()
            return 200, {"Content-Type": "application/xml"}, body
        if record.method == "PUT":
            return 200, {}, b""
        return (
            405,
            {"Content-Type": "application/xml"},
            xml_error("MethodNotAllowed", "Method not allowed"),
        )

    def _object_request(self, record: S3Request):
        key = record.key
        if record.method == "PUT":
            metadata = {}
            for name, value in record.headers:
                lname = name.lower()
                if lname.startswith("x-amz-meta-"):
                    metadata[lname[len("x-amz-meta-") :]] = value
            content_type = "application/octet-stream"
            for name, value in record.headers:
                if name.lower() == "content-type":
                    content_type = value
                    break
            obj = self.put_object(key, record.body, metadata)
            obj.content_type = content_type
            record.metadata = dict(metadata)
            return 200, {"ETag": f'"{obj.etag}"'}, b""
        if record.method in ("GET", "HEAD"):
            obj = self.object(key)
            if obj is None:
                record.code = "NoSuchKey"
                return (
                    404,
                    {"Content-Type": "application/xml"},
                    xml_error("NoSuchKey", "The specified key does not exist.", key),
                )
            headers = {
                "Content-Type": obj.content_type,
                "ETag": f'"{obj.etag}"',
                "Last-Modified": datetime.datetime.fromtimestamp(
                    obj.last_modified, datetime.timezone.utc
                ).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            }
            for name, value in obj.metadata.items():
                headers[f"x-amz-meta-{name}"] = value
            record.metadata = dict(obj.metadata)
            # HEAD carries the object's real Content-Length; the handler
            # suppresses the body write for the HEAD method.
            return 200, headers, obj.body
        if record.method == "DELETE":
            self.delete_object(key)
            record.metadata = {}
            return 204, {}, b""
        return (
            405,
            {"Content-Type": "application/xml"},
            xml_error("MethodNotAllowed", "Method not allowed"),
        )


# ---------------------------------------------------------------------------
# gate / backend / HTTP helpers
# ---------------------------------------------------------------------------


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
    def __init__(self, s3: MockS3, *extra: str, env: dict | None = None):
        self.s3 = s3
        self.store = f"s3api:{s3.bucket}"
        self.extra = list(extra)
        self.env = env if env is not None else s3.gate_env()
        self.port = free_port()
        self.proc = None
        self._lines: list[str] = []
        self._reader = None

    def __enter__(self) -> Gate:
        args = [
            GATE,
            "--listen",
            f"127.0.0.1:{self.port}",
            "--target",
            f"http://127.0.0.1:{BACKEND_PORT}",
            "--difficulty",
            DIFFICULTY,
            "--store",
            self.store,
            *self.extra,
        ]
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env,
        )
        self._reader = threading.Thread(target=self._drain, name="gate-log", daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                time.sleep(0.05)
                raise RuntimeError(
                    f"gate exited {self.proc.returncode} during startup: {self.log()[-800:]}"
                )
            if can_connect(self.port):
                return self
            time.sleep(0.02)
        raise RuntimeError(f"gate did not listen within 12s: {self.log()[-800:]}")

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

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b"<html><body><h1>PAST THE WALL</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def request(port: int, method: str, path: str, headers=None, timeout: float = 10):
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


def run_process(args, env: dict, timeout: float):
    started = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
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


def _json_body(req: S3Request):
    try:
        return json.loads(req.body.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None


def _expiry_ms(req: S3Request) -> int | None:
    value = req.metadata.get("x-anubis-expiry-ms")
    if value is None:
        for name, meta_value in req.metadata.items():
            if name.lower().endswith("x-anubis-expiry-ms"):
                value = meta_value
                break
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_challenge_put(mock: MockS3, challenge, since: int = 0, timeout: float = 4.0):
    def predicate():
        for req in mock.puts(CH_PREFIX, since):
            data = _json_body(req)
            if not isinstance(data, dict):
                continue
            if data.get("randomData") == challenge.random_data:
                return req
            if challenge.id and req.key.rstrip("/").endswith("/" + challenge.id):
                return req
        return None

    return wait_for(predicate, timeout)


def _challenge_key(mock: MockS3, challenge) -> str | None:
    for req in mock.puts(CH_PREFIX):
        data = _json_body(req)
        if isinstance(data, dict) and (
            data.get("randomData") == challenge.random_data
            or (challenge.id and req.key.rstrip("/").endswith("/" + challenge.id))
        ):
            return req.key
    for obj in mock.objects(CH_PREFIX):
        data = obj.body
        if challenge.id and obj.key.rstrip("/").endswith("/" + challenge.id):
            return obj.key
        try:
            decoded = json.loads(data.decode("utf-8", "replace"))
        except ValueError:
            continue
        if isinstance(decoded, dict) and decoded.get("randomData") == challenge.random_data:
            return obj.key
    return None


def _spent_puts(mock: MockS3, key: str, since: int = 0):
    return [
        req
        for req in mock.puts(key, since)
        if isinstance(_json_body(req), dict) and _json_body(req).get("spent") is True
    ]


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def case_startup(_provider):
    with MockS3() as mock:
        with Gate(mock):
            requests = wait_for(lambda: mock.requests(), 8)
            require(requests, "gate never contacted the S3 endpoint during startup")
            signed = [req for req in requests if req.sig_ok]
            require(
                signed,
                "startup request(s) did not carry a valid SigV4 signature: "
                + ", ".join(
                    f"{r.method} {r.path} -> {r.status} {r.auth_error}" for r in requests[-3:]
                ),
            )
            first = signed[0]
            return (
                f"{len(requests)} startup request(s); first {first.method} "
                f"{first.path or '/'} -> {first.status}; valid signature"
            )


def case_issue(_provider):
    with MockS3() as mock:
        want_expiry_min = int(time.time() * 1000)
        with Gate(mock) as gate:
            challenge, _ = get_challenge(gate.port)
            _ = solve(challenge)
            put = _find_challenge_put(mock, challenge, timeout=5)
            require(
                put is not None,
                "no PUT of betteranusbis/challenge/... with the challenge JSON "
                f"(randomData {challenge.random_data[:16]}...)",
            )
            require(put.status == 200, f"challenge PUT returned {put.status}")
            data = _json_body(put)
            require(isinstance(data, dict), f"challenge object is not JSON: {put.body[:120]!r}")
            for field in ("randomData", "algorithm", "difficulty"):
                require(field in data, f"challenge JSON lacks {field!r}: {data}")
            require(
                data.get("randomData") == challenge.random_data,
                "stored challenge randomData does not match the issued page",
            )
            require(":" not in put.key, f"challenge key was not normalized: {put.key!r}")
            expiry = _expiry_ms(put)
            require(expiry is not None, "challenge PUT lacks x-amz-meta-x-anubis-expiry-ms")
            require(
                want_expiry_min <= expiry <= want_expiry_min + COOKIE_TTL_MS + 120000,
                f"expiry metadata {expiry} is not now+TTL (now {want_expiry_min})",
            )
            ttl = (expiry - want_expiry_min) / 1000.0
            return f"{put.key}; spent={data.get('spent')}; expiry {expiry} (~{ttl:.0f}s)"


def case_restart(_provider):
    with MockS3() as mock:
        with Gate(mock) as first:
            challenge, _ = get_challenge(first.port)
            solution = solve(challenge)
            key = None
            for _ in range(100):
                key = _challenge_key(mock, challenge)
                if key:
                    break
                time.sleep(0.02)
            require(key is not None, "challenge was never PUT before the restart")

        mark = mock.mark()
        with Gate(mock) as second:
            wait_for(lambda: mock.requests(since=mark), 5)
            status, _, body = pass_challenge(second.port, challenge, solution)
            require(
                status == 302,
                f"pass-challenge after restart returned {status}: {body[:180]!r}",
            )
            gets = wait_for(
                lambda: [
                    r
                    for r in mock.requests(since=mark)
                    if r.method in ("GET", "HEAD") and r.key == key
                ],
                3,
            )
            require(gets, f"restarted gate never fetched {key} from S3")
            return f"{key}: challenge issued before restart verified on a new process (302)"


def case_sharing(_provider):
    with MockS3() as mock:
        with Gate(mock) as gate_a, Gate(mock) as gate_b:
            wait_for(lambda: len(mock.requests()) >= 2, 6)
            challenge, _ = get_challenge(gate_a.port)
            solution = solve(challenge)
            key = wait_for(lambda: _challenge_key(mock, challenge), 3)
            require(key is not None, "gate A did not PUT its challenge to S3")
            mark = mock.mark()
            status, _, body = pass_challenge(gate_b.port, challenge, solution)
            require(
                status == 302,
                f"gate B rejected gate A's challenge with {status}: {body[:180]!r}",
            )
            gets = wait_for(
                lambda: [
                    r
                    for r in mock.requests(since=mark)
                    if r.method in ("GET", "HEAD") and r.key == key
                ],
                3,
            )
            require(gets, f"gate B never fetched gate A's object {key}")
            return f"B fetched {key} and accepted A's challenge (302)"


def case_replay(_provider):
    with MockS3() as mock:
        with Gate(mock) as gate:
            challenge, _ = get_challenge(gate.port)
            solution = solve(challenge)
            key = wait_for(lambda: _challenge_key(mock, challenge), 3)
            require(key is not None, "no challenge object in S3 before the replay test")

            first, _, body1 = pass_challenge(gate.port, challenge, solution)
            require(first == 302, f"the first submission returned {first}: {body1[:180]!r}")
            mark = mock.mark()
            second, headers2, body2 = pass_challenge(gate.port, challenge, solution)
            require(
                second == 403,
                f"the replay returned {second} instead of 403 "
                f"(Location={header(headers2, 'Location')!r}): {body2[:180]!r}",
            )
            spent = wait_for(lambda: _spent_puts(mock, key), 3)
            require(
                spent,
                f"no spent=true PUT for {key} after the replay; got "
                + ", ".join(f"{r.method} {r.path} {r.status}" for r in mock.requests(mark)),
            )
            return f"first 302, replay 403, {key} rewritten with spent=true"


def case_honeypot(_provider):
    with MockS3() as mock:
        with Gate(mock) as gate_a, Gate(mock) as gate_b:
            wait_for(lambda: len(mock.requests()) >= 2, 6)
            status, _, body = request(gate_a.port, "GET", "/")
            require(status == 200, f"challenge page returned {status}")
            match = re.search(rb"honeypot/([^/\"'\s<>]+)/init", body)
            require(match is not None, "challenge page has no honeypot trap link")
            trap = match.group(1).decode("utf-8", "replace")

            status, _, _ = request(gate_a.port, "GET", API + f"honeypot/{trap}/init", timeout=15)
            require(status in (200, 302, 303), f"honeypot init returned {status}")

            put = wait_for(lambda: mock.puts(HP_PREFIX), 4)
            require(put, "no betteranusbis/honeypot/... object was PUT")
            put = put[-1]
            data = _json_body(put)
            require(isinstance(data, dict), f"honeypot object is not JSON: {put.body[:120]!r}")
            require(int(data.get("count", 0)) >= 1, f"honeypot count is not >= 1: {data}")
            require(int(data.get("last", 0)) > 0, f"honeypot last is missing: {data}")
            expiry = _expiry_ms(put)
            require(expiry is not None, "honeypot PUT lacks x-amz-meta-x-anubis-expiry-ms")
            require(
                expiry >= int(time.time() * 1000),
                f"honeypot expiry metadata is already in the past: {expiry}",
            )
            key = put.key

            mark = mock.mark()
            request(gate_b.port, "GET", "/")
            gets = wait_for(
                lambda: [
                    r
                    for r in mock.requests(since=mark)
                    if r.method in ("GET", "HEAD") and r.key == key
                ],
                4,
            )
            require(gets, f"gate B never fetched the honeypot object {key}")
            return f"{key}; count={data.get('count')}; B fetched it"


def case_wrong_secret(_provider):
    with MockS3() as mock:
        env = mock.gate_env(secret_key="definitely-the-wrong-secret")
        args = [
            GATE,
            "--listen",
            f"127.0.0.1:{free_port()}",
            "--target",
            f"http://127.0.0.1:{BACKEND_PORT}",
            "--difficulty",
            DIFFICULTY,
            "--store",
            f"s3api:{mock.bucket}",
        ]
        returncode, output, elapsed = run_process(args, env, timeout=10)
        require(
            returncode is not None,
            "gate was still running 10s after a signature failure (no startup probe?)",
        )
        require(returncode != 0, f"a wrong secret did not fail startup (exit {returncode})")
        require(
            elapsed <= 8.0,
            f"rejecting a wrong secret took {elapsed:.1f}s (want <= ~8s)",
        )
        require(
            re.search(r"signature|sigv4|auth|credential|denied|403|s3", output, re.I),
            f"error does not mention signature/auth/S3: {output.strip()[-240:]!r}",
        )
        rejected = [r for r in mock.requests() if not r.sig_ok]
        detail = f"exit {returncode} in {elapsed:.2f}s"
        if rejected:
            detail += f"; mock rejected with {rejected[-1].code or '403'} SignatureDoesNotMatch"
        return f"{detail}: {output.strip().splitlines()[-1][:120]}"


def case_unreachable(_provider):
    port = free_port()  # nothing is listening there
    env = MockS3().gate_env(AWS_ENDPOINT_URL_S3=f"http://127.0.0.1:{port}")
    args = [
        GATE,
        "--listen",
        f"127.0.0.1:{free_port()}",
        "--target",
        f"http://127.0.0.1:{BACKEND_PORT}",
        "--difficulty",
        DIFFICULTY,
        "--store",
        f"s3api:{DEFAULT_BUCKET}",
    ]
    returncode, output, elapsed = run_process(args, env, timeout=8)
    require(
        returncode is not None,
        "gate was still running 8s after an unreachable S3 endpoint",
    )
    require(
        returncode != 0,
        f"an unreachable S3 endpoint did not fail startup (exit {returncode})",
    )
    require(
        elapsed <= 6.0,
        f"rejecting an unreachable endpoint took {elapsed:.1f}s (want <= ~5s)",
    )
    require(
        re.search(r"s3|connect|refused|unreachable|endpoint|network|request", output, re.I),
        f"error does not mention S3/connect: {output.strip()[-240:]!r}",
    )
    return f"exit {returncode} in {elapsed:.2f}s: {output.strip().splitlines()[-1][:120]}"


def case_missing_config(_provider):
    details = []
    with MockS3() as mock:
        full_env = mock.gate_env()
        no_region = mock.gate_env()
        for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            no_region.pop(key, None)
        no_creds = mock.gate_env()
        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            no_creds.pop(key, None)

        variants = [
            ("empty bucket", ["--store", "s3api:"], full_env, r"bucket|s3api|store|invalid|empty"),
            ("missing region", [], no_region, r"region"),
            ("missing credentials", [], no_creds, r"credential|access|secret|key|auth|environment"),
        ]
        for name, extra, env, pattern in variants:
            args = [
                GATE,
                "--listen",
                f"127.0.0.1:{free_port()}",
                "--target",
                f"http://127.0.0.1:{BACKEND_PORT}",
                "--difficulty",
                DIFFICULTY,
                "--store",
                f"s3api:{mock.bucket}",
                *extra,
            ]
            returncode, output, elapsed = run_process(args, env, timeout=6)
            require(returncode is not None, f"{name}: gate hung instead of rejecting the config")
            require(returncode != 0, f"{name}: accepted (exit {returncode})")
            require(output.strip(), f"{name}: exited silently")
            require(
                re.search(pattern, output, re.I),
                f"{name}: message is not clear: {output.strip()[-220:]!r}",
            )
            details.append(f"{name} -> exit {returncode} in {elapsed:.2f}s")
    return "; ".join(details)


def case_expired(_provider):
    with MockS3() as mock:
        with Gate(mock) as first:
            challenge, _ = get_challenge(first.port)
            solution = solve(challenge)
            key = wait_for(lambda: _challenge_key(mock, challenge), 3)
            require(key is not None, "challenge was never PUT, cannot test expiry")
            require(
                mock.set_metadata(key, {"x-anubis-expiry-ms": str(int(time.time() * 1000) - 5000)}),
                f"could not rewrite metadata on {key}",
            )

        mark = mock.mark()
        with Gate(mock) as second:
            wait_for(lambda: mock.requests(since=mark), 5)
            status, _, body = pass_challenge(second.port, challenge, solution)
            require(
                status != 302,
                f"an object whose metadata says it expired was still accepted (302): {body[:160]!r}",
            )
            require(
                400 <= status < 600,
                f"expired-challenge submission returned unexpected {status}: {body[:160]!r}",
            )
            gets = wait_for(
                lambda: [
                    r
                    for r in mock.requests(since=mark)
                    if r.method in ("GET", "HEAD") and r.key == key
                ],
                3,
            )
            require(gets, f"restarted gate never fetched the expired object {key}")
            spent = _spent_puts(mock, key, since=mark)
            require(
                not spent,
                f"expired object {key} was resurrected and marked spent instead of not-found",
            )
            deleted = any(r.method == "DELETE" and r.key == key for r in mock.requests(since=mark))
            detail = f"expired metadata -> {status} (not 302, no spent write)"
            if deleted:
                detail += "; object deleted best-effort"
            return detail


CASES = [
    ("startup probe reaches S3 with a valid SigV4 signature", case_startup),
    ("issuing a challenge PUTs JSON with expiry metadata", case_issue),
    ("a challenge survives a gate restart (remote GET)", case_restart),
    ("two gates share one bucket (B gets A's object)", case_sharing),
    ("replaying a solved challenge is rejected with a spent PUT", case_replay),
    ("honeypot object carries expiry metadata and B reads it", case_honeypot),
    ("a wrong secret fails startup with a signature error", case_wrong_secret),
    ("an unreachable S3 endpoint fails startup quickly", case_unreachable),
    ("missing bucket / region / credentials are rejected", case_missing_config),
    ("expired metadata is treated as not found (no resurrection)", case_expired),
]


# ---------------------------------------------------------------------------
# verifier self-test against AWS's published SigV4 vectors
# ---------------------------------------------------------------------------


def _parse_req_file(path: str):
    raw = open(path, "rb").read()
    if b"\r\n\r\n" in raw:
        head, _, body = raw.partition(b"\r\n\r\n")
    elif b"\n\n" in raw:
        head, _, body = raw.partition(b"\n\n")
    else:
        head, body = raw, b""
    lines = head.replace(b"\r\n", b"\n").split(b"\n")
    request_line = lines[0].decode("latin-1")
    method, target, _ = request_line.split(" ", 2)
    target_parts = urlsplit(target)
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        text = line.decode("latin-1")
        if text[:1] in (" ", "\t") and headers:
            name, value = headers[-1]
            headers[-1] = (name, value + " " + text.strip())
        else:
            name, sep, value = text.partition(":")
            if sep:
                headers.append((name.strip(), value))
    return method, target_parts.path, target_parts.query, headers, body


def run_selftest(vectors_dir: str | None = None) -> int:
    candidates = [
        vectors_dir,
        "/tmp/sigv4suite/botocore-develop/tests/unit/auth/aws4_testsuite",
    ]
    root = next((path for path in candidates if path and os.path.isdir(path)), None)
    if root is None:
        print("SigV4 self-test: SKIP (no aws4_testsuite directory found)")
        return 0
    pattern = os.path.join(root, "*", "*.req")
    import glob

    files = sorted(glob.glob(pattern))
    passed = failed = 0
    for path in files:
        method, uri, query, headers, body = _parse_req_file(path)
        # The .req files are byte streams read as latin-1; recover UTF-8 paths.
        uri = uri.encode("latin-1", "surrogateescape").decode("utf-8", "surrogateescape")
        query = query.encode("latin-1", "surrogateescape").decode("utf-8", "surrogateescape")
        authz_path = path[: -len(".req")] + ".authz"
        try:
            with open(authz_path, encoding="latin-1") as stream:
                authz = stream.read().strip()
        except OSError:
            authz = ""
        if authz and not any(n.lower() == "authorization" for n, _ in headers):
            headers.append(("Authorization", authz))
        # Some vectors (session token) carry headers only in .creq; add any
        # signed header that .req omits.
        creq_path = path[: -len(".req")] + ".creq"
        try:
            creq_text = open(creq_path, encoding="latin-1").read()
        except OSError:
            creq_text = ""
        if creq_text and authz:
            present = {n.lower() for n, _ in headers}
            creq_lines = creq_text.replace("\r\n", "\n").split("\n")
            for line in creq_lines:
                name, sep, value = line.partition(":")
                if sep and name == name.lower() and name.strip() and name not in present:
                    headers.append((name, value))
                    present.add(name)
        auth = next((v for n, v in headers if n.lower() == "authorization"), None)
        if not auth:
            failed += 1
            print(f"FAIL sigv4 {os.path.basename(path)}: no Authorization header")
            continue
        ok, code, message = verify_sigv4(
            method,
            uri,
            query,
            headers,
            body,
            region="us-east-1",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            service="service",
        )
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"FAIL sigv4 {os.path.basename(path)}: {code}: {message}")
    print(f"SigV4 self-test: {passed} passed, {failed} failed ({len(files)} vectors)")
    return 1 if failed else 0


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


def s3_advertised() -> bool:
    text = help_text()
    if not text:
        return False
    flat = " ".join(text.split())
    if re.search(r"s3api[^.]{0,80}not implemented", flat, re.I):
        return False
    if re.search(r"S3 is not implemented", flat, re.I):
        return False
    if re.search(r"--store[^.]{0,200}s3api", flat, re.I):
        return True
    if "s3api:" in text or "AWS_ENDPOINT_URL_S3" in text:
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

    parser = argparse.ArgumentParser(description="S3 store-backend tests for betterAnusbis")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="also check the mock's independent SigV4 verifier against AWS vectors",
    )
    parser.add_argument(
        "--vectors",
        metavar="DIR",
        help="aws4_testsuite directory for --selftest",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        rc = run_selftest(args.vectors)
        if rc:
            return rc

    if not os.path.exists(GATE):
        print(f"FAIL 0. {GATE} is missing; run make first")
        return 1

    if not s3_advertised():
        reason = "--store s3api is not advertised in ./betterAnusbis --help"
        print(f"S3 store backend not implemented yet: {reason}\n")
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

    print("store: in-process threaded S3 mock (independent SigV4 verification)\n")
    try:
        for number, (title, function) in enumerate(CASES, 1):
            run_case(number, title, function, None)
    finally:
        backend.shutdown()
        backend.server_close()
        backend_thread.join(2)

    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
