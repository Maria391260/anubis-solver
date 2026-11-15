#!/usr/bin/env bash
#
# impersonate_test.sh — prove that the C fetch layer presents a browser TLS
# fingerprint when libcurl-impersonate is installed, and falls back cleanly when
# it is not asked to.
#
# The evidence is local: an `openssl s_server` we start ourselves, its -msg dump
# of our ClientHello, and what the services report as the transport. No traffic
# leaves the machine.
#
# The decisive assertion is GREASE. Chrome inserts the GREASE values (0x0a0a,
# 0x1a1a, ...) into the cipher list, supported groups and signature algorithms;
# stock OpenSSL never sends them. A ClientHello containing two or more distinct
# GREASE values is not OpenSSL's, whatever else it looks like.
#
# Exit codes: 0 everything passed (SKIP counts as pass), 1 otherwise.
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/anubis_impersonate_test.XXXXXX") || {
    echo "FAIL cannot create temp directory" >&2
    exit 1
}
PORT_TLS=8446
PORT_API=8096
PASS=0
FAIL=0
SKIP=0
TLS_PID=""
API_PID=""

cleanup() {
    trap - EXIT INT TERM
    for p in "$API_PID" "$TLS_PID"; do
        [ -n "$p" ] && kill "$p" 2>/dev/null
    done
    for p in "$API_PID" "$TLS_PID"; do
        [ -n "$p" ] && wait "$p" 2>/dev/null
    done
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

ok() { PASS=$((PASS + 1)); printf 'PASS %s\n' "$*"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$*"; }
ck() { if [ "$2" = 1 ]; then ok "$1"; else bad "$1"; fi; }
has() { if grep -q -- "$1" "$2" 2>/dev/null; then echo 1; else echo 0; fi; }

# --------------------------------------------------------------- library
LIB=$(ls "$ROOT"/3rd_party/curl-impersonate/lib/*.so* 2>/dev/null | head -1)
[ -z "$LIB" ] && LIB=$(ls /usr/lib/*/libcurl-impersonate-chrome.so* 2>/dev/null | head -1)
if [ -z "$LIB" ]; then
    echo "SKIP no libcurl-impersonate found (make impersonate-lib CURL_IMPERSONATE_LIB=...)"
    exit 0
fi
echo "using $LIB"

if [ ! -f "$ROOT/libanubis_solver.so" ]; then
    (cd "$ROOT" && make -j8 libanubis_solver.so) >"$TMP/build.log" 2>&1 ||
        { cat "$TMP/build.log" >&2; echo "FAIL build .so"; exit 1; }
fi
(cd "$ROOT" && make anubis_mcp anubis_api) >"$TMP/build2.log" 2>&1 || {
    cat "$TMP/build2.log" >&2
    echo "FAIL build services"
    exit 1
}

# --------------------------------------------------------------- TLS server
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=localhost \
    -keyout "$TMP/key.pem" -out "$TMP/cert.pem" >/dev/null 2>&1 ||
    { echo "FAIL cannot make a test certificate"; exit 1; }
openssl s_server -accept "$PORT_TLS" -cert "$TMP/cert.pem" -key "$TMP/key.pem" \
    -www -msg >"$TMP/hello.log" 2>&1 &
TLS_PID=$!
sleep 1
if ! kill -0 "$TLS_PID" 2>/dev/null; then
    echo "FAIL s_server did not start (port $PORT_TLS in use?)"
    exit 1
fi

# One MCP fetch: mcp_fetch <http url> <profile json> <outfile>
mcp_fetch() {
    local url=$1 target=$2 out=$3
    local req
    req=$(printf '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"anubis_fetch","arguments":{"url":"%s","max_chars":200,"impersonate":"%s"}}}' "$url" "$target")
    printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
                  "$req" |
        ANUBIS_INSECURE=1 timeout 25 "$ROOT/anubis_mcp" 2>/dev/null >"$out.raw"
    python3 - "$out.raw" "$out" <<'PY'
import json, sys
try:
    lines = [l for l in open(sys.argv[1]) if l.strip()]
    reply = json.loads(lines[-1])
    content = reply["result"]["content"]
    text = content[0]["text"] if content else ""
except Exception as exc:  # noqa: BLE001 - a broken reply is a test failure
    text = "MCP reply unreadable: %s" % exc
open(sys.argv[2], "w").write(text)
PY
}

# Count GREASE cipher suites in the first ClientHello of an s_server -msg dump.
# GREASE is 0x?a?a: both bytes equal and the low nibble 0xa. Chrome puts one in
# the cipher list; OpenSSL never does. Parsing the list itself (instead of
# scanning the whole hexdump) keeps random bytes from producing false positives.
grease_count() { # grease_count <slice file>
    python3 - "$1" <<'PY'
import re, sys

data = open(sys.argv[1], "rb").read().decode("latin1")
hellos = data.split("ClientHello")[1:]
if not hellos:
    print(0)
    raise SystemExit
b = bytes(int(x, 16) for x in re.findall(r"\b[0-9a-f]{2}\b", hellos[0]))

# The record header is printed on its own line, before the "ClientHello" label,
# so the chunk usually starts at the handshake header (0x01). Accept both forms.
start = -1
if b and b[0] == 0x01:                  # handshake header right after the label
    start = 0
else:
    for k in range(len(b) - 5):         # record header included in the chunk
        if b[k] == 0x16 and b[k + 1] == 0x03:
            start = k + 5
            break
if start < 0:
    print(0)
    raise SystemExit
i = start
if i + 39 >= len(b):
    print(0)
    raise SystemExit
i += 4 + 2 + 32                         # handshake header, legacy version, random
i += 1 + b[i]                           # session id
if i + 1 >= len(b):
    print(0)
    raise SystemExit
cslen = (b[i] << 8) | b[i + 1]
i += 2
ciphers = [(b[i + k] << 8) | b[i + k + 1]
           for k in range(0, max(cslen - 1, 0), 2) if i + k + 1 < len(b)]
print(sum(1 for c in ciphers if (c >> 8) == (c & 0xFF) and (c & 0x0F) == 0x0A))
PY
}

slice_since() { # slice_since <log> <byte offset> <dest>
    tail -c +"$(( $2 + 1 ))" "$1" >"$3"
}

# ---- 1. the version tool reports the library -------------------------------
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"anubis_version","arguments":{}}}' |
    timeout 15 "$ROOT/anubis_mcp" 2>/dev/null | tail -1 >"$TMP/version.raw"
python3 - "$TMP/version.raw" >"$TMP/version.json" <<'PY'
import json, sys
try:
    reply = json.loads([l for l in open(sys.argv[1]) if l.strip()][-1])
    sys.stdout.write(reply["result"]["content"][0]["text"])
except Exception:
    sys.stdout.write("{}")
PY
ck "anubis_version reports impersonation available" \
   "$(has '"impersonate_available":true' "$TMP/version.json")"
ck "anubis_version names a Chrome profile" "$(has '"impersonate":"chrome' "$TMP/version.json")"

# ---- 2. with impersonation: TLS works, transport says so, GREASE present ---
OFF1=$(wc -c <"$TMP/hello.log")
mcp_fetch "https://127.0.0.1:$PORT_TLS/" chrome "$TMP/chrome.txt"
slice_since "$TMP/hello.log" "$OFF1" "$TMP/chrome.hello"
ck "impersonated fetch of a local TLS server returns 200" \
   "$(has 'status: 200' "$TMP/chrome.txt")"
ck "transport is curl-impersonate/chrome..." \
   "$(has 'transport: curl-impersonate/chrome' "$TMP/chrome.txt")"
GREASE=$(grease_count "$TMP/chrome.hello")
ck "the ClientHello carries a GREASE cipher suite (got $GREASE)" \
   "$([ "${GREASE:-0}" -ge 1 ] && echo 1 || echo 0)"

# ---- 3. with impersonation off: OpenSSL, and no GREASE --------------------
OFF2=$(wc -c <"$TMP/hello.log")
mcp_fetch "https://127.0.0.1:$PORT_TLS/" off "$TMP/openssl.txt"
slice_since "$TMP/hello.log" "$OFF2" "$TMP/openssl.hello"
ck "openssl fallback fetch of the same server returns 200" \
   "$(has 'status: 200' "$TMP/openssl.txt")"
ck "transport is openssl when impersonation is off" \
   "$(has 'transport: openssl' "$TMP/openssl.txt")"
GREASE_OFF=$(grease_count "$TMP/openssl.hello")
ck "the OpenSSL ClientHello carries no GREASE (got $GREASE_OFF)" \
   "$([ "${GREASE_OFF:-1}" -eq 0 ] && echo 1 || echo 0)"

# ---- 4. a profile the library does not have is an error, not silent fallback
mcp_fetch "https://127.0.0.1:$PORT_TLS/" firefox "$TMP/firefox.txt"
ck "unknown profile fails loudly" \
   "$(has 'impersonation requested but unavailable' "$TMP/firefox.txt")"

# ---- 5. the HTTP API takes the profile per request -----------------------
ANUBIS_INSECURE=1 "$ROOT/anubis_api" --listen "127.0.0.1:$PORT_API" >"$TMP/api.log" 2>&1 &
API_PID=$!
sleep 1
ck "API /v1/version reports the active profile" \
   "$(curl -s "localhost:$PORT_API/v1/version" | grep -c '"impersonate":"chrome' |
       sed 's/^[1-9][0-9]*$/1/')"
BODY=$(curl -s "localhost:$PORT_API/v1/fetch" -d \
    "{\"url\":\"https://127.0.0.1:$PORT_TLS/\",\"impersonate\":\"chrome110\"}" \
    -H 'Content-Type: application/json')
ck "API /v1/fetch with chrome110 reports that transport" \
   "$(printf '%s' "$BODY" | grep -c '"transport":"curl-impersonate/chrome110"' |
       sed 's/^[1-9][0-9]*$/1/')"
BODY=$(curl -s "localhost:$PORT_API/v1/fetch" -d \
    "{\"url\":\"https://127.0.0.1:$PORT_TLS/\",\"impersonate\":\"off\"}" \
    -H 'Content-Type: application/json')
ck "API /v1/fetch with off reports openssl" \
   "$(printf '%s' "$BODY" | grep -c '"transport":"openssl"' |
       sed 's/^[1-9][0-9]*$/1/')"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
