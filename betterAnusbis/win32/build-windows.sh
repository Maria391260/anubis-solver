#!/bin/sh
# build-windows.sh — cross-build the Windows release of betterAnusbis.
#
# The gate is POSIX C with three Windows-shaped problems: winsock instead of
# BSD sockets, no POSIX regex, and no OpenSSL for mingw on the average Linux
# box. This script solves all three without Windows:
#
#   * win32-compat.h maps the sockets/threads/signals, win32/include/ maps the
#     headers Windows does not ship;
#   * PCRE2 is cross-built and provides the POSIX regcomp()/regexec() wrapper
#     the policy code uses;
#   * OpenSSL is cross-built (winstore enabled by configuration) for TLS, the
#     JWTs and the Ed25519 signing.
#
# Output, in $OUT (default build/windows):
#
#   betterAnusbis.exe                 the gate, statically linked except for
#   anubis_solver.dll                 the engine, which is the pair
#   betterAnusbis-<ver>-x86_64-windows.zip
#   betterAnusbis-<ver>-x86_64-windows.msi
#
# The two binaries must stay together: betterAnusbis.exe imports the DLL by
# name, which is what the SxS-free Windows loader wants.
#
# Requirements: x86_64-w64-mingw32-gcc-posix (gcc-mingw-w64-x86-64-posix),
# make, curl or wget. wine is optional, for --verify/--smoke. wixl (msitools)
# is optional, for the MSI.
#
# Usage:
#   ./build-windows.sh                 # build, verify with wine if present
#   ./build-windows.sh --deps          # also fetch and build OpenSSL + PCRE2
#   ./build-windows.sh --smoke         # also run a challenge/pass round trip
#   ./build-windows.sh --no-msi        # zip only
#   DEPS=/path ./build-windows.sh      # reuse a prepared deps directory
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
GATE=$ROOT/betterAnusbis
OUT=${OUT:-$ROOT/build/windows}
DEPS=${DEPS:-$OUT/deps}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
CROSS=${CROSS:-x86_64-w64-mingw32-}
CC=${CC:-${CROSS}gcc-posix}
OPENSSL_VER=${OPENSSL_VER:-3.5.3}
PCRE2_VER=${PCRE2_VER:-10.45}

BUILD_DEPS=0
MAKE_MSI=1
VERIFY=1
SMOKE=0
for arg in "$@"; do
    case $arg in
        --deps) BUILD_DEPS=1 ;;
        --no-msi) MAKE_MSI=0 ;;
        --no-verify) VERIFY=0 ;;
        --smoke) SMOKE=1 ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "build-windows.sh: unknown argument $arg" >&2; exit 2 ;;
    esac
done

command -v "$CC" >/dev/null 2>&1 || {
    echo "build-windows.sh: $CC not found; install gcc-mingw-w64-x86-64-posix" >&2
    exit 1
}
command -v make >/dev/null 2>&1 || {
    echo "build-windows.sh: make not found" >&2
    exit 1
}

VERSION=$(sed -n 's/^#define VERSION "\([^"]*\)".*/\1/p' "$GATE/betterAnusbis.c" | head -1)
[ -n "$VERSION" ] || {
    echo "build-windows.sh: cannot find VERSION in betterAnusbis.c" >&2
    exit 1
}

fetch() {
    url=$1
    dest=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest" "$url"
    else
        echo "build-windows.sh: need curl or wget to fetch $url" >&2
        exit 1
    fi
}

build_deps() {
    mkdir -p "$DEPS"
    if [ ! -f "$DEPS/openssl/lib64/libssl.a" ]; then
        echo "==> OpenSSL $OPENSSL_VER for mingw (win64, static)"
        tmp=$DEPS/openssl-src
        rm -rf "$tmp"
        fetch "https://github.com/openssl/openssl/releases/download/openssl-$OPENSSL_VER/openssl-$OPENSSL_VER.tar.gz" \
              "$DEPS/openssl.tar.gz"
        mkdir -p "$tmp"
        tar xzf "$DEPS/openssl.tar.gz" -C "$tmp" --strip-components=1
        (
            cd "$tmp"
            ./Configure mingw64 --cross-compile-prefix="$CROSS" \
                no-shared no-tests no-docs --prefix="$DEPS/openssl" --libdir=lib64
            make -j"$JOBS"
            make install_sw
        )
        rm -rf "$tmp" "$DEPS/openssl.tar.gz"
    fi
    if [ ! -f "$DEPS/pcre2/lib/libpcre2-8.a" ]; then
        echo "==> PCRE2 $PCRE2_VER for mingw (static, POSIX wrapper)"
        tmp=$DEPS/pcre2-src
        rm -rf "$tmp"
        fetch "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-$PCRE2_VER/pcre2-$PCRE2_VER.tar.gz" \
              "$DEPS/pcre2.tar.gz"
        mkdir -p "$tmp"
        tar xzf "$DEPS/pcre2.tar.gz" -C "$tmp" --strip-components=1
        (
            cd "$tmp"
            ./configure --host="${CROSS%-}" --prefix="$DEPS/pcre2" \
                --disable-shared --enable-static --disable-pcre2grep \
                --disable-pcre2test --disable-jit
            make -j"$JOBS"
            make install
        )
        rm -rf "$tmp" "$DEPS/pcre2.tar.gz"
    fi
}

if [ "$BUILD_DEPS" = 1 ]; then
    build_deps
fi

OPENSSL_DIR=${OPENSSL_DIR:-$DEPS/openssl}
PCRE2_DIR=${PCRE2_DIR:-$DEPS/pcre2}
for f in "$OPENSSL_DIR/lib64/libssl.a" "$OPENSSL_DIR/lib64/libcrypto.a" \
         "$PCRE2_DIR/lib/libpcre2-8.a" "$PCRE2_DIR/lib/libpcre2-posix.a"; do
    [ -f "$f" ] || {
        echo "build-windows.sh: missing $f" >&2
        echo "  run $0 --deps, or point DEPS/OPENSSL_DIR/PCRE2_DIR at a prepared tree" >&2
        exit 1
    }
done

mkdir -p "$OUT" "$OUT/obj"
# The top-level Makefile writes objects into $(BUILD)/<source dir>, so make the
# mirror directories before it runs (a plain `make` on a clean tree does the
# same with mkdir -p in its pattern rules; cross builds skip that).
for d in core hashx argon2/blake2; do mkdir -p "$OUT/obj/$d"; done

echo "==> engine: anubis_solver.dll (CPU kernels, no GPU backends)"
make -C "$ROOT" -j"$JOBS" \
    CC="$CC" ARCH=none NATIVE= \
    CUDA=0 VULKAN=0 L0=0 OPENCL=0 \
    BUILD="$OUT/obj" LIB="$OUT/anubis_solver.dll" \
    CFLAGS="-O2 -std=gnu17 -Wall -Wextra -DANUBIS_BUILD_SHARED" \
    LDFLAGS="-shared -static -static-libgcc -pthread -Wl,--out-implib,$OUT/libanubis_solver.dll.a"

echo "==> gate: betterAnusbis.exe (HTTP/1.1; optional h2/h3 omitted)"
# shellcheck disable=SC2086
"$CC" -O2 -g -std=gnu17 -Wall -Wextra \
    -I"$GATE/win32/include" -I"$GATE" -I"$ROOT/include" \
    -I"$OPENSSL_DIR/include" -I"$PCRE2_DIR/include" \
    -o "$OUT/betterAnusbis.exe" "$GATE/betterAnusbis.c" \
    "$OUT/libanubis_solver.dll.a" \
    -L"$OPENSSL_DIR/lib64" -L"$PCRE2_DIR/lib" \
    -lssl -lcrypto -lpcre2-posix -lpcre2-8 \
    -lbcrypt -lws2_32 -lcrypt32 -ladvapi32 -luser32 \
    -static -static-libgcc -pthread

cat >"$OUT/README-WINDOWS.txt" <<EOF
betterAnusbis $VERSION for Windows (x86-64)
==========================================

betterAnusbis.exe          the gate
anubis_solver.dll          the proof-of-work engine it links against
LICENSE, ATTRIBUTION.md, MIGRATE.md, README-WINDOWS.txt

Keep the .exe and the .dll in the same directory: the executable imports
anubis_solver.dll by name, so Windows looks for it next to the .exe.

Run it from a Command Prompt or PowerShell:

  betterAnusbis.exe --target http://127.0.0.1:3000 --listen 127.0.0.1:8080

then open http://127.0.0.1:8080/ and the challenge page solves itself.

This build is HTTP/1.1 only. HTTP/2 needs nghttp2 and HTTP/3 needs nghttp3 +
OpenSSL QUIC, which are not part of the Windows release. Everything else is
the same gate, including the six algorithms, the policy file, the honeypot,
the stores, TLS with --tls-cert/--tls-key, and the metrics listener.

Windows specifics:

  * the TLS backend verifies against the Windows certificate store when
    OpenSSL's winstore support is available; set SSL_CERT_FILE to a PEM
    bundle to use one explicitly;
  * HashX uses the interpreter, not the System V JIT, because the generated
    code takes its argument in rdi and Windows passes it in rcx;
  * SIGHUP does not exist here, so the settings dump that "kill -HUP" prints
    on Linux is not available; /metrics and --healthcheck are;
  * SIGTERM/SIGINT become the console's Ctrl+C/Ctrl+Break handling, which
    flushes a file/bbolt store on the way out.

No installer required: unzip anywhere. The .msi installs the same files under
Program Files\betterAnusbis and adds that directory to PATH; run it with

  msiexec /i betterAnusbis-$VERSION-x86_64-windows.msi          (/qn = silent)

On Wine, launching the .msi on its own also installs it, but prints lines like
"fixme:msi:internal_ui_handler internal UI not implemented" -- Wine's MSI UI is
incomplete, the install itself is not affected.
EOF

stage=$OUT/stage
rm -rf "$stage"
mkdir -p "$stage"
cp "$OUT/betterAnusbis.exe" "$OUT/anubis_solver.dll" "$stage/"
cp "$OUT/README-WINDOWS.txt" "$stage/"
cp "$ROOT/LICENSE" "$ROOT/ATTRIBUTION.md" "$ROOT/MIGRATE.md" "$stage/"

ZIP=$OUT/betterAnusbis-$VERSION-x86_64-windows.zip
rm -f "$ZIP"
if command -v zip >/dev/null 2>&1; then
    (cd "$stage" && zip -q -9 "$ZIP" ./*)
else
    python3 - "$stage" "$ZIP" <<'PY'
import os, sys, zipfile
stage, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in sorted(os.listdir(stage)):
        z.write(os.path.join(stage, name), name)
PY
fi
echo "    $ZIP"

if [ "$MAKE_MSI" = 1 ]; then
    if command -v wixl >/dev/null 2>&1; then
        MSI=$OUT/betterAnusbis-$VERSION-x86_64-windows.msi
        rm -f "$MSI"
        # Version is already 3-part numeric (e.g. 1.50.3); wixl's preprocessor
        # takes the source paths through -D so the .wxs stays path-free.
        wixl -a x64 -o "$MSI" \
            -D "Version=$VERSION" \
            -D "StageDir=$stage" \
            "$GATE/win32/betterAnusbis.wxs"
        echo "    $MSI"
    else
        echo "    (wixl not found; skipping the .msi — install msitools + wixl)" >&2
    fi
fi

if command -v wine >/dev/null 2>&1 && [ "$VERIFY" = 1 ]; then
    echo "==> wine: version"
    WINEDEBUG=-all wine "$OUT/betterAnusbis.exe" --version
fi

if command -v wine >/dev/null 2>&1 && [ "$SMOKE" = 1 ]; then
    echo "==> wine: challenge + pass round trip"
    python3 - "$OUT" <<'PY'
import hashlib, http.cookiejar, json, os, re, socket, subprocess, sys, time
import urllib.parse, urllib.request

out = sys.argv[1]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# http.server prints the port it chose to stderr, which is not worth parsing
# here: reserve a free port, close it, and hand it to the server.
bport = free_port()
backend = subprocess.Popen(
    [sys.executable, "-m", "http.server", str(bport), "--bind", "127.0.0.1"],
    cwd="/tmp", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
gport = free_port()
gate = subprocess.Popen(
    ["wine", os.path.join(out, "betterAnusbis.exe"),
     "--listen", "127.0.0.1:%d" % gport,
     "--target", "http://127.0.0.1:%d" % bport,
     "--algorithm", "sha256", "--difficulty", "4"],
    env=dict(os.environ, WINEDEBUG="-all"),
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ok = False
try:
    base = "http://127.0.0.1:%d" % gport
    for _ in range(60):
        try:
            html = urllib.request.urlopen(base + "/", timeout=2).read().decode()
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("smoke: gate did not answer")
    p = json.loads(re.search(
        r'<script id="anubis_challenge" type="application/json">(.*?)</script>',
        html, re.S).group(1))
    ch, rules = p["challenge"], p["rules"]
    data = bytes.fromhex(ch["randomData"])
    little = data[-1] >= 0x80
    bits = int(rules["difficulty"])
    nonce = 0
    while True:
        digest = hashlib.sha256(
            data + nonce.to_bytes(4, "little" if little else "big")).digest()
        if int.from_bytes(digest, "big") >> (256 - bits) == 0:
            break
        nonce += 1
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    url = (base + "/.within.website/x/cmd/anubis/api/pass-challenge"
           "?id=%s&response=%s&nonce=%d&redir=/" % (
               urllib.parse.quote(ch["id"]), digest.hex(), nonce))
    r = op.open(url, timeout=20)
    ok = r.status == 200 and op.open(base + "/", timeout=10).status == 200
    print("    gate solved sha256 at difficulty %d, passed: %s" % (bits, ok))
finally:
    gate.terminate()
    backend.terminate()
    try:
        gate.wait(timeout=10)
    except Exception:
        gate.kill()
    backend.wait()
if not ok:
    raise SystemExit("smoke: pass did not succeed")
PY
fi

echo
echo "built: $OUT/betterAnusbis.exe ($(wc -c <"$OUT/betterAnusbis.exe") bytes)"
echo "done. The .exe and .dll must ship together."
