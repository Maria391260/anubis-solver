#!/usr/bin/env bash
#
# tests/test_universal.sh — the one-file build: every backend compiled in, no
# vendor library linked, the same .so runnable on a machine with no GPU at all.
#
# What it proves, in order:
#   1. the .so exports the backend inventory and `easter`;
#   2. it has no DT_NEEDED on any accelerator library (cuda, cudart, vulkan,
#      OpenCL, Level Zero, libtpu) — so a missing driver cannot stop the load;
#   3. the compiled-in mask agrees with what `make report` says, i.e. the
#      Makefile flags and the C preprocessor did not drift apart;
#   4. probing the machine reports at least the CPU and never crashes or hangs;
#   5. the C header is usable from C and C++ (it is the public API);
#   6. the Python library and the C API describe the same inventory.
#
# Usage: tests/test_universal.sh        (run `make universal` first)
# Exit:  0 when every assertion passed, 1 otherwise.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LIB="$ROOT/libanubis_solver.so"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/anubis_universal.XXXXXX")"
APID=""

cleanup() {
    trap - EXIT INT TERM
    if [ -n "$APID" ]; then
        kill "$APID" 2>/dev/null || true
        wait "$APID" 2>/dev/null || true
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    printf '  ok   %s\n' "$1"
}

bad() {
    FAIL=$((FAIL + 1))
    printf '  FAIL %s%s\n' "$1" "${2:+ ($2)}"
}

check() { # name, condition-result
    if [ "$2" -eq 0 ]; then ok "$1"; else bad "$1"; fi
}

if [ ! -f "$LIB" ]; then
    echo "missing $LIB — run: make universal" >&2
    exit 1
fi

echo "export surface"
for sym in anubis_backends_compiled anubis_backends_available \
           anubis_backend_name easter anubis_solver_version \
           anubis_set_tpu_verify anubis_tpu_verify \
           anubis_tpu_available anubis_tpu_info anubis_tpu_solve; do
    nm -D --defined-only "$LIB" 2>/dev/null | grep -q " T $sym$"
    check "exports $sym" $?
done

echo "running the .so directly is handled, not a SIGSEGV"
DIRECT_OUT="$("$LIB" 2>&1)"
DIRECT_RC=$?
[ "$DIRECT_RC" -eq 1 ]
check "direct run exits 1" $?
case "$DIRECT_OUT" in
*"not an executable"*) ok "direct run says it is a shared library" ;;
*) bad "direct run says it is a shared library" "$(echo "$DIRECT_OUT" | head -1)" ;;
esac
case "$DIRECT_OUT" in
*"celikfana and DeepSeek V4.1 Flash"*) ok "direct run prints the credit line" ;;
*) bad "direct run prints the credit line" ;;
esac

echo "no accelerator in DT_NEEDED (everything is dlopen()ed)"
NEEDED="$(readelf -d "$LIB" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')"
printf '       deps: %s\n' "$(echo $NEEDED)"
for bad_lib in libcuda.so libcudart.so libnvidia-ml.so libvulkan.so libOpenCL.so \
               libze_loader.so libtpu.so libpjrt.so; do
    if echo "$NEEDED" | grep -q "$bad_lib"; then
        bad "linked against $bad_lib" "the .so must dlopen it"
    else
        ok "not linked against $bad_lib"
    fi
done

echo "C header from C"
cat > "$TMP/probe.c" <<'EOF'
#include <stdio.h>
#include "anubis-solver.h"

int main(void) {
    int c = anubis_backends_compiled();
    int a = anubis_backends_available();
    int bit;
    if (c & ~0x3f) {
        return 2;
    }
    printf("compiled=%d\n", c);
    printf("available=%d\n", a);
    for (bit = 1; bit <= 0x20; bit <<= 1) {
        const char *name = anubis_backend_name(bit);
        if ((c & bit) && name == NULL) {
            return 3;
        }
        printf("%s:%s\n", name ? name : "-", (a & bit) ? "yes" : "no");
    }
    if (anubis_backend_name(0x1000) != NULL) {
        return 4;
    }
    if (easter() == NULL) {
        return 5;
    }
    return 0;
}
EOF
if cc -O2 -Wall -Wextra -I"$ROOT/include" -o "$TMP/probe" "$TMP/probe.c" \
      -L"$ROOT" -lanubis_solver -Wl,-rpath,"$ROOT" 2>"$TMP/probe.err"; then
    ok "header compiles as C"
else
    bad "header compiles as C" "$(tail -1 "$TMP/probe.err")"
fi

if [ -x "$TMP/probe" ]; then
    if timeout 120 "$TMP/probe" > "$TMP/probe.out" 2>"$TMP/probe.err"; then
        ok "C probe runs (no crash without a GPU)"
    else
        bad "C probe runs" "exit $?"
    fi
    grep -q '^cpu:yes$' "$TMP/probe.out"
    check "cpu is available" $?
    COMPILED_C="$(sed -n 's/^compiled=//p' "$TMP/probe.out")"
    AVAIL_C="$(sed -n 's/^available=//p' "$TMP/probe.out")"
else
    COMPILED_C=""; AVAIL_C=""
fi

echo "C++ sees the same header"
cat > "$TMP/probe.cpp" <<'EOF'
#include "anubis-solver.h"
int main() { return (anubis_backends_compiled() & ANUBIS_BACKEND_CPU) ? 0 : 1; }
EOF
if c++ -O2 -Wall -Wextra -I"$ROOT/include" -o "$TMP/probe_cpp" "$TMP/probe.cpp" \
      -L"$ROOT" -lanubis_solver -Wl,-rpath,"$ROOT" 2>"$TMP/cpp.err"; then
    ok "header compiles as C++"
    "$TMP/probe_cpp"
    check "extern \"C\" link works" $?
else
    bad "header compiles as C++" "$(tail -1 "$TMP/cpp.err")"
fi

echo "the Python library and the CLI agree with the library"
python3 - "$ROOT" "$COMPILED_C" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import anubis_solver as a

mask = int(sys.argv[2])
names = {"cpu": 1, "cuda": 2, "levelzero": 4, "vulkan": 8, "opencl": 16, "tpu": 32}
want = sorted(n for n, b in names.items() if mask & b)

s = a.AnubisSolver()
compiled = sorted(s.backends_compiled())
available = list(s.backends())
report = s.backend_report()
bad = []
if compiled != want:
    bad.append(f"compiled {compiled} != {want}")
if not set(available) <= set(want):
    bad.append(f"available {available} not a subset of compiled")
if "cpu" not in available:
    bad.append("cpu missing from available")
for name in want:
    if name not in report:
        bad.append(f"{name} missing from backend_report()")
for line in bad:
    print("  FAIL", line)
sys.exit(1 if bad else 0)
EOF
check "AnubisSolver().backends*() and backend_report() match the .so" $?

CLI_OUT="$(cd "$ROOT" && python3 anubis_solver.py --backends 2>&1)"
if echo "$CLI_OUT" | grep -q "cpu.*compiled=yes" &&
   echo "$CLI_OUT" | grep -q "libanubis_solver "; then
    ok "--backends prints the inventory"
else
    bad "--backends prints the inventory" "$(echo "$CLI_OUT" | head -1)"
fi

echo "Makefile flags agree with the compiled-in mask"
# `make report` is the human view of the same variables the compiler used.
REPORT="$(cd "$ROOT" && make -s report 2>/dev/null)"
printf '       %s\n' "$REPORT"
WANT=0
for name in cuda levelzero vulkan opencl tpu; do
    case "$name" in
    cuda) bit=2 ;;
    levelzero) bit=4 ;;
    vulkan) bit=8 ;;
    opencl) bit=16 ;;
    tpu) bit=32 ;;
    esac
    if echo "$REPORT" | grep -qE "^[[:space:]]+$name[[:space:]]+yes$"; then
        WANT=$((WANT | bit))
    fi
done
WANT=$((WANT | 1))
if [ "$WANT" = "$COMPILED_C" ]; then
    ok "make report matches anubis_backends_compiled() ($WANT)"
else
    bad "make report matches anubis_backends_compiled()" "report=$WANT lib=$COMPILED_C"
fi

echo "probing is repeatable and bounded"
START=$(date +%s)
"$TMP/probe" > "$TMP/probe2.out" 2>/dev/null
END=$(date +%s)
AVAIL2="$(sed -n 's/^available=//p' "$TMP/probe2.out")"
[ "$AVAIL2" = "$AVAIL_C" ]
check "second probe reports the same backends" $?
[ $((END - START)) -le 60 ]
check "second probe took $((END - START))s (<60)" $?

echo "the API service describes the same inventory"
"$ROOT/anubis_api" --listen 127.0.0.1:18123 > "$TMP/api.log" 2>&1 &
APID=$!
for _ in $(seq 1 50); do
    if curl -sf --max-time 1 http://127.0.0.1:18123/healthz >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
if curl -sf --max-time 60 http://127.0.0.1:18123/v1/backends > "$TMP/backends.json" 2>/dev/null; then
    ok "GET /v1/backends answers"
    python3 - "$TMP/backends.json" "$COMPILED_C" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
mask = int(sys.argv[2])
names = {"cpu": 1, "cuda": 2, "levelzero": 4, "vulkan": 8, "opencl": 16, "tpu": 32}
want = sorted(n for n, b in names.items() if mask & b)
compiled = sorted(data["compiled"])
available = data["available"]
bad = []
if compiled != want:
    bad.append(f"compiled {compiled} != {want}")
if "cpu" not in available:
    bad.append("cpu missing from available")
if not set(available) <= set(compiled):
    bad.append(f"available {available} not a subset of compiled")
for b in bad:
    print("  FAIL", b)
sys.exit(1 if bad else 0)
EOF
    check "JSON inventory matches the library" $?
else
    bad "GET /v1/backends answers"
fi
kill "$APID" 2>/dev/null || true
wait "$APID" 2>/dev/null || true
APID=""

echo
printf 'universal: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
