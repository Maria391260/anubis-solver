#!/bin/sh
# Argon2id graph debugging on a TPU: print word 0 of every block so the
# matrix can be compared word-for-word with the Python/ref model offline.
set -e
cd "$(dirname "$0")"
CC=${CC:-cc}
OUT=${OUT:-/tmp/argon2_pjrt}
$CC -O2 -I. -o "$OUT" argon2_pjrt.c -ldl

LIBTPU_PATH=$(python3 -c "import glob;c=glob.glob('/usr/local/lib/python3*/dist-packages/libtpu/libtpu.so')+glob.glob('/usr/local/lib/python3*/site-packages/libtpu/libtpu.so')+glob.glob('/usr/lib/python3*/dist-packages/libtpu/libtpu.so');print(c[0] if c else '')")
if [ -z "$LIBTPU_PATH" ]; then
    echo "libtpu.so not found - is this a TPU runtime?" >&2
    exit 1
fi
echo "libtpu: $LIBTPU_PATH"

echo "== schedule only (rounds removed on both sides), m=8 t=2 lanes=1"
"$OUT" --lib "$LIBTPU_PATH" --run-matrix --sched --m 8 --t 2 --lanes 1

echo "== full fill, m=8 t=1 lanes=1"
"$OUT" --lib "$LIBTPU_PATH" --run-matrix --m 8 --t 1 --lanes 1
