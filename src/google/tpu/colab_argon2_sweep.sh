#!/bin/sh
# Lane sweep for the Argon2id TPU client: lanes are independent nonces, so the
# only question left is how far the vector unit can be filled before the
# per-window time starts to grow. Each lane costs 19.4 MiB of HBM.
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

for L in 8 64 128 256; do
    echo "== lanes=$L (matrix = $((L * 19456 * 1024 / 1000000)) MB)"
    "$OUT" --lib "$LIBTPU_PATH" --lanes "$L" --difficulty 32 --stress 6 --json \
        || echo "lanes=$L failed"
done
