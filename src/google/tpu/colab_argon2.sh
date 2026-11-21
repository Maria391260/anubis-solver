#!/bin/sh
# Argon2id on a TPU, from a Colab (or any TPU VM) shell: build the PJRT client,
# check the host reference, run the StableHLO fill against it, then measure.
#
#   git clone -q -b collab_push https://gitlab.com/celikfana/anubis-solver.git /content/as
#   sh /content/as/src/google/tpu/colab_argon2.sh
#
# Everything the device computes is checked against the host reference in this
# same binary, so a wrong graph cannot report a rate.
set -e
cd "$(dirname "$0")"

CC=${CC:-cc}
OUT=${OUT:-/tmp/argon2_pjrt}
$CC -O2 -I. -o "$OUT" argon2_pjrt.c -ldl

LIBTPU_PATH=$(python3 - <<'PY'
import glob
c = (glob.glob('/usr/local/lib/python3*/dist-packages/libtpu/libtpu.so') +
     glob.glob('/usr/local/lib/python3*/site-packages/libtpu/libtpu.so') +
     glob.glob('/usr/lib/python3*/dist-packages/libtpu/libtpu.so') +
     glob.glob('/opt/**/libtpu.so', recursive=True))
print(c[0] if c else '')
PY
)
if [ -z "$LIBTPU_PATH" ]; then
    echo "libtpu.so not found - is this a TPU runtime?" >&2
    exit 1
fi
echo "libtpu: $LIBTPU_PATH"

echo "== host reference (no TPU)"
"$OUT" --unit
"$OUT" --hosttest

echo "== graph vs host reference, several sizes"
"$OUT" --lib "$LIBTPU_PATH" --selftest

echo "== full size (m=19456, t=2), difficulty 8, verified hit"
"$OUT" --lib "$LIBTPU_PATH" --difficulty 8 --lanes 8

echo "== throughput"
"$OUT" --lib "$LIBTPU_PATH" --lanes 32 --difficulty 32 --bench-hashes 64 --json
