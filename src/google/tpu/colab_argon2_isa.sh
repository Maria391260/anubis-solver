#!/bin/sh
# Ask libtpu to dump what it actually builds for the Argon2id fill: the ISA
# program (the machine code it would push to the device) and the HLO text.
# The flags come from libtpu's own symbol table:
#   --xla_jf_dump_isa_program_proto, --xla_jf_dump_hlo_text
# plus the usual --xla_dump_to / --xla_dump_hlo_as_text.
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

D=/tmp/isadump
rm -rf "$D"
mkdir -p "$D"
cd "$D"
export XLA_FLAGS="--xla_jf_dump_isa_program_proto=true --xla_jf_dump_hlo_text=true --xla_dump_to=$D --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*"
echo "XLA_FLAGS=$XLA_FLAGS"

# one BLAKE2 column round is the smallest module that still unpacks, rounds,
# repacks: enough to see what the emitter does with the fill
"$OUT" --lib "$LIBTPU_PATH" --run-matrix --filltest1 --m 8 --t 1 --lanes 1 || true

echo "== files"
ls -la "$D" | head -40
for f in "$D"/*; do
    [ -f "$f" ] || continue
    echo "== $f ($(stat -c %s "$f") bytes)"
    case "$f" in
    *.txt|*.hlo|*.mlir|*hlo*)
        head -60 "$f"
        ;;
    *)
        strings -a -n 6 "$f" | head -40
        ;;
    esac
done
