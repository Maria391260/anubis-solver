#!/bin/sh
# Ask XLA to print the TPU assembly it generated for our kernels.
#
# The compiler inside libtpu.so carries its own dumper (LloDumper::DumpBundles
# IfEnabled -> BundlesToTextStream), gated by flags that only LIBTPU_INIT_ARGS
# can set. That makes the ISA layer readable without reimplementing a driver:
# per-bundle mnemonics with the issue slot each op landed in.
#
# Two questions it answers, both of which change a kernel:
#   sha256  - are the 64 rounds really 3 vector ops per bundle, or is the
#             scheduler leaving slots empty (51 B/bundle, 9 issue slots)?
#   argon2id - is the ~7 cycles/vector-op a dependent-ALU chain, or padding
#             that a latency-hiding scheduler could remove?
#
#   sh colab_llo_dump.sh
set -u
cd "$(dirname "$0")"

CC=${CC:-cc}
TMP=${TMP:-/tmp}
DUMP=${DUMP:-$TMP/llo}
SHA=$TMP/sha256_pjrt
A2=$TMP/argon2_pjrt

$CC -O2 -I. -o "$SHA" sha256_pjrt.c -ldl || exit 1
$CC -O2 -I. -o "$A2" argon2_pjrt.c -ldl || exit 1

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
rm -rf "$DUMP"
mkdir -p "$DUMP/sha" "$DUMP/a2" || exit 1

# Small shapes on purpose: a 262144-lane module would produce a dump measured
# in gigabytes, and the bundle structure does not depend on the lane count.
echo "== sha256, 8192 lanes (one execute), dumping LLO"
LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=$DUMP/sha --xla_jf_dump_llo_pass_label_regex=.*"
export LIBTPU_INIT_ARGS
"$SHA" --lib "$LIBTPU_PATH" --window 8192 --bench-hashes 8192 --json 2> "$DUMP/sha.log"
sha_rc=$?

echo "== argon2id, m=64 t=2 lanes=4, dumping LLO"
LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=$DUMP/a2 --xla_jf_dump_llo_pass_label_regex=.*"
export LIBTPU_INIT_ARGS
"$A2" --lib "$LIBTPU_PATH" --m 64 --t 2 --lanes 4 --difficulty 4 2> "$DUMP/a2.log"
a2_rc=$?

unset LIBTPU_INIT_ARGS

n=$(find "$DUMP" -type f | wc -l)
echo
echo "exit: sha256=$sha_rc argon2id=$a2_rc   files under $DUMP: $n"
find "$DUMP" -type f -size +0 | head -20 | while read -r f; do
    printf '%9s  %s\n' "$(wc -c < "$f")" "$f"
done

if [ "$n" -le 2 ]; then
    cat <<'TXT'

No dump files, only the two logs. That means libtpu refused the flags; the log
below says which one and why (the jf_* family is internal, and the allowlist
differs between releases). Read the first lines of the logs:
TXT
    for f in "$DUMP/sha.log" "$DUMP/a2.log"; do
        [ -s "$f" ] || continue
        echo "--- $f"
        head -6 "$f"
    done
    exit 0
fi

echo
echo "== what the dump looks like (first 25 non-empty lines of the sha256 dump)"
first=$(find "$DUMP/sha" -type f -size +0 | head -1)
[ -n "$first" ] || first=$(find "$DUMP" -name '*.log' -prune -o -type f -size +0 -print | head -1)
[ -n "$first" ] && grep -v '^[[:space:]]*$' "$first" | head -25

echo
echo "== mnemonic histogram (sha256 dump, top 20)"
python3 - "$DUMP/sha" <<'PY'
import collections, os, re, sys
words = collections.Counter()
for root, _, files in os.walk(sys.argv[1]):
    for name in files:
        try:
            with open(os.path.join(root, name), errors="replace") as f:
                for line in f:
                    for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", line):
                        if tok.islower() or tok.isupper() or "_" in tok:
                            words[tok] += 1
        except OSError:
            continue
for tok, n in words.most_common(20):
    print("%8d  %s" % (n, tok))
PY

cat <<'TXT'

What to look for:
  * bundle count vs round count: sha256 does 64 rounds x 4 lanes-worth of work;
    if the bundle count is far below 3 x 64, the scheduler is packing well and
    343 MHash/s is ALU-bound, not issue-bound.
  * slot names in the dump (VEX, VLD, VST, VRES): the vector core has two ALUs
    plus load/store/result pipes, and a bundle has 9 slots. Empty slots at the
    end of a round are the cheap win: unroll two lanes' rounds interleaved.
  * argon2id: if consecutive bundles pair a dependent op with an independent
    one, the chain is already interleaved; if each bundle carries one op and
    waits, the 7 cycles are ALU latency and no flag will fix it.
TXT
