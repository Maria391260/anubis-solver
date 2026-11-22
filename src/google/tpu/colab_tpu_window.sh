#!/bin/sh
# Is the sha256/hashx rate still climbing at the window we shipped?
#
# The window is lanes-per-execute. The published table stopped at 262144 lanes
# because that was the client's hard cap, not because the device asked for it:
# 38.0 -> 111.9 -> 251.3 -> 342.8 MHash/s was still rising when the assertion
# tripped (increments +74, +139, +92). The cap is now 1048576, so this sweep
# asks the only question left on that axis.
#
# It also A/Bs the tensor-v copy-back with --dmamap, because copy-back is 77%
# of that tier's e2e time and libtpu bounces non-premapped destinations.
#
#   sh colab_tpu_window.sh
set -u
cd "$(dirname "$0")"

CC=${CC:-cc}
TMP=${TMP:-/tmp}
SHA=$TMP/sha256_pjrt
HX=$TMP/hashx_pjrt
TS=$TMP/tensor_pjrt
ROWS=$TMP/tpu_window_rows.txt

$CC -O2 -I. -o "$SHA" sha256_pjrt.c -ldl || exit 1
$CC -O2 -I. -I../../hashx -o "$HX" hashx_pjrt.c ../../hashx/hashx.c -ldl || exit 1
$CC -O2 -I. -o "$TS" tensor_pjrt.c -ldl || exit 1

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
: > "$ROWS"

rec() { # tier window output
    t=$1
    w=$2
    o=$3
    json=$(printf '%s\n' "$o" | grep -o '{"[^}]*}' | tail -1)
    if [ -z "$json" ]; then
        note=$(printf '%s\n' "$o" | grep -v '^$' | head -1 | cut -c1-80)
        echo "   ($t/$w) no json: ${note:-no output}" >&2
        printf '%s|%s|%s|%s|%s\n' "$t" "$w" "-" "failed" "-" >> "$ROWS"
        return
    fi
    hs=$(printf '%s\n' "$json" | sed -n 's/.*"mhash_s":\([0-9.]*\).*/\1/p')
    cm=$(printf '%s\n' "$json" | sed -n 's/.*"compile_ms":\([0-9.]*\).*/\1/p')
    n=$(printf '%s\n' "$json" | sed -n 's/.*"hashes":\([0-9]*\).*/\1/p')
    printf '%s|%s|%s|%s|%s\n' "$t" "$w" "${hs:--}" "${cm:--}" "${n:--}" >> "$ROWS"
}

echo "== sha256: window sweep"
for w in 262144 524288 1048576; do
    out=$("$SHA" --lib "$LIBTPU_PATH" --window "$w" --difficulty 32 \
               --bench-hashes 67108864 --json 2>&1)
    rec sha256 "$w" "$out"
    printf '%s\n' "$out" | grep -o '{"[^}]*}' | tail -1 | sed 's/^/   /'
done

echo "== hashx: window sweep"
for w in 32768 131072 262144 524288 1048576; do
    out=$("$HX" --lib "$LIBTPU_PATH" --window "$w" --difficulty 32 \
               --bench-hashes 33554432 --json 2>&1)
    rec hashx "$w" "$out"
    printf '%s\n' "$out" | grep -o '{"[^}]*}' | tail -1 | sed 's/^/   /'
done

echo "== tensor-v d16: plain vs dmamap"
for mode in plain dmamap; do
    if [ "$mode" = "dmamap" ]; then
        out=$("$TS" --lib "$LIBTPU_PATH" --tensor-v --difficulty 16 --n 1024 \
                   --rounds 5 --dmamap --json 2>&1)
    else
        out=$("$TS" --lib "$LIBTPU_PATH" --tensor-v --difficulty 16 --n 1024 \
                   --rounds 5 --json 2>&1)
    fi
    json=$(printf '%s\n' "$out" | grep -o '{"[^}]*}' | tail -1)
    if [ -z "$json" ]; then
        echo "   ($mode) no json: $(printf '%s\n' "$out" | grep -v '^$' | head -1 | cut -c1-80)" >&2
        printf '%s|%s|%s|%s|%s\n' tensor-v "$mode" "-" "failed" "-" >> "$ROWS"
        continue
    fi
    e2e=$(printf '%s\n' "$json" | sed -n 's/.*"e2e_ms":\([0-9.]*\).*/\1/p')
    gm=$(printf '%s\n' "$json" | sed -n 's/.*"gmacs_e2e":\([0-9.]*\).*/\1/p')
    dm=$(printf '%s\n' "$json" | sed -n 's/.*"dma_mapped":"\?\([a-z]*\)"\?.*/\1/p')
    printf '%s|%s|%s|%s|%s\n' tensor-v "$mode" "$e2e" "$gm" "$dm" >> "$ROWS"
    echo "   $mode: e2e ${e2e} ms, ${gm} GMAC/s, dma_mapped=$dm"
done

python3 - "$ROWS" <<'PY'
import sys

rows = [l.rstrip("\n").split("|") for l in open(sys.argv[1]) if l.strip()]
print()
print("tier        window      MHash/s   compile ms   hashes/exec")
print("----------  ----------  --------  -----------   ------------")
for tier, w, hs, cm, n in rows:
    if tier.startswith("tensor"):
        continue
    print("%-10s  %10s  %8s  %11s   %s" % (tier, w, hs, cm, n))
print()
print("tensor-v d16 (epoch, not a hash rate):")
for tier, mode, e2e, gm, dm in rows:
    if not tier.startswith("tensor"):
        continue
    print("  %-8s e2e %s ms   %s GMAC/s   dma_mapped=%s" % (mode, e2e, gm, dm))
print()
print("If MHash/s keeps climbing to 1048576 lanes, the shipped 343 MHash/s")
print("figure is a software cap, not the device: raise it and re-record.")
PY

cat <<'TXT'

Reading it:
  * sha256: the published increments were +74, +139, +92 MHash/s per doubling,
    so the curve is flattening - expect a plateau near 380-420, not a doubling.
    Any climb past 400 MHash/s is a new headline number.
  * hashx: same axis, and it had more headroom (75.3 MHash/s at 32768 lanes).
  * tensor-v: if dma_mapped=true and e2e drops ~30%, the staging copy was real
    and --dmamap is the fix. If DmaMap is refused, libtpu prints why on stderr.
TXT
