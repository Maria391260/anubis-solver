#!/bin/sh
# Use every chip in a multi-chip TPU slice.
#
# A Colab/v5e-1 slice exposes one addressable device, so the lane sweep there
# tops out at what fits in one chip. A v3-8 or v5e-8 slice exposes eight, and
# lanes are independent nonces, so eight processes are eight independent
# problems: the rates add. No sharding, no collectives, no changes to the
# graph - each process compiles once and owns one chip.
#
#   sh colab_argon2_multidev.sh            # 32 lanes per chip
#   LANES=64 SECS=20 sh colab_argon2_multidev.sh
#
# Works the same on a Kaggle TPU v3-8 VM or a TRC slice; it only needs
# libtpu.so, cc and python3.
set -u
cd "$(dirname "$0")"

CC=${CC:-cc}
TMP=${TMP:-/tmp}
OUT=${OUT:-$TMP/argon2_pjrt}
LANES=${LANES:-32}
SECS=${SECS:-20}
DIFF=${DIFF:-32}

$CC -O2 -I. -o "$OUT" argon2_pjrt.c -ldl || exit 1

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

# How many chips? An out-of-range index makes the client print the count and
# exit, which is cheaper than parsing a device list out of the PJRT API twice.
NDEV=$("$OUT" --lib "$LIBTPU_PATH" --device-index 999 --m 8 --t 1 --lanes 1 2>&1 |
       sed -n 's/.*out of range: \([0-9]*\) addressable.*/\1/p' | tail -1)
case "$NDEV" in
    ''|*[!0-9]*) echo "could not determine device count" >&2; exit 1 ;;
esac
echo "devices: $NDEV   lanes/device: $LANES   matrix/device: $((LANES * 19456 / 1024)) MiB"

rm -f "$TMP"/a2dev.*.json "$TMP"/a2dev.*.log
i=0
while [ "$i" -lt "$NDEV" ]; do
    "$OUT" --lib "$LIBTPU_PATH" --device-index "$i" --lanes "$LANES" \
           --difficulty "$DIFF" --stress "$SECS" --json \
           > "$TMP/a2dev.$i.json" 2> "$TMP/a2dev.$i.log" &
    i=$((i + 1))
done
wait

python3 - "$TMP" "$NDEV" "$LANES" <<'PY'
import glob, json, os, sys

tmp, ndev, lanes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows, total, hbm, ok = [], 0.0, 0.0, 0
for i in range(ndev):
    path = os.path.join(tmp, "a2dev.%d.json" % i)
    line = ""
    if os.path.exists(path):
        with open(path) as f:
            for l in f:
                l = l.strip()
                if l.startswith("{"):
                    line = l
    if not line:
        err = os.path.join(tmp, "a2dev.%d.log" % i)
        note = ""
        if os.path.exists(err):
            with open(err) as f:
                note = next((l.strip() for l in f if l.strip()), "")[:60]
        rows.append((i, None, None, note or "no json"))
        continue
    d = json.loads(line)
    ok += 1
    total += d["steady_h_s"]
    hbm += lanes * 19456 * 1024 / 2**30
    ver = d.get("verified")
    rows.append((i, d["steady_h_s"], d["win_p50_ms"],
                 "ok" if ver is True else ("MISMATCH" if ver is False else ver)))

print()
print("device   steady H/s   p50 window   verified")
print("------   ----------   ----------   --------")
for i, hs, p50, ver in rows:
    if hs is None:
        print("%6d   %10s   %10s   %s" % (i, "-", "-", ver))
    else:
        print("%6d   %10.3f   %9.1f ms   %s" % (i, hs, p50, ver))
print()
print("aggregate: %.2f H/s from %d of %d device(s), %.1f GiB of matrix in flight"
      % (total, ok, ndev, hbm))
if ok:
    print("per-chip:  %.3f H/s   (%.1f ms per nonce chain)"
          % (total / ok, 1000.0 / (total / ok)))
PY

cat <<'TXT'

Reading it:
  * a chip is added, not shared: no collectives, no sharding, no graph change,
    so the aggregate should be ndev x one chip within a couple of percent.
  * each process compiles its own executable (~13 s, done in parallel), so the
    wall time is one compile plus the stress window, not ndev compiles.
  * memory: 19.9 MiB per lane. A v3-8 has 8 x 16 GiB, so 32 lanes/chip uses
    5 GiB of 128 GiB and 256 lanes/chip would still fit.
TXT
