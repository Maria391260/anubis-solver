#!/bin/sh
# Does libtpu.so have a "go faster" switch?
#
# The library carries 2797 absl flags, 1079 of them xla_tpu_*. They are not
# reachable through XLA_FLAGS; libtpu parses its own flag table from the
# LIBTPU_INIT_ARGS environment variable at plugin init (the string is in the
# binary next to the plugin's init path). None of them is a clock, voltage or
# power knob - there is no such flag in the 451 MB image.
#
# This script runs the same serialized Argon2id executable, in the same
# 12-second steady-state window, under a handful of those flags. Only the flag
# changes, so any difference in H/s is the flag (or a rejected flag, which
# shows up as an init error - printed instead of a number).
#
#   git clone -q -b collab_push https://gitlab.com/celikfana/anubis-solver.git /content/as
#   sh /content/as/src/google/tpu/colab_argon2_flags.sh
set -u
cd "$(dirname "$0")"

CC=${CC:-cc}
OUT=${OUT:-/tmp/argon2_pjrt}
SER=${SER:-/tmp/argon2_19456x2x32.pjrt}
SECS=${SECS:-12}

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

# Same shapes as colab_argon2.sh, so its serialized executable is reusable.
if [ ! -f "$SER" ]; then
    echo "== compiling once (13 s) so every variation below loads the same code"
    "$OUT" --lib "$LIBTPU_PATH" --lanes 32 --difficulty 8 --bench-hashes 33 \
           --serialize "$SER" --json || exit 1
fi

printf '%-14s %10s %6s  %s\n' "config" "steady H/s" "nonces" "note"
printf '%-14s %10s %6s  %s\n' "------" "----------" "------" "----"

# label|LIBTPU_INIT_ARGS
while IFS='|' read -r label flags; do
    [ -n "$label" ] || continue
    if [ -n "$flags" ]; then
        LIBTPU_INIT_ARGS="$flags"
        export LIBTPU_INIT_ARGS
    else
        unset LIBTPU_INIT_ARGS
    fi
    out=$("$OUT" --lib "$LIBTPU_PATH" --load "$SER" --lanes 32 --difficulty 32 \
               --stress "$SECS" --json 2>&1)
    json=$(printf '%s\n' "$out" | grep -o '{"hashes".*}' | tail -1)
    if [ -z "$json" ]; then
        note=$(printf '%s\n' "$out" | grep -v '^$' | head -1 | cut -c1-90)
        printf '%-14s %10s %6s  %s\n' "$label" "-" "-" "${note:-no output}"
        continue
    fi
    hs=$(printf '%s\n' "$json" | sed -n 's/.*"steady_h_s":\([0-9.]*\).*/\1/p')
    n=$(printf '%s\n' "$json" | sed -n 's/.*"hashes":\([0-9]*\).*/\1/p')
    printf '%-14s %10s %6s  %s\n' "$label" "$hs" "$n" "$flags"
done <<'EOF'
baseline|
critpath|--xla_jf_critical_path_scheduler=true
vliw|--xla_jf_vliw_scheduler=true
opt2|--xla_optimization_level=2
opt3|--xla_optimization_level=3
naive-bundle|--xla_jf_naive_bundle_packer=true
llvm-emit|--xla_llvm_isa_emitter=true --xla_llvm_isa_emitter_force=true
no-hlo-pipe|--xla_jf_enable_hlo_pipeline=false
EOF

cat <<'TXT'

How to read this:
  * a number equal to baseline (9.38 H/s) means the flag did not change codegen
    for this graph, or does not apply to it;
  * a line with a note instead of a number means libtpu rejected the flag at
    init ("unknown flag") - that is a fact about the flag, not about the TPU;
  * a big win is unlikely, and here is why: the fill is one dependent chain of
    ~87 us per block, and measured HBM traffic is 1.42 GiB/s out of 819 GB/s
    (0.19%). There is no idle resource for a flag to unlock - the chain is the
    wall. A flag can only help by scheduling that chain better (critpath, vliw,
    optimization_level) or by emitting different instructions (llvm-emit).
TXT
