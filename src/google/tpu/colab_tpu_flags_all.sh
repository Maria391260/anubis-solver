#!/bin/sh
# Do libtpu's scheduler/emitter flags move any tier, or only Argon2id?
#
# Same idea as colab_argon2_flags.sh, widened to every TPU tier the project
# has: sha256 (343 MHash/s), hashx (75 MHash/s), argon2id (9.38 H/s, the
# latency-bound one), tensor and tensor-v (11.5 GMAC/s on the MXU).
#
# One process per (tier, flag set), because LIBTPU_INIT_ARGS is read at plugin
# init. Argon2id deliberately recompiles per flag set here: --load replays a
# serialized executable and would ignore every codegen flag by construction,
# which would make all five rows identical for the boring reason.
#
#   git clone -q -b collab_push https://gitlab.com/celikfana/anubis-solver.git /content/as
#   sh /content/as/src/google/tpu/colab_tpu_flags_all.sh
#
# Expect ~5 minutes: 22 plugin inits, 15 compiles (argon2id ~13 s, hashx ~14.6 s,
# sha256 ~10 s each) and 5 x 8 s of Argon2id stress. RENDER_ONLY=1 reprints the
# table from the last run without touching the TPU.
set -u
cd "$(dirname "$0")"

CC=${CC:-cc}
TMP=${TMP:-/tmp}
SHA=$TMP/sha256_pjrt
HX=$TMP/hashx_pjrt
A2=$TMP/argon2_pjrt
TS=$TMP/tensor_pjrt
ROWS=$TMP/tpu_flags_rows.txt
SECS=${SECS:-8}

rec() { # tier cfg metric-key output
    t=$1
    c=$2
    k=$3
    o=$4
    json=$(printf '%s\n' "$o" | grep -o '{"[^}]*}' | tail -1)
    if [ -z "$json" ]; then
        note=$(printf '%s\n' "$o" | grep -v '^$' | head -1 | cut -c1-90)
        echo "   ($t/$c) no JSON: ${note:-no output}" >&2
        printf '%s|%s|%s|%s|%s\n' "$t" "$c" "$k" "-" "refused" >> "$ROWS"
        return
    fi
    v=$(printf '%s\n' "$json" | sed -n "s/.*\"$k\":\([0-9.]*\).*/\1/p")
    cm=$(printf '%s\n' "$json" | sed -n 's/.*"compile_ms":\([0-9.]*\).*/\1/p')
    printf '%s|%s|%s|%s|%s\n' "$t" "$c" "$k" "${v:--}" "${cm:--}" >> "$ROWS"
}

sweep() {
    $CC -O2 -I. -o "$SHA" sha256_pjrt.c -ldl || exit 1
    # hashx needs the interpreter that verifies its hits, so it links hashx.c
    $CC -O2 -I. -I../../hashx -o "$HX" hashx_pjrt.c ../../hashx/hashx.c -ldl || exit 1
    $CC -O2 -I. -o "$A2" argon2_pjrt.c -ldl || exit 1
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

    while IFS='|' read -r cfg flags; do
        [ -n "$cfg" ] || continue
        if [ -n "$flags" ]; then
            LIBTPU_INIT_ARGS="$flags"
            export LIBTPU_INIT_ARGS
        else
            unset LIBTPU_INIT_ARGS
        fi
        printf '== %-10s %s\n' "$cfg" "${flags:-<none>}"

        out=$("$SHA" --lib "$LIBTPU_PATH" --window 262144 \
                   --bench-hashes 67108864 --json 2>&1)
        rec sha256 "$cfg" mhash_s "$out"

        out=$("$HX" --lib "$LIBTPU_PATH" --window 32768 \
                   --bench-hashes 33554432 --json 2>&1)
        rec hashx "$cfg" mhash_s "$out"

        out=$("$A2" --lib "$LIBTPU_PATH" --lanes 32 --difficulty 32 \
                   --stress "$SECS" --json 2>&1)
        rec argon2id "$cfg" steady_h_s "$out"

        out=$("$TS" --lib "$LIBTPU_PATH" --difficulty 16 --n 1024 \
                   --rounds 5 --json 2>&1)
        rec tensor "$cfg" gmacs_execute "$out"

        out=$("$TS" --lib "$LIBTPU_PATH" --tensor-v --difficulty 16 \
                   --n 1024 --rounds 5 --json 2>&1)
        rec tensor-v "$cfg" gmacs_execute "$out"
    done <<'EOF'
baseline|
critpath|--xla_jf_critical_path_scheduler=true
vliw|--xla_jf_vliw_scheduler=true
opt3|--xla_optimization_level=3
llvm-emit|--xla_llvm_isa_emitter=true --xla_llvm_isa_emitter_force=true
EOF
}

if [ "${RENDER_ONLY:-0}" = "1" ]; then
    [ -s "$ROWS" ] || { echo "RENDER_ONLY=1 but $ROWS is empty" >&2; exit 1; }
else
    sweep
fi

python3 - "$ROWS" <<'PY'
import sys

rows = [l.rstrip("\n").split("|") for l in open(sys.argv[1]) if l.strip()]
cfgs, tiers, cell = [], [], {}
for tier, cfg, key, val, extra in rows:
    if cfg not in cfgs:
        cfgs.append(cfg)
    if tier not in tiers:
        tiers.append(tier)
    cell[(tier, cfg)] = (val, extra)

unit = {"sha256": "MHash/s", "hashx": "MHash/s", "argon2id": "H/s",
        "tensor": "GMAC/s", "tensor-v": "GMAC/s"}

w = 14
print()
print("tier".ljust(w) + "".join(c.rjust(20) for c in cfgs))
print("".ljust(w) + "".join(("-" * min(len(c), 18)).rjust(20) for c in cfgs))
for tier in tiers:
    line = tier.ljust(w)
    for cfg in cfgs:
        val, extra = cell.get((tier, cfg), ("-", "-"))
        if val == "-":
            line += extra[:18].rjust(20)
        elif cfg == "baseline":
            line += ("%s %s" % (val, unit.get(tier, ""))).rjust(20)
        else:
            base, _ = cell.get((tier, "baseline"), ("-", "-"))
            try:
                d = (float(val) / float(base) - 1.0) * 100.0
                line += ("%s (%+.1f%%)" % (val, d)).rjust(20)
            except (ValueError, ZeroDivisionError):
                line += ("%s" % val).rjust(20)
    print(line)
print()
print("baseline is absolute; the other cells are the same number with the")
print("percentage versus baseline. A cell holding a sentence is a flag libtpu")
print("refused at init. Compile times are in %s." % sys.argv[1])
PY

cat <<'TXT'

Reading it:
  * sha256 / hashx / tensor / tensor-v are throughput-bound, so a codegen win
    shows up directly; vliw and llvm-emit are the likely candidates there.
  * argon2id is one dependent chain at ~7 cycles per vector op, and it is
    recompiled per flag set here, so a scheduler that actually reshapes the
    chain would show up too. "identical" is the expected answer and the point
    of measuring it.
  * run-to-run noise on these tiers is ~1% (Argon2id's windows are flat to
    0.006%, sha256/hashx are a few ms per run), so read anything inside +-1%
    as "no change" rather than as a win.
TXT
