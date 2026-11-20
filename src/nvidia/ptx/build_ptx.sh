#!/bin/bash
# Build the hand-written PTX kernels to cubin with ptxas.
#
# Usage: ./build_ptx.sh [sm_75|sm_80|sm_86|sm_90] [ILP=1|2|4] [MAXRREG=0]
#
# sha256_pow.ptx is compiled as-is. sha256_opt.ptx is generated for the
# requested ILP by gen_opt_ptx.py (the checked-in file is the ILP=2 output),
# and sha256_opt_loop_ilp<N>.cubin is the grid-stride variant used when
# ANUBIS_PTX_LOOP > 1 (one thread hashes several nonce groups). hashx_interp.ptx
# is generated too.
#
# MAXRREG caps registers per thread (-maxrregcount). The kernel needs 42 at
# ILP=1, which rounds up to a 48-register allocation and costs occupancy on
# Turing: 32 gives the full 64 warps/SM with no spills and the same instruction
# count, so it is worth an A/B on real hardware.
#
# ENV: PTXAS (assembler), PTX_MAXRREG (same as the 3rd arg), PTX_OUT_DIR
# (where the cubins go, default ./cubin -- kaggle_ab.py builds one directory
# per configuration).
set -euo pipefail

PTXAS="${PTXAS:-ptxas}"
ARCH="${1:-sm_75}"
ILP="${2:-2}"
MAXRREG="${3:-${PTX_MAXRREG:-0}}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${PTX_OUT_DIR:-$SRC_DIR/cubin}"
mkdir -p "$OUT_DIR"

PTXAS_OPTS=(-arch="$ARCH")
if [ "$MAXRREG" != "0" ] && [ -n "$MAXRREG" ]; then
    PTXAS_OPTS+=("-maxrregcount=$MAXRREG")
fi

echo "Building PTX for $ARCH (ILP=$ILP, maxrreg=$MAXRREG) using $PTXAS"

echo "Compiling sha256_pow.ptx..."
"$PTXAS" "${PTXAS_OPTS[@]}" -o "$OUT_DIR/sha256_pow.cubin" "$SRC_DIR/sha256_pow.ptx"

echo "Generating and compiling sha256_opt.ptx (ILP=$ILP)..."
TMP="$OUT_DIR/sha256_opt_ilp${ILP}.ptx"
python3 "$SRC_DIR/gen_opt_ptx.py" --ilp "$ILP" > "$TMP"
"$PTXAS" "${PTXAS_OPTS[@]}" -o "$OUT_DIR/sha256_opt_ilp${ILP}.cubin" "$TMP"
rm -f "$TMP"

echo "Generating and compiling sha256_opt_loop.ptx (ILP=$ILP, grid-stride)..."
TMP="$OUT_DIR/sha256_opt_loop_ilp${ILP}.ptx"
python3 "$SRC_DIR/gen_opt_ptx.py" --ilp "$ILP" --loop > "$TMP"
"$PTXAS" "${PTXAS_OPTS[@]}" -o "$OUT_DIR/sha256_opt_loop_ilp${ILP}.cubin" "$TMP"
rm -f "$TMP"

echo "Generating and compiling sha256_opt_imad_ilp${ILP}.cubin (FMA-pipe adds)..."
TMP="$OUT_DIR/sha256_opt_imad_ilp${ILP}.ptx"
python3 "$SRC_DIR/gen_opt_ptx.py" --ilp "$ILP" --imad-add > "$TMP"
"$PTXAS" "${PTXAS_OPTS[@]}" -o "$OUT_DIR/sha256_opt_imad_ilp${ILP}.cubin" "$TMP"
rm -f "$TMP"

echo "Generating and compiling hashx_interp.ptx..."
python3 "$SRC_DIR/gen_hashx_ptx.py" > "$OUT_DIR/hashx_interp.ptx"
"$PTXAS" "${PTXAS_OPTS[@]}" -o "$OUT_DIR/hashx_interp.cubin" "$OUT_DIR/hashx_interp.ptx"
rm -f "$OUT_DIR/hashx_interp.ptx"

echo "Done. Cubins in $OUT_DIR:"
ls -la "$OUT_DIR/"

if command -v nvdisasm &>/dev/null; then
    for k in "sha256_opt_ilp${ILP}" "sha256_opt_loop_ilp${ILP}" "sha256_opt_imad_ilp${ILP}"; do
        if [ -f "$OUT_DIR/$k.cubin" ]; then
            echo ""
            echo "$k.cubin: $(nvdisasm -c "$OUT_DIR/$k.cubin" 2>/dev/null \
                | grep -cE '^[[:space:]]+/\*[0-9a-f]{4}\*/') SASS instructions, $(nvdisasm -c "$OUT_DIR/$k.cubin" 2>/dev/null \
                | sed -n 's/.*SHI_REGISTERS=\([0-9]*\).*/\1/p' | head -1) registers"
        fi
    done
fi
