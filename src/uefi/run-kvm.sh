#!/bin/sh
# Boot anubis.efi under OVMF + KVM, stream the console to a window, and pop a
# notification when the run ends. This is the harness the EFI work lives in:
# rebuild, stage, boot, watch, report.
#
#   ./run-kvm.sh            # 45 s, 4 vCPUs, port 8642
#   SECS=90 SMP=8 ./run-kvm.sh
set -u
cd "$(dirname "$0")"

SECS=${SECS:-45}
SMP=${SMP:-4}
PORT=${PORT:-8642}
LOG=${LOG:-/tmp/anubis-uefi.log}
DBG=${DBG:-/tmp/anubis-uefi.dbg}
VARS=/tmp/anubis-vars.fd

pkill -x qemu-system-x86 2>/dev/null  # a stale run owns port 8642 and /dev/kvm
sleep 1
make -s || exit 1

if [ ! -f esp.img ]; then
    dd if=/dev/zero of=esp.img bs=1M count=24 status=none
    mkfs.vfat -F 16 -n ANUBIS esp.img >/dev/null
    mmd -i esp.img ::/EFI ::/EFI/BOOT
fi
mcopy -o -i esp.img anubis.efi ::/EFI/BOOT/BOOTX64.EFI
cp /usr/share/OVMF/OVMF_VARS_4M.fd "$VARS"
: > "$LOG"
: > "$DBG"

xterm -geometry 120x38 -fa Monospace -fs 11 \
      -title "anubis-uefi: live boot log" \
      -e sh -c "cd '$PWD' && ./logwatch.sh '$LOG' '$DBG'" &
TERM_WIN=$!

printf '=== anubis-uefi boot %s (smp=%s port=%s) ===\n' \
       "$(date +%H:%M:%S)" "$SMP" "$PORT" >> "$LOG"

timeout "$SECS" qemu-system-x86_64 -M q35 -enable-kvm -cpu host -smp "$SMP" \
    -m 512 -display none -serial "file:$LOG" -debugcon "file:$DBG" \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
    -drive if=pflash,format=raw,file="$VARS" \
    -netdev "user,id=n0,hostfwd=tcp:127.0.0.1:$PORT-:$PORT" \
    -device virtio-net-pci,netdev=n0 \
    -drive file=esp.img,format=raw,if=virtio >>"$LOG" 2>&1

printf '=== run finished %s ===\n' "$(date +%H:%M:%S)" >> "$LOG"

saw_app=$(grep -ac 'anubis-uefi' "$LOG" 2>/dev/null || true)
saw_dbg=$(grep -ac 'anubis' "$DBG" 2>/dev/null || true)
saw_exc=$(strings "$LOG" 2>/dev/null | grep -ac 'Exception Type' || true)
started=$(strings "$LOG" 2>/dev/null | grep -ac 'starting Boot0001' || true)

paplay /usr/share/sounds/freedesktop/stereo/complete.oga >/dev/null 2>&1 &
zenity --info --width=560 --title="anubis-uefi run" --text="\
started by BdsDxe: $started
app lines on serial: $saw_app
bytes of 0xE9 debug: $saw_dbg
exceptions: $saw_exc

log: $LOG
" >/dev/null 2>&1 &

wait "$TERM_WIN" 2>/dev/null
exit 0
