#!/bin/sh
# Stream both firmware consoles of a run into one window: the serial console
# (where BdsDxe and our own anubis_console_write go) and the QEMU debug port
# 0xE9 (where our first byte goes, before any firmware protocol is touched).
#
#   ./logwatch.sh [/tmp/anubis-uefi.log] [/tmp/anubis-uefi.dbg]
LOG=${1:-/tmp/anubis-uefi.log}
DBG=${2:-/tmp/anubis-uefi.dbg}

tail -n +1 -f "$LOG" 2>/dev/null | stdbuf -oL awk '{ printf "%s  %s\n", strftime("%H:%M:%S"), $0; fflush() }' &
tail -n +1 -f "$DBG" 2>/dev/null | stdbuf -oL awk '{ printf "%s  DBG %s\n", strftime("%H:%M:%S"), $0; fflush() }' &
wait
