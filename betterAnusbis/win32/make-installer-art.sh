#!/bin/sh
# make-installer-art.sh — regenerate the branded MSI dialog bitmap.
#
# The Windows installer's license page is the WixUI_Minimal WelcomeEulaDlg
# dialog. Its left-hand art is a 547x312 DIB (a 410x270 dialog-unit window at
# 4/3 pixels per unit); the dialog is widened by build-windows.sh so the title
# "Please read the betterAnusbis License Agreement" fits on one line. This
# script rebuilds that bitmap from the anti-anti-ai mascot: the art sits in the
# left 164 pixels, the right side stays white for the license text.
#
# The generated file is committed because the Windows build must not depend on
# ImageMagick (or on the mascot being present at the exact size). Run this only
# when the mascot or the dialog geometry changes.
#
#   ./make-installer-art.sh        # needs ImageMagick (convert/magick)
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)
mascot=$root/mascot/anti-anti-ai.png
out=$here/installer/dlgbmp.bmp

[ -f "$mascot" ] || { echo "$0: missing $mascot" >&2; exit 1; }
if command -v magick >/dev/null 2>&1; then
    conv=magick
else
    conv=convert
fi
command -v "$conv" >/dev/null 2>&1 || {
    echo "$0: need ImageMagick (convert or magick)" >&2
    exit 1
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# white dialog background, art cropped to fill the left strip exactly
"$conv" -size 547x312 xc:white "$tmp/bg.png"
"$conv" "$mascot" -resize '164x312^' -gravity center -crop 164x312+0+0 +repage \
    "$tmp/art.png"
# 256 colours keeps the bitmap at ~68 KiB and is what WiX's own bitmaps use
"$conv" "$tmp/bg.png" "$tmp/art.png" -geometry +0+0 -composite \
    -colors 256 -type Palette -define bmp:format=bmp3 "$out"

echo "wrote $out"
