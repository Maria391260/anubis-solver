# RELEASING.md

How the binaries on the release page are produced. Two architectures, one
machine, no containers: a Debian 12 sysroot for each, plus qemu for the ARM one.

The reason for the sysroots is not paranoia. This host's gcc 15 defaults to
`-std=gnu23`, and glibc headers redirect `strtol` and friends to `__isoc23_*`
under `_GNU_SOURCE`, which jacks the minimum glibc up to **2.38**. A binary that
only runs on Ubuntu 24.04 and newer is not a release, it is a souvenir. Building
against bookworm's glibc 2.36 puts the floor at **2.34**, which covers Debian 12,
Ubuntu 22.04+, Raspberry Pi OS bookworm and anything newer.

Keep `-std=gnu17` in both parts of the tree anyway: it is the C standard the
code is written in, and leaving it implicit means the compiler picks.

## The sysroots

Both are Debian 12 (bookworm), unpacked with `dpkg-deb -x`, no root needed.

```sh
# amd64
for d in libc6-dev linux-libc-dev libssl-dev libssl3; do
    # take the bookworm amd64 .deb for $d from deb.debian.org
    dpkg-deb -x "$d"*.deb /tmp/sys64
done
printf '#!/bin/sh\nexec gcc --sysroot=/tmp/sys64 "$@"\n' > /tmp/cc64 && chmod +x /tmp/cc64

# arm64
for d in libc6-dev linux-libc-dev libssl-dev libssl3 libc6; do
    # take the bookworm arm64 .deb for $d
    dpkg-deb -x "$d"*.deb /tmp/armsysroot
done
cat > /tmp/cc-arm <<'EOF'
#!/bin/sh
INC=$(aarch64-linux-gnu-gcc -print-file-name=include)
exec aarch64-linux-gnu-gcc --sysroot=/tmp/armsysroot -nostdinc \
  -isystem /tmp/armsysroot/usr/include \
  -isystem /tmp/armsysroot/usr/include/aarch64-linux-gnu \
  -isystem "$INC" "$@"
EOF
chmod +x /tmp/cc-arm
```

`-nostdinc` is not optional for the cross build: the toolchain keeps its own
glibc headers in `/usr/aarch64-linux-gnu/include`, outside any sysroot, and
without `-nostdinc` they win and put glibc 2.38 back.

## The builds

```sh
# amd64: CPU kernels only, which is what the release ships
make clean && make -j"$(nproc)" CC=/tmp/cc64 CUDA=0 VULKAN=0 L0=0 OPENCL=0
cd betterAnusbis && make CC=/tmp/cc64
cd ..

# arm64
make clean && make ARCH=aarch64 CC=/tmp/cc-arm NATIVE= \
    CUDA=0 VULKAN=0 L0=0 OPENCL=0 BUILD=/tmp/arm/obj libanubis_solver.so
cd betterAnusbis && /tmp/cc-arm -O2 -g -Wall -Wextra -pthread -std=gnu17 \
    -I../include -o /tmp/arm/betterAnusbis betterAnusbis.c \
    -L/tmp/armsysroot/usr/lib/aarch64-linux-gnu -L.. \
    -lanubis_solver -lssl -lcrypto -Wl,-rpath,'$ORIGIN'
```

`--sysroot` changes where the compiler looks, not where the linker puts things;
`-Wl,-rpath,'$ORIGIN'` is what lets the pair work from any directory.

## Checking the floor before shipping

```sh
objdump -T betterAnusbis/betterAnusbis | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1
aarch64-linux-gnu-objdump -T /tmp/arm/betterAnusbis | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1
```

Both must say `GLIBC_2.34`. If either says `GLIBC_2.38`, a `__isoc23_*` symbol
got in, which means the headers are the host's and not the sysroot's.

## Running the ARM build without ARM hardware

```sh
# qemu-user-static for the host (amd64), arm64 libc already in the sysroot
curl -O http://deb.debian.org/debian/pool/main/q/qemu/qemu-user-static_*_amd64.deb
dpkg-deb -x qemu-user-static_*_amd64.deb /tmp/qtools

mkdir -p /tmp/armrun/opt/anubis /tmp/armrun/usr/lib
cp -r /tmp/armsysroot/lib /tmp/armrun/
cp -r /tmp/armsysroot/usr/lib/aarch64-linux-gnu /tmp/armrun/usr/lib/
cp /tmp/arm/betterAnusbis libanubis_solver.so /tmp/armrun/opt/anubis/

cd /tmp/armrun/opt/anubis
/tmp/qtools/usr/bin/qemu-aarch64-static -L /tmp/armrun ./betterAnusbis --version
/tmp/qtools/usr/bin/qemu-aarch64-static -L /tmp/armrun ./betterAnusbis \
    --target http://127.0.0.1:9 --listen 127.0.0.1:8713 --difficulty 4
curl -i http://127.0.0.1:8713/     # 200, and a challenge page
```

That is a link check and a start check, not a benchmark: qemu is a translator
and the numbers it produces mean nothing. It does prove the ARM pair loads,
listens and answers, which is what a release artifact has to do.

## What goes on the release

Per architecture: a tarball with both binaries plus LICENSE, ATTRIBUTION.md and
MIGRATE.md, and the two raw files. The x86-64 raw files keep the canonical names
(`betterAnusbis`, `libanubis_solver.so`); the aarch64 ones carry a suffix, and
`libanubis_solver-aarch64.so` has to be renamed to `libanubis_solver.so` next to
the binary, because that is the name in `DT_NEEDED`.

The GPU backends are compiled per machine (CUDA, Level Zero, Vulkan, OpenCL are
dlopen-based and need local drivers and headers), so the release build is CPU
only, and says so.
