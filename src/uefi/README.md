# Anubis on bare metal (UEFI)

A solver that runs **before any operating system**: the firmware loads this PE
image, it brings up `EFI_TCP4` on its own, accepts JSON frames, scans nonces on
the CPU (all APs, via `EFI_MP_SERVICES_PROTOCOL`) and answers with the hit and
the time it took.

```
libanubis_solver.so  -- TCP, JSON  -->  anubis.efi  -->  SHA-256 on the APs
```

## Build

```bash
make            # -> anubis.efi (~98 KB PE32+)
make esp        # stage it as \EFI\BOOT\BOOTX64.EFI on a directory
make run        # boot it in QEMU with OVMF
```

Needs `gnu-efi` (Debian: `apt install gnu-efi`) and, for `make run`, `qemu-system-x86`
plus `ovmf`.

## Protocol

No HTTP: a firmware has no room for a parser and no need for one. Each frame is
a 4-byte big-endian length followed by that many bytes of JSON. One connection
at a time; the client can pipeline frames.

```json
{"op":"caps"}
  -> {"ok":true,"caps":{"model":"...","kernel":"sha-ni","cpus":8,"solved":3}}

{"op":"ping"}
  -> {"ok":true,"pong":true}

{"op":"scan","algo":"sha256","challenge":"<hex>","difficulty":24,
 "start":0,"count":2000}
  -> {"ok":true,"hit":true,"algo":"sha256","kernel":"sha-ni","nonce":1374,
      "attempts":1375,"us":91,"cycles":243000,"threads":8,"digest":"..."}
  -> {"ok":true,"hit":false,"algo":"sha256","kernel":"sha-ni",
      "attempts":2000,"us":88,"cycles":240000,"threads":8}
```

`us`/`cycles` come from the TSC, calibrated at boot against a 10 ms firmware
stall; the clock starts at the first nonce and stops on the hit, so it is the
solve time, not the network time.

## What it knows about itself

1. **CPUID at boot**: `SHA-NI`, `AVX2`, `AVX-512F/VL` (plus the XCR0 check, so an
   OS-less boot does not execute AVX because a bit was set), AES.
2. **`\anubis-caps.json` on the EFI System Partition**: the profile and a solved
   counter, written at first boot and reused next time when the CPU matches.
3. **MP Services**: processor count, and work handed out with a locked add, so
   the BSP and every AP pull chunks from one queue with no per-CPU arguments.

## Status

| piece | state |
|---|---|
| builds a PE32+ image | yes |
| CPUID ladder | verified on the host (`sha_ni=1 avx2=1 avx512f=0 -> sha-ni`) |
| SHA-256 + Anubis nonce order (`use_le` when the last challenge byte >= 0x80) | verified against the NIST vector |
| TSC timing in the reply | implemented |
| MP Services scan | implemented, not yet run on hardware |
| boot under OVMF | **not yet**: image is accepted and jumps to `0xB0000` |

Two things this build already taught us, which are worth keeping in the Makefile
comments: gnu-efi's sample `objcopy` list omits `.rodata`, and an image without
it prints nothing at all; and the sample `crt0` faults on this toolchain. Both
are documented where they bit.
