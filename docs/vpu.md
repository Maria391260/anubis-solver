# The VPU: what it is, and why `sha256` is not ported to it

> "VPU" is overloaded. This page is about **Intel's** VPU (the Core Ultra NPU).
> The other two are not devices you can target: the **vector unit inside a
> Google TPU** (you reach it only through PJRT, which calls the whole chip a
> TPU) and **Google's Video Processing Unit / Argos VCU** (a fixed-function
> video transcoder internal to Google). Both are covered in
> [`src/google/tpu/README.md`](../src/google/tpu/README.md).

"VPU" is the name Intel's kernel driver still uses for the neural accelerator
that ships in Core Ultra (Meteor Lake and later) client CPUs. The stack is:

| layer | what you see |
|---|---|
| kernel | `intel_vpu.ko`, `/dev/accel/accel0`, class `accel` |
| user space | `libze_intel_npu.so.1` (formerly `libze_intel_vpu.so.1`), a Level Zero driver |
| compute model | **graphs**: `zeGraphCreate(blob)` + `zeGraphExecute`; the blob is produced by Intel's Driver Compiler (`vcl*` API) from an IR network |
| API traces | Level Zero's `ZE_INIT_FLAG_VPU_ONLY` and `ZE_DEVICE_TYPE_VPU` in `ze_api.h` |

That last row is where the confusion starts: the VPU speaks Level Zero, so
`src/intel/levelzero/` looks like a template. It is not one. On the iGPU
`zeModuleCreate` compiles **our SPIR-V**; on the VPU the Level Zero surface
carries **pre-compiled graph blobs**. There is no public SPIR-V → VPU path, so
`sha256_pow.spv` and the OpenCL kernels in this tree cannot be pointed at it
the way `levelzero_sha256.c` points at the iGPU.

The older, unrelated VPU is the Movidius Myriad X (Neural Compute Stick 2).
OpenVINO cancelled MYRIAD/HDDL support in 2023.0 — 2022.3 is the last release
that runs it — and its SHAVE toolchain is not public either. Between the two,
there is no supported "run my kernel on a VPU" interface.

## The probe

`tools/vpu_probe.c` is first contact, the way `src/google/tpu/pjrt_probe.c` was
for the TPU. It answers the three questions that decide whether a port is
possible on the machine it runs on:

1. does Level Zero see a device through `zeInit(ZE_INIT_FLAG_VPU_ONLY)`?
2. what does `zeDeviceGetProperties` say it is (`type == 5`,
   `ZE_DEVICE_TYPE_VPU`)?
3. does the driver accept our SPIR-V (`zeModuleCreate(IL_SPIRV)`) and the
   `sha256_pow` entry point (`zeKernelCreate`)?

```bash
make vpu-probe
./build/vpu-probe                  # add --spv FILE, --kernel NAME, --all
```

On the development machine (i7-1260P, Alder Lake — the kernel ships
`intel_vpu.ko` but there is no VPU silicon, no `/dev/accel`, and no Level Zero
driver installed):

```
kernel side
  /dev/accel/                        no
  /sys/class/accel/                  yes
  /sys/module/intel_vpu              no
user space
  libze_loader.so.1                  yes
  libze_intel_npu.so.1               no
  libze_intel_vpu.so.1               no
level zero
  zeInit(ZE_INIT_FLAG_VPU_ONLY)      0x78000001 (ZE_RESULT_ERROR_UNINITIALIZED)
result: no Level Zero driver answered - no VPU here
```

Exit codes: `0` VPU found (and SPIR-V accepted, when `--spv` was given), `2` no
VPU, `3` module rejected, `4` kernel entry point rejected, `1` usage/loader.

## What a port would take

1. Run `make vpu-probe` on a Core Ultra machine with
   [`intel-npu-driver`](https://github.com/intel/linux-npu-driver) installed
   and the firmware package present. That is the only machine on which the
   answer exists.
2. If `zeModuleCreate(IL_SPIRV)` succeeds, the port is mechanical: a copy of
   `src/intel/levelzero/levelzero_sha256.c` that initialises with
   `ZE_INIT_FLAG_VPU_ONLY` and filters `ZE_DEVICE_TYPE_VPU`, then the same
   dispatch, verification and fallback rules as every other backend.
3. If it fails — the expected answer, because the NPU driver is a graph
   driver — the only remaining route is to express SHA-256 as an OpenVINO IR
   graph and let the NPU plugin compile it. Opset 15 has the bitwise ops and
   the shifts this needs, but the NPU compiler's coverage for a
   64-round bit-serial workload is unknown, and it would be a Python/OpenVINO
   experiment, not a C backend in this `.so`. Even if it compiled, a bit-serial
   hash on an NPU built for dense matrix work would lose to the CPU by orders
   of magnitude, so it would be a measurement, not a gate backend.

Either way the useful next step is the same: run the probe on real VPU
hardware. Until then `src/google/tpu/` remains the only accelerator path in
this tree where a non-GPU device runs this project's own kernels.
