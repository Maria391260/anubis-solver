# `tensor` on a TPU, through the PJRT C API

This is the measurement that answers "what would the tier do on TPU silicon",
not a supported backend of the gate: the gate's browser paths are WebNN, wasm
and JavaScript, and `tensor` verifies on the CPU. What is here is a complete C
client for libtpu.

| file | what it is |
|---|---|
| `pjrt_c_api.h` | vendored from `openxla/xla` (API 0.118); TPU images ship the plugin but not the header |
| `pjrt_probe.c` | first contact: `dlopen`, `GetPjrtApi`, plugin init, client, devices, compile a dot |
| `tensor_pjrt.c` | the real thing: derive A/B, upload as S8, execute, copy back, digest, benchmark |
| `kaggle/` | a Kaggle TPU kernel that builds and runs the probe (Kaggle's CLI accepted the TPU accelerator but attached no TPU; Colab did) |

## What it produces

At the tier's sizes the dot is launch-bound (~0.2 ms regardless of N), and at
N=4096 the same code measures 180 TMAC/s, about 91% of a v5e chip's int8 spec.
The end-to-end number is dominated by the 4-64 MiB copy back and the host
SHA-256, not by the GEMM. Numbers, tables and caveats:
[`SPEED.md`](../../../SPEED.md#tpu-the-tensor-tier-through-pjrt-no-python).

## Three things that are not obvious

1. **Initialize before anything else.** libtpu needs `PJRT_Plugin_Initialize`
   first; skip it and the crash is a check-failure inside libtpu
   (`InitGoogle() has not finished yet`), not a PJRT error.
2. **One process owns the chip.** If JAX imported earlier in the same runtime,
   libtpu refuses with "already in use by process with pid N". Restart the
   runtime, or run this before any Python TPU work.
3. **Empty compile options mean zero replicas.** `Requested total cores: 0` is
   the TPU backend declining to invent a device assignment; six bytes of
   `CompileOptionsProto{ executable_build_options{ num_replicas=1,
   num_partitions=1 } }` (field 3 -> 4, 5) fix it.

The dot itself is `stablehlo.dot_general` from `tensor<NxNxi8>` to
`tensor<NxNxi32>`. StableHLO's C20 says the result element type must match the
operands, but the TPU backend accepts the widening form and runs it on the MXU;
that is what makes an int8 tier measurable here at all.

## Build and run

```bash
cc -O2 -I. -o tensor_pjrt tensor_pjrt.c -ldl
./tensor_pjrt --lib /path/to/libtpu.so --difficulty 4,8,12,16 --rounds 5
./tensor_pjrt --lib /path/to/libtpu.so --n 4096 --rounds 5 --no-verify --json
```
