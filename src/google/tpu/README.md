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
| `sha256_pjrt.c` | the `sha256` tier: batched nonces, 343 MHash/s at 262,144 lanes |
| `pjrt_common.h` | host SHA-256, hex and clock, shared by the clients |
| `kaggle/` | a Kaggle TPU kernel that builds and runs the probe (Kaggle's CLI accepted the TPU accelerator but attached no TPU; Colab did) |
| `tpu_backend.c` | the same search as library code: no stdout, no `exit()`, one cached executable per challenge, `anubis_tpu_solve()` |
| `tpu_backend_selftest.c` | the graceful path without libtpu (`make test-tpu`) |

## Library ABI

`tpu_backend.c` is linked into `libanubis_solver.so` and its three entry points
are part of the public ABI, declared in `include/anubis-solver.h`:

```c
int anubis_tpu_available(void);
const char *anubis_tpu_info(void);
int anubis_tpu_solve(const uint8_t *challenge, uint32_t challenge_len,
                     int difficulty, int use_le, uint64_t max_nonce,
                     double budget_ms, uint32_t *out_nonce,
                     char *out_hex, double *out_ms);
```

They exist in every build (a `TPU=0` build links stubs that report
"unavailable"), matching `anubis_l0_solve` / `anubis_vulkan_solve`. Return
codes: `0` hit, `1` exhausted, `-1` backend failure, outputs untouched on
failure.

The core's dispatch (`anubis_solve_bits()`) is still the recommended path: it
applies `ANUBIS_TPU_MIN_WORK` (2^28 expected nonces by default) and falls back
to the CPU. A direct caller owns both decisions, and `anubis_tpu_verify()` does
not apply — `anubis_tpu_solve()` always derives the winning digest on the host.

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

## The sha256 tier

`sha256_pjrt` is the tier this hardware is actually good at: independent
one-block messages, no chaining. Window size dominates everything (see
[`SPEED.md`](../../../SPEED.md#tpu-the-sha256-tier-343-mhashs)); at 262,144
lanes it measures 343 MHash/s, ~43x the browser's wasm kernel. Three bugs cost
real Colab round trips and are worth recording because each was invisible in a
different way:

1. **Nonce byte position inside a big-endian word.** The block words are
   `b0<<24 | b1<<16 | b2<<8 | b3`, so the first nonce byte is the most
   significant one. Writing it to the least significant byte produced digests
   for a different message - caught by the CPU re-check, not by the device.
2. **`PJRT_Buffer_ToHostBuffer` copies the physical layout.** With
   `host_layout = NULL` it does not mean "row-major": the TPU had chosen
   dim-0-fastest for `tensor<Nx8xi32>`, so 32 bytes per lane arrived
   transposed. Returning a flat `tensor<N*8xi32>>` removed the choice.
3. **`comparison_type UNSIGNED` did not compare unsigned on this plugin.**
   `H0 < 2^(32-d)` therefore treated every negative word as a pass. The test is
   now `shift_right_logical(H0, 32-d) == 0`, where signedness cannot enter.

## Build and run

```bash
cc -O2 -I. -o tensor_pjrt tensor_pjrt.c -ldl
./tensor_pjrt --lib /path/to/libtpu.so --difficulty 4,8,12,16 --rounds 5
./tensor_pjrt --lib /path/to/libtpu.so --n 4096 --rounds 5 --no-verify --json

make sha256_pjrt
./sha256_pjrt --lib /path/to/libtpu.so --difficulty 20 --window 262144 --json
```
