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
| `argon2_pjrt.c` | the `argon2id` tier: the 19 MiB Argon2id fill as StableHLO, lanes are independent nonces |
| `colab_argon2.sh` | build + host check + fill selftest + throughput, from one Colab cell |
| `pjrt_common.h` | host SHA-256, hex and clock, shared by the clients |
| `kaggle/` | a Kaggle TPU kernel that builds and runs the probe (Kaggle's CLI accepted the TPU accelerator but attached no TPU; Colab did) |
| `tpu_backend.c` | the same search as library code: no stdout, no `exit()`, one cached executable per challenge, `anubis_tpu_solve()` |
| `tpu_backend_selftest.c` | the graceful path without libtpu (`make test-tpu`) |

## TPU, VPU, or "the other VPU"?

The names collide, so, briefly:

- **The VPU inside a TPU.** Every TensorCore is MXU(s) + a **vector unit** +
  a scalar unit ([Cloud TPU architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm));
  TPU 8t's "more balanced vector processing" is that block. It is not a
  separately addressable device: PJRT exposes the TPU as a whole, StableHLO is
  the only input, and XLA decides per op whether it lands on an MXU or the
  vector unit. `sha256_pjrt.c` is 64 rounds of elementwise integer ops and no
  matmuls - which is exactly the vector unit's job - so this directory already
  runs on the VPU; it just cannot name it, and there is no "VPU plugin" to
  load instead of libtpu.
- **Google's Video Processing Unit (VCU).** A separate, fixed-function video
  transcoder: the 2018 "VPU" from Google's custom-compute history
  ([blog](https://cloud.google.com/blog/topics/systems/the-past-present-and-future-of-custom-compute-at-google)),
  then Argos (2021: two ASICs per PCIe card, ten encoder cores each, H.264 /
  VP9 / AV1). It lives inside Google's datacenters, with no public driver, no
  PJRT plugin and no general-purpose compute API - there is nothing here to
  port a hash onto.

So this backend is TPU-only by construction, and "run sha256 on the Google VPU"
is already what the `sha256` tier does, through the TPU compiler. The video
VPU is not reachable, and Intel's VPU is a different device entirely
([docs/vpu.md](../../../docs/vpu.md)).

## Which unit does the work

| tier | HLO | unit |
|---|---|---|
| `tensor` (`tensor_pjrt.c`) | `stablehlo.dot_general` (S8 x S8 -> S32) | MXU |
| `sha256` (`sha256_pjrt.c`) | elementwise `tensor<Nxi32>` (add / xor / and / shift / compare / convert), no dot | vector unit |

So the 343 MHash/s `sha256` figure is a vector-unit measurement, and the `tensor`
figure is an MXU one. Neither names the unit: PJRT exposes the TPU as a whole
and XLA schedules the ops.

HashX would be a vector-unit job too, and a better fit than on the GPUs in this
tree: its program is fixed per challenge, so the whole (up to) 512-instruction
program can be unrolled into StableHLO with the 8 virtual registers as SSA
values - no indexed register file, which is exactly what forces the CUDA and
Vulkan interpreters into local memory (`r[dst]` is data there). Two costs have
to be paid instead:

- HashX registers are u64 and the TPU lane is 32-bit, so every register is an
  i32 pair and mul / umulh / smulh / rotl are emulated in 32-bit ops, the way
  `src/vulkan/hashx.comp` does with `uvec2`. That is several vector ops per
  HashX instruction, so expect roughly 2k-6k i32 ops per hash against ~640 for
  the SHA-256 tier.
- `INSTR_BRANCH` jumps back to the last `INSTR_TARGET` at most once per lane,
  so the loop body is emitted twice and the second pass is blended in with a
  per-lane `select` on the branch condition - branch-free, and exact.

Implemented in [`hashx_pjrt.c`](hashx_pjrt.c), compiled once per challenge and
measured on one Colab TPU device: **75.3 MHash/s** (2,097,152 hashes in 27.8 ms,
compile 14.6 s), 7.8x one CPU thread and ~65% of the two-T4 PTX interpreter.
The serialized executable is a `xla.TpuExecutableProto` with a
`jellyfish.TpuCompilationEnvironment`. See
[`SPEED.md`](../../../SPEED.md#hashx-on-a-tpu-colab-pjrt).

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

## Argon2id: 19 MiB of serial memory on a systolic array

`argon2_pjrt.c` runs the gate's `argon2id` tier (m=19456 KiB, t=2, p=1,
32-byte tag, salt = the nonce) with the fill on the device.

The matrix is one int32 tensor, `lanes * m * 256`: each 1 KiB block is 256
int32, laid out as lo/hi pairs of its 128 u64, so a block is already
little-endian on the wire. A `stablehlo.while` walks the 38,912 block
computations of a challenge; each iteration

1. gathers the previous block and, from its first word, computes
   `index_alpha` and the reference block (u64 as an i32 pair, the signed
   `mulhi32` trick from `sha256_pjrt.c`; no divides, no unsigned compares),
2. runs the 16 BLAKE2 rounds of `fill_block` (128 `G`s, `fBlaMka`) on 128 u64
   SSA values, unpacked from and repacked into the block tensor,
3. scatters the new block back into the matrix.

Two things keep the graph small enough to compile:

- **The addressing schedule is not data.** Argon2id's first pass uses
  data-independent addressing for the first half of memory, so those
  pseudo-random words are generated on the host and uploaded once - they only
  depend on `(m, t)`, not on the challenge or the nonce.
- **lanes are independent nonces.** All lanes follow the same schedule, so one
  iteration advances one block in every lane at once; the memory cost is
  `lanes * 19 MiB`, which is why `--lanes` defaults low and the device limit,
  not the loop, is what bounds it.

H0, B0 and B1 are BLAKE2b, so they stay on the host (68 bytes of BLAKE2b per
nonce), and so does the final 1 KiB tag. Unlike `hashx_pjrt`, the module does
not depend on the challenge: one compiled executable serves every challenge on
that device.

```sh
./argon2_pjrt --unit                 # u64 pair primitives, no device
./argon2_pjrt --hosttest             # the host Argon2id vs fixed vectors
./argon2_pjrt --lib $LIBTPU --selftest   # the graph vs the host reference
./argon2_pjrt --lib $LIBTPU --difficulty 8 --lanes 8
./argon2_pjrt --lib $LIBTPU --lanes 32 --difficulty 32 --bench-hashes 64 --json
```

Measured on one v5e-1 chip at m=19456, t=2, 32 lanes: **9.38 H/s** (192 nonces
in 6 windows, 20.5 s, steady state 9.379, per-window device time 3411.6-3411.8
ms). That is 3.8x slower than one CPU core running the reference engine, and
the point of the number is that the 819 GB/s HBM is not reachable: a window
moves 4.86 GiB in 3.41 s, 1.42 GiB/s or 0.19% of the HBM, because the next
block cannot start until the previous one is written. The compile is 11.9 s
once per `(m, t, lanes)` and serializes to 20.7 MB; the host side is 4.1 ms of
input blocks and 0.5 ms of tags per window. `--stress SECONDS` reports the
steady-state rate with the first window discarded, and `--load FILE` reuses a
serialized executable so no compile is in the measurement.

`--dump DIR` writes the module plus its inputs and the expected tag, so the
same StableHLO can be compiled and run on a CPU XLA client (jaxlib's MLIR
bindings parse it; `jaxlib.xla_client.make_cpu_client()` executes it) before
spending time on a TPU. `--dump-all` returns the whole matrix instead of the
last block, and `--filltest` builds a single fill, both for bisecting a
mismatch.

## Build and run

```bash
cc -O2 -I. -o tensor_pjrt tensor_pjrt.c -ldl
./tensor_pjrt --lib /path/to/libtpu.so --difficulty 4,8,12,16 --rounds 5
./tensor_pjrt --lib /path/to/libtpu.so --n 4096 --rounds 5 --no-verify --json

make sha256_pjrt
./sha256_pjrt --lib /path/to/libtpu.so --difficulty 20 --window 262144 --json
```
