# PTX kernels for Anubis SHA-256

Hand-written PTX for the two SHA-256 proof-of-work paths and for the HashX
interpreter. PTX is NVIDIA's documented intermediate ISA, so `ptxas` can
assemble these directly; unlike SASS there is no control-word/scheduler problem
(see `../sass/README.md`).

These are complete kernels: all 64 SHA-256 rounds, the message schedule,
the difficulty check and result reporting. They are not stubs.

## Files

| File | Description |
|---|---|
| `sha256_pow.ptx` | Both modes (`fast`/`slow` decimal nonce and `sha256` u32 nonce). Block + schedule in local memory. |
| `sha256_opt.ptx` | Mode 1 only, register-only message schedule, `ILP=2` by default. **Generated** -- see below. |
| `sha256_opt_loop.ptx` | The same kernel wrapped in a grid-stride loop, so a thread hashes many nonce groups. **Generated**. |
| `hashx_interp.ptx` | The HashX interpreter: SipHash expansion, the 11 opcodes, finalisation, leading-zero-bit test. **Generated**. |
| `gen_opt_ptx.py` | Generates `sha256_opt.ptx` for `--ilp 1|2|4`. |
| `gen_hashx_ptx.py` | Generates `hashx_interp.ptx`. |
| `build_ptx.sh` | Assembles both kernels to `cubin/` with `ptxas`. |
| `ptx_loader.c` | Driver-API loader (`cuModuleLoadData`) with the scalar ABI below, SHA-256 and HashX. |
| `hashx_shim.c` | ctypes wrapper around the CPU HashX reference (`../../hashx/hashx.c`). |
| `ptxsim.py` | Minimal PTX interpreter used only by the offline tests. |
| `test_ptx.py` | Executes both SHA-256 kernels and checks every digest against `hashlib`. |
| `test_hashx_ptx.py` | Executes the HashX kernel and checks every digest against `hashx_hash()`. |
| `kaggle_bench_ptx.py` | SHA-256 benchmark: builds, then reports exact MH/s and digests. |
| `kaggle_ab.py` | A/B harness: rebuilds per configuration, runs each in its own process, reports kernel-only GH/s and the SM clock. |
| `kaggle_bench_hashx_ptx.py` | HashX benchmark: same, plus the CPU reference rate and the speedup. |
| `kaggle_hashx/` | `kaggle kernels push` job that clones and runs the HashX benchmark on a T4. |
| `cubin/` | Built cubins. |

`sha256_opt.ptx` and `hashx_interp.ptx` are generated. Do not edit them;
change the generators and regenerate:

```bash
python3 gen_opt_ptx.py --ilp 2 > sha256_opt.ptx
python3 gen_opt_ptx.py --ilp 2 --loop > sha256_opt_loop.ptx
python3 gen_hashx_ptx.py > hashx_interp.ptx
```

`--imad-add` is experimental: it emits every register+register add as
`mad.lo.u32 d, a, %one, b` with `one` a kernel parameter fixed at 1, so ptxas
cannot fold it and the add issues as `IMAD` on the FMA pipe instead of `IADD3`
on the ALU pipe. SHA-256 is ALU-bound (the 637 shifts and 348 `LOP3` can only
run there), so this is the only lever left on IPC. It changes the kernel
signature (one extra `u32` before `results_ptr`), so cubins built with it need a
loader that passes that argument.

## Building

```bash
export PATH="/path/to/cuda/bin:$PATH"
cd src/nvidia/ptx
./build_ptx.sh sm_75 2          # T4: sm_75, ILP=2
# -> cubin/sha256_pow.cubin, cubin/sha256_opt_ilp2.cubin,
#    cubin/sha256_opt_loop_ilp2.cubin, cubin/hashx_interp.cubin
```

The third argument is a `-maxrregcount` cap (`0` = let ptxas choose), and
`PTX_OUT_DIR` redirects the output, which is what `kaggle_ab.py` uses to keep one
directory per configuration:

```bash
./build_ptx.sh sm_75 1 32        # 32 regs, 0 spills at ILP=1
PTX_OUT_DIR=/tmp/cfg ./build_ptx.sh sm_75 1
```

`ptxas` from the `nvidia-cuda-nvcc-cu12` wheel works if there is no toolkit:

```bash
PTXAS=$(python3 -c 'import nvidia.cuda_nvcc, os; print(os.path.join(os.path.dirname(nvidia.cuda_nvcc.__file__), "bin", "ptxas"))')
```

## Kernel ABI

Both entry points take flat scalar parameters and write hits through
`results_ptr`:

```
sha256_pow_kernel / sha256_opt_kernel(
    u32 mid0..mid7, u32 prefix_len, u32 mode, u32 use_le,
    u32 difficulty, u32 check_bits, u32 base, u32 count, u64 results_ptr)
```

`mode` is 0 for the decimal-nonce challenge, 1 for the fixed u32 nonce.
`use_le` selects little- or big-endian nonce bytes (mode 1 only).
`check_bits` is 0 for difficulty in nibbles and 1 for bits.
`base + count` never exceeds `2^32 - 1`.

```c
typedef struct {
    uint32_t count;
    uint32_t pad[3];
    uint32_t nonces[8];
    uint32_t hashes[8][8];   /* 8 big-endian words, as u32 */
} ptx_results;               /* 304 bytes */
```

A thread claims a slot with `atom.global.add.u32` on `count`; at most eight
hits are stored per launch. Every digest is re-verified by
`src/core/anubis_solver.c` before it is accepted.

HashX uses the same results struct. `hashx_interp.ptx` takes the four SipHash
keys, a pointer to the exported program (`hashx_export()`, 16 bytes per
instruction), the instruction count, the difficulty in leading zero bits and
the nonce range:

```
hashx_kernel(u64 k0..k3, u64 prog_ptr, u32 code_size, u32 difficulty,
             u32 base, u32 count, u64 results_ptr)
```

The driver API for it is:

```c
int anubis_ptx_hashx_available(void);
int anubis_ptx_hashx_set_program(const uint64_t keys[4], const uint32_t *prog,
                                 uint32_t instructions);
int anubis_ptx_hashx_solve(int difficulty, uint64_t start, uint64_t max_nonce,
                           uint32_t *out_nonce, char out_hex[65], double *out_ms);
```

## Using the cubins

```bash
gcc -O2 -shared -fPIC -o libanubis_ptx_loader.so ptx_loader.c -lcuda
```

If the machine has `ptxas` but no driver headers (Kaggle images do), gcc needs
`-I` on the CUDA include directory, or `-DANUBIS_MIN_CUDA_H` to use the
bundled `cuda_min.h`:

```bash
gcc -O2 -shared -fPIC -DANUBIS_MIN_CUDA_H -o libanubis_ptx_loader.so \
    ptx_loader.c -lcuda
```

If there is also no `libcuda.so` to link against (Kaggle again: the driver
library is only present at run time), build with `-DANUBIS_DLOPEN_CUDA` and no
`-lcuda`. The loader then resolves the driver entry points with
`dlopen("libcuda.so.1")` + `dlsym`:

```bash
gcc -O2 -shared -fPIC -DANUBIS_DLOPEN_CUDA -o libanubis_ptx_loader.so \
    ptx_loader.c
```

Then set `ANUBIS_PTX_DIR` to the directory holding the cubins and call
`anubis_ptx_solve()` / `anubis_ptx_available()`.

| Variable | Default | Effect |
|---|---|---|
| `ANUBIS_PTX_DIR` | `.` | Directory holding the cubins (`sha256_*.cubin`, `hashx_interp.cubin`) |
| `ANUBIS_PTX_ILP` | `2` | Selects `sha256_opt_ilp<N>.cubin` (1, 2 or 4) |
| `ANUBIS_PTX_CHUNK` | `1<<22` | Nonces per launch per device (max `1<<24`) |
| `ANUBIS_PTX_BLOCK` | `256` | Threads per block, clamped to 32..1024 and to a multiple of 32 |
| `ANUBIS_PTX_LOOP` | `1` | Nonce groups per thread. Above 1 it loads `sha256_opt_loop_ilp<N>.cubin` and sizes the grid to `count/(block*ILP*loop)`; if that cubin is missing it falls back to one group per thread rather than under-covering the range |

## Why PTX instead of SASS

| | SASS | PTX |
|---|---|---|
| Documented | no | yes (PTX ISA) |
| Official assembler | no | `ptxas` |
| Portable across SMs | no | yes (recompiled per arch) |
| Control word | chosen by the scheduler | `ptxas` handles it |

## Verification

No GPU is required to check these kernels. `test_ptx.py` runs the actual PTX
through `ptxsim.py` and compares to `hashlib`:

```bash
python3 src/nvidia/ptx/test_ptx.py
```

It checks:

- mode 1 with little- and big-endian nonces, including `0xfffffffe`;
- mode 0 decimal nonces of 1, 2, 3, 6 and 10 digits;
- nibble and bit difficulty, that each stored hit really has the required
  leading zeros and the reported digest is correct;
- the optimized kernel at ILP=2, both lanes, with 1 and 2 active nonces;
- the grid-stride loop build (`sha256_opt_loop.ptx`) covers a range exactly at
  loop depths 1, 2, 3, 4 and 8, including ranges that are not a multiple of the
  block width, in both nonce byte orders, and its bit-difficulty hits are real;
- that the midstate + padded-tail path agrees with `hashlib` on the full
  message.

The HashX kernel has its own test, which needs `gcc` but no GPU:

```bash
python3 src/nvidia/ptx/test_hashx_ptx.py
```

It builds `libhashx_shim.so` from `src/hashx/hashx.c`, exports a real generated
program, runs the PTX in `ptxsim.py` and checks:

- difficulty 0 digests against `hashx_hash()` for single nonces, the top of the
  u32 range, and eight nonces at once;
- that the number of hits over 400 nonces equals the number of real solutions
  at 4 and 8 leading zero bits, and that every stored hit is one of them;
- the two-block launch geometry (200 nonces) and the eight result slots.

`ptxsim.py` is a test aid, not a general emulator: it implements only the
instruction subset these kernels use.

## Performance

Measured on a Kaggle T4 (2x, `sm_75`, `monktelv31/anubis-ptx-t4-ab`, 2026-11-18,
`results/ptx-t4-ab-2026-11-18.json`): **2.75 GH/s kernel-only** for the `sha256`
tier on one T4 at 1470 MHz (2.69 GH/s wall, 5.3 GH/s on two).

The interesting part is that 21 configurations land inside 7 % of each other, so
this kernel is at a plateau rather than a tuning point:

| knob | range tried | result |
|---|---|---|
| ILP | 1, 2 | flat (ILP 1 uses 42 regs, ILP 2 uses 64 for 2 nonces) |
| grid-stride loop (`ANUBIS_PTX_LOOP`) | 1, 2, 8, 32 | flat: 2.71 (loop 8) vs 2.73 (loop 1) at ILP 2 / 256 |
| threads per block | 128, 256, 512, 1024 | 512 is ~1 % better than 256, 1024 is worse |
| `-maxrregcount` (occupancy) | 42 (default), 40, 32 | **32 regs is 5 % slower**, so it is not occupancy-starved |

Why it is flat: the integer pipe is the wall. Turing has 64 INT32 lanes per SM,
so an INT instruction retires for two warps per clock, and the kernel needs
1440 SASS instructions per hash:

    40 SM x 1.47 GHz x 2 warp-instr/clk x 32 lanes / 1440 = 2.6 GH/s

and we measure 2.75 GH/s (the ~5 % above the model is the 101 `IMAD` adds that
run on the FP32/FMA pipe instead). 1440 is also the arithmetic floor: 64 rounds
x 15 instructions (three shifts + one `LOP3` per sigma, one `LOP3` each for Ch
and Maj, five adds) + 48 schedule words x 10. So there is no scalar instruction
left to delete, which is why the loop and occupancy knobs do nothing.

Two consequences for anyone tuning this further:

- **Do not chase the "uniform schedule" idea.** Only `w17`, `w19` and `w21` of
  the expanded message are non-varying (`w[j]` depends on `w[j-7]` and `w[j-15]`,
  so an odd word stops being constant as soon as `j-7 >= 16`). The grid-stride
  loop therefore hoists nothing, and the SASS confirms it: 1448 instructions for
  the loop build against 1440 for one hash per thread.
- **Forcing adds onto the FMA pipe does not work.** `ptxas` canonicalises
  `mad.lo.u32 d, a, 1, b` back to `IADD3`, byte-identical SASS.

Compare per clock, not per run: the T4 idles at 300 MHz and boosts to 1590, and
this session ran at 1425-1485 under load. The best configuration is 1.87 MH/s
per MHz; published hashcat 1400 numbers (3.0-3.34 GH/s) work out to 2.0-2.1
MH/s per MHz, i.e. 10-12 % more per clock — and 3.34 GH/s at 1.87 MH/s/MHz would
need 1786 MHz, above the card's 1590 MHz maximum, which is why a same-session,
clock-logged comparison is the only honest one. Use `kaggle_ab.py` for that; it
reports kernel-only GH/s and the SM clock per row.

The HashX interpreter was measured in the same way on 2026-11-17 (kernel
`monktelv31/anubis-hashx-ptx`): **115.4 MH/s** on two T4s over a fixed 128 Mi
nonce range, against 9.7 MH/s for one CPU thread (11.9×). That is level with
the CUDA interpreter (~108 MH/s under the library's dispatch), which is what
the work says it should be. Every digest was checked against `hashx_hash()`.
See `results/hashx-ptx-t4-2026-11-17.json`.

Measure on real hardware by loading the cubins with `ptx_loader.c`; the CUDA
kernels in `../cuda/` are the reference numbers (`fast` ~922 MH/s and `sha256`
~1.7 GH/s on one T4, difficulty 6 / 24 bits, from `results/cuda-t4.json`).
