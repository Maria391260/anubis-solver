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
| `kaggle_bench_hashx_ptx.py` | HashX benchmark: same, plus the CPU reference rate and the speedup. |
| `kaggle_hashx/` | `kaggle kernels push` job that clones and runs the HashX benchmark on a T4. |
| `cubin/` | Built cubins. |

`sha256_opt.ptx` and `hashx_interp.ptx` are generated. Do not edit them;
change the generators and regenerate:

```bash
python3 gen_opt_ptx.py --ilp 2 > sha256_opt.ptx
python3 gen_hashx_ptx.py > hashx_interp.ptx
```

## Building

```bash
export PATH="/path/to/cuda/bin:$PATH"
cd src/nvidia/ptx
./build_ptx.sh sm_75 2          # T4: sm_75, ILP=2
# -> cubin/sha256_pow.cubin, cubin/sha256_opt_ilp2.cubin, cubin/hashx_interp.cubin
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
| `ANUBIS_PTX_CHUNK` | `1<<22` | Nonces per launch per device |

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

The SHA-256 kernels were measured on a Kaggle T4 on 2026-11-17 (kernel
`monktelv31/anubis-ptx-t4`): **2.69 GH/s** `sha256` on one T4 and **5.29 GH/s**
on two, essentially flat across ILP 1/2/4. The earlier claim of `2.4 GH/s` was
attached to a stub that computed a single round, so it was meaningless and has
been removed.

The HashX interpreter is benchmarked by `kaggle_bench_hashx_ptx.py`
(`monktelv31/anubis-hashx-ptx`); see `results/hashx-ptx-t4-*.json`.

Measure on real hardware by loading the cubins with `ptx_loader.c`; the CUDA
kernels in `../cuda/` are the reference numbers (`fast` ~922 MH/s and `sha256`
~1.7 GH/s on one T4, difficulty 6 / 24 bits, from `results/cuda-t4.json`).
