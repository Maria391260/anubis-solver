/*
 * anubis-solver.h — public C/C++ API for libanubis_solver.
 *
 * C:   #include <anubis-solver.h>
 * C++: #include <anubis-solver.h>   (functions are declared extern "C")
 *
 * Build:  cc  demo.c   -Iinclude -L. -lanubis_solver -Wl,-rpath,'$ORIGIN'
 *         c++ demo.cpp -Iinclude -L. -lanubis_solver -Wl,-rpath,'$ORIGIN'
 */

#ifndef ANUBIS_SOLVER_H
#define ANUBIS_SOLVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define ANUBIS_OK 0
#define ANUBIS_ERR_ARG 1
#define ANUBIS_ERR_UNSOLVED 2
#define ANUBIS_ERR_INTERNAL 3

#if defined(_WIN32) && defined(ANUBIS_BUILD_SHARED)
#  define ANUBIS_API __declspec(dllexport)
#elif defined(_WIN32)
#  define ANUBIS_API __declspec(dllimport)
#else
#  define ANUBIS_API
#endif

/* ------------------------------------------------------------------ *
 * Introspection
 * ------------------------------------------------------------------ */

/* Library version, e.g. "2.0.2". */
ANUBIS_API const char *anubis_solver_version(void);

/* Release codename, e.g. "shaoogle". */
ANUBIS_API const char *anubis_solver_codename(void);

/* ------------------------------------------------------------------ *
 * Backends
 *
 * One .so carries every backend that could be compiled in; at run time the
 * library probes the machine and solves on whatever it finds. Vendor APIs are
 * dlopen()ed, so a build with all of them loads on a machine with none.
 * ------------------------------------------------------------------ */

#define ANUBIS_BACKEND_CPU 0x01          /* always */
#define ANUBIS_BACKEND_CUDA 0x02         /* NVIDIA, needs nvcc at build time */
#define ANUBIS_BACKEND_LEVELZERO 0x04    /* Intel Xe */
#define ANUBIS_BACKEND_VULKAN 0x08       /* any Vulkan ICD */
#define ANUBIS_BACKEND_OPENCL 0x10       /* any OpenCL ICD */
#define ANUBIS_BACKEND_TPU 0x20          /* Google, PJRT / libtpu */

/* Bitmask of the backends compiled into this .so. */
ANUBIS_API int anubis_backends_compiled(void);

/* Bitmask of those that are usable right now. Probes the hardware once per
 * backend and caches; a missing driver or device is not an error here. */
ANUBIS_API int anubis_backends_available(void);

/* Name of one backend bit ("cpu", "cuda", "levelzero", "vulkan", "opencl",
 * "tpu"), or NULL when the argument is not a single backend bit. */
ANUBIS_API const char *anubis_backend_name(int backend);

/* Easter egg. Returns the same sentence the chip answers to a probe of BAR0
 * at 0x0FC in `beyond/` — a static string, no arguments, never NULL, never
 * freed. Symbol name is exactly `easter` (no anubis_ prefix) because that is
 * the name the callers look for. */
ANUBIS_API const char *easter(void);

/* Name of the active GPU backend (e.g. "CUDA: Tesla T4 x2"), or NULL. */
ANUBIS_API const char *anubis_gpu_info(void);

/* Sensible default worker count (physical cores). */
ANUBIS_API int anubis_default_threads(void);

/* Human-readable CPU affinity, e.g. "P-cores (0,2,4,...)" or "off". */
ANUBIS_API const char *anubis_cpu_affinity_info(void);

/* SHA-256 batch width: 16 (AVX-512), 8 (AVX2), 0 (scalar). */
ANUBIS_API int anubis_batch_width(void);

/* GPU hit verification (default on). Disabling it trusts the GPU result as-is
 * and skips one reference hash per hit; useful for benchmarking only. */
ANUBIS_API void anubis_set_gpu_verify(int enable);
ANUBIS_API int anubis_gpu_verify(void);
/* TPU hit verification (default on). Resolves as: the value set here, then
 * ANUBIS_TPU_NO_VERIFY=1, then anubis_gpu_verify(). Pass -1 to clear the
 * override and fall back to the env / inherited switch. "Off" skips only the
 * second reference hash in the core: anubis_tpu_solve() always derives the
 * winning digest itself, since the device reports just a lane flag. */
ANUBIS_API void anubis_set_tpu_verify(int enable);
ANUBIS_API int anubis_tpu_verify(void);

/* ------------------------------------------------------------------ *
 * SHA-256 helpers
 * ------------------------------------------------------------------ */

/* Hex SHA-256 of `data`; `out_hex` must hold 65 bytes. */
ANUBIS_API int anubis_sha256_hex(const char *data, size_t len, char *out_hex);

/* SHA-256 midstate after `len` bytes (len must be a multiple of 64). */
ANUBIS_API int anubis_sha256_midstate(const char *data, size_t len,
                                      uint32_t out[8]);

/* ------------------------------------------------------------------ *
 * Solving
 *
 * Each returns ANUBIS_OK on a hit, ANUBIS_ERR_UNSOLVED when max_nonce or
 * budget_ms is exhausted, or another ANUBIS_* error code.
 *
 * threads:   0 = anubis_default_threads()
 * max_nonce: 0 = no nonce limit (nibbles family only)
 * budget_ms: <= 0 = no time limit
 * out_hex:   65-byte buffer, receives the hash as lowercase hex
 * out_ms:    optional, receives the elapsed time
 * ------------------------------------------------------------------ */

/* `fast` / `slow` (leading zero nibbles). */
ANUBIS_API int anubis_solve_nibbles(const char *challenge, int difficulty,
                                    int threads, int64_t max_nonce,
                                    double budget_ms, uint64_t *out_nonce,
                                    char *out_hex, double *out_ms);

/* `sha256` / `wasm-sha256` (leading zero bits). */
ANUBIS_API int anubis_solve_bits(const char *challenge_hex, int difficulty,
                                 int threads, double budget_ms,
                                 uint32_t *out_nonce, char *out_hex,
                                 double *out_ms);

/* `argon2id` (leading zero bits, Argon2id v1.3). */
ANUBIS_API int anubis_solve_argon2id(const char *challenge_hex, int difficulty,
                                     int threads, double budget_ms,
                                     uint32_t *out_nonce, char *out_hex,
                                     double *out_ms);

/* `hashx` (leading zero bits, HashX program). */
ANUBIS_API int anubis_solve_hashx(const char *challenge_hex, int difficulty,
                                  int threads, double budget_ms,
                                  uint32_t *out_nonce, char *out_hex,
                                  double *out_ms);

/* ------------------------------------------------------------------ *
 * Verification (re-check a nonce; returns ANUBIS_OK when valid)
 * ------------------------------------------------------------------ */

ANUBIS_API int anubis_verify_nibbles(const char *challenge, uint64_t nonce,
                                     int difficulty, char *out_hex);
ANUBIS_API int anubis_verify_bits(const char *challenge_hex, uint32_t nonce,
                                  int difficulty, char *out_hex);
ANUBIS_API int anubis_verify_argon2id(const char *challenge_hex, uint32_t nonce,
                                      int difficulty, char *out_hex);
ANUBIS_API int anubis_verify_hashx(const char *challenge_hex, uint32_t nonce,
                                   int difficulty, char *out_hex);

/* `tensor`: one integer GEMM, digest of the result. Deterministic, so the
 * proof is the digest itself and the nonce is always 0; verification
 * recomputes one GEMM and costs no more than one solve. */
ANUBIS_API int anubis_verify_tensor(const char *challenge_hex, uint32_t nonce,
                                    int difficulty, char *out_hex);
ANUBIS_API int anubis_tensor_digest(const char *challenge_hex, int difficulty,
                                    char *out_hex);

/* `tensor` solve. The proof is deterministic (nonce 0), so this is the same
 * digest as anubis_tensor_digest; when a CUDA device is present the GEMM runs
 * there and the digest is still checked against the CPU reference unless
 * anubis_set_gpu_verify(0). threads and budget_ms do not apply. */
ANUBIS_API int anubis_solve_tensor(const char *challenge_hex, int difficulty,
                                   int threads, double budget_ms,
                                   uint32_t *out_nonce, char *out_hex,
                                   double *out_ms);

/* ------------------------------------------------------------------ *
 * Google TPU (PJRT C API), the `sha256` tier
 *
 * libtpu.so is dlopen()ed, so these symbols exist in every build: without the
 * plugin anubis_tpu_available() is 0 and anubis_tpu_solve() returns -1.
 *
 * anubis_solve_bits() is still the recommended entry point: it applies the
 * ANUBIS_TPU_MIN_WORK threshold and falls back to the CPU. A direct caller
 * owns both decisions. This call always re-derives the winning digest on the
 * host, so anubis_tpu_verify() does not apply to it.
 *
 * challenge       raw bytes, 1..51 (one 64-byte block with the nonce)
 * use_le          non-zero: little-endian nonce, else big-endian
 * max_nonce       inclusive nonce bound, 0 = the whole 32-bit space
 * budget_ms       wall clock for the call, compile included; <= 0 = none
 * Returns 0 on a hit, 1 when the range/budget ran out, -1 on backend failure
 * (no plugin, no device, compile or execute error).
 * ------------------------------------------------------------------ */
ANUBIS_API int anubis_tpu_available(void);
ANUBIS_API const char *anubis_tpu_info(void);   /* NULL when unavailable */
ANUBIS_API int anubis_tpu_solve(const uint8_t *challenge, uint32_t challenge_len,
                                int difficulty, int use_le, uint64_t max_nonce,
                                double budget_ms, uint32_t *out_nonce,
                                char *out_hex, double *out_ms);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ANUBIS_SOLVER_H */
