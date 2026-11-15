/*
 * tpu_backend.h — the Anubis `sha256` proof of work on a TPU, as library code.
 *
 * The standalone client in sha256_pjrt.c proves the mechanism; this header is
 * the surface libanubis_solver.so links against. Everything here is safe to
 * call from an arbitrary host process: no output, no exit(), no required
 * environment, and a backend that is simply unavailable when libtpu is not.
 *
 * Return codes follow the engine convention: 0 = hit, 1 = exhausted,
 * -1 = backend failure (no plugin, no devices, compile/execute failed, or the
 * TPU is owned by somebody else).
 */
#ifndef ANUBIS_TPU_BACKEND_H
#define ANUBIS_TPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These three are for the solver core, not for host processes: the shared
 * library exports the backend inventory (anubis_backends_*) instead. Hiding
 * them keeps the .so's dynamic symbol table to the public ABI. */
#if defined(__GNUC__)
#  define ANUBIS_TPU_INTERNAL __attribute__((visibility("hidden")))
#else
#  define ANUBIS_TPU_INTERNAL
#endif

/* 1 when libtpu loaded and a PJRT client with at least one addressable device
 * exists. The probe runs once per process; later calls are a cached read. */
ANUBIS_TPU_INTERNAL int anubis_tpu_available(void);

/* "TPU: PJRT <major>.<minor>, <n> devices" once available, NULL otherwise.
 * The returned pointer has static storage and is valid for the process. */
ANUBIS_TPU_INTERNAL const char *anubis_tpu_info(void);

/*
 * Search for a nonce whose sha256(challenge || u32 nonce) has `difficulty`
 * leading zero bits.
 *
 *   challenge/challenge_len  raw challenge bytes, 1..51 bytes
 *   difficulty               1..256 leading zero bits
 *   use_le                   non-zero: nonce is little-endian, else big-endian
 *   max_nonce                inclusive nonce bound, 0 = unlimited
 *   budget_ms                wall-clock budget for the whole call, <= 0 = none;
 *                            a new challenge pays its compile inside this
 *   out_nonce                mandatory; written only on a hit
 *   out_hex                  optional 65-byte buffer for the digest hex
 *   out_ms                   optional; the call's wall clock, compile included
 *
 * Every device-side hit is re-derived with the host SHA-256 before it is
 * returned, so a miscomputing plugin cannot produce a false hit. Outputs are
 * untouched on failure. Safe to call repeatedly and from multiple threads
 * (calls are serialized; the compiled executable is cached per challenge).
 */
ANUBIS_TPU_INTERNAL int anubis_tpu_solve(const uint8_t *challenge,
                                         uint32_t challenge_len,
                                         int difficulty, int use_le,
                                         uint64_t max_nonce, double budget_ms,
                                         uint32_t *out_nonce, char *out_hex,
                                         double *out_ms);

#ifdef __cplusplus
}
#endif

#endif /* ANUBIS_TPU_BACKEND_H */
