/*
 * hashx_shim.c -- a flat ctypes-friendly wrapper around the CPU HashX
 * reference (src/hashx/hashx.c), used by test_hashx_ptx.py and
 * kaggle_bench_hashx_ptx.py to produce the keys/program the GPU interprets
 * and the digests it must match.
 *
 *   gcc -O2 -shared -fPIC -o libhashx_shim.so \
 *       ../../hashx/hashx.c hashx_shim.c
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../../hashx/hashx.h"

int hx_shim_init(const unsigned char *data, size_t len,
                 void **state, size_t *size)
{
    return hashx_init(data, len, state, size);
}

int hx_shim_export(const void *state, size_t size, uint64_t keys[4],
                   uint32_t *prog, uint32_t *count)
{
    size_t n = 0;
    int rc = hashx_export(state, size, keys, prog, &n);
    *count = (uint32_t)n;
    return rc;
}

void hx_shim_hash(const void *state, size_t size, uint64_t nonce,
                  unsigned char out[32])
{
    hashx_hash(state, size, nonce, out);
}

/* Hash `count` nonces with the C reference; used for the CPU rate. The
 * digests are folded into out[32] so the loop cannot be optimised away. */
void hx_shim_scan(const void *state, size_t size, uint32_t count,
                  unsigned char out[32])
{
    unsigned char dg[32];
    uint32_t i;
    memset(out, 0, 32);
    for (i = 0; i < count; i++) {
        hashx_hash(state, size, (unsigned long long)i, dg);
        for (int j = 0; j < 32; j++) {
            out[j] ^= dg[j];
        }
    }
}

void hx_shim_free(void *state)
{
    hashx_free(state);
}
