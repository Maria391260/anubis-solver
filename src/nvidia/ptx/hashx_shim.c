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

void hx_shim_free(void *state)
{
    hashx_free(state);
}
