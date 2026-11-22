/* SHA-256, portable C. Only the `sha256` tier's challenge block is hashed, so
 * this is the reference path; the selected kernel (SHA-NI/AVX2/AVX-512) is used
 * for the search loop in solve.c. */
#pragma once

#include <stddef.h>
#include <stdint.h>

void anubis_sha256(const void *data, size_t len, uint8_t out[32]);

/* Raw compression over 16 big-endian words, exposed for the search loop. */
void anubis_sha256_block(uint32_t h[8], const uint8_t block[64]);
