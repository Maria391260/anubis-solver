/*
 * pjrt_common.h — host-side helpers shared by the PJRT clients: SHA-256 (the
 * reference for every digest these programs print), hex, and a monotonic clock.
 *
 * Kept separate from pjrt_c_api.h so the plugin header stays a verbatim copy of
 * the upstream one. Header-only, no state.
 */
#ifndef PJRT_COMMON_H
#define PJRT_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t h[8];
    uint64_t total;
    uint8_t buf[64];
    size_t used;
} pjrt_sha256;

static const uint32_t pjrt_k256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static inline uint32_t pjrt_rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

static inline void pjrt_sha256_block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, hh;
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
               ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = pjrt_rotr(w[i - 15], 7) ^ pjrt_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = pjrt_rotr(w[i - 2], 17) ^ pjrt_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = pjrt_rotr(e, 6) ^ pjrt_rotr(e, 11) ^ pjrt_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + S1 + ch + pjrt_k256[i] + w[i];
        uint32_t S0 = pjrt_rotr(a, 2) ^ pjrt_rotr(a, 13) ^ pjrt_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

static inline void pjrt_sha256_init(pjrt_sha256 *c) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u};
    memcpy(c->h, iv, sizeof iv);
    c->total = 0;
    c->used = 0;
}

static inline void pjrt_sha256_update(pjrt_sha256 *c, const void *data, size_t len) {
    const uint8_t *p = data;
    c->total += len;
    while (len > 0) {
        size_t take = 64 - c->used;
        if (take > len) {
            take = len;
        }
        memcpy(c->buf + c->used, p, take);
        c->used += take;
        p += take;
        len -= take;
        if (c->used == 64) {
            pjrt_sha256_block(c->h, c->buf);
            c->used = 0;
        }
    }
}

static inline void pjrt_sha256_final(pjrt_sha256 *c, uint8_t out[32]) {
    uint64_t bits = c->total * 8u;
    uint8_t pad[72];
    size_t padlen;
    c->buf[c->used++] = 0x80;
    padlen = (c->used > 56) ? (64 - c->used + 56) : (56 - c->used);
    memset(pad, 0, sizeof pad);
    pjrt_sha256_update(c, pad, padlen);
    for (int i = 0; i < 8; i++) {
        pad[i] = (uint8_t)(bits >> (8 * (7 - i)));
    }
    pjrt_sha256_update(c, pad, 8);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(c->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(c->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(c->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)c->h[i];
    }
}

static inline void pjrt_sha256_buf(const void *data, size_t len, uint8_t out[32]) {
    pjrt_sha256 c;
    pjrt_sha256_init(&c);
    pjrt_sha256_update(&c, data, len);
    pjrt_sha256_final(&c, out);
}

static inline double pjrt_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static inline int pjrt_hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static inline size_t pjrt_hex_decode(const char *hex, uint8_t *out, size_t cap) {
    size_t n = strlen(hex) / 2, i;
    if (n > cap) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        int hi = pjrt_hexval(hex[2 * i]), lo = pjrt_hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return 0;
        }
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return n;
}

static inline void pjrt_to_hex(const uint8_t *in, size_t len, char *out) {
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = d[in[i] >> 4];
        out[2 * i + 1] = d[in[i] & 15];
    }
    out[2 * len] = '\0';
}

#endif /* PJRT_COMMON_H */
