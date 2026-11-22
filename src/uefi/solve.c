#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "json.h"
#include "sha256.h"
#include "solve.h"

static int hexval(int c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

static int hex_decode(const char *hex, uint8_t *out, int max) {
    int n = 0;

    while (hex[0] != '\0' && hex[1] != '\0' && n < max) {
        int hi = hexval((unsigned char)hex[0]);
        int lo = hexval((unsigned char)hex[1]);
        if (hi < 0 || lo < 0) {
            return -1;
        }
        out[n++] = (uint8_t)((hi << 4) | lo);
        hex += 2;
    }
    return n;
}

int solve_parse(const char *json, solve_req_t *r) {
    memset(r, 0, sizeof *r);
    if (json_str(json, "algo", r->algo, sizeof r->algo) != 0) {
        strcpy(r->algo, "sha256");
    }
    if (json_str(json, "challenge", r->challenge, sizeof r->challenge) != 0) {
        return -1;
    }
    r->difficulty = (int)json_int(json, "difficulty", -1);
    r->start = (unsigned long long)json_int(json, "start", 0);
    r->count = (unsigned long long)json_int(json, "count", 1);
    if (r->count == 0) {
        r->count = 1;
    }
    if (r->difficulty < 1 || r->difficulty > 32) {
        return -1;
    }
    return 0;
}

static int leading_zero_bits(const uint8_t d[32], int bits) {
    int i, n = 0;

    for (i = 0; i < 32 && n < bits; i++) {
        if (d[i] == 0) {
            n += 8;
            continue;
        }
        for (int b = 7; b >= 0 && n < bits; b--) {
            if (d[i] & (1u << b)) {
                return n >= bits;
            }
            n++;
        }
    }
    return n >= bits;
}

int solve_caps_json(const char *model, int kernel, unsigned long long solved,
                    char *out, size_t n) {
    return snprintf(out, n,
                    "{\"ok\":true,\"caps\":{\"model\":\"%s\",\"kernel\":\"%s\","
                    "\"solved\":%llu}}",
                    model, anubis_kernel_name(kernel), solved);
}

int solve_run(const solve_req_t *r, const anubis_cpu_t *cpu, char *out, size_t n) {
    static const uint32_t h0[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u};
    uint8_t raw[64];
    uint8_t block[64];
    uint8_t digest[32];
    uint32_t h[8];
    int raw_len, i, use_le;
    unsigned long long nonce, attempts = 0;

    if (strcmp(r->algo, "sha256") != 0) {
        return snprintf(out, n,
                        "{\"ok\":false,\"error\":\"algo '%s' is not wired into the "
                        "UEFI build yet\"}",
                        r->algo);
    }
    raw_len = hex_decode(r->challenge, raw, 51);
    if (raw_len <= 0 || raw_len + 4 > 55) {
        return snprintf(out, n, "{\"ok\":false,\"error\":\"bad challenge\"}");
    }
    use_le = raw[raw_len - 1] >= 0x80;

    memset(block, 0, sizeof block);
    memcpy(block, raw, (size_t)raw_len);
    block[raw_len + 4] = 0x80;                    /* padding byte */
    block[62] = (uint8_t)((raw_len + 4) >> 5);    /* length in bits, high */
    block[63] = (uint8_t)((raw_len + 4) << 3);    /* and low */

    for (nonce = r->start; nonce < r->start + r->count; nonce++) {
        if (use_le) {
            block[raw_len + 0] = (uint8_t)nonce;
            block[raw_len + 1] = (uint8_t)(nonce >> 8);
            block[raw_len + 2] = (uint8_t)(nonce >> 16);
            block[raw_len + 3] = (uint8_t)(nonce >> 24);
        } else {
            block[raw_len + 0] = (uint8_t)(nonce >> 24);
            block[raw_len + 1] = (uint8_t)(nonce >> 16);
            block[raw_len + 2] = (uint8_t)(nonce >> 8);
            block[raw_len + 3] = (uint8_t)nonce;
        }
        for (i = 0; i < 8; i++) {
            h[i] = h0[i];
        }
        anubis_sha256_block(h, block);
        for (i = 0; i < 8; i++) {
            digest[i * 4 + 0] = (uint8_t)(h[i] >> 24);
            digest[i * 4 + 1] = (uint8_t)(h[i] >> 16);
            digest[i * 4 + 2] = (uint8_t)(h[i] >> 8);
            digest[i * 4 + 3] = (uint8_t)h[i];
        }
        attempts++;
        if (leading_zero_bits(digest, r->difficulty)) {
            char hex[65];
            for (i = 0; i < 32; i++) {
                hex[i * 2] = "0123456789abcdef"[digest[i] >> 4];
                hex[i * 2 + 1] = "0123456789abcdef"[digest[i] & 15];
            }
            hex[64] = '\0';
            return snprintf(out, n,
                            "{\"ok\":true,\"hit\":true,\"algo\":\"sha256\","
                            "\"kernel\":\"%s\",\"nonce\":%llu,\"attempts\":%llu,"
                            "\"digest\":\"%s\"}",
                            anubis_kernel_name(anubis_cpu_pick(cpu)), nonce,
                            attempts, hex);
        }
    }
    return snprintf(out, n,
                    "{\"ok\":true,\"hit\":false,\"algo\":\"sha256\","
                    "\"kernel\":\"%s\",\"attempts\":%llu}",
                    anubis_kernel_name(anubis_cpu_pick(cpu)), attempts);
}
