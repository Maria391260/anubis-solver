#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "json.h"
#include "mp.h"
#include "sha256.h"
#include "solve.h"
#include "timer.h"

#define CHUNK 8192u

static const uint32_t H0[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

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

int solve_caps_json(const char *model, int kernel, unsigned long long solved,
                    char *out, size_t n) {
    return snprintf(out, n,
                    "{\"ok\":true,\"caps\":{\"model\":\"%s\",\"kernel\":\"%s\","
                    "\"cpus\":%d,\"solved\":%llu}}",
                    model, anubis_kernel_name(kernel), mp_count(), solved);
}

/* ---- the scan ---------------------------------------------------------- */

typedef struct {
    const solve_req_t *r;
    uint8_t block[64];
    int raw_len;
    int use_le;
    volatile long next_chunk;
    volatile int found;
    unsigned long long nonce;
    uint8_t digest[32];
} scan_job;

static int scan_once(scan_job *j, unsigned long long nonce, uint8_t digest[32]) {
    uint32_t h[8];
    int i;

    if (j->use_le) {
        j->block[j->raw_len + 0] = (uint8_t)nonce;
        j->block[j->raw_len + 1] = (uint8_t)(nonce >> 8);
        j->block[j->raw_len + 2] = (uint8_t)(nonce >> 16);
        j->block[j->raw_len + 3] = (uint8_t)(nonce >> 24);
    } else {
        j->block[j->raw_len + 0] = (uint8_t)(nonce >> 24);
        j->block[j->raw_len + 1] = (uint8_t)(nonce >> 16);
        j->block[j->raw_len + 2] = (uint8_t)(nonce >> 8);
        j->block[j->raw_len + 3] = (uint8_t)nonce;
    }
    for (i = 0; i < 8; i++) {
        h[i] = H0[i];
    }
    anubis_sha256_block(h, j->block);
    for (i = 0; i < 8; i++) {
        digest[i * 4 + 0] = (uint8_t)(h[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(h[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(h[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)h[i];
    }
    return leading_zero_bits(digest, j->r->difficulty);
}

/* Runs on the BSP and on every AP. Chunks are claimed with one locked add, so
 * no two workers scan the same nonce and no per-CPU argument is needed. */
static void scan_worker(void *arg) {
    scan_job *j = arg;
    unsigned long long end = j->r->start + j->r->count;

    for (;;) {
        long c;
        unsigned long long base, hi, n;
        uint8_t d[32];

        if (j->found) {
            return;
        }
        c = mp_atomic_add(&j->next_chunk, 1);
        base = j->r->start + (unsigned long long)c * CHUNK;
        if (base >= end) {
            return;
        }
        hi = base + CHUNK;
        if (hi > end) {
            hi = end;
        }
        for (n = base; n < hi; n++) {
            if (scan_once(j, n, d)) {
                if (mp_atomic_cas(&j->found, 0, 1)) {
                    j->nonce = n;
                    memcpy(j->digest, d, 32);
                }
                return;
            }
        }
    }
}

static int hex32(const uint8_t d[32], char out[65]) {
    int i;

    for (i = 0; i < 32; i++) {
        out[i * 2] = "0123456789abcdef"[d[i] >> 4];
        out[i * 2 + 1] = "0123456789abcdef"[d[i] & 15];
    }
    out[64] = '\0';
    return 0;
}

int solve_run(const solve_req_t *r, const anubis_cpu_t *cpu, char *out, size_t n) {
    static scan_job job;
    uint8_t raw[64];
    uint64_t t0, t1, cycles;
    int raw_len, i;

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

    memset(&job, 0, sizeof job);
    job.r = r;
    job.raw_len = raw_len;
    job.use_le = raw[raw_len - 1] >= 0x80;
    memcpy(job.block, raw, (size_t)raw_len);
    job.block[raw_len + 4] = 0x80;
    job.block[62] = (uint8_t)((raw_len + 4) >> 5);
    job.block[63] = (uint8_t)((raw_len + 4) << 3);

    t0 = timer_now();
    if (mp_count() > 1) {
        mp_run(scan_worker, &job);
    } else {
        scan_worker(&job);
    }
    t1 = timer_now();
    cycles = t1 - t0;

    if (job.found) {
        char hex[65];
        hex32(job.digest, hex);
        return snprintf(out, n,
                        "{\"ok\":true,\"hit\":true,\"algo\":\"sha256\","
                        "\"kernel\":\"%s\",\"nonce\":%llu,\"attempts\":%llu,"
                        "\"us\":%llu,\"cycles\":%llu,\"threads\":%d,"
                        "\"digest\":\"%s\"}",
                        anubis_kernel_name(anubis_cpu_pick(cpu)), job.nonce,
                        job.nonce - r->start + 1,
                        (unsigned long long)timer_us(cycles),
                        (unsigned long long)cycles, mp_count(), hex);
    }
    (void)i;
    return snprintf(out, n,
                    "{\"ok\":true,\"hit\":false,\"algo\":\"sha256\","
                    "\"kernel\":\"%s\",\"attempts\":%llu,\"us\":%llu,"
                    "\"cycles\":%llu,\"threads\":%d}",
                    anubis_kernel_name(anubis_cpu_pick(cpu)), r->count,
                    (unsigned long long)timer_us(cycles),
                    (unsigned long long)cycles, mp_count());
}
