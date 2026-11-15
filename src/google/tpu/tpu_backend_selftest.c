/*
 * tpu_backend_selftest.c — prove the TPU backend's two paths.
 *
 * On a machine with no libtpu (this checkout's normal case) the graceful path
 * is the whole test: available() says no, info() is NULL, solve() returns -1
 * without crashing, and a rejected challenge never touches the outputs. That
 * is the path every host process takes when it is not on a TPU.
 *
 * If ANUBIS_TPU_LIB names a real libtpu, the real path is tested instead: the
 * backend reports available, and a known challenge at difficulty 4 yields a
 * nonce whose digest the host reference reproduces and whose leading bits meet
 * the target. A second solve proves the cached executable still works.
 *
 *   cc -O2 -I. -o tpu_backend_selftest tpu_backend_selftest.c tpu_backend.o \
 *       -ldl -lpthread
 *   ./tpu_backend_selftest            # graceful path on this box
 *   ANUBIS_TPU_LIB=/path/libtpu.so ./tpu_backend_selftest
 *
 * One line per check; exit 0 when every check passed.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_common.h"
#include "tpu_backend.h"

/* The stand-alone client's default challenge: 32 bytes, last byte < 0x80, so
 * the nonce is big-endian. */
static const char KNOWN_CHALLENGE[] =
    "9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b";

static int failures;

static void check(int ok, const char *what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static int leading_zero_bits(const uint8_t d[32]) {
    int n = 0;
    int i;

    for (i = 0; i < 32; i++) {
        if (d[i] == 0) {
            n += 8;
            continue;
        }
        n += __builtin_clz((unsigned)d[i]) - 24;
        break;
    }
    return n;
}

/* The host reference the backend must agree with: SHA-256(raw || nonce). */
static void reference_digest(const uint8_t *raw, size_t raw_len, uint32_t nonce,
                             int use_le, uint8_t out[32]) {
    uint8_t msg[64];

    memcpy(msg, raw, raw_len);
    if (use_le) {
        msg[raw_len + 0] = (uint8_t)nonce;
        msg[raw_len + 1] = (uint8_t)(nonce >> 8);
        msg[raw_len + 2] = (uint8_t)(nonce >> 16);
        msg[raw_len + 3] = (uint8_t)(nonce >> 24);
    } else {
        msg[raw_len + 0] = (uint8_t)(nonce >> 24);
        msg[raw_len + 1] = (uint8_t)(nonce >> 16);
        msg[raw_len + 2] = (uint8_t)(nonce >> 8);
        msg[raw_len + 3] = (uint8_t)nonce;
    }
    pjrt_sha256_buf(msg, raw_len + 4u, out);
}

static void graceful_checks(void) {
    uint8_t raw[64];
    size_t raw_len;
    uint32_t nonce = 0xDEADBEEFu;
    double ms = -1.0;
    char hex[65];
    int rc;

    raw_len = pjrt_hex_decode(KNOWN_CHALLENGE, raw, sizeof raw);
    check(raw_len == 32, "test challenge decodes to 32 bytes");

    check(anubis_tpu_available() == 0,
          "available() == 0 without libtpu");
    check(anubis_tpu_info() == NULL,
          "info() == NULL without libtpu");

    memset(hex, 'x', sizeof hex);
    hex[64] = '\0';
    rc = anubis_tpu_solve(raw, (uint32_t)raw_len, 4, 0, 0, 0.0, &nonce, hex,
                          &ms);
    check(rc == -1, "solve() == -1 without libtpu");
    check(nonce == 0xDEADBEEFu && ms == -1.0 && hex[0] == 'x',
          "a failed solve leaves its outputs untouched");

    check(anubis_tpu_solve(raw, 0, 4, 0, 0, 0.0, &nonce, hex, &ms) == -1,
          "solve() rejects an empty challenge");
    check(anubis_tpu_solve(raw, 52, 4, 0, 0, 0.0, &nonce, hex, &ms) == -1,
          "solve() rejects challenge_len + 4 > 55");
    check(anubis_tpu_available() == 0,
          "available() still 0 on a repeated call");
}

static void real_checks(void) {
    const char *info = anubis_tpu_info();
    uint8_t raw[64];
    size_t raw_len;
    int use_le;
    int rc, rc2;
    uint32_t nonce = 0, nonce2 = 0;
    char hex[65] = {0}, hex2[65] = {0};
    double ms = -1.0, ms2 = -1.0;

    check(info != NULL, "info() != NULL with a live backend");
    if (info != NULL) {
        check(strncmp(info, "TPU: PJRT ", 10) == 0,
              "info() is \"TPU: PJRT <major>.<minor>, <n> devices\"");
    }

    raw_len = pjrt_hex_decode(KNOWN_CHALLENGE, raw, sizeof raw);
    check(raw_len == 32, "test challenge decodes to 32 bytes");
    use_le = raw_len > 0 && raw[raw_len - 1] >= 0x80;

    rc = anubis_tpu_solve(raw, (uint32_t)raw_len, 4, use_le, 0, 0.0, &nonce,
                          hex, &ms);
    check(rc == 0, "solve() at difficulty 4 returns a hit");
    if (rc == 0) {
        uint8_t ref[32];
        char refhex[65];

        reference_digest(raw, raw_len, nonce, use_le, ref);
        pjrt_to_hex(ref, 32, refhex);
        check(ms >= 0.0, "out_ms is non-negative");
        check(strcmp(hex, refhex) == 0,
              "returned digest matches the host reference");
        check(leading_zero_bits(ref) >= 4,
              "returned nonce meets difficulty 4");

        rc2 = anubis_tpu_solve(raw, (uint32_t)raw_len, 4, use_le, 0, 0.0,
                               &nonce2, hex2, &ms2);
        check(rc2 == 0 && nonce2 == nonce && strcmp(hex2, hex) == 0,
              "second call reuses the cached executable and agrees");
    }
}

int main(void) {
    const char *env = getenv("ANUBIS_TPU_LIB");
    int configured = env != NULL && *env != '\0';
    int avail = anubis_tpu_available();

    if (configured) {
        /* A configured plugin that does not come up is a real failure, not the
         * graceful path: the caller asked for a TPU. */
        check(avail == 1, "available() == 1 with ANUBIS_TPU_LIB");
        if (avail == 1) {
            real_checks();
        }
    } else if (avail == 0) {
        graceful_checks();
    } else {
        /* libtpu is installed system-wide and a device answered; there is no
         * graceful path to prove, but the real one is testable. */
        real_checks();
    }

    printf("%s: %d check(s) failed\n", failures == 0 ? "PASS" : "FAIL",
           failures);
    return failures == 0 ? 0 : 1;
}
