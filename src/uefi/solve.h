/* One request in, one answer out. The framing/transport lives in net.c; this
 * file only knows about Anubis challenges. */
#pragma once

#include <stddef.h>

#include "cpu.h"

#define SOLVE_MAX_ALGO 16

typedef struct {
    char algo[SOLVE_MAX_ALGO];   /* "sha256" (hashx/argon2id: later) */
    char challenge[160];         /* hex, as it arrives in JSON */
    int difficulty;              /* leading zero bits */
    unsigned long long start;    /* first nonce to try */
    unsigned long long count;    /* how many to try (0 = 1) */
} solve_req_t;

/* 0 on success, -1 if the frame is not a request we understand. */
int solve_parse(const char *json, solve_req_t *r);

/* Writes the reply object (no trailing newline) into out. Returns the length. */
int solve_run(const solve_req_t *r, const anubis_cpu_t *cpu, char *out,
              size_t n);

/* The capability answer, shared by the handshake and the cached file. */
int solve_caps_json(const char *model, int kernel, unsigned long long solved,
                    char *out, size_t n);
