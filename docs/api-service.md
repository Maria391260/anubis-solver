# The C API service

`anubis_api` is a small HTTP/1.1 JSON service around the solver and the fetch
layer, both from this repository. It exists so a program in any language can
solve Anubis challenges, verify answers, or fetch a protected URL without
linking C.

## Build and run

```bash
make -j8 libanubis_solver.so
cc -O2 -Isrc/fetch -Iinclude -o anubis_api \
   src/api/anubis_api.c src/fetch/anubis_fetch.c \
   -L. -lanubis_solver -Wl,-rpath,'$ORIGIN' -lssl -lcrypto -lpthread
./anubis_api --listen 127.0.0.1:8088
```

It binds localhost by default in the documented command. It has **no
authentication**: bind it to loopback or put it behind your own auth. Anyone who
can reach it can spend your CPU on proofs of work.

## Endpoints

### `GET /healthz`

```json
{"ok":true,"version":"2.0.0","codename":"shaoogle"}
```

### `GET /v1/version`

```json
{"version":"2.0.0","codename":"shaoogle",
 "algorithms":["fast","slow","sha256","wasm-sha256","argon2id","hashx","tensor"]}
```

### `POST /v1/solve`

```bash
curl -s localhost:8088/v1/solve -d '{
  "algorithm": "sha256",
  "challenge": "9f3c1a7b...e0f1a2b",
  "difficulty": 12
}'
```

```json
{"nonce":2381,"response":"00ab...","elapsed_ms":3.4}
```

Unknown algorithm is a 400 with `{"error":"unknown algorithm"}`. `threads` and
`budget_ms` are optional and map straight onto the solver's arguments.

### `POST /v1/verify`

```bash
curl -s localhost:8088/v1/verify -d '{
  "algorithm": "sha256", "challenge": "9f3c...", "difficulty": 12, "nonce": 2381
}'
```

```json
{"ok":true,"digest":"00ab..."}
```

### `POST /v1/fetch`

```bash
curl -s localhost:8088/v1/fetch -d '{"url":"https://example.com/"}'
```

```json
{"status":200,"final_url":"https://example.com/","content":"<!doctype html>..."}
```

This is the same client the MCP server uses: it solves whatever challenge the
site issues, keeps the cookies, and returns the final page. A failure to reach
or solve the site is a 502 with `{"error":"..."}`.

## Limits

A request body over `--max-body` (default 1 MiB) is a 413. Responses are JSON
with proper escaping. The service is single-process; concurrency is worth
benchmarking before putting it behind a load balancer, because `argon2id`
solves need ~19 MiB per worker thread.
