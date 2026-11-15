# The C API service

`anubis_api` is a small HTTP/1.1 JSON service around the solver and the fetch
layer, both from this repository. It exists so a program in any language can
solve Anubis challenges, verify answers, or fetch a protected URL without
linking C.

## Build and run

```bash
make -j8 libanubis_solver.so
make anubis_api
./anubis_api --listen 127.0.0.1:8088
```

The fetch layer is part of `libanubis_solver.so` (2.0.2 and later), so the
service is one source file plus that library.

It binds localhost by default in the documented command. It has **no
authentication**: bind it to loopback or put it behind your own auth. Anyone who
can reach it can spend your CPU on proofs of work.

## Endpoints

### `GET /healthz`

```json
{"ok":true,"version":"2.0.2","codename":"shaoogle"}
```

### `GET /v1/version`

```json
{"version":"2.0.2","codename":"shaoogle",
 "algorithms":["fast","slow","sha256","wasm-sha256","argon2id","hashx","tensor"],
 "impersonate":"chrome116","impersonate_available":true,
 "backends_compiled":["cpu","levelzero","vulkan","opencl","tpu"]}
```

### `GET /v1/backends`

What the `.so` carries and what this machine can actually use. Separate from
`/v1/version` because `available` probes the drivers (a cold Vulkan or OpenCL
stack can take a second), while the version endpoint must stay cheap.

```json
{"compiled":["cpu","levelzero","vulkan","opencl","tpu"],
 "available":["cpu","vulkan","opencl"]}
```

`available` is always a subset of `compiled`, and `cpu` is always in both.

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
{"status":200,"final_url":"https://example.com/",
 "transport":"curl-impersonate/chrome116","content":"<!doctype html>..."}
```

This is the same client the MCP server uses: it solves whatever challenge the
site issues, keeps the cookies, and returns the final page. A failure to reach
or solve the site is a 502 with `{"error":"..."}`.

`transport` says which client actually made the request. `impersonate` picks a
browser TLS fingerprint for this request only — the service is multi-threaded,
so the profile is per call and never global:

```bash
curl -s localhost:8088/v1/fetch -d '{"url":"https://example.com/","impersonate":"chrome110"}'
curl -s localhost:8088/v1/fetch -d '{"url":"https://example.com/","impersonate":"off"}'
```

Values are `chrome`, `edge`, `safari` (newest of each family), a concrete
profile such as `chrome116`, `auto`, or `off` for the built-in OpenSSL client.
Impersonation needs a [libcurl-impersonate](../3rd_party/curl-impersonate/README.md)
installed; without it the default is OpenSSL and an explicit profile is a 502
rather than a silent downgrade. `ANUBIS_IMPERSONATE` sets the default profile
for the process.

## Limits

A request body over `--max-body` (default 1 MiB) is a 413. Responses are JSON
with proper escaping. The service is single-process; concurrency is worth
benchmarking before putting it behind a load balancer, because `argon2id`
solves need ~19 MiB per worker thread.
