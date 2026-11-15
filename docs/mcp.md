# The C MCP server

`anubis_mcp` is a Model Context Protocol server written in C, with no Python
and no Node in the loop. It speaks newline-delimited JSON-RPC 2.0 on stdio and
exposes three tools: fetch a URL through an Anubis challenge, solve a challenge
directly, and report the version. Both the solving and the HTTP work come from
this repository: `libanubis_solver.so` and `src/fetch/anubis_fetch.c`.

## Build

```bash
make -j8 libanubis_solver.so
cc -O2 -Isrc/fetch -Iinclude -o anubis_mcp \
   src/mcp/anubis_mcp.c src/fetch/anubis_fetch.c \
   -L. -lanubis_solver -Wl,-rpath,'$ORIGIN' -lssl -lcrypto
```

(`make anubis_mcp` does the same thing.)

## Register it

Any MCP client that can launch a stdio server will do. Claude Desktop:

```json
{
  "mcpServers": {
    "anubis": {
      "command": "/absolute/path/to/anubis_mcp",
      "args": []
    }
  }
}
```

DeepSeek Harness, in `.dsh/config.json` (or the equivalent plugin config):

```json
{
  "mcpServers": {
    "anubis": { "command": "/absolute/path/to/anubis_mcp" }
  }
}
```

Because the binary links `libanubis_solver.so` with `$ORIGIN` in its runpath,
keep the two files in the same directory (the repo root, or an installed pair
under `/usr/local/bin` + `/usr/local/lib`).

## Tools

| tool | arguments | returns |
|---|---|---|
| `anubis_fetch` | `url` (required), `max_chars` (default 100000) | `status: <code>`, `url: <final>`, then the page as text |
| `anubis_solve` | `algorithm`, `challenge` (hex), `difficulty`, `threads` optional | JSON `{"nonce":n,"response":"<hex>","algorithm":"..."}` |
| `anubis_version` | none | JSON `{"version":"2.0.0","codename":"shaoogle","algorithms":[...]}` |

`anubis_fetch` follows the challenge the same way the browser does: it reads the
`anubis_challenge` JSON, solves it with the algorithm it names (`fast`, `slow`,
`sha256`, `argon2id`, `hashx`, `tensor`, and the time-based `metarefresh` /
`preact`), submits `pass-challenge` with the cookies the server set, and
re-fetches. TLS is verified; `ANUBIS_INSECURE=1` disables verification for
testing against a self-signed gate.

## Test

```bash
tests/mcp_test.sh
```

It starts a local gate (`betterAnusbis`) and a static backend, then drives the
server's stdio with real requests: initialize, tools/list, solve for `sha256`
and `tensor`, a version call, an end-to-end fetch through the gate, and an
unknown method. Every assertion prints PASS/FAIL and the script exits non-zero
on failure.
