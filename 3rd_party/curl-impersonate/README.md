# libcurl-impersonate (runtime only)

The C fetch layer in [`src/fetch/`](../../src/fetch/) can present a real browser
TLS fingerprint — Chrome's cipher list, extension order, ALPN h2 and GREASE
values — instead of OpenSSL's own ClientHello. It does that with
[curl-impersonate](https://github.com/lwthiker/curl-impersonate), a patched
libcurl whose TLS stack is BoringSSL.

Nothing here is required to build. `libanubis_solver.so` and the two services
build and run with the built-in OpenSSL client; they only look for this library
at runtime, with `dlopen()`, from the paths listed in
[`src/fetch/fetch_impersonate.c`](../../src/fetch/fetch_impersonate.c).

## Install it

```bash
make impersonate-lib CURL_IMPERSONATE_LIB=/path/to/libcurl-impersonate-chrome.so.4
```

That copies the file into `lib/` here (which git ignores), where the fetch layer
looks first after `$ANUBIS_IMPERSONATE_LIB`. A curl_cffi wheel also contains a
suitable library: `python3 -c 'import curl_cffi, os; print(os.path.dirname(curl_cffi.__file__))'`
then `/lib64/libcurl-impersonate-chrome.so.4` in that directory.

## Use it

```bash
# Chrome (newest profile the library has)
./anubis_api --listen 127.0.0.1:8088
curl -s localhost:8088/v1/fetch -d '{"url":"https://example.com/","impersonate":"chrome"}'
# {"status":200, ..., "transport":"curl-impersonate/chrome116", "content":"..."}

# Pin a profile, or turn it off
ANUBIS_IMPERSONATE=chrome110 ./anubis_mcp
ANUBIS_IMPERSONATE=off ./anubis_mcp
```

Profiles depend on the build. The one used in this repository's tests has
`chrome99`, `chrome99_android`, `chrome100`, `chrome101`, `chrome104`,
`chrome107`, `chrome110`, `chrome116`, `edge99`, `edge101`, `safari15_3` and
`safari15_5`; `chrome`, `edge` and `safari` resolve to the newest of each.
An unknown profile fails loudly rather than silently falling back.

## Verify it

```bash
make test-impersonate
```

The test starts a local `openssl s_server`, fetches it through the MCP server
and the API service, and checks the ClientHello in the server's `-msg` dump for
a GREASE cipher suite — something OpenSSL never sends. With `impersonate: off`
the same test requires GREASE to be absent.

## Licence

curl-impersonate is MIT-licensed and wraps curl, BoringSSL, nghttp2 and brotli,
each under its own licence. Because the library is installed locally rather than
redistributed here, its licences apply to your copy of it.
