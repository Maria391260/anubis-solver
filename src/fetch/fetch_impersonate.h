/*
 * fetch_impersonate.h — a browser TLS fingerprint for the C fetch layer.
 *
 * The hand-written client in anubis_fetch.c speaks HTTP/1.1 with OpenSSL's own
 * ClientHello. That is fine for a gate, and useless in front of anything that
 * looks at the handshake: the cipher list, extension order, ALPN and GREASE
 * values are OpenSSL's, not Chrome's.
 *
 * Real impersonation needs a patched libcurl (curl-impersonate). We do not want
 * it at build time — it is not in every distribution, and this library must
 * keep building with nothing but a C compiler. So this file loads it at
 * runtime with dlopen() and calls the handful of functions it needs through
 * direct function pointers. No headers from the library are required, and no
 * libcurl is linked.
 *
 * The library comes from, in order:
 *   - $ANUBIS_IMPERSONATE_LIB (explicit path)
 *   - ./3rd_party/curl-impersonate/lib/libcurl-impersonate-chrome.so.4,
 *     ../3rd_party/... and ../../3rd_party/... (the repo layout)
 *   - next to the running executable, then beside its 3rd_party/
 *   - libcurl-impersonate-chrome.so.4 / libcurl-impersonate.so.4 on the system
 *
 * Every function here is prefixed anubis_imp_ and stays internal to the fetch
 * layer; the public surface is in anubis_fetch.h.
 */
#ifndef ANUBIS_FETCH_IMPERSONATE_H
#define ANUBIS_FETCH_IMPERSONATE_H

#include <stddef.h>

#include "anubis_fetch.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load and inspect the library. Idempotent. Returns 0 when a library with
 * curl_easy_impersonate is available, -1 otherwise (reason in the error
 * buffer, if given). */
int anubis_imp_load(char *err, size_t err_cap);

/* Path of the loaded library, or NULL. */
const char *anubis_imp_path(void);

/* "libcurl/8.1.1 BoringSSL ..." as the library reports itself, or NULL. */
const char *anubis_imp_version(void);

/* Resolve a profile name to one the loaded build actually has.
 * "chrome" picks the newest chrome, "edge"/"safari" likewise; an exact name
 * such as "chrome116" is used as given. Returns 0 and writes the concrete
 * name into out, or -1 when the profile does not exist. */
int anubis_imp_resolve_target(const char *name, char *out, size_t cap);

/* One easy handle: cookies persist across requests on it, which is what the
 * Anubis pass-challenge dance needs. */
typedef struct anubis_imp_handle anubis_imp_handle;

/* The handle remembers its impersonation profile; close() frees it. */
anubis_imp_handle *anubis_imp_open(const char *target, char *err, size_t err_cap);
void anubis_imp_close(anubis_imp_handle *h);

/* Concrete profile the handle was opened with (e.g. "chrome116"). */
const char *anubis_imp_handle_target(const anubis_imp_handle *h);

/* GET url on this handle. `insecure` skips certificate verification.
 * Fills out->status/body/body_len/final_url, or out->error on failure.
 * Redirects are followed by the library; cookies are handled by it too.
 *
 * When `headers` is not NULL it receives a malloc'd copy of every response
 * header block the transfer produced (status lines included) and `headers_len`
 * its size; both are NULL/0 when the library gave none. Free the pointer. */
int anubis_imp_get(anubis_imp_handle *h, const char *url, int insecure,
                   anubis_page *out, char **headers, size_t *headers_len);

/* Seed the handle's Cookie header for the first transfer (the cookie engine
 * still records Set-Cookie on its own). NULL clears it. */
void anubis_imp_set_cookies(anubis_imp_handle *h, const char *cookies);

/* Serialize the handle's cookie engine as "name=value; name2=value2".
 * Returns 0 and a malloc'd string (never NULL; "" when empty), -1 when the
 * library has no CURLINFO_COOKIELIST symbol. Free the pointer. */
int anubis_imp_cookies(anubis_imp_handle *h, char **out);

#ifdef __cplusplus
}
#endif

#endif /* ANUBIS_FETCH_IMPERSONATE_H */
