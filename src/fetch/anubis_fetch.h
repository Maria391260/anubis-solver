/*
 * anubis_fetch.h — fetch an Anubis-protected URL from C.
 *
 * The library behind the C MCP server and the C API service: one call that
 * follows a challenge page, solves it with libanubis_solver, submits the
 * answer with the cookies the server set, and returns the real page.
 *
 *   anubis_page page;
 *   if (anubis_fetch_get("https://example.com/", &page) == 0) { ... }
 *
 * TLS is OpenSSL, verification on by default (set ANUBIS_INSECURE=1 to skip,
 * for testing against a self-signed gate). The solver is the .so this project
 * already builds, so every algorithm it supports is supported here.
 *
 * When a curl-impersonate library is present — $ANUBIS_IMPERSONATE_LIB, or
 * 3rd_party/curl-impersonate/lib/ in this repository — the requests go through
 * it instead, with a real browser TLS fingerprint (Chrome by default). The
 * hand-written OpenSSL client stays as the fallback and is still what runs when
 * no such library exists.
 */
#ifndef ANUBIS_FETCH_H
#define ANUBIS_FETCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int status;        /* final HTTP status */
    char *body;        /* NUL-terminated; may contain NULs, use body_len */
    size_t body_len;
    char *final_url;   /* after redirects */
    char error[256];   /* empty on success */
} anubis_page;

/* Fetch a URL, solving any Anubis challenge on the way.
 * Returns 0 on success (out->status is the final status), -1 on failure with
 * out->error set. Free with anubis_page_free(). */
int anubis_fetch_get(const char *url, anubis_page *out);
void anubis_page_free(anubis_page *page);

/* Headers seen along the way, next to the page. A separate struct because
 * anubis_page is allocated by the caller: growing it would write past the end
 * of every program compiled against an older release. */
typedef struct {
    anubis_page page;   /* status, body, final_url, error as usual */
    char *headers;      /* every response header block seen, in order, CRLF */
    size_t headers_len; /* NUL-terminated anyway, but may contain NULs */
} anubis_page_dump;

/* Same fetch as anubis_fetch_get_with(), plus the raw header blocks of every
 * response it went through: the challenge 401, the pass-challenge reply, any
 * redirect and the final page. Each block starts with its own status line, so
 * concatenation is self-describing. Free with anubis_page_dump_free(). */
int anubis_fetch_get_dump(const char *url, const char *target,
                          anubis_page_dump *out);
void anubis_page_dump_free(anubis_page_dump *dump);

/* Same, but choose the impersonation profile for this call only, so a
 * multi-threaded caller (the API service) never races on global state.
 * `target` may be NULL for the default, "off" for the built-in OpenSSL
 * client, "auto" to impersonate when a library is present, or a profile name
 * such as "chrome", "chrome116", "edge101", "safari15_5". Thread-safe. */
int anubis_fetch_get_with(const char *url, const char *target, anubis_page *out);

/* Default profile for anubis_fetch_get(): NULL/"auto" (impersonate when
 * available), "off" (never), or a profile name. The default is also settable
 * through $ANUBIS_IMPERSONATE, which takes precedence over this setting.
 * Returns 0, or -1 when the profile is not in the loaded library. */
int anubis_fetch_set_impersonate(const char *target);

/* The configured profile, "" when impersonation is disabled. */
const char *anubis_fetch_impersonate(void);

/* 1 when a usable libcurl-impersonate is installed, 0 otherwise. */
int anubis_fetch_impersonate_available(void);

/* Transport used by the calling thread's most recent fetch:
 * "openssl", or "curl-impersonate/<profile>". Never NULL. */
const char *anubis_fetch_transport(void);

/* Version of this layer, for MCP serverInfo and API /version. */
const char *anubis_fetch_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ANUBIS_FETCH_H */
