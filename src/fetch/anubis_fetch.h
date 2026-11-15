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

/* Version of this layer, for MCP serverInfo and API /version. */
const char *anubis_fetch_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ANUBIS_FETCH_H */
