/*
 * anubis_fetch.c — the C client behind the MCP server and the API service.
 *
 * One job: GET a URL that may be behind Anubis, solve whatever it asks for
 * with libanubis_solver, submit the answer with the cookies the server set,
 * and hand back the real page. The challenge protocol here is the one the gate
 * implements: a JSON payload in <script id="anubis_challenge">, a
 * pass-challenge GET with id/response/nonce, and a cookie that follows.
 *
 * Deliberately small: HTTP/1.1, Content-Length or chunked, one request per
 * connection (Connection: close), a cookie list, and redirects. No HTTP/2, no
 * connection reuse. This is a client for a tool, not a browser.
 */
#include "anubis_fetch.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "anubis-solver.h"

#define MAX_HEADER 65536
#define MAX_BODY (32u << 20)
#define UA "anubis-c/2.0.0"

/* ------------------------------------------------------------------ util */

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

typedef struct {
    char *data;
    size_t len, cap;
} buf;

static int buf_add(buf *b, const void *p, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        char *d = realloc(b->data, cap);
        if (!d) return -1;
        b->data = d;
        b->cap = cap;
    }
    memcpy(b->data + b->len, p, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 0;
}

static void seterr(anubis_page *out, const char *msg) {
    snprintf(out->error, sizeof out->error, "%s", msg);
}

static char *url_encode(const char *s) {
    static const char hex[] = "0123456789ABCDEF";
    size_t n = strlen(s), o = 0;
    char *r = malloc(n * 3 + 1);
    if (!r) return NULL;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            r[o++] = (char)c;
        } else {
            r[o++] = '%';
            r[o++] = hex[c >> 4];
            r[o++] = hex[c & 15];
        }
    }
    r[o] = '\0';
    return r;
}

/* --------------------------------------------------------------- cookies */

typedef struct {
    char **items;
    size_t n, cap;
} jar;

static void jar_set(jar *j, const char *pair) {
    char name[128];
    const char *eq = strchr(pair, '=');
    size_t nl;
    if (!eq) return;
    nl = (size_t)(eq - pair);
    if (nl == 0 || nl >= sizeof name) return;
    memcpy(name, pair, nl);
    name[nl] = '\0';
    for (size_t i = 0; i < j->n; i++) {
        if (strncmp(j->items[i], name, nl) == 0 && j->items[i][nl] == '=') {
            free(j->items[i]);
            j->items[i] = xstrdup(pair);
            return;
        }
    }
    if (j->n == j->cap) {
        size_t cap = j->cap ? j->cap * 2 : 8;
        char **p = realloc(j->items, cap * sizeof *p);
        if (!p) return;
        j->items = p;
        j->cap = cap;
    }
    j->items[j->n++] = xstrdup(pair);
}

static char *jar_header(jar *j) {
    size_t total = 1;
    for (size_t i = 0; i < j->n; i++) total += strlen(j->items[i]) + 2;
    char *h = malloc(total);
    if (!h) return NULL;
    h[0] = '\0';
    for (size_t i = 0; i < j->n; i++) {
        if (i) strcat(h, "; ");
        strcat(h, j->items[i]);
    }
    return h;
}

static void jar_free(jar *j) {
    for (size_t i = 0; i < j->n; i++) free(j->items[i]);
    free(j->items);
    j->items = NULL;
    j->n = j->cap = 0;
}

/* ------------------------------------------------------------------- URL */

typedef struct {
    char scheme[8];
    char host[256];
    int port;
    char path[4096];
} url_t;

static int parse_url(const char *url, url_t *u) {
    const char *p = strstr(url, "://");
    const char *rest, *slash;
    size_t sl;
    if (!p) return -1;
    sl = (size_t)(p - url);
    if (sl >= sizeof u->scheme) return -1;
    memcpy(u->scheme, url, sl);
    u->scheme[sl] = '\0';
    rest = p + 3;
    slash = strchr(rest, '/');
    {
        const char *host_end = slash ? slash : rest + strlen(rest);
        const char *colon = memchr(rest, ':', (size_t)(host_end - rest));
        size_t hl = (size_t)((colon ? colon : host_end) - rest);
        if (hl == 0 || hl >= sizeof u->host) return -1;
        memcpy(u->host, rest, hl);
        u->host[hl] = '\0';
        u->port = strcmp(u->scheme, "https") == 0 ? 443 : 80;
        if (colon) u->port = atoi(colon + 1);
        snprintf(u->path, sizeof u->path, "%s", slash ? slash : "/");
    }
    if (strcmp(u->scheme, "http") && strcmp(u->scheme, "https")) return -1;
    return 0;
}

/* --------------------------------------------------------- one HTTP GET */

static int tcp_connect(const char *host, int port) {
    struct addrinfo hints, *res = NULL, *ai;
    char portstr[16];
    int fd = -1;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(portstr, sizeof portstr, "%d", port);
    if (getaddrinfo(host, portstr, &hints, &res) != 0) return -1;
    for (ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        {
            struct timeval tv = {20, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        }
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static int ssl_read_all(SSL *ssl, buf *out) {
    char tmp[16384];
    for (;;) {
        int n = SSL_read(ssl, tmp, sizeof tmp);
        if (n <= 0) {
            int e = SSL_get_error(ssl, n);
            if (e == SSL_ERROR_ZERO_RETURN) return 0;
            if (e == SSL_ERROR_SYSCALL && ERR_peek_error() == 0) return 0;
            return -1;
        }
        if (buf_add(out, tmp, (size_t)n) != 0) return -1;
    }
}

static int read_all(int fd, buf *out) {
    char tmp[16384];
    for (;;) {
        ssize_t n = read(fd, tmp, sizeof tmp);
        if (n == 0) return 0;
        if (n < 0) return (errno == EINTR) ? 0 : -1;
        if (buf_add(out, tmp, (size_t)n) != 0) return -1;
    }
}

static int decode_chunked(const char *in, size_t inlen, buf *out) {
    size_t i = 0;
    for (;;) {
        char sz[32];
        size_t k = 0, n;
        while (i < inlen && k < sizeof sz - 1 && in[i] != '\r' && in[i] != '\n') sz[k++] = in[i++];
        sz[k] = '\0';
        if (k == 0) break;
        n = (size_t)strtoul(sz, NULL, 16);
        while (i < inlen && in[i] != '\n') i++;
        if (i < inlen) i++;
        if (n == 0) break;
        if (i + n > inlen) n = inlen - i;
        if (buf_add(out, in + i, n) != 0) return -1;
        i += n;
        while (i < inlen && (in[i] == '\r' || in[i] == '\n')) i++;
    }
    return 0;
}

static int http_get(const char *url, jar *j, const char *extra_headers,
                    anubis_page *out) {
    url_t u;
    int fd = -1, rc = -1;
    SSL_CTX *ctx = NULL;
    SSL *ssl = NULL;
    buf raw = {0};
    char *cookie = NULL;
    char *req = NULL;

    if (parse_url(url, &u) != 0) {
        seterr(out, "bad url");
        return -1;
    }
    fd = tcp_connect(u.host, u.port);
    if (fd < 0) {
        seterr(out, "connect failed");
        return -1;
    }
    if (strcmp(u.scheme, "https") == 0) {
        const char *insecure = getenv("ANUBIS_INSECURE");
        ctx = SSL_CTX_new(TLS_client_method());
        if (!ctx) {
            seterr(out, "tls init failed");
            goto done;
        }
        SSL_CTX_set_default_verify_paths(ctx);
        SSL_CTX_set_verify(ctx, (insecure && *insecure == '1') ? SSL_VERIFY_NONE
                                                               : SSL_VERIFY_PEER,
                           NULL);
        ssl = SSL_new(ctx);
        if (!ssl || SSL_set_fd(ssl, fd) != 1) {
            seterr(out, "tls setup failed");
            goto done;
        }
        SSL_set_tlsext_host_name(ssl, u.host);
        SSL_set1_host(ssl, u.host);
        if (SSL_connect(ssl) != 1) {
            seterr(out, "tls handshake failed");
            goto done;
        }
    }

    cookie = jar_header(j);
    {
        size_t cap = 2048 + strlen(u.path) + strlen(u.host) + (cookie ? strlen(cookie) : 0);
        req = malloc(cap);
        snprintf(req, cap,
                 "GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: %s\r\n"
                 "Accept: text/html,application/json;q=0.9,*/*;q=0.8\r\n"
                 "Accept-Encoding: identity\r\n%s%s%s"
                 "Connection: close\r\n\r\n",
                 u.path, u.host, UA, cookie ? "Cookie: " : "", cookie ? cookie : "",
                 extra_headers ? extra_headers : "");
    }
    if (ssl) {
        if (SSL_write(ssl, req, (int)strlen(req)) <= 0) {
            seterr(out, "tls write failed");
            goto done;
        }
        if (ssl_read_all(ssl, &raw) != 0) {
            seterr(out, "tls read failed");
            goto done;
        }
    } else {
        if (write(fd, req, strlen(req)) < 0) {
            seterr(out, "write failed");
            goto done;
        }
        if (read_all(fd, &raw) != 0) {
            seterr(out, "read failed");
            goto done;
        }
    }

    {
        char *hdr_end = strstr(raw.data, "\r\n\r\n");
        char *body = hdr_end ? hdr_end + 4 : raw.data;
        size_t body_len = hdr_end ? raw.len - (size_t)(body - raw.data) : raw.len;
        char *line = raw.data;
        int chunked = 0;
        long content_length = -1;

        out->status = 0;
        if (strncmp(line, "HTTP/", 5) == 0) {
            char *sp = strchr(line, ' ');
            if (sp) out->status = atoi(sp + 1);
        }
        /* headers */
        {
            char *p = strstr(raw.data, "\r\n");
            while (p && p < body) {
                char *next = strstr(p + 2, "\r\n");
                size_t len = next ? (size_t)(next - (p + 2)) : 0;
                if (strncasecmp(p + 2, "Set-Cookie:", 11) == 0) {
                    char v[1024];
                    size_t k = 0;
                    const char *s = p + 2 + 11;
                    while (*s == ' ') s++;
                    while (k < sizeof v - 1 && s + k < (next ? next : body) && s[k] != ';') {
                        v[k] = s[k];
                        k++;
                    }
                    v[k] = '\0';
                    if (k) jar_set(j, v);
                } else if (strncasecmp(p + 2, "Content-Length:", 15) == 0) {
                    content_length = atol(p + 2 + 15);
                } else if (strncasecmp(p + 2, "Transfer-Encoding:", 18) == 0 &&
                           strcasestr(p + 2, "chunked")) {
                    chunked = 1;
                } else if (strncasecmp(p + 2, "Location:", 9) == 0) {
                    char v[2048];
                    size_t k = 0;
                    const char *s = p + 2 + 9;
                    while (*s == ' ') s++;
                    while (k < sizeof v - 1 && s + k < (next ? next : body) && s[k] != '\r') {
                        v[k] = s[k];
                        k++;
                    }
                    v[k] = '\0';
                    free(out->final_url);
                    out->final_url = xstrdup(v);
                }
                if (!next) break;
                p = next;
                if (len > MAX_HEADER) break;
            }
        }
        if (chunked) {
            buf dec = {0};
            if (decode_chunked(body, body_len, &dec) == 0) {
                out->body = dec.data;
                out->body_len = dec.len;
            } else {
                seterr(out, "chunked decode failed");
                goto done;
            }
        } else {
            size_t take = body_len;
            if (content_length >= 0 && (size_t)content_length < take) take = (size_t)content_length;
            if (take > MAX_BODY) take = MAX_BODY;
            out->body = malloc(take + 1);
            if (!out->body) {
                seterr(out, "oom");
                goto done;
            }
            memcpy(out->body, body, take);
            out->body[take] = '\0';
            out->body_len = take;
        }
    }
    rc = 0;

done:
    if (ssl) SSL_free(ssl);
    if (ctx) SSL_CTX_free(ctx);
    if (fd >= 0) close(fd);
    free(raw.data);
    free(cookie);
    free(req);
    return rc;
}

/* ------------------------------------------------------------------ JSON */

static int json_string(const char *json, const char *key, char *out, size_t cap) {
    char pat[64];
    const char *p;
    size_t k = 0;
    snprintf(pat, sizeof pat, "\"%s\"", key);
    p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ':') p++;
    if (*p != '"') return 0;
    p++;
    while (*p && *p != '"' && k < cap - 1) {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': out[k++] = '\n'; break;
                case 't': out[k++] = '\t'; break;
                case '"': out[k++] = '"'; break;
                case '\\': out[k++] = '\\'; break;
                default: out[k++] = *p; break;
            }
            p++;
        } else {
            out[k++] = *p++;
        }
    }
    out[k] = '\0';
    return k > 0;
}

static int json_int(const char *json, const char *key, long *out) {
    char pat[64];
    const char *p;
    snprintf(pat, sizeof pat, "\"%s\"", key);
    p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ':') p++;
    if (!isdigit((unsigned char)*p) && *p != '-') return 0;
    *out = strtol(p, NULL, 10);
    return 1;
}

/* ------------------------------------------------------------------ solve */

static int solve_algorithm(const char *algorithm, const char *challenge,
                           int difficulty, uint32_t *nonce, char *digest,
                           size_t digest_cap) {
    int rc;
    if (!strcasecmp(algorithm, "fast") || !strcasecmp(algorithm, "slow")) {
        uint64_t n64 = 0;
        rc = anubis_solve_nibbles(challenge, difficulty, 0, 0, 0.0, &n64, digest, NULL);
        *nonce = (uint32_t)n64;
    } else if (!strcasecmp(algorithm, "sha256") || !strcasecmp(algorithm, "wasm-sha256")) {
        rc = anubis_solve_bits(challenge, difficulty, 0, 0, nonce, digest, NULL);
    } else if (!strcasecmp(algorithm, "argon2id")) {
        rc = anubis_solve_argon2id(challenge, difficulty, 0, 0, nonce, digest, NULL);
    } else if (!strcasecmp(algorithm, "hashx")) {
        rc = anubis_solve_hashx(challenge, difficulty, 0, 0, nonce, digest, NULL);
    } else if (!strcasecmp(algorithm, "tensor")) {
        rc = anubis_solve_tensor(challenge, difficulty, 0, 0, nonce, digest, NULL);
    } else {
        return -1;
    }
    (void)digest_cap;
    return rc == ANUBIS_OK ? 0 : -1;
}

static int is_challenge_page(const char *body) {
    return body && (strstr(body, "anubis_challenge") != NULL ||
                    strstr(body, "Anubis") != NULL);
}

static char *origin_of(const char *url) {
    url_t u;
    char *o;
    if (parse_url(url, &u) != 0) return NULL;
    o = malloc(300);
    if (!o) return NULL;
    if ((strcmp(u.scheme, "https") == 0 && u.port == 443) ||
        (strcmp(u.scheme, "http") == 0 && u.port == 80)) {
        snprintf(o, 300, "%s://%s", u.scheme, u.host);
    } else {
        snprintf(o, 300, "%s://%s:%d", u.scheme, u.host, u.port);
    }
    return o;
}

static char *join_url(const char *base, const char *path) {
    size_t n;
    char *r;
    if (strstr(path, "://")) return xstrdup(path);
    if (path[0] == '/') {
        char *origin = origin_of(base);
        n = strlen(origin) + strlen(path) + 1;
        r = malloc(n);
        snprintf(r, n, "%s%s", origin, path);
        free(origin);
        return r;
    }
    n = strlen(base) + strlen(path) + 2;
    r = malloc(n);
    snprintf(r, n, "%s/%s", base, path);
    return r;
}

static void sleep_ms(long ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

/* ------------------------------------------------------------------- API */

const char *anubis_fetch_version(void) {
    return "2.0.0 \"shaoogle\"";
}

void anubis_page_free(anubis_page *page) {
    if (!page) return;
    free(page->body);
    free(page->final_url);
    memset(page, 0, sizeof *page);
}

int anubis_fetch_get(const char *url, anubis_page *out) {
    jar j = {0};
    char *current = xstrdup(url);
    int attempt;
    if (!out) return -1;
    memset(out, 0, sizeof *out);
    if (!current) return -1;

    for (attempt = 0; attempt < 8; attempt++) {
        anubis_page page = {0};
        if (http_get(current, &j, NULL, &page) != 0) {
            snprintf(out->error, sizeof out->error, "%s", page.error);
            anubis_page_free(&page);
            free(current);
            jar_free(&j);
            return -1;
        }
        if (page.final_url && (page.status == 301 || page.status == 302 ||
                               page.status == 303 || page.status == 307 ||
                               page.status == 308)) {
            char *next = join_url(current, page.final_url);
            anubis_page_free(&page);
            free(current);
            current = next;
            continue;
        }
        if (is_challenge_page(page.body) && strstr(page.body, "anubis_challenge")) {
            char algorithm[32] = "sha256", random_data[1024] = "", id[256] = "";
            long difficulty = 0;
            uint32_t nonce = 0;
            char digest[65] = "";
            char *origin, *pass, *enc_id, *enc_rd, *enc_resp, *enc_redir;
            anubis_page solved = {0};

            json_string(page.body, "algorithm", algorithm, sizeof algorithm);
            json_string(page.body, "randomData", random_data, sizeof random_data);
            json_string(page.body, "id", id, sizeof id);
            json_int(page.body, "difficulty", &difficulty);

            if (random_data[0] == '\0') {
                snprintf(out->error, sizeof out->error, "challenge without randomData");
                anubis_page_free(&page);
                free(current);
                jar_free(&j);
                return -1;
            }
            origin = origin_of(current);
            if (!origin) {
                snprintf(out->error, sizeof out->error, "bad url");
                anubis_page_free(&page);
                free(current);
                jar_free(&j);
                return -1;
            }
            enc_id = url_encode(id);
            enc_rd = url_encode(random_data);
            enc_redir = url_encode(current);

            if (!strcasecmp(algorithm, "metarefresh")) {
                sleep_ms((long)difficulty * 800 + 200);
                pass = malloc(strlen(origin) + strlen(enc_rd) + strlen(enc_redir) +
                              strlen(enc_id) + 256);
                sprintf(pass,
                        "%s/.within.website/x/cmd/anubis/api/pass-challenge"
                        "?challenge=%s&id=%s&redir=%s&elapsedTime=%ld",
                        origin, enc_rd, enc_id, enc_redir, difficulty * 800 + 200);
            } else if (!strcasecmp(algorithm, "preact")) {
                char hash[65];
                anubis_sha256_hex(random_data, strlen(random_data), hash);
                sleep_ms((long)difficulty * 100 + 100);
                enc_resp = url_encode(hash);
                pass = malloc(strlen(origin) + strlen(enc_resp) + strlen(enc_redir) +
                              strlen(enc_id) + 256);
                sprintf(pass,
                        "%s/.within.website/x/cmd/anubis/api/pass-challenge"
                        "?result=%s&id=%s&redir=%s&elapsedTime=%ld",
                        origin, enc_resp, enc_id, enc_redir, difficulty * 100 + 100);
                free(enc_resp);
            } else {
                if (solve_algorithm(algorithm, random_data, (int)difficulty, &nonce,
                                    digest, sizeof digest) != 0) {
                    snprintf(out->error, sizeof out->error,
                             "cannot solve %s difficulty %ld", algorithm, difficulty);
                    free(origin); free(enc_id); free(enc_rd); free(enc_redir);
                    anubis_page_free(&page);
                    free(current);
                    jar_free(&j);
                    return -1;
                }
                enc_resp = url_encode(digest);
                pass = malloc(strlen(origin) + strlen(enc_resp) + strlen(enc_redir) +
                              strlen(enc_id) + 256);
                sprintf(pass,
                        "%s/.within.website/x/cmd/anubis/api/pass-challenge"
                        "?id=%s&response=%s&nonce=%u&redir=%s&elapsedTime=1",
                        origin, enc_id, enc_resp, nonce, enc_redir);
                free(enc_resp);
            }
            free(origin);
            free(enc_id);
            free(enc_rd);
            free(enc_redir);

            if (http_get(pass, &j, NULL, &solved) != 0) {
                snprintf(out->error, sizeof out->error, "pass-challenge failed: %.200s",
                         solved.error);
                free(pass);
                anubis_page_free(&solved);
                anubis_page_free(&page);
                free(current);
                jar_free(&j);
                return -1;
            }
            free(pass);
            if (solved.final_url && (solved.status == 301 || solved.status == 302 ||
                                     solved.status == 303 || solved.status == 307)) {
                /* the cookie is what matters; keep the redirect target unused */
            }
            anubis_page_free(&solved);
            anubis_page_free(&page);
            continue; /* re-fetch the original URL with the new cookie */
        }

        *out = page;
        out->final_url = xstrdup(current);
        free(current);
        jar_free(&j);
        return 0;
    }
    snprintf(out->error, sizeof out->error, "too many challenge rounds");
    free(current);
    jar_free(&j);
    return -1;
}
