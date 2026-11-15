/*
 * fetch_impersonate.c — runtime binding to libcurl-impersonate.
 *
 * Why dlopen and not a link: the library is a patched libcurl (curl-impersonate)
 * whose TLS stack is BoringSSL, and it is not packaged by distributions. Linking
 * it would make every build of this project depend on a library most machines do
 * not have, and would drag BoringSSL into the .so that people load for solving
 * proofs. Instead this file loads it when it is there and the rest of the fetch
 * layer falls back to its own OpenSSL client when it is not.
 *
 * The ABI used is the stable libcurl 4 ABI: curl_global_init, curl_easy_* and
 * curl_easy_impersonate, all declared here by hand so no curl headers are
 * needed at build time. The numeric option values below are part of that ABI
 * (libcurl never renumbers them); each is followed by the curl.h expression it
 * corresponds to, for review.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* strcasestr, strdup, readlink */
#endif
#include "fetch_impersonate.h"

#include <dlfcn.h>
#include <glob.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>

/* ---- the stable libcurl ABI, by hand ---------------------------------- */

typedef void CURL;
typedef int CURLcode;

#define CURLE_OK 0
#define CURL_GLOBAL_ALL 3
#define CURLINFO_EFFECTIVE_URL 0x100001   /* CURLINFO_STRING + 1 */
#define CURLINFO_RESPONSE_CODE 0x200002   /* CURLINFO_LONG   + 2 */

/* CURLOPTTYPE_OBJECTPOINT = 10000, _FUNCTIONPOINT = 20000, _LONG = 0 */
#define CURLOPT_WRITEDATA 10001           /* OBJECTPOINT + 1  */
#define CURLOPT_URL 10002                 /* OBJECTPOINT + 2  */
#define CURLOPT_ERRORBUFFER 10010         /* OBJECTPOINT + 10 */
#define CURLOPT_WRITEFUNCTION 20011       /* FUNCTIONPOINT + 11 */
#define CURLOPT_COOKIEFILE 10031          /* OBJECTPOINT + 31 */
#define CURLOPT_FOLLOWLOCATION 52         /* LONG + 52 */
#define CURLOPT_SSL_VERIFYPEER 64         /* LONG + 64 */
#define CURLOPT_MAXREDIRS 68              /* LONG + 68 */
#define CURLOPT_SSL_VERIFYHOST 81         /* LONG + 81 */
#define CURLOPT_NOSIGNAL 99               /* LONG + 99 */
#define CURLOPT_ACCEPT_ENCODING 10102     /* OBJECTPOINT + 102 */
#define CURLOPT_TIMEOUT_MS 155            /* LONG + 155 */
#define CURLOPT_CONNECTTIMEOUT_MS 156     /* LONG + 156 */

struct imp_api {
    void *lib;
    char path[PATH_MAX];
    const char *version;
    CURLcode (*global_init)(long);
    void (*global_cleanup)(void);
    CURL *(*easy_init)(void);
    void (*easy_cleanup)(CURL *);
    CURLcode (*easy_setopt)(CURL *, int, ...);
    CURLcode (*easy_perform)(CURL *);
    CURLcode (*easy_getinfo)(CURL *, int, ...);
    CURLcode (*easy_impersonate)(CURL *, const char *, int);
    const char *(*easy_strerror)(CURLcode);
};

static struct imp_api g_imp;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_loaded = -1; /* -1 not tried, 0 loaded, 1 not found */
static char g_err[512];

/* ---- loading ---------------------------------------------------------- */

static int bind_symbols(void *lib, const char *path) {
    struct imp_api a;
    memset(&a, 0, sizeof a);
    a.lib = lib;
    snprintf(a.path, sizeof a.path, "%s", path);

    *(void **)&a.global_init = dlsym(lib, "curl_global_init");
    *(void **)&a.global_cleanup = dlsym(lib, "curl_global_cleanup");
    *(void **)&a.easy_init = dlsym(lib, "curl_easy_init");
    *(void **)&a.easy_cleanup = dlsym(lib, "curl_easy_cleanup");
    *(void **)&a.easy_setopt = dlsym(lib, "curl_easy_setopt");
    *(void **)&a.easy_perform = dlsym(lib, "curl_easy_perform");
    *(void **)&a.easy_getinfo = dlsym(lib, "curl_easy_getinfo");
    *(void **)&a.easy_strerror = dlsym(lib, "curl_easy_strerror");
    *(void **)&a.easy_impersonate = dlsym(lib, "curl_easy_impersonate");
    {
        const char *(*fn_version)(void) = dlsym(lib, "curl_version");
        a.version = fn_version ? fn_version() : NULL;
    }

    if (!a.global_init || !a.easy_init || !a.easy_cleanup || !a.easy_setopt ||
        !a.easy_perform || !a.easy_getinfo) {
        snprintf(g_err, sizeof g_err, "%s: not a libcurl (%s)", path, dlerror());
        return -1;
    }
    if (!a.easy_impersonate) {
        snprintf(g_err, sizeof g_err,
                 "%s: plain libcurl without curl_easy_impersonate; "
                 "this must be a curl-impersonate build", path);
        return -1;
    }
    if (a.global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
        snprintf(g_err, sizeof g_err, "%s: curl_global_init failed", path);
        return -1;
    }
    g_imp = a;
    return 0;
}

static int try_path(const char *path) {
    void *lib;
    if (!path || !*path) return -1;
    lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) return -1;
    if (bind_symbols(lib, path) != 0) {
        dlclose(lib);
        return -1;
    }
    return 0;
}

/* Directory of the running executable, without a trailing slash.
 * Linux-only (/proc); elsewhere the caller just gets -1 and we rely on the
 * other candidates. */
static int exe_dir(char *out, size_t cap) {
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    char *slash;
    if (n <= 0) return -1;
    self[n] = '\0';
    slash = strrchr(self, '/');
    if (!slash) return -1;
    *slash = '\0';
    snprintf(out, cap, "%s", self);
    return 0;
}

/* A directory may hold the library under its full versioned name
 * (libcurl-impersonate-chrome.so.4.8.0) without the .so.4 symlink. glob() finds
 * whichever name is there, so `make impersonate-lib` only has to copy the file. */
static int try_glob(const char *pattern) {
    glob_t g;
    int rc = -1;
    if (glob(pattern, 0, NULL, &g) != 0) return -1;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        if (try_path(g.gl_pathv[i]) == 0) {
            rc = 0;
            break;
        }
    }
    globfree(&g);
    return rc;
}

static void load_once(void) {
    static const char *const rel[] = {
        "3rd_party/curl-impersonate/lib/libcurl-impersonate-chrome.so.4",
        "3rd_party/curl-impersonate/lib/libcurl-impersonate-chrome.so",
        "../3rd_party/curl-impersonate/lib/libcurl-impersonate-chrome.so.4",
        "../../3rd_party/curl-impersonate/lib/libcurl-impersonate-chrome.so.4",
    };
    static const char *const sys[] = {
        "libcurl-impersonate-chrome.so.4",
        "libcurl-impersonate-chrome.so",
        "libcurl-impersonate.so.4",
        "libcurl-impersonate.so",
    };
    const char *env = getenv("ANUBIS_IMPERSONATE_LIB");
    char dir[PATH_MAX];
    char cand[PATH_MAX + 128];

    if (env && *env && try_path(env) == 0) {
        g_loaded = 0;
        return;
    }
    for (size_t i = 0; i < sizeof rel / sizeof rel[0]; i++) {
        if (try_path(rel[i]) == 0) {
            g_loaded = 0;
            return;
        }
    }
    /* Then the same layout relative to the executable, and finally every
     * versioned name in ./3rd_party/curl-impersonate/lib. */
    for (size_t i = 0; i < sizeof rel / sizeof rel[0]; i++) {
        const char *slash = strrchr(rel[i], '/');
        char pat[PATH_MAX + 128];
        if (!slash) continue;
        snprintf(pat, sizeof pat, "%.*s*.so*", (int)(slash - rel[i] + 1), rel[i]);
        if (try_glob(pat) == 0) {
            g_loaded = 0;
            return;
        }
    }
    if (exe_dir(dir, sizeof dir) == 0) {
        for (size_t i = 0; i < sizeof rel / sizeof rel[0]; i++) {
            snprintf(cand, sizeof cand, "%s/%s", dir, rel[i]);
            if (try_path(cand) == 0) {
                g_loaded = 0;
                return;
            }
        }
    }
    for (size_t i = 0; i < sizeof sys / sizeof sys[0]; i++) {
        if (try_path(sys[i]) == 0) {
            g_loaded = 0;
            return;
        }
    }
    if (g_err[0] == '\0')
        snprintf(g_err, sizeof g_err,
                 "no libcurl-impersonate found (looked at $ANUBIS_IMPERSONATE_LIB, "
                 "./3rd_party/curl-impersonate/lib and the system loader path)");
    g_loaded = 1;
}

int anubis_imp_load(char *err, size_t err_cap) {
    pthread_once(&g_once, load_once);
    if (g_loaded != 0 && err && err_cap) {
        snprintf(err, err_cap, "%s", g_err[0] ? g_err : "no impersonation library");
    }
    return g_loaded == 0 ? 0 : -1;
}

const char *anubis_imp_path(void) {
    pthread_once(&g_once, load_once);
    return g_loaded == 0 ? g_imp.path : NULL;
}

const char *anubis_imp_version(void) {
    pthread_once(&g_once, load_once);
    return g_loaded == 0 ? g_imp.version : NULL;
}

/* ---- profiles --------------------------------------------------------- */

struct alias {
    const char *family;
    const char *const *names;
};

static const char *const chrome_names[] = {
    "chrome116", "chrome110", "chrome107", "chrome104", "chrome101",
    "chrome100", "chrome99", "chrome99_android", NULL,
};
static const char *const edge_names[] = {"edge101", "edge99", NULL};
static const char *const safari_names[] = {"safari15_5", "safari15_3", NULL};

static const struct alias aliases[] = {
    {"chrome", chrome_names},
    {"edge", edge_names},
    {"safari", safari_names},
};

/* Does this build accept the exact profile? A throwaway handle answers that
 * without committing to one; impersonate() on a fresh handle is cheap. */
static int profile_exists(const char *name) {
    CURL *h = g_imp.easy_init();
    CURLcode rc;
    if (!h) return 0;
    rc = g_imp.easy_impersonate(h, name, 1);
    g_imp.easy_cleanup(h);
    return rc == CURLE_OK;
}

int anubis_imp_resolve_target(const char *name, char *out, size_t cap) {
    pthread_once(&g_once, load_once);
    if (g_loaded != 0 || !name || !*name) return -1;
    if (profile_exists(name)) {
        snprintf(out, cap, "%s", name);
        return 0;
    }
    for (size_t i = 0; i < sizeof aliases / sizeof aliases[0]; i++) {
        if (strcmp(aliases[i].family, name) != 0) continue;
        for (const char *const *p = aliases[i].names; *p; p++) {
            if (profile_exists(*p)) {
                snprintf(out, cap, "%s", *p);
                return 0;
            }
        }
        return -1;
    }
    return -1;
}

/* ---- easy handles ----------------------------------------------------- */

struct anubis_imp_handle {
    CURL *curl;
    char target[64];
};

struct sink {
    char *data;
    size_t len, cap;
    int overflow;
};

#define IMP_MAX_BODY (32u << 20)

static size_t write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    struct sink *s = userdata;
    size_t n = size * nmemb;
    if (s->overflow) return 0; /* abort the transfer */
    if (s->len + n + 1 > IMP_MAX_BODY) {
        s->overflow = 1;
        return 0;
    }
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 4096;
        char *d;
        while (cap < s->len + n + 1) cap *= 2;
        d = realloc(s->data, cap);
        if (!d) {
            s->overflow = 1;
            return 0;
        }
        s->data = d;
        s->cap = cap;
    }
    memcpy(s->data + s->len, ptr, n);
    s->len += n;
    s->data[s->len] = '\0';
    return n;
}

anubis_imp_handle *anubis_imp_open(const char *target, char *err, size_t err_cap) {
    anubis_imp_handle *h;
    char resolved[64];
    CURLcode rc;

    if (anubis_imp_load(err, err_cap) != 0) return NULL;
    if (anubis_imp_resolve_target(target, resolved, sizeof resolved) != 0) {
        if (err && err_cap)
            snprintf(err, err_cap,
                     "impersonation profile '%s' is not in this build of "
                     "libcurl-impersonate", target ? target : "(null)");
        return NULL;
    }
    h = calloc(1, sizeof *h);
    if (!h) {
        if (err && err_cap) snprintf(err, err_cap, "oom");
        return NULL;
    }
    h->curl = g_imp.easy_init();
    if (!h->curl) {
        free(h);
        if (err && err_cap) snprintf(err, err_cap, "curl_easy_init failed");
        return NULL;
    }
    rc = g_imp.easy_impersonate(h->curl, resolved, 1);
    if (rc != CURLE_OK) {
        g_imp.easy_cleanup(h->curl);
        free(h);
        if (err && err_cap)
            snprintf(err, err_cap, "impersonate %s failed: %s", resolved,
                     g_imp.easy_strerror ? g_imp.easy_strerror(rc) : "error");
        return NULL;
    }
    snprintf(h->target, sizeof h->target, "%s", resolved);
    g_imp.easy_setopt(h->curl, CURLOPT_FOLLOWLOCATION, 1L);
    g_imp.easy_setopt(h->curl, CURLOPT_MAXREDIRS, 10L);
    g_imp.easy_setopt(h->curl, CURLOPT_TIMEOUT_MS, 30000L);
    g_imp.easy_setopt(h->curl, CURLOPT_CONNECTTIMEOUT_MS, 10000L);
    g_imp.easy_setopt(h->curl, CURLOPT_NOSIGNAL, 1L);
    /* Empty string enables the in-memory cookie engine: the pass-challenge
     * cookie must travel on the next request without touching disk. */
    g_imp.easy_setopt(h->curl, CURLOPT_COOKIEFILE, "");
    g_imp.easy_setopt(h->curl, CURLOPT_ACCEPT_ENCODING, "");
    g_imp.easy_setopt(h->curl, CURLOPT_WRITEFUNCTION, write_cb);
    return h;
}

void anubis_imp_close(anubis_imp_handle *h) {
    if (!h) return;
    if (h->curl) g_imp.easy_cleanup(h->curl);
    free(h);
}

const char *anubis_imp_handle_target(const anubis_imp_handle *h) {
    return h ? h->target : NULL;
}

int anubis_imp_get(anubis_imp_handle *h, const char *url, int insecure,
                   anubis_page *out) {
    struct sink s = {0};
    char errbuf[256] = {0};
    CURLcode rc;
    long code = 0;
    char *effective = NULL;

    if (!h || !url || !out) return -1;
    if (g_imp.easy_setopt(h->curl, CURLOPT_URL, url) != CURLE_OK ||
        g_imp.easy_setopt(h->curl, CURLOPT_WRITEDATA, &s) != CURLE_OK ||
        g_imp.easy_setopt(h->curl, CURLOPT_ERRORBUFFER, errbuf) != CURLE_OK) {
        snprintf(out->error, sizeof out->error, "curl_easy_setopt failed");
        free(s.data);
        return -1;
    }
    g_imp.easy_setopt(h->curl, CURLOPT_SSL_VERIFYPEER, insecure ? 0L : 1L);
    g_imp.easy_setopt(h->curl, CURLOPT_SSL_VERIFYHOST, insecure ? 0L : 2L);

    rc = g_imp.easy_perform(h->curl);
    if (rc != CURLE_OK) {
        if (s.overflow)
            snprintf(out->error, sizeof out->error, "response larger than 32 MiB");
        else
            snprintf(out->error, sizeof out->error, "%s",
                     errbuf[0] ? errbuf
                               : (g_imp.easy_strerror ? g_imp.easy_strerror(rc)
                                                      : "transfer failed"));
        free(s.data);
        return -1;
    }
    g_imp.easy_getinfo(h->curl, CURLINFO_RESPONSE_CODE, &code);
    g_imp.easy_getinfo(h->curl, CURLINFO_EFFECTIVE_URL, &effective);

    out->status = (int)code;
    out->body = s.data ? s.data : calloc(1, 1);
    out->body_len = s.len;
    out->final_url = effective ? strdup(effective) : NULL;
    return 0;
}
