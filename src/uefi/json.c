#include <stddef.h>

#include "json.h"

/* Find "key" at object depth 1, then the value after the colon. Deliberately
 * dumb: it scans, it does not build a tree. Payloads come from our own client,
 * not from the internet, and the frame is length-checked before we get here. */
static const char *find_key(const char *js, const char *key) {
    size_t klen = 0;
    const char *p;
    int depth = 0;
    int instr = 0;

    while (key[klen] != '\0') {
        klen++;
    }
    for (p = js; *p != '\0'; p++) {
        if (instr) {
            if (*p == '\\' && p[1] != '\0') {
                p++;
            } else if (*p == '"') {
                instr = 0;
            }
            continue;
        }
        if (*p == '"') {
            if (depth == 1 && (size_t)(p - js) >= 0) {
                /* candidate key? */
                const char *q = p + 1;
                size_t i = 0;
                while (i < klen && q[i] == key[i]) {
                    i++;
                }
                /* exact match, quote-terminated, followed by ':' */
                if (i == klen && q[i] == '"') {
                    const char *r = q + i + 1;
                    while (*r == ' ' || *r == '\t') {
                        r++;
                    }
                    if (*r == ':') {
                        return r + 1;
                    }
                }
            }
            instr = 1;
            continue;
        }
        if (*p == '{' || *p == '[') {
            depth++;
        } else if (*p == '}' || *p == ']') {
            depth--;
            if (depth <= 0) {
                return NULL;
            }
        }
    }
    return NULL;
}

int json_str(const char *js, const char *key, char *out, size_t n) {
    const char *v = find_key(js, key);
    size_t i = 0;

    if (v == NULL) {
        return -1;
    }
    while (*v == ' ' || *v == '\t') {
        v++;
    }
    if (*v != '"') {
        return -1;
    }
    v++;
    while (*v != '\0' && *v != '"' && i + 1 < n) {
        if (*v == '\\' && v[1] != '\0') {
            v++;
        }
        out[i++] = *v++;
    }
    out[i] = '\0';
    return 0;
}

long long json_int(const char *js, const char *key, long long def) {
    const char *v = find_key(js, key);
    long long sign = 1, val = 0;
    int any = 0;

    if (v == NULL) {
        return def;
    }
    while (*v == ' ' || *v == '\t') {
        v++;
    }
    if (*v == '-') {
        sign = -1;
        v++;
    } else if (*v == '+') {
        v++;
    }
    while (*v >= '0' && *v <= '9') {
        val = val * 10 + (*v - '0');
        any = 1;
        v++;
    }
    return any ? sign * val : def;
}

size_t json_extent(const char *p, size_t n) {
    int depth = 0;
    int instr = 0;
    size_t i;

    for (i = 0; i < n; i++) {
        char c = p[i];

        if (instr) {
            if (c == '\\' && i + 1 < n) {
                i++;
            } else if (c == '"') {
                instr = 0;
            }
            continue;
        }
        if (c == '"') {
            instr = 1;
        } else if (c == '{' || c == '[') {
            depth++;
        } else if (c == '}' || c == ']') {
            depth--;
            if (depth == 0) {
                return i + 1;
            }
        }
    }
    return 0;
}
