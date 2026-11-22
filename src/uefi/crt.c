/* The "libc" the firmware does not have.
 *
 * Everything here is a deliberate subset: pool-backed malloc, the memory and
 * string routines the hashing code needs, and a printf that copies to the UEFI
 * console. No files, no signals, no threads - the app is single-threaded by
 * design, because a TPU-free PoW check has nothing to overlap.
 */
#include <efi.h>
#include <efilib.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <stdio.h>
#include <efiser.h>
#include <string.h>

#include "shim.h"

static EFI_SYSTEM_TABLE *g_st;
static EFI_SERIAL_IO_PROTOCOL *g_ser;

void shim_init(EFI_SYSTEM_TABLE *st) {
    g_st = st;
    g_ser = NULL;
    /* Headless boot has no ConOut worth trusting, so log to COM1 as well.
     * QEMU with -serial file: captures this without a graphics console. */
    if (BS->LocateProtocol(&gEfiSerialIoProtocolGuid, NULL, (void **)&g_ser) ==
            EFI_SUCCESS && g_ser != NULL) {
        g_ser->SetAttributes(g_ser, 115200, 0, 0, 0, 8, 1);
    }
}

/* ---- console ---------------------------------------------------------- */

void anubis_console_write(const char *s) {
    CHAR16 buf[512];
    size_t i = 0;
    size_t len = 0;

    /* QEMU's debug console (port 0xE9). Independent of every firmware
     * protocol, so it works even when ConOut is a dead graphics console. */
    {
        const char *q = s;
        while (*q != '\0') {
            __asm__ volatile("outb %0, %1" : : "a"((unsigned char)*q), "Nd"((unsigned short)0xE9));
            q++;
        }
    }
    if (g_ser != NULL) {
        while (s[len] != '\0') {
            len++;
        }
        if (len > 0) {
            UINTN n = len;
            g_ser->Write(g_ser, &n, (void *)s);
        }
    }
    if (g_st == NULL || g_st->ConOut == NULL) {
        return;
    }
    while (*s != '\0') {
        if (i == 511) {
            buf[i] = 0;
            g_st->ConOut->OutputString(g_st->ConOut, buf);
            i = 0;
        }
        buf[i++] = (CHAR16)(unsigned char)*s++;
    }
    buf[i] = 0;
    g_st->ConOut->OutputString(g_st->ConOut, buf);
}

/* ---- memory ----------------------------------------------------------- */

#define HDR 16u

void *malloc(size_t n) {
    void *p = NULL;
    uint8_t *raw;

    if (g_st == NULL || g_st->BootServices == NULL) {
        return NULL;
    }
    raw = NULL;
    if (g_st->BootServices->AllocatePool(EfiLoaderData, n + HDR, &p) !=
            EFI_SUCCESS || p == NULL) {
        return NULL;
    }
    raw = (uint8_t *)p;
    *(size_t *)raw = n;
    return raw + HDR;
}

void free(void *p) {
    if (p != NULL && g_st != NULL) {
        g_st->BootServices->FreePool((uint8_t *)p - HDR);
    }
}

void *calloc(size_t n, size_t sz) {
    size_t total = n * sz;
    void *p = malloc(total);

    if (p != NULL) {
        memset(p, 0, total);
    }
    return p;
}

void *realloc(void *p, size_t n) {
    size_t old;
    void *q;

    if (p == NULL) {
        return malloc(n);
    }
    old = *(size_t *)((uint8_t *)p - HDR);
    q = malloc(n);
    if (q == NULL) {
        return NULL;
    }
    memcpy(q, p, old < n ? old : n);
    free(p);
    return q;
}

/* ---- string ----------------------------------------------------------- */

/* memcpy/memset come from gnu-efi's libefi, which provides them for its own
 * runtime; defining them here would collide at link time. */

void *memmove(void *dst, const void *src, size_t n) {
    uint8_t *d = dst;
    const uint8_t *s = src;

    if (d < s) {
        while (n-- > 0) {
            *d++ = *s++;
        }
    } else {
        while (n-- > 0) {
            d[n] = s[n];
        }
    }
    return dst;
}

int memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *x = a;
    const uint8_t *y = b;

    while (n-- > 0) {
        if (*x != *y) {
            return (int)*x - (int)*y;
        }
        x++;
        y++;
    }
    return 0;
}

void *memchr(const void *s, int c, size_t n) {
    const uint8_t *p = s;

    while (n-- > 0) {
        if (*p == (uint8_t)c) {
            return (void *)p;
        }
        p++;
    }
    return NULL;
}

size_t strlen(const char *s) {
    size_t n = 0;

    while (s[n] != '\0') {
        n++;
    }
    return n;
}

int strcmp(const char *a, const char *b) {
    while (*a != '\0' && *a == *b) {
        a++;
        b++;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n) {
    while (n-- > 0) {
        if (*a != *b) {
            return (int)(unsigned char)*a - (int)(unsigned char)*b;
        }
        if (*a == '\0') {
            return 0;
        }
        a++;
        b++;
    }
    return 0;
}

char *strcpy(char *dst, const char *src) {
    char *d = dst;

    while ((*d++ = *src++) != '\0') {
    }
    return dst;
}

char *strncpy(char *dst, const char *src, size_t n) {
    size_t i = 0;

    for (; i < n && src[i] != '\0'; i++) {
        dst[i] = src[i];
    }
    for (; i < n; i++) {
        dst[i] = '\0';
    }
    return dst;
}

char *strcat(char *dst, const char *src) {
    strcpy(dst + strlen(dst), src);
    return dst;
}

char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c) {
            return (char *)s;
        }
        if (*s == '\0') {
            return NULL;
        }
    }
}

char *strrchr(const char *s, int c) {
    const char *last = NULL;

    for (;; s++) {
        if (*s == (char)c) {
            last = s;
        }
        if (*s == '\0') {
            return (char *)last;
        }
    }
}

char *strstr(const char *h, const char *n) {
    size_t ln = strlen(n);

    if (ln == 0) {
        return (char *)h;
    }
    for (; *h != '\0'; h++) {
        if (*h == *n && memcmp(h, n, ln) == 0) {
            return (char *)h;
        }
    }
    return NULL;
}

char *strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);

    if (p != NULL) {
        memcpy(p, s, n);
    }
    return p;
}

/* ---- numbers ---------------------------------------------------------- */

long strtol(const char *s, char **end, int base) {
    long v = 0;
    int neg = 0;

    while (*s == ' ') {
        s++;
    }
    if (*s == '-') {
        neg = 1;
        s++;
    } else if (*s == '+') {
        s++;
    }
    if (base == 0) {
        base = 10;
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            base = 16;
            s += 2;
        } else if (s[0] == '0') {
            base = 8;
        }
    }
    for (;; s++) {
        int d;
        if (*s >= '0' && *s <= '9') {
            d = *s - '0';
        } else if (*s >= 'a' && *s <= 'z') {
            d = *s - 'a' + 10;
        } else if (*s >= 'A' && *s <= 'Z') {
            d = *s - 'A' + 10;
        } else {
            break;
        }
        if (d >= base) {
            break;
        }
        v = v * base + d;
    }
    if (end != NULL) {
        *end = (char *)s;
    }
    return neg ? -v : v;
}

unsigned long strtoul(const char *s, char **end, int base) {
    return (unsigned long)strtol(s, end, base);
}

int atoi(const char *s) { return (int)strtol(s, NULL, 10); }

long atol(const char *s) { return strtol(s, NULL, 10); }

void qsort(void *base, size_t n, size_t sz,
           int (*cmp)(const void *, const void *)) {
    uint8_t *b = base;
    uint8_t tmp[64];
    size_t i, j;

    for (i = 1; i < n; i++) {
        for (j = i; j > 0; j--) {
            if (cmp(b + (j - 1) * sz, b + j * sz) <= 0) {
                break;
            }
            if (sz > sizeof tmp) {
                return;
            }
            memcpy(tmp, b + (j - 1) * sz, sz);
            memcpy(b + (j - 1) * sz, b + j * sz, sz);
            memcpy(b + j * sz, tmp, sz);
        }
    }
}

char *getenv(const char *name) { return NULL; }

void abort(void) {
    anubis_console_write("\nanubis: abort()\n");
    for (;;) {
        __asm__ volatile("hlt");
    }
}

/* ---- printf ----------------------------------------------------------- */

static int emit(char *buf, size_t n, size_t *pos, char c) {
    if (*pos + 1 < n) {
        buf[*pos] = c;
    }
    (*pos)++;
    return 0;
}

int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap) {
    size_t pos = 0;

    while (*fmt != '\0') {
        if (*fmt != '%') {
            emit(buf, n, &pos, *fmt++);
            continue;
        }
        fmt++;
        switch (*fmt) {
        case 's': {
            const char *s = va_arg(ap, const char *);
            if (s == NULL) {
                s = "(null)";
            }
            while (*s != '\0') {
                emit(buf, n, &pos, *s++);
            }
            break;
        }
        case 'c':
            emit(buf, n, &pos, (char)va_arg(ap, int));
            break;
        case 'd':
        case 'i': {
            long v = va_arg(ap, long);
            char tmp[24];
            int k = 0;
            unsigned long u = v < 0 ? (unsigned long)-v : (unsigned long)v;
            if (v < 0) {
                emit(buf, n, &pos, '-');
            }
            do {
                tmp[k++] = (char)('0' + u % 10);
                u /= 10;
            } while (u != 0);
            while (k-- > 0) {
                emit(buf, n, &pos, tmp[k]);
            }
            break;
        }
        case 'u':
        case 'x':
        case 'X': {
            unsigned long u = va_arg(ap, unsigned long);
            int base = (*fmt == 'u') ? 10 : 16;
            char tmp[24];
            int k = 0;
            const char *digits = (*fmt == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
            do {
                tmp[k++] = digits[u % base];
                u /= base;
            } while (u != 0);
            while (k-- > 0) {
                emit(buf, n, &pos, tmp[k]);
            }
            break;
        }
        case 'p': {
            unsigned long u = (unsigned long)va_arg(ap, void *);
            char tmp[24];
            int k = 0;
            emit(buf, n, &pos, '0');
            emit(buf, n, &pos, 'x');
            do {
                tmp[k++] = "0123456789abcdef"[u % 16];
                u /= 16;
            } while (u != 0);
            while (k-- > 0) {
                emit(buf, n, &pos, tmp[k]);
            }
            break;
        }
        case '%':
            emit(buf, n, &pos, '%');
            break;
        default:
            emit(buf, n, &pos, '%');
            emit(buf, n, &pos, *fmt);
            break;
        }
        fmt++;
    }
    if (n > 0) {
        buf[pos < n ? pos : n - 1] = '\0';
    }
    return (int)pos;
}

int snprintf(char *buf, size_t n, const char *fmt, ...) {
    va_list ap;
    int r;

    va_start(ap, fmt);
    r = vsnprintf(buf, n, fmt, ap);
    va_end(ap);
    return r;
}

int vprintf(const char *fmt, va_list ap) {
    char buf[1024];

    vsnprintf(buf, sizeof buf, fmt, ap);
    anubis_console_write(buf);
    return (int)strlen(buf);
}

int printf(const char *fmt, ...) {
    va_list ap;
    int r;

    va_start(ap, fmt);
    r = vprintf(fmt, ap);
    va_end(ap);
    return r;
}

int fprintf(FILE *f, const char *fmt, ...) {
    va_list ap;
    int r;

    (void)f;
    va_start(ap, fmt);
    r = vprintf(fmt, ap);
    va_end(ap);
    return r;
}

int puts(const char *s) {
    anubis_console_write(s);
    anubis_console_write("\n");
    return 0;
}

int putchar(int c) {
    char s[2];

    s[0] = (char)c;
    s[1] = '\0';
    anubis_console_write(s);
    return c;
}

int fflush(FILE *f) {
    (void)f;
    return 0;
}
