/*
 * win32-compat.h — the small POSIX-to-Win32 layer betterAnusbis needs.
 *
 * betterAnusbis is written against POSIX sockets, pthreads and OpenSSL. Three
 * of those four exist on Windows already: winsock2 is BSD-sockets-shaped,
 * winpthreads is a real pthreads implementation, and OpenSSL is cross-built
 * for mingw. The missing piece is the few POSIX conveniences the code uses as
 * if they were free: poll(), socketpair(), fcntl(O_NONBLOCK), close() that
 * works on both files and sockets, sigaction(), and a handful of mcsvcrt
 * names that differ by one underscore.
 *
 * Everything below is a thin shim, not an abstraction: the socket values stay
 * real SOCKET handles (cast to int, which is what OpenSSL's int-fd API does on
 * Windows too), so SSL_set_fd() and friends get exactly what they expect.
 *
 * The socket registry exists because close() in this code is used for files
 * and sockets alike. close() asks the registry which one it is, and forwards
 * to closesocket() or _close(). Every socket this file creates is registered
 * and unregistered exactly once.
 */
#ifndef BETTERANUSBIS_WIN32_COMPAT_H
#define BETTERANUSBIS_WIN32_COMPAT_H

#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0A00
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <afunix.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <process.h>
#include <signal.h>
#include <time.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

/* strings.h is a BSD-ism; MSVCRT spells the same two functions differently. */
#define strcasecmp _stricmp
#define strncasecmp _strnicmp

/* strcasestr(3) is a GNU extension with no MSVCRT equivalent. */
static char *strcasestr(const char *hay, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) return (char *)hay;
    for (; *hay; hay++)
        if (_strnicmp(hay, needle, n) == 0) return (char *)hay;
    return NULL;
}

/* Signals that MSVCRT does not define. BetterAnusbis sets SIGPIPE to SIG_IGN
 * and hangs SIGHUP off a print-the-settings handler; neither exists on
 * Windows, and w_sigaction() below treats both as successfully ignored. */
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
#ifndef SIGHUP
#define SIGHUP 1
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef SHUT_RD
#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH
#endif
#ifndef EAI_NODATA
#define EAI_NODATA EAI_NONAME
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0x4000
#endif
#ifndef F_GETFL
#define F_GETFL 3
#endif
#ifndef F_SETFL
#define F_SETFL 4
#endif
#ifndef F_SETFD
#define F_SETFD 2
#endif
#ifndef FD_CLOEXEC
#define FD_CLOEXEC 1
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif

#ifndef _NFDS_T
#define _NFDS_T
typedef unsigned long nfds_t;
#endif

/* ------------------------------------------------------------------ *
 * Winsock lifecycle and the socket registry
 * ------------------------------------------------------------------ */

static INIT_ONCE ba_wsa_once = INIT_ONCE_STATIC_INIT;
static CRITICAL_SECTION ba_sock_lock;
static int ba_sock_tab[4096];
static int ba_sock_n;

static BOOL CALLBACK ba_wsa_init_once(PINIT_ONCE once, PVOID p, PVOID *ctx) {
    WSADATA w;
    (void)once;
    (void)p;
    (void)ctx;
    WSAStartup(MAKEWORD(2, 2), &w);
    InitializeCriticalSection(&ba_sock_lock);
    return TRUE;
}

static void ba_wsa_init(void) {
    InitOnceExecuteOnce(&ba_wsa_once, ba_wsa_init_once, NULL, NULL);
}

static void ba_sock_add(int fd) {
    EnterCriticalSection(&ba_sock_lock);
    if (ba_sock_n < (int)(sizeof ba_sock_tab / sizeof ba_sock_tab[0]))
        ba_sock_tab[ba_sock_n++] = fd;
    LeaveCriticalSection(&ba_sock_lock);
}

static int ba_sock_has(int fd) {
    int found = 0;
    EnterCriticalSection(&ba_sock_lock);
    for (int i = 0; i < ba_sock_n; i++)
        if (ba_sock_tab[i] == fd) {
            found = 1;
            break;
        }
    LeaveCriticalSection(&ba_sock_lock);
    return found;
}

static void ba_sock_del(int fd) {
    EnterCriticalSection(&ba_sock_lock);
    for (int i = 0; i < ba_sock_n; i++)
        if (ba_sock_tab[i] == fd) {
            ba_sock_tab[i] = ba_sock_tab[--ba_sock_n];
            break;
        }
    LeaveCriticalSection(&ba_sock_lock);
}

static SOCKET ba_sock(int fd) {
    return (SOCKET)(uintptr_t)(unsigned int)fd;
}

static void ba_errno_from_wsa(int e) {
    switch (e) {
    case 0: return;
    case WSAEINTR: errno = EINTR; return;
    case WSAEWOULDBLOCK: errno = EWOULDBLOCK; return;
    case WSAEINPROGRESS: errno = EINPROGRESS; return;
    case WSAEALREADY: errno = EALREADY; return;
    case WSAENOTSOCK: errno = EBADF; return;
    case WSAEDESTADDRREQ: errno = EDESTADDRREQ; return;
    case WSAEMSGSIZE: errno = EMSGSIZE; return;
    case WSAEPROTOTYPE: errno = EPROTOTYPE; return;
    case WSAENOPROTOOPT: errno = ENOPROTOOPT; return;
    case WSAEPROTONOSUPPORT: errno = EPROTONOSUPPORT; return;
    case WSAEOPNOTSUPP: errno = EOPNOTSUPP; return;
    case WSAEAFNOSUPPORT: errno = EAFNOSUPPORT; return;
    case WSAEADDRINUSE: errno = EADDRINUSE; return;
    case WSAEADDRNOTAVAIL: errno = EADDRNOTAVAIL; return;
    case WSAENETDOWN: errno = ENETDOWN; return;
    case WSAENETUNREACH: errno = ENETUNREACH; return;
    case WSAENETRESET: errno = ENETRESET; return;
    case WSAECONNABORTED: errno = ECONNABORTED; return;
    case WSAECONNRESET: errno = ECONNRESET; return;
    case WSAENOBUFS: errno = ENOBUFS; return;
    case WSAEISCONN: errno = EISCONN; return;
    case WSAENOTCONN: errno = ENOTCONN; return;
    case WSAETIMEDOUT: errno = ETIMEDOUT; return;
    case WSAECONNREFUSED: errno = ECONNREFUSED; return;
    case WSAEHOSTUNREACH: errno = EHOSTUNREACH; return;
    case WSAEACCES: errno = EACCES; return;
    case WSAEFAULT: errno = EFAULT; return;
    case WSAEINVAL: errno = EINVAL; return;
    case WSAESHUTDOWN: errno = EPIPE; return;
    default: errno = EIO; return;
    }
}

/* ------------------------------------------------------------------ *
 * The calls the gate makes
 * ------------------------------------------------------------------ */

static int w_socket(int af, int type, int proto) {
    SOCKET s;
    ba_wsa_init();
    s = WSASocketA(af, type, proto, NULL, 0, 0);
    if (s == INVALID_SOCKET) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    ba_sock_add((int)(uintptr_t)s);
    return (int)(uintptr_t)s;
}

static int w_accept(int fd, struct sockaddr *addr, socklen_t *alen) {
    SOCKET s = accept(ba_sock(fd), addr, alen);
    if (s == INVALID_SOCKET) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    ba_sock_add((int)(uintptr_t)s);
    return (int)(uintptr_t)s;
}

static int w_connect(int fd, const struct sockaddr *addr, socklen_t alen) {
    if (connect(ba_sock(fd), addr, alen) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return 0;
}

static int w_bind(int fd, const struct sockaddr *addr, socklen_t alen) {
    if (bind(ba_sock(fd), addr, alen) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return 0;
}

static int w_listen(int fd, int backlog) {
    if (listen(ba_sock(fd), backlog) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return 0;
}

static int w_getpeername(int fd, struct sockaddr *addr, socklen_t *alen) {
    if (getpeername(ba_sock(fd), addr, alen) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return 0;
}

static int w_shutdown(int fd, int how) {
    if (shutdown(ba_sock(fd), how) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return 0;
}

static int w_setsockopt(int fd, int level, int name, const void *val,
                        socklen_t len) {
    /* POSIX puts a struct timeval here; winsock wants milliseconds as a
     * DWORD. This is the one option whose representation genuinely differs. */
    if (level == SOL_SOCKET && (name == SO_RCVTIMEO || name == SO_SNDTIMEO) &&
        val != NULL && len == (socklen_t)sizeof(struct timeval)) {
        const struct timeval *tv = (const struct timeval *)val;
        DWORD ms = (DWORD)(tv->tv_sec * 1000 + tv->tv_usec / 1000);
        if (ms == 0 && (tv->tv_sec != 0 || tv->tv_usec != 0)) ms = 1;
        if (setsockopt(ba_sock(fd), level, name, (const char *)&ms,
                       (int)sizeof ms) == SOCKET_ERROR) {
            ba_errno_from_wsa(WSAGetLastError());
            return -1;
        }
        return 0;
    }
    if (setsockopt(ba_sock(fd), level, name, (const char *)val, (int)len) ==
        SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return 0;
}

static int w_getsockopt(int fd, int level, int name, void *val,
                        socklen_t *len) {
    int l = len ? (int)*len : 0;
    if (getsockopt(ba_sock(fd), level, name, (char *)val, &l) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    if (len) *len = (socklen_t)l;
    return 0;
}

static int w_send(int fd, const void *buf, size_t len, int flags) {
    int r;
    ba_wsa_init();
    r = send(ba_sock(fd), (const char *)buf, (int)len, flags);
    if (r == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return r;
}

static int w_recv(int fd, void *buf, size_t len, int flags) {
    int r;
    ba_wsa_init();
    r = recv(ba_sock(fd), (char *)buf, (int)len, flags);
    if (r == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        return -1;
    }
    return r;
}

static int w_close(int fd) {
    if (ba_sock_has(fd)) {
        ba_sock_del(fd);
        if (closesocket(ba_sock(fd)) == SOCKET_ERROR) {
            ba_errno_from_wsa(WSAGetLastError());
            return -1;
        }
        return 0;
    }
    return _close(fd);
}

/* read()/write() are used on socketpairs as well as on files. */
static ssize_t w_read(int fd, void *buf, size_t n) {
    if (ba_sock_has(fd)) return (ssize_t)w_recv(fd, buf, n, 0);
    return (ssize_t)_read(fd, buf, (unsigned)n);
}

static ssize_t w_write(int fd, const void *buf, size_t n) {
    if (ba_sock_has(fd)) return (ssize_t)w_send(fd, buf, n, 0);
    return (ssize_t)_write(fd, buf, (unsigned)n);
}

static int w_poll(struct pollfd *fds, nfds_t n, int timeout_ms) {
    WSAPOLLFD stack[64];
    WSAPOLLFD *p = stack;
    int i, r;
    if ((size_t)n > sizeof stack / sizeof stack[0]) {
        p = (WSAPOLLFD *)malloc((size_t)n * sizeof *p);
        if (p == NULL) {
            errno = ENOMEM;
            return -1;
        }
    }
    for (i = 0; i < (int)n; i++) {
        p[i].fd = (fds[i].fd < 0) ? INVALID_SOCKET : ba_sock(fds[i].fd);
        p[i].events = fds[i].events;
        p[i].revents = 0;
    }
    ba_wsa_init();
    r = WSAPoll(p, (ULONG)n, timeout_ms);
    if (r == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        if (p != stack) free(p);
        return -1;
    }
    for (i = 0; i < (int)n; i++) fds[i].revents = p[i].revents;
    if (p != stack) free(p);
    return r;
}

static int w_fcntl(int fd, int cmd, intptr_t arg) {
    (void)arg;
    if (cmd == F_GETFL) return 0;
    if (cmd == F_SETFL) {
        u_long nonblock = (arg & O_NONBLOCK) ? 1 : 0;
        if (ba_sock_has(fd)) {
            if (ioctlsocket(ba_sock(fd), FIONBIO, &nonblock) == SOCKET_ERROR) {
                ba_errno_from_wsa(WSAGetLastError());
                return -1;
            }
        }
        return 0;
    }
    return 0;
}

/* socketpair(2) does not exist on Windows. Two connected loopback stream
 * sockets are the same contract for the use here: a full-duplex pipe that
 * poll() and read()/write() work on. */
static int w_socketpair(int domain, int type, int proto, int sv[2]) {
    struct sockaddr_in addr;
    int l = -1, a = -1, b = -1, one = 1;
    socklen_t alen = (socklen_t)sizeof addr;
    (void)domain;
    (void)type;
    (void)proto;
    l = w_socket(AF_INET, SOCK_STREAM, 0);
    if (l < 0) goto fail;
    w_setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, (socklen_t)sizeof one);
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (w_bind(l, (struct sockaddr *)&addr, (socklen_t)sizeof addr) < 0) goto fail;
    if (w_listen(l, 1) < 0) goto fail;
    if (getsockname(ba_sock(l), (struct sockaddr *)&addr, &alen) == SOCKET_ERROR) {
        ba_errno_from_wsa(WSAGetLastError());
        goto fail;
    }
    b = w_socket(AF_INET, SOCK_STREAM, 0);
    if (b < 0) goto fail;
    if (w_connect(b, (struct sockaddr *)&addr, (socklen_t)sizeof addr) < 0) goto fail;
    a = w_accept(l, NULL, NULL);
    if (a < 0) goto fail;
    w_close(l);
    sv[0] = a;
    sv[1] = b;
    return 0;
fail:
    if (l >= 0) w_close(l);
    if (a >= 0) w_close(a);
    if (b >= 0) w_close(b);
    return -1;
}

/* ------------------------------------------------------------------ *
 * signals
 * ------------------------------------------------------------------ */

typedef int sigset_t;

struct sigaction {
    void (*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
};

static int sigemptyset(sigset_t *set) {
    if (set) *set = 0;
    return 0;
}

static int w_sigaction(int sig, const struct sigaction *act,
                       struct sigaction *old) {
    if (old) memset(old, 0, sizeof *old);
    if (sig == SIGPIPE || sig == SIGHUP) return 0; /* nothing delivers them */
    if (act != NULL && act->sa_handler != NULL) {
        if (signal(sig, act->sa_handler) == SIG_ERR) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ *
 * names that differ by an underscore or a signature
 * ------------------------------------------------------------------ */

#ifndef HAVE_USLEEP
#define usleep(usec) Sleep((DWORD)((usec) / 1000))
#endif

#define gmtime_r(t, tm) (gmtime_s((tm), (t)) == 0 ? (tm) : NULL)
#define timegm(tm) _mkgmtime(tm)
#define fsync(fd) _commit(fd)
#define fchmod(fd, mode) (0)
#define rename(from, to) w_rename((from), (to))

static int w_rename(const char *from, const char *to) {
    if (MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return 0;
    errno = EACCES;
    return -1;
}

/* mkstemp: create a unique file next to the template. O_EXCL makes the create
 * atomic, so the loop cannot race itself. */
static int w_mkstemp(char *tmpl) {
    size_t n = strlen(tmpl);
    static LONG counter;
    unsigned seed = (unsigned)GetTickCount64() ^ (unsigned)_getpid();
    int attempt;
    if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX") != 0) {
        errno = EINVAL;
        return -1;
    }
    for (attempt = 0; attempt < 256; attempt++) {
        unsigned v = seed + (unsigned)InterlockedIncrement(&counter) * 2654435761u;
        int i, fd;
        for (i = 0; i < 6; i++) {
            tmpl[n - 6 + i] = "abcdefghijklmnopqrstuvwxyz0123456789"[v % 36];
            v /= 36;
        }
        fd = _open(tmpl, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                   _S_IREAD | _S_IWRITE);
        if (fd >= 0) return fd;
        if (errno != EEXIST) return -1;
    }
    errno = EEXIST;
    return -1;
}
#define mkstemp(tmpl) w_mkstemp(tmpl)

/* The macros that make the existing source compile unchanged. */
#define socket w_socket
#define accept w_accept
#define connect w_connect
#define bind w_bind
#define listen w_listen
#define getpeername w_getpeername
#define shutdown w_shutdown
#define setsockopt w_setsockopt
#define getsockopt w_getsockopt
#define send w_send
#define recv w_recv
#define close w_close
#define read w_read
#define write w_write
#define poll w_poll
#define fcntl w_fcntl
#define socketpair w_socketpair
/* Function-like: an object-like macro would also rewrite `struct sigaction`. */
#define sigaction(sig, act, old) w_sigaction((sig), (act), (old))

#endif /* BETTERANUSBIS_WIN32_COMPAT_H */
