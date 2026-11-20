/*
 * nvidia_boost.c -- pin NVIDIA GPUs at their maximum SM clock.
 *
 * A Tesla T4 parks at 300 MHz when idle and only boosts under sustained load,
 * so a benchmark that starts, runs for a few hundred milliseconds per row and
 * then idles measures the clock ramp, not the kernel: the same configuration
 * reads 1.9 GH/s in one row and 3.1 GH/s in the next. This file gives callers
 * a way to remove that variable before timing anything.
 *
 * NVML is loaded at runtime with dlopen(), so the library still loads on a
 * machine with no driver at all, and the symbol stays free for a CPU-only box.
 *
 * What it does, per visible device:
 *   1. nvmlDeviceSetGpuLockedClocks(dev, max_sm, max_sm)   (R450+, Turing+)
 *   2. nvmlDeviceSetApplicationsClocks(dev, mem, sm)        (fallback)
 *   3. nvmlDeviceSetPersistenceMode(dev, ENABLED)          (best effort)
 *
 * Locking clocks needs CAP_SYS_ADMIN (root, or a container with the device
 * cgroup and the capability). Everything else is read-only.
 *
 * The lock lasts until the process exits, the driver is reset, or a caller
 * unlocks it with nvmlDeviceResetGpuLockedClocks(); we do not reset it behind
 * the caller's back, because that would make a finished benchmark slow again.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "anubis-solver.h"

#ifdef __GNUC__
#define ANUBIS_EXPORT __attribute__((visibility("default")))
#else
#define ANUBIS_EXPORT
#endif

/* NVML types and constants, spelled out so this file needs no headers. */
typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;

#define NVML_SUCCESS 0
#define NVML_ERROR_UNINITIALIZED 1
#define NVML_ERROR_INVALID_ARGUMENT 2
#define NVML_ERROR_NOT_SUPPORTED 3
#define NVML_ERROR_NO_PERMISSION 4
#define NVML_ERROR_NOT_FOUND 6
#define NVML_CLOCK_SM 1
#define NVML_CLOCK_MEM 2
#define NVML_FEATURE_DISABLED 0
#define NVML_FEATURE_ENABLED 1

typedef nvmlReturn_t (*fn_init)(void);
typedef nvmlReturn_t (*fn_shutdown)(void);
typedef nvmlReturn_t (*fn_count)(unsigned *);
typedef nvmlReturn_t (*fn_handle)(unsigned, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_max_clock)(nvmlDevice_t, int, unsigned *);
typedef nvmlReturn_t (*fn_clock)(nvmlDevice_t, int, unsigned *);
typedef nvmlReturn_t (*fn_lock)(nvmlDevice_t, unsigned, unsigned);
typedef nvmlReturn_t (*fn_app_clocks)(nvmlDevice_t, unsigned, unsigned);
typedef nvmlReturn_t (*fn_reset_app)(nvmlDevice_t);
typedef nvmlReturn_t (*fn_persist)(nvmlDevice_t, int);
typedef nvmlReturn_t (*fn_name)(nvmlDevice_t, char *, unsigned);

static struct {
    pthread_once_t once;
    int loaded;
    void *lib;
    fn_init init;
    fn_shutdown shutdown;
    fn_count count;
    fn_handle handle;
    fn_max_clock max_clock;
    fn_clock clock;
    fn_lock lock_clocks;
    fn_app_clocks app_clocks;
    fn_reset_app reset_app;
    fn_persist persist;
    fn_name name;
} g_nv;

/* Result of the most recent force call, for anubis_nvidia_boost_info(). */
static char g_info[512];
static pthread_mutex_t g_info_mu = PTHREAD_MUTEX_INITIALIZER;

static void set_info(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    pthread_mutex_lock(&g_info_mu);
    vsnprintf(g_info, sizeof(g_info), fmt, ap);
    pthread_mutex_unlock(&g_info_mu);
    va_end(ap);
}

static void *sym(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    return p;
}

static void nvml_load_once(void) {
    static const char *const paths[] = {
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
        "/usr/local/nvidia/lib64/libnvidia-ml.so.1",
        NULL
    };
    for (int i = 0; paths[i] != NULL && g_nv.lib == NULL; i++) {
        g_nv.lib = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
    }
    if (g_nv.lib == NULL) {
        return;
    }
    g_nv.init = (fn_init)sym(g_nv.lib, "nvmlInit_v2");
    if (g_nv.init == NULL) {
        g_nv.init = (fn_init)sym(g_nv.lib, "nvmlInit");
    }
    g_nv.shutdown = (fn_shutdown)sym(g_nv.lib, "nvmlShutdown");
    g_nv.count = (fn_count)sym(g_nv.lib, "nvmlDeviceGetCount_v2");
    if (g_nv.count == NULL) {
        g_nv.count = (fn_count)sym(g_nv.lib, "nvmlDeviceGetCount");
    }
    g_nv.handle = (fn_handle)sym(g_nv.lib, "nvmlDeviceGetHandleByIndex_v2");
    if (g_nv.handle == NULL) {
        g_nv.handle = (fn_handle)sym(g_nv.lib, "nvmlDeviceGetHandleByIndex");
    }
    g_nv.max_clock = (fn_max_clock)sym(g_nv.lib, "nvmlDeviceGetMaxClockInfo");
    g_nv.clock = (fn_clock)sym(g_nv.lib, "nvmlDeviceGetClockInfo");
    /* R450+; absent on older drivers, which is why there is a fallback. */
    g_nv.lock_clocks = (fn_lock)sym(g_nv.lib, "nvmlDeviceSetGpuLockedClocks");
    g_nv.app_clocks = (fn_app_clocks)sym(g_nv.lib, "nvmlDeviceSetApplicationsClocks");
    g_nv.reset_app = (fn_reset_app)sym(g_nv.lib, "nvmlDeviceResetApplicationsClocks");
    g_nv.persist = (fn_persist)sym(g_nv.lib, "nvmlDeviceSetPersistenceMode");
    g_nv.name = (fn_name)sym(g_nv.lib, "nvmlDeviceGetName");
    if (g_nv.init == NULL || g_nv.count == NULL || g_nv.handle == NULL) {
        dlclose(g_nv.lib);
        g_nv.lib = NULL;
        return;
    }
    g_nv.loaded = 1;
}

static int nvml_ready(void) {
    pthread_once(&g_nv.once, nvml_load_once);
    if (!g_nv.loaded) {
        set_info("libnvidia-ml.so not found (no NVIDIA driver on this machine?)");
        return 0;
    }
    if (g_nv.init() != NVML_SUCCESS) {
        set_info("nvmlInit failed (driver/container mismatch?)");
        return 0;
    }
    return 1;
}

/*
 * Lock every visible GPU's SM clock at its maximum. Returns 0 when at least one
 * device was locked, or a negative code when nothing could be:
 *
 *   -1  libnvidia-ml.so (NVML) is not available
 *   -2  NVML reports no devices
 *   -3  permission denied: clock control needs root (CAP_SYS_ADMIN)
 *   -4  this driver/GPU cannot lock clocks (older than R450)
 *   -5  the driver refused the value (power or thermal policy)
 *
 * The per-device outcome is in anubis_nvidia_boost_info().
 */
ANUBIS_EXPORT
int anubis_nvidia_force_boost_max(void) {
    if (!nvml_ready()) {
        return -1;
    }
    unsigned count = 0;
    if (g_nv.count(&count) != NVML_SUCCESS || count == 0) {
        set_info("NVML reports no devices");
        return -2;
    }
    int locked = 0;
    int perm = 0, unsupported = 0, refused = 0, max_seen = 0;
    char detail[320];
    detail[0] = '\0';
    for (unsigned i = 0; i < count; i++) {
        nvmlDevice_t dev = NULL;
        char name[96] = "GPU";
        if (g_nv.handle(i, &dev) != NVML_SUCCESS || dev == NULL) {
            continue;
        }
        if (g_nv.name != NULL) {
            (void)g_nv.name(dev, name, sizeof(name));
        }
        unsigned max_sm = 0;
        if (g_nv.max_clock == NULL ||
            g_nv.max_clock(dev, NVML_CLOCK_SM, &max_sm) != NVML_SUCCESS ||
            max_sm == 0) {
            unsupported++;
            continue;
        }
        if (g_nv.persist != NULL) {
            (void)g_nv.persist(dev, NVML_FEATURE_ENABLED);
        }
        nvmlReturn_t rc = NVML_ERROR_NOT_SUPPORTED;
        if (g_nv.lock_clocks != NULL) {
            rc = g_nv.lock_clocks(dev, max_sm, max_sm);
        }
        if (rc != NVML_SUCCESS && g_nv.app_clocks != NULL) {
            unsigned mem = 0;
            if (g_nv.max_clock(dev, NVML_CLOCK_MEM, &mem) == NVML_SUCCESS) {
                rc = g_nv.app_clocks(dev, mem, max_sm);
            }
        }
        if (rc == NVML_SUCCESS) {
            unsigned now = 0;
            if (g_nv.clock != NULL &&
                g_nv.clock(dev, NVML_CLOCK_SM, &now) == NVML_SUCCESS && now > 0) {
                int n = snprintf(detail + strlen(detail),
                                 sizeof(detail) - strlen(detail),
                                 "%s%s %u MHz (max %u)",
                                 detail[0] ? "; " : "", name, now, max_sm);
                (void)n;
            }
            if ((int)max_sm > max_seen) {
                max_seen = (int)max_sm;
            }
            locked++;
        } else if (rc == NVML_ERROR_NO_PERMISSION) {
            perm++;
        } else if (rc == NVML_ERROR_NOT_SUPPORTED) {
            unsupported++;
        } else {
            refused++;
        }
    }
    if (g_nv.shutdown != NULL) {
        (void)g_nv.shutdown();
    }
    if (locked > 0) {
        set_info("locked %d device(s) at max SM clock: %s", locked,
                 detail[0] ? detail : "(clock read-back unavailable)");
        return 0;
    }
    if (perm > 0) {
        set_info("clock locking needs root (CAP_SYS_ADMIN); %d device(s) refused",
                 perm);
        return -3;
    }
    if (unsupported > 0) {
        set_info("this driver/GPU cannot lock clocks (needs R450+): %d device(s)",
                 unsupported);
        return -4;
    }
    set_info("driver refused the maximum clock on %d device(s) (power/thermal)",
             refused ? refused : (int)count);
    return -5;
}

/* Same thing under the name the harnesses use. Kept as a real export rather
 * than an alias so `ctypes` on any platform finds it by name. */
ANUBIS_EXPORT
int force_nvidia_boost_max(void) {
    return anubis_nvidia_force_boost_max();
}

/* Current SM clock of the first visible device, in MHz, or -1. Cheap enough to
 * call between benchmark rows; nothing is cached, so it always reads the
 * driver. */
ANUBIS_EXPORT
int anubis_nvidia_clock_mhz(void) {
    if (!nvml_ready()) {
        return -1;
    }
    nvmlDevice_t dev = NULL;
    int mhz = -1;
    unsigned now = 0;
    if (g_nv.handle(0, &dev) == NVML_SUCCESS && dev != NULL && g_nv.clock != NULL &&
        g_nv.clock(dev, NVML_CLOCK_SM, &now) == NVML_SUCCESS) {
        mhz = (int)now;
    }
    if (g_nv.shutdown != NULL) {
        (void)g_nv.shutdown();
    }
    return mhz;
}

/* What the last force call did (never NULL). */
ANUBIS_EXPORT
const char *anubis_nvidia_boost_info(void) {
    pthread_mutex_lock(&g_info_mu);
    const char *s = g_info[0] ? g_info : "not called";
    pthread_mutex_unlock(&g_info_mu);
    return s;
}
