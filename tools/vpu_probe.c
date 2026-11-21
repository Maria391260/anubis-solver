/*
 * vpu_probe.c — first contact with an Intel VPU/NPU, the way pjrt_probe.c was
 * first contact with the TPU.
 *
 * The "VPU" is Intel's name for the neural accelerator in Core Ultra (Meteor
 * Lake and later) client CPUs: the kernel driver is intel_vpu
 * (/dev/accel/accel0), the user-space driver is a Level Zero implementation
 * (libze_intel_npu.so.1, formerly libze_intel_vpu.so.1) and the compute model
 * is a graph, not a kernel: zeGraphCreate takes a blob produced by Intel's
 * Driver Compiler (the vcl* API) from an IR network. There is no public
 * SPIR-V -> VPU path, which is why the GPU Level Zero backends in this tree
 * cannot simply be pointed at it.
 *
 * This tool answers the questions that decide whether a SHA-256 port is
 * possible at all, on the machine it is run on:
 *
 *   1. is there a VPU device, through Level Zero's ZE_INIT_FLAG_VPU_ONLY?
 *   2. what does zeDeviceGetProperties say it is (type 5 = ZE_DEVICE_TYPE_VPU)?
 *   3. does its driver accept our SPIR-V kernel (zeModuleCreate, IL_SPIRV) and
 *      the "sha256_pow" entry point (zeKernelCreate)?
 *
 * A "no" at step 3 is the answer the code expects on today's drivers; a "yes"
 * would be the green light to write the backend next to levelzero_sha256.c.
 *
 * Build: make vpu-probe
 * Usage: ./build/vpu-probe [--spv PATH] [--kernel NAME] [--all]
 * Exit:  0 VPU found (and SPIR-V accepted, if --spv was given)
 *        2 no VPU device
 *        3 VPU found, SPIR-V module rejected
 *        4 module built but kernel entry point rejected
 *        1 usage or loader failure
 */
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "level_zero/ze_api.h"

/* ---- Level Zero entry points, all resolved from libze_loader.so.1 ------- */
typedef ze_result_t (*fn_zeInit)(ze_init_flags_t);
typedef ze_result_t (*fn_zeDriverGet)(uint32_t *, ze_driver_handle_t *);
typedef ze_result_t (*fn_zeDeviceGet)(ze_driver_handle_t, uint32_t *,
                                      ze_device_handle_t *);
typedef ze_result_t (*fn_zeDeviceGetProperties)(ze_device_handle_t,
                                                ze_device_properties_t *);
typedef ze_result_t (*fn_zeContextCreate)(ze_driver_handle_t,
                                          const ze_context_desc_t *,
                                          ze_context_handle_t *);
typedef ze_result_t (*fn_zeContextDestroy)(ze_context_handle_t);
typedef ze_result_t (*fn_zeModuleCreate)(ze_context_handle_t,
                                         ze_device_handle_t,
                                         const ze_module_desc_t *,
                                         ze_module_handle_t *,
                                         ze_module_build_log_handle_t *);
typedef ze_result_t (*fn_zeModuleDestroy)(ze_module_handle_t);
typedef ze_result_t (*fn_zeModuleBuildLogGetString)(
    ze_module_build_log_handle_t, size_t *, char *);
typedef ze_result_t (*fn_zeModuleBuildLogDestroy)(ze_module_build_log_handle_t);
typedef ze_result_t (*fn_zeKernelCreate)(ze_module_handle_t,
                                         const ze_kernel_desc_t *,
                                         ze_kernel_handle_t *);
typedef ze_result_t (*fn_zeKernelDestroy)(ze_kernel_handle_t);

static fn_zeInit p_zeInit;
static fn_zeDriverGet p_zeDriverGet;
static fn_zeDeviceGet p_zeDeviceGet;
static fn_zeDeviceGetProperties p_zeDeviceGetProperties;
static fn_zeContextCreate p_zeContextCreate;
static fn_zeContextDestroy p_zeContextDestroy;
static fn_zeModuleCreate p_zeModuleCreate;
static fn_zeModuleDestroy p_zeModuleDestroy;
static fn_zeModuleBuildLogGetString p_zeBuildLogGetString;
static fn_zeModuleBuildLogDestroy p_zeBuildLogDestroy;
static fn_zeKernelCreate p_zeKernelCreate;
static fn_zeKernelDestroy p_zeKernelDestroy;

static const char *result_name(ze_result_t r)
{
    switch (r) {
    case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
        return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
    default: return NULL;
    }
}

static void print_result(const char *what, ze_result_t r)
{
    const char *name = result_name(r);

    if (name != NULL) {
        printf("  %-34s %#x (%s)\n", what, (unsigned)r, name);
    } else {
        printf("  %-34s %#x\n", what, (unsigned)r);
    }
}

static const char *device_type_name(ze_device_type_t t)
{
    switch (t) {
    case ZE_DEVICE_TYPE_GPU: return "GPU";
    case ZE_DEVICE_TYPE_CPU: return "CPU";
    case ZE_DEVICE_TYPE_FPGA: return "FPGA";
    case ZE_DEVICE_TYPE_MCA: return "MCA";
    case ZE_DEVICE_TYPE_VPU: return "VPU";
    default: return "unknown";
    }
}

static int exists(const char *path)
{
    struct stat st;

    return stat(path, &st) == 0;
}

static void check_path(const char *label, const char *path)
{
    printf("  %-34s %s\n", label, exists(path) ? "yes" : "no");
}

static void check_dlopen(const char *label, const char *lib)
{
    void *h = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);

    printf("  %-34s %s\n", label, h ? "yes" : "no");
    if (h != NULL) {
        dlclose(h);
    }
}

static int load_syms(void *h)
{
#define SYM(field, name)                                                     \
    do {                                                                     \
        field = (void *)dlsym(h, name);                                      \
        if (field == NULL) {                                                 \
            return 0;                                                        \
        }                                                                    \
    } while (0)
    SYM(p_zeInit, "zeInit");
    SYM(p_zeDriverGet, "zeDriverGet");
    SYM(p_zeDeviceGet, "zeDeviceGet");
    SYM(p_zeDeviceGetProperties, "zeDeviceGetProperties");
    SYM(p_zeContextCreate, "zeContextCreate");
    SYM(p_zeModuleCreate, "zeModuleCreate");
    SYM(p_zeKernelCreate, "zeKernelCreate");
    /* Destroy and build-log calls are optional: a probe may leak at exit. */
    p_zeContextDestroy = (void *)dlsym(h, "zeContextDestroy");
    p_zeModuleDestroy = (void *)dlsym(h, "zeModuleDestroy");
    p_zeBuildLogGetString = (void *)dlsym(h, "zeModuleBuildLogGetString");
    p_zeBuildLogDestroy = (void *)dlsym(h, "zeModuleBuildLogDestroy");
    p_zeKernelDestroy = (void *)dlsym(h, "zeKernelDestroy");
#undef SYM
    return 1;
}

static void print_build_log(ze_module_build_log_handle_t log)
{
    size_t size = 0;
    char *text;

    if (log == NULL || p_zeBuildLogGetString == NULL) {
        return;
    }
    if (p_zeBuildLogGetString(log, &size, NULL) != ZE_RESULT_SUCCESS ||
        size == 0) {
        return;
    }
    text = malloc(size + 1);
    if (text == NULL) {
        return;
    }
    if (p_zeBuildLogGetString(log, &size, text) == ZE_RESULT_SUCCESS) {
        text[size] = '\0';
        printf("  build log:\n%s\n", text);
    }
    free(text);
}

static uint8_t *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    long len;
    uint8_t *buf;

    if (f == NULL) {
        fprintf(stderr, "vpu-probe: cannot open %s: %s\n", path,
                strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "vpu-probe: cannot size %s\n", path);
        fclose(f);
        return NULL;
    }
    buf = malloc((size_t)len);
    if (buf == NULL) {
        fprintf(stderr, "vpu-probe: out of memory reading %s\n", path);
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "vpu-probe: short read on %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)len;
    return buf;
}

static void describe_spv(const char *path, const uint8_t *buf, size_t size)
{
    uint32_t words[5];

    printf("  %-34s %s (%zu bytes)\n", "kernel file", path, size);
    if (size < sizeof words) {
        printf("  %-34s %s\n", "format", "too short to be SPIR-V");
        return;
    }
    memcpy(words, buf, sizeof words);
    if (words[0] != 0x07230203u) {
        printf("  %-34s %s\n", "format", "not SPIR-V (bad magic)");
        return;
    }
    printf("  %-34s SPIR-V %u.%u\n", "format",
           (words[1] >> 16) & 0xffu, (words[1] >> 8) & 0xffu);
}

static int try_module(ze_driver_handle_t driver, ze_device_handle_t device,
                      const char *spv_path, const char *kernel_name)
{
    ze_context_desc_t cdesc = { .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                                .pNext = NULL, .flags = 0 };
    ze_context_handle_t ctx = NULL;
    ze_module_handle_t module = NULL;
    ze_module_build_log_handle_t log = NULL;
    ze_module_desc_t mdesc = { 0 };
    ze_kernel_handle_t kernel = NULL;
    ze_kernel_desc_t kdesc = { 0 };
    uint8_t *spv;
    size_t spv_size = 0;
    int rc = 1;

    spv = read_file(spv_path, &spv_size);
    if (spv == NULL) {
        return 1;
    }
    describe_spv(spv_path, spv, spv_size);

    print_result("zeContextCreate",
                 p_zeContextCreate(driver, &cdesc, &ctx));
    if (ctx == NULL) {
        free(spv);
        return 1;
    }

    mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mdesc.pNext = NULL;
    mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    mdesc.inputSize = spv_size;
    mdesc.pInputModule = spv;
    mdesc.pBuildFlags = NULL;
    mdesc.pConstants = NULL;

    ze_result_t r = p_zeModuleCreate(ctx, device, &mdesc, &module, &log);
    print_result("zeModuleCreate(IL_SPIRV)", r);
    print_build_log(log);
    if (p_zeBuildLogDestroy != NULL && log != NULL) {
        p_zeBuildLogDestroy(log);
    }
    if (r != ZE_RESULT_SUCCESS || module == NULL) {
        if (p_zeContextDestroy != NULL) {
            p_zeContextDestroy(ctx);
        }
        free(spv);
        return 3;
    }

    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pNext = NULL;
    kdesc.flags = 0;
    kdesc.pKernelName = kernel_name;
    r = p_zeKernelCreate(module, &kdesc, &kernel);
    print_result("zeKernelCreate", r);
    if (r != ZE_RESULT_SUCCESS) {
        rc = 4;
    } else {
        printf("  %-34s %s\n", "entry point", kernel_name);
        if (p_zeKernelDestroy != NULL) {
            p_zeKernelDestroy(kernel);
        }
        rc = 0;
    }

    if (p_zeModuleDestroy != NULL) {
        p_zeModuleDestroy(module);
    }
    if (p_zeContextDestroy != NULL) {
        p_zeContextDestroy(ctx);
    }
    free(spv);
    return rc;
}

int main(int argc, char **argv)
{
    const char *spv_path = "src/intel/levelzero/sha256_pow.spv";
    const char *kernel_name = "sha256_pow";
    ze_init_flags_t flags = ZE_INIT_FLAG_VPU_ONLY;
    ze_driver_handle_t *drivers = NULL;
    uint32_t ndrivers = 0;
    ze_driver_handle_t vpu_driver = NULL;
    ze_device_handle_t vpu_device = NULL;
    uint32_t vpu_count = 0;
    uint32_t device_count = 0;
    void *loader;
    ze_result_t r;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--spv") == 0 && i + 1 < argc) {
            spv_path = argv[++i];
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (strcmp(argv[i], "--all") == 0) {
            flags = 0;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            printf("usage: %s [--spv PATH] [--kernel NAME] [--all]\n"
                   "  --spv     SPIR-V to try to load (default: the L0 "
                   "sha256 kernel in this tree)\n"
                   "  --kernel  entry point to look up (default: sha256_pow)\n"
                   "  --all     zeInit(0) instead of ZE_INIT_FLAG_VPU_ONLY\n",
                   argv[0]);
            return 0;
        } else {
            fprintf(stderr, "vpu-probe: unknown argument %s (try --help)\n",
                    argv[i]);
            return 1;
        }
    }

    printf("vpu_probe - Intel VPU/NPU first contact\n\n");

    printf("kernel side\n");
    check_path("/dev/accel/", "/dev/accel");
    check_path("/sys/class/accel/", "/sys/class/accel");
    check_path("intel_vpu.ko loaded", "/sys/module/intel_vpu");

    printf("\nuser space\n");
    loader = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    printf("  %-34s %s\n", "libze_loader.so.1", loader ? "yes" : "no");
    if (loader == NULL) {
        printf("\nresult: no Level Zero loader (%s)\n", dlerror());
        return 2;
    }
    check_dlopen("libze_intel_npu.so.1", "libze_intel_npu.so.1");
    check_dlopen("libze_intel_vpu.so.1", "libze_intel_vpu.so.1");

    if (!load_syms(loader)) {
        printf("\nresult: libze_loader.so.1 is too old for this probe\n");
        return 1;
    }

    printf("\nlevel zero\n");
    r = p_zeInit(flags);
    print_result(flags ? "zeInit(ZE_INIT_FLAG_VPU_ONLY)" : "zeInit(0)", r);
    if (r != ZE_RESULT_SUCCESS) {
        printf("\nresult: no Level Zero driver answered - no VPU here\n");
        return 2;
    }

    p_zeDriverGet(&ndrivers, NULL);
    if (ndrivers == 0) {
        printf("  %-34s %u\n", "drivers", ndrivers);
        printf("\nresult: no driver - no VPU here%s\n",
               flags ? " (try --all)" : "");
        return 2;
    }
    drivers = calloc(ndrivers, sizeof *drivers);
    if (drivers == NULL) {
        return 1;
    }
    if (p_zeDriverGet(&ndrivers, drivers) != ZE_RESULT_SUCCESS) {
        free(drivers);
        return 1;
    }
    printf("  %-34s %u\n", "drivers", ndrivers);

    for (uint32_t d = 0; d < ndrivers; d++) {
        ze_device_handle_t *devices = NULL;
        uint32_t ndevices = 0;

        p_zeDeviceGet(drivers[d], &ndevices, NULL);
        if (ndevices == 0) {
            continue;
        }
        devices = calloc(ndevices, sizeof *devices);
        if (devices == NULL) {
            continue;
        }
        if (p_zeDeviceGet(drivers[d], &ndevices, devices) !=
            ZE_RESULT_SUCCESS) {
            free(devices);
            continue;
        }
        for (uint32_t e = 0; e < ndevices; e++) {
            ze_device_properties_t props = { 0 };
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            props.pNext = NULL;
            if (p_zeDeviceGetProperties(devices[e], &props) !=
                ZE_RESULT_SUCCESS) {
                continue;
            }
            printf("  device[%u/%u] %-22s type=%d (%s) vendor=%#x device=%#x\n",
                   d, e, props.name, (int)props.type,
                   device_type_name(props.type), props.vendorId,
                   props.deviceId);
            if (props.type == ZE_DEVICE_TYPE_VPU && vpu_device == NULL) {
                vpu_driver = drivers[d];
                vpu_device = devices[e];
            }
            if (props.type == ZE_DEVICE_TYPE_VPU) {
                vpu_count++;
            }
            device_count++;
        }
        free(devices);
    }

    printf("\n");
    if (vpu_device == NULL) {
        printf("result: %u Level Zero device(s), none of them a VPU\n",
               device_count);
        free(drivers);
        return 2;
    }
    printf("result: %u VPU device(s)\n", vpu_count);

    printf("\ntrying the kernel\n");
    int rc = try_module(vpu_driver, vpu_device, spv_path, kernel_name);
    free(drivers);
    return rc;
}
