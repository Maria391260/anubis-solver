/*
 * pjrt_probe.c — does this machine have a PJRT device, and which API version?
 *
 * The rest of this directory has to be written against a header we cannot run
 * locally, so the first thing to establish on a real TPU is the boring part:
 * libtpu loads, GetPjrtApi() returns a struct, the major version matches, and
 * a client comes up with addressable devices. Everything prints, so a failure
 * is evidence rather than a blank screen.
 *
 *   cc -O2 -I. -o pjrt_probe pjrt_probe.c -ldl
 *   ./pjrt_probe /path/to/libtpu.so
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"

typedef PJRT_Api *(*get_pjrt_api_fn)(void);

static void show_error(PJRT_Api *api, PJRT_Error *e, const char *what) {
    if (e == NULL) {
        return;
    }
    PJRT_Error_Message_Args ma = {0};
    ma.struct_size = sizeof ma;
    ma.error = e;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "%s: %.*s\n", what, (int)ma.message_size, ma.message);
    PJRT_Error_Destroy_Args da = {0};
    da.struct_size = sizeof da;
    da.error = e;
    api->PJRT_Error_Destroy(&da);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : getenv("LIBTPU");
    if (path == NULL) {
        fprintf(stderr, "usage: %s /path/to/libtpu.so\n", argv[0]);
        return 2;
    }
    printf("library: %s\n", path);
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    get_pjrt_api_fn get = (get_pjrt_api_fn)dlsym(h, "GetPjrtApi");
    if (get == NULL) {
        fprintf(stderr, "dlsym(GetPjrtApi): %s\n", dlerror());
        return 1;
    }
    PJRT_Api *api = get();
    if (api == NULL) {
        fprintf(stderr, "GetPjrtApi returned NULL\n");
        return 1;
    }
    printf("api: struct_size=%zu major=%d minor=%d (header %d.%d)\n",
           api->struct_size, api->pjrt_api_version.major_version,
           api->pjrt_api_version.minor_version, PJRT_API_MAJOR, PJRT_API_MINOR);
    if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        fprintf(stderr, "major mismatch, refusing to call further\n");
        return 1;
    }

    /* Documented PJRT lifecycle: the plugin is initialized before any other
     * call. Without this libtpu reaches Client_Create with InitGoogle still
     * pending and aborts with "InitGoogle() has not finished yet". */
    PJRT_Plugin_Initialize_Args ia = {0};
    PJRT_Error *e = NULL;
    ia.struct_size = sizeof ia;
    e = api->PJRT_Plugin_Initialize(&ia);
    if (e != NULL) {
        show_error(api, e, "plugin initialize");
        return 1;
    }
    printf("plugin: initialized\n");

    PJRT_Client_Create_Args ca = {0};
    ca.struct_size = sizeof ca;
    e = api->PJRT_Client_Create(&ca);
    if (e != NULL) {
        show_error(api, e, "client create");
        return 1;
    }
    printf("client: %p\n", (void *)ca.client);

    PJRT_Client_AddressableDevices_Args da = {0};
    da.struct_size = sizeof da;
    da.client = ca.client;
    e = api->PJRT_Client_AddressableDevices(&da);
    if (e != NULL) {
        show_error(api, e, "addressable devices");
        return 1;
    }
    printf("devices: %zu\n", da.num_addressable_devices);
    for (size_t i = 0; i < da.num_addressable_devices; i++) {
        printf("  device %zu: %p\n", i, (void *)da.addressable_devices[i]);
    }

    /* Can the plugin compile the shape we need? HLO text is the one format the
     * C API accepts as text, and a dot whose result element type is wider than
     * its operands is exactly how XLA spells an int8 matmul that accumulates in
     * int32 (the instruction's shape *is* the preferred element type). */
    static const char hlo_text[] =
        "HloModule tensor\n"
        "\n"
        "ENTRY %main (a: s8[128,128], b: s8[128,128]) -> s32[128,128] {\n"
        "  %a = s8[128,128]{1,0} parameter(0)\n"
        "  %b = s8[128,128]{1,0} parameter(1)\n"
        "  ROOT %d = s32[128,128]{1,0} dot(%a, %b), "
        "lhs_contracting_dims={1}, rhs_contracting_dims={0}\n"
        "}\n";

    PJRT_Program prog = {0};
    prog.struct_size = sizeof prog;
    prog.code = (char *)hlo_text;
    prog.code_size = sizeof hlo_text - 1;
    prog.format = "hlo_text";
    prog.format_size = 8;

    PJRT_Client_Compile_Args cc = {0};
    cc.struct_size = sizeof cc;
    cc.client = ca.client;
    cc.program = &prog;
    e = api->PJRT_Client_Compile(&cc);
    if (e != NULL) {
        show_error(api, e, "compile hlo_text dot");
        return 1;
    }
    printf("compiled: executable=%p\n", (void *)cc.executable);
    return 0;
}
