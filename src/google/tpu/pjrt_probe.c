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

    /* Which program format does this plugin accept, and will it compile the
     * int8 dot we need? The C API documents "hlo" (HloModuleProto) and
     * "mlir"; libtpu rejected "hlo_text" outright, so try the MLIR spellings
     * of the same operation and print each answer. One run, three facts:
     *   - whether MLIR is accepted at all
     *   - whether StableHLO accepts an s8 dot with an s32 result (its C20 says
     *     the result element type must equal the operands', but implementations
     *     can be ahead of the spec - this settles it)
     *   - whether the legacy mhlo dialect is registered, as a fallback
     */
    static const char stablehlo_dot[] =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>)\n"
        "      -> tensor<128x128xi32> {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0],\n"
        "        precision = [DEFAULT]\n"
        "        : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>\n"
        "    return %0 : tensor<128x128xi32>\n"
        "  }\n"
        "}\n";
    static const char stablehlo_i8[] =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>)\n"
        "      -> tensor<128x128xi8> {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0],\n"
        "        precision = [DEFAULT]\n"
        "        : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi8>\n"
        "    return %0 : tensor<128x128xi8>\n"
        "  }\n"
        "}\n";
    static const char mhlo_dot[] =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>)\n"
        "      -> tensor<128x128xi32> {\n"
        "    %0 = \"mhlo.dot_general\"(%a, %b)\n"
        "        {dot_dimension_numbers = #mhlo.dot<\n"
        "            lhs_contracting_dimensions = [1],\n"
        "            rhs_contracting_dimensions = [0]>}\n"
        "        : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>\n"
        "    return %0 : tensor<128x128xi32>\n"
        "  }\n"
        "}\n";

    struct {
        const char *label;
        const char *format;
        const char *code;
    } variants[] = {
        {"stablehlo dot i8->i32", "mlir", stablehlo_dot},
        {"stablehlo dot i8->i8", "mlir", stablehlo_i8},
        {"mhlo dot_general i8->i32", "mlir", mhlo_dot},
    };

    /* CompileOptionsProto{ executable_build_options{ num_replicas=1,
     * num_partitions=1 } } - field 3 (message), inside it fields 4 and 5 as
     * varint 1. Without it libtpu asks for a device assignment of zero
     * replicas and refuses: "Requested total cores: 0". */
    static const unsigned char compile_options[] = {0x1a, 0x04, 0x20, 0x01, 0x28, 0x01};

    for (size_t v = 0; v < sizeof variants / sizeof variants[0]; v++) {
        PJRT_Program prog = {0};
        prog.struct_size = sizeof prog;
        prog.code = (char *)variants[v].code;
        prog.code_size = strlen(variants[v].code);
        prog.format = variants[v].format;
        prog.format_size = strlen(variants[v].format);

        PJRT_Client_Compile_Args cc = {0};
        cc.struct_size = sizeof cc;
        cc.client = ca.client;
        cc.program = &prog;
        cc.compile_options = (const char *)compile_options;
        cc.compile_options_size = sizeof compile_options;
        PJRT_Error *ce = api->PJRT_Client_Compile(&cc);
        if (ce != NULL) {
            PJRT_Error_Message_Args ma = {0};
            ma.struct_size = sizeof ma;
            ma.error = ce;
            api->PJRT_Error_Message(&ma);
            printf("compile[%s]: FAIL %.*s\n", variants[v].label,
                   (int)ma.message_size, ma.message);
            PJRT_Error_Destroy_Args ed = {0};
            ed.struct_size = sizeof ed;
            ed.error = ce;
            api->PJRT_Error_Destroy(&ed);
        } else {
            printf("compile[%s]: OK executable=%p\n", variants[v].label,
                   (void *)cc.executable);
        }
    }
    return 0;
}
