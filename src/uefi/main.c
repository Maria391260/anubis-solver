/* anubis-uefi: a challenge solver that runs before any operating system does.
 *
 *   UEFI firmware -> this PE image -> EFI_TCP4 -> JSON frames -> SHA-256
 *
 * The point is the detection ladder: CPUID decides whether the machine has
 * SHA-NI, AVX2 or AVX-512, the answer is written to \anubis-caps.json on the
 * ESP so the next boot does not have to ask again, and the solver uses the
 * best kernel the *runtime* says is safe (not the one the build happened on).
 */
#include <efi.h>
#include <stdio.h>
#include <string.h>
#include <efilib.h>

#include "caps.h"
#include "cpu.h"
#include "json.h"
#include "net.h"
#include "shim.h"
#include "mp.h"
#include "timer.h"
#include "solve.h"

#define ANUBIS_PORT 8642

static anubis_profile_t g_prof;
static EFI_HANDLE g_image;

static int same_cpu(const anubis_cpu_t *a, const anubis_cpu_t *b) {
    return a->sha_ni == b->sha_ni && a->avx2 == b->avx2 &&
           a->avx512f == b->avx512f && a->avx512vl == b->avx512vl;
}

static int handle_frame(const char *req, int len, char *resp, int cap, void *ctx) {
    static char buf[8192];
    char op[16];

    (void)ctx;
    if (len <= 0 || len >= (int)sizeof buf) {
        return snprintf(resp, (size_t)cap, "{\"ok\":false,\"error\":\"frame size\"}");
    }
    memcpy(buf, req, (size_t)len);
    buf[len] = '\0';

    if (json_str(buf, "op", op, sizeof op) != 0) {
        strcpy(op, "scan");
    }
    if (strcmp(op, "ping") == 0) {
        return snprintf(resp, (size_t)cap, "{\"ok\":true,\"pong\":true}");
    }
    if (strcmp(op, "caps") == 0) {
        char tmp[128];
        int n = solve_caps_json(g_prof.cpu.model, g_prof.kernel, g_prof.solved,
                                resp, (size_t)cap);
        (void)tmp;
        return n;
    }
    {
        solve_req_t r;
        int n;

        if (solve_parse(buf, &r) != 0) {
            return snprintf(resp, (size_t)cap,
                            "{\"ok\":false,\"error\":\"bad request\"}");
        }
        n = solve_run(&r, &g_prof.cpu, resp, (size_t)cap);
        if (n > 0 && strstr(resp, "\"hit\":true") != NULL) {
            g_prof.solved++;
            caps_save(g_image, &g_prof); /* best effort: the counter survives */
        }
        return n;
    }
}

EFI_STATUS EFIAPI efi_main(EFI_HANDLE image, EFI_SYSTEM_TABLE *st) {
    anubis_cpu_t now;
    EFI_STATUS cached;

    /* Reached from entry.S: no gnu-efi crt0, no InitializeLib. Fill in the
     * globals the firmware-facing helpers use and go. */
    ST = st;
    BS = st->BootServices;
    RT = st->RuntimeServices;
    shim_init(st);
    timer_init();
    mp_init();
    g_image = image;

    anubis_console_write("\nanubis-uefi: boot\n");
    anubis_cpu_detect(&now);
    cached = caps_load(image, &g_prof);
    if (!EFI_ERROR(cached) && same_cpu(&g_prof.cpu, &now)) {
        anubis_console_write("anubis-uefi: profile restored from caps cache\n");
        g_prof.cpu = now; /* keep the freshly measured model string */
    } else {
        g_prof.cpu = now;
        g_prof.kernel = anubis_cpu_pick(&now);
        g_prof.solved = 0;
        caps_save(image, &g_prof);
        anubis_console_write("anubis-uefi: profile detected and saved\n");
    }

    {
        char line[256];
        snprintf(line, sizeof line,
                 "anubis-uefi: cpu=%s\nanubis-uefi: sha-ni=%d avx2=%d avx512f=%d "
                 "aes=%d kernel=%s\nanubis-uefi: port=%d\n",
                 g_prof.cpu.model, g_prof.cpu.sha_ni, g_prof.cpu.avx2,
                 g_prof.cpu.avx512f, g_prof.cpu.aes,
                 anubis_kernel_name(g_prof.kernel), ANUBIS_PORT);
        anubis_console_write(line);
    }

    net_serve(ANUBIS_PORT, handle_frame, NULL);

    anubis_console_write("anubis-uefi: no network, press a key\n");
    {
        EFI_INPUT_KEY k;
        UINTN idx;
        BS->WaitForEvent(1, &st->ConIn->WaitForKey, &idx);
        st->ConIn->ReadKeyStroke(st->ConIn, &k);
    }
    return EFI_SUCCESS;
}
