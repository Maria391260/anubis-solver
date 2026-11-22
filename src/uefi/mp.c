#include <efi.h>
#include <efilib.h>

#include "mp.h"

/* gnu-efi has no MP Services definitions (EDK2 does), so define the protocol
 * here. Field order is fixed by the UEFI spec. */
#define EFI_MP_SERVICES_PROTOCOL_GUID \
    { 0x3fdda605, 0xa76e, 0x4f46, {0xad, 0x29, 0x12, 0xf4, 0x53, 0x1b, 0x3d, 0x08} }

typedef void(EFIAPI *EFI_AP_PROCEDURE)(void *arg);

typedef struct _EFI_MP_SERVICES EFI_MP_SERVICES;
struct _EFI_MP_SERVICES {
    EFI_STATUS(EFIAPI *GetNumberOfProcessors)(EFI_MP_SERVICES *, UINTN *, UINTN *);
    EFI_STATUS(EFIAPI *GetProcessorInfo)(EFI_MP_SERVICES *, UINTN, void *);
    EFI_STATUS(EFIAPI *StartupAllAPs)(EFI_MP_SERVICES *, EFI_AP_PROCEDURE, BOOLEAN,
                                      EFI_EVENT, UINTN, void *, UINTN **);
    EFI_STATUS(EFIAPI *StartupThisAP)(EFI_MP_SERVICES *, EFI_AP_PROCEDURE, UINTN,
                                      BOOLEAN, EFI_EVENT, UINTN, void *, BOOLEAN *);
};

static const EFI_GUID gMpGuid = EFI_MP_SERVICES_PROTOCOL_GUID;

static EFI_MP_SERVICES *g_mp;
static UINTN g_cpus = 1;

int mp_init(void) {
    UINTN enabled = 0;
    EFI_STATUS st;

    st = BS->LocateProtocol(&gMpGuid, NULL, (void **)&g_mp);
    if (EFI_ERROR(st) || g_mp == NULL) {
        g_mp = NULL;
        g_cpus = 1;
        return 1;
    }
    if (EFI_ERROR(g_mp->GetNumberOfProcessors(g_mp, &g_cpus, &enabled))) {
        g_cpus = 1;
    }
    if (g_cpus == 0) {
        g_cpus = 1;
    }
    return (int)g_cpus;
}

int mp_count(void) { return (int)g_cpus; }

int mp_run(void (*fn)(void *), void *arg) {
    EFI_EVENT done = NULL;
    UINTN idx;

    if (g_mp != NULL && g_cpus > 1) {
        EFI_STATUS st = g_mp->StartupAllAPs(
            g_mp, (EFI_AP_PROCEDURE)fn, FALSE, &done, 0, arg, NULL);
        if (EFI_ERROR(st)) {
            done = NULL;
        }
    }
    fn(arg); /* the BSP pulls its weight; otherwise a 4-core box runs 3 */
    if (done != NULL) {
        BS->WaitForEvent(1, &done, &idx);
        BS->CloseEvent(done);
    }
    return 0;
}

long mp_atomic_add(volatile long *p, long v) {
    long old = v;

    __asm__ volatile("lock xaddq %0, %1" : "+r"(old), "+m"(*p) : : "memory");
    return old;
}

int mp_atomic_cas(volatile int *p, int want, int set) {
    unsigned char ok;

    __asm__ volatile("lock cmpxchgl %3, %1; sete %0"
                     : "=q"(ok), "+m"(*p)
                     : "a"(want), "r"(set)
                     : "memory");
    return ok != 0;
}
