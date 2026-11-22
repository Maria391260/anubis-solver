/* CPUID is the only truth on a booting machine: the firmware may run us under
 * a different CPU than the one in the datasheet, and in a VM the answer can be
 * "none of the above". Everything is decided here, once, and then cached. */
#include <stdint.h>

#include "cpu.h"

typedef struct {
    uint32_t eax, ebx, ecx, edx;
} cpuid_t;

static void cpuid(uint32_t leaf, uint32_t sub, cpuid_t *r) {
    __asm__ volatile("cpuid"
                     : "=a"(r->eax), "=b"(r->ebx), "=c"(r->ecx), "=d"(r->edx)
                     : "a"(leaf), "c"(sub));
}

static uint64_t read_xcr0(void) {
    uint32_t lo, hi;

    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

void anubis_cpu_detect(anubis_cpu_t *c) {
    cpuid_t l1, l7;
    int i, k;

    c->sha_ni = 0;
    c->avx2 = 0;
    c->avx512f = 0;
    c->avx512vl = 0;
    c->aes = 0;
    c->model[0] = '\0';

    cpuid(0, 0, &l1);
    cpuid(1, 0, &l1);
    c->aes = (l1.ecx >> 25) & 1;

    cpuid(7, 0, &l7);
    c->sha_ni = (int)((l7.ebx >> 29) & 1);
    c->avx2 = (int)((l7.ebx >> 5) & 1);
    c->avx512f = (int)((l7.ebx >> 16) & 1);
    c->avx512vl = (int)((l7.ebx >> 31) & 1);

    /* The bits are necessary but not sufficient: the firmware must have
     * enabled the state in XCR0, or the first AVX instruction faults. */
    if ((l1.ecx & (1u << 27)) == 0) { /* OSXSAVE */
        c->avx2 = 0;
        c->avx512f = 0;
        c->avx512vl = 0;
    } else {
        uint64_t xcr0 = read_xcr0();
        if ((xcr0 & 0x6) != 0x6) {
            c->avx2 = 0;
        }
        if ((xcr0 & 0xE6) != 0xE6) {
            c->avx512f = 0;
            c->avx512vl = 0;
        }
    }

    /* Brand string: leaf 0x80000000 tells us how much of it exists. */
    cpuid(0x80000000u, 0, &l1);
    if (l1.eax >= 0x80000004u) {
        uint32_t *w = (uint32_t *)c->model;
        for (i = 0; i < 3; i++) {
            cpuid(0x80000002u + (uint32_t)i, 0, &l7);
            w[i * 4 + 0] = l7.eax;
            w[i * 4 + 1] = l7.ebx;
            w[i * 4 + 2] = l7.ecx;
            w[i * 4 + 3] = l7.edx;
        }
        c->model[48] = '\0';
        /* collapse runs of spaces and trim */
        k = 0;
        for (i = 0; i < 48; i++) {
            if (c->model[i] == ' ' && (k == 0 || c->model[k - 1] == ' ')) {
                continue;
            }
            c->model[k++] = c->model[i];
        }
        while (k > 0 && c->model[k - 1] == ' ') {
            k--;
        }
        c->model[k] = '\0';
    }
}

int anubis_cpu_pick(const anubis_cpu_t *c) {
    if (c->sha_ni) {
        return ANUBIS_KERNEL_SHA_NI;
    }
    if (c->avx512f && c->avx512vl) {
        return ANUBIS_KERNEL_AVX512;
    }
    if (c->avx2) {
        return ANUBIS_KERNEL_AVX2;
    }
    return ANUBIS_KERNEL_PORTABLE;
}

const char *anubis_kernel_name(int k) {
    switch (k) {
    case ANUBIS_KERNEL_SHA_NI:
        return "sha-ni";
    case ANUBIS_KERNEL_AVX512:
        return "avx512";
    case ANUBIS_KERNEL_AVX2:
        return "avx2";
    default:
        return "portable";
    }
}
