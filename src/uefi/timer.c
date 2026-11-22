#include <efi.h>
#include <efilib.h>
#include <stdint.h>

#include "timer.h"

static uint64_t g_per_us;

uint64_t timer_now(void) {
    uint32_t lo, hi;

    /* RDTSC: available on every x86_64 the firmware will run. The TSC is not
     * guaranteed invariant by the spec, but every machine UEFI runs on today
     * has an invariant TSC, and the calibration below makes the ratio explicit
     * rather than assumed. */
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void timer_init(void) {
    uint64_t a, b;

    if (g_per_us != 0) {
        return;
    }
    a = timer_now();
    BS->Stall(10000); /* microseconds; 10 ms is long enough to average noise */
    b = timer_now();
    g_per_us = (b - a) / 10000ULL;
    if (g_per_us == 0) {
        g_per_us = 1;
    }
}

uint64_t timer_us(uint64_t delta) { return delta / g_per_us; }
