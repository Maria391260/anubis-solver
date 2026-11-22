/* UEFI has no threads, so this is not pthreads: it is EFI_MP_SERVICES_PROTOCOL,
 * the firmware's own application-processor wakeup. The BSP does its share too,
 * so an N-processor box gets N workers, not N-1.
 *
 * APs cannot be given per-CPU arguments (the procedure gets one shared pointer),
 * so the work is handed out with an atomic counter: whoever grabs chunk k does
 * chunk k. */
#pragma once

#include <efi.h>

/* Returns the number of processors (1 if MP services are missing). */
int mp_init(void);
int mp_count(void);

/* Runs fn(arg) on every AP and on the BSP, then waits. */
int mp_run(void (*fn)(void *), void *arg);

/* lock xadd, for the chunk counter. */
long mp_atomic_add(volatile long *p, long v);

/* cmpxchg, for claiming the hit exactly once. */
int mp_atomic_cas(volatile int *p, int want, int set);
