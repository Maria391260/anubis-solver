/* The capability cache. "Detect once, write it down, trust it next boot" is the
 * whole point: a firmware boot on the same box never has to re-probe, and the
 * profile is visible to whoever reads the ESP later. */
#pragma once

#include <efi.h>

#include "cpu.h"

typedef struct {
    anubis_cpu_t cpu;
    int kernel;          /* anubis_cpu_pick() result, possibly from the cache */
    unsigned long long solved;
} anubis_profile_t;

/* Reads \anubis-caps.json from the volume this image was loaded from. Returns
 * EFI_SUCCESS only if the file existed and parsed. */
EFI_STATUS caps_load(EFI_HANDLE image, anubis_profile_t *p);

/* Writes the same file; creates it if needed. */
EFI_STATUS caps_save(EFI_HANDLE image, const anubis_profile_t *p);
