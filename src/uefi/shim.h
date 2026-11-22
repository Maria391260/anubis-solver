/* The tiny platform layer the firmware build exposes to the rest of the app. */
#pragma once

#include <efi.h>

void shim_init(EFI_SYSTEM_TABLE *st);
void anubis_console_write(const char *s);
