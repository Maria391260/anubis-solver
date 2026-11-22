/* Minimal <stdio.h> for the UEFI build. Output goes to the firmware console
 * through a callback installed by main(), so library code that "prints" ends
 * up on the serial/OVMF console instead of nowhere. */
#pragma once

#include <stddef.h>
#include <stdarg.h>

typedef struct _anubis_file FILE;

extern FILE *stdout;
extern FILE *stderr;

int snprintf(char *buf, size_t n, const char *fmt, ...);
int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap);
int printf(const char *fmt, ...);
int vprintf(const char *fmt, va_list ap);
int fprintf(FILE *f, const char *fmt, ...);
int puts(const char *s);
int putchar(int c);
int fflush(FILE *f);
