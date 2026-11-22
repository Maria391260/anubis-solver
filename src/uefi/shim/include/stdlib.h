/* Minimal <stdlib.h> for the UEFI build: firmware pool memory, no OS. */
#pragma once

#include <stddef.h>

void *malloc(size_t n);
void *calloc(size_t n, size_t sz);
void *realloc(void *p, size_t n);
void free(void *p);

int atoi(const char *s);
long atol(const char *s);
long strtol(const char *s, char **end, int base);
unsigned long strtoul(const char *s, char **end, int base);
void qsort(void *base, size_t n, size_t sz,
           int (*cmp)(const void *, const void *));
char *getenv(const char *name);
void abort(void);

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
