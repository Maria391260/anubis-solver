/* What this machine can actually do, decided by CPUID at boot. */
#pragma once

#define ANUBIS_KERNEL_PORTABLE 0
#define ANUBIS_KERNEL_AVX2 1
#define ANUBIS_KERNEL_SHA_NI 2
#define ANUBIS_KERNEL_AVX512 3

typedef struct {
    int sha_ni;    /* CPUID.7.0:EBX[29] */
    int avx2;      /* CPUID.7.0:EBX[5]  */
    int avx512f;   /* CPUID.7.0:EBX[16] */
    int avx512vl;  /* CPUID.7.0:EBX[31] */
    int aes;       /* CPUID.1:ECX[25]   */
    char model[64]; /* brand string, spaces trimmed */
} anubis_cpu_t;

void anubis_cpu_detect(anubis_cpu_t *c);

/* The fastest kernel the *runtime* detection says is safe, and the name of it
 * as it appears in the JSON. */
int anubis_cpu_pick(const anubis_cpu_t *c);
const char *anubis_kernel_name(int k);
