#include <efi.h>
#include <string.h>
#include <stdio.h>
#include <efilib.h>

#include "caps.h"
#include "json.h"

#define CAPS_NAME L"\\anubis-caps.json"
#define CAPS_MAX 512

static EFI_STATUS open_root(EFI_HANDLE image, EFI_FILE **root) {
    EFI_LOADED_IMAGE *li = NULL;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *fs = NULL;
    EFI_STATUS st;

    st = BS->HandleProtocol(image, &gEfiLoadedImageProtocolGuid, (void **)&li);
    if (EFI_ERROR(st)) {
        return st;
    }
    st = BS->HandleProtocol(li->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid,
                            (void **)&fs);
    if (EFI_ERROR(st)) {
        return st;
    }
    return fs->OpenVolume(fs, root);
}

EFI_STATUS caps_load(EFI_HANDLE image, anubis_profile_t *p) {
    EFI_FILE *root = NULL, *f = NULL;
    EFI_STATUS st;
    char buf[CAPS_MAX];
    UINTN len = sizeof buf - 1;
    char k[24];

    st = open_root(image, &root);
    if (EFI_ERROR(st)) {
        return st;
    }
    st = root->Open(root, &f, CAPS_NAME, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(st)) {
        return st;
    }
    st = f->Read(f, &len, buf);
    f->Close(f);
    if (EFI_ERROR(st) || len == 0) {
        return EFI_LOAD_ERROR;
    }
    buf[len] = '\0';

    p->cpu.sha_ni = (int)json_int(buf, "sha_ni", 0);
    p->cpu.avx2 = (int)json_int(buf, "avx2", 0);
    p->cpu.avx512f = (int)json_int(buf, "avx512f", 0);
    p->cpu.avx512vl = (int)json_int(buf, "avx512vl", 0);
    p->solved = (unsigned long long)json_int(buf, "solved", 0);
    if (json_str(buf, "model", p->cpu.model, sizeof p->cpu.model) != 0) {
        p->cpu.model[0] = '\0';
    }
    p->kernel = ANUBIS_KERNEL_PORTABLE;
    if (json_str(buf, "kernel", k, sizeof k) == 0) {
        if (strcmp(k, "sha-ni") == 0) {
            p->kernel = ANUBIS_KERNEL_SHA_NI;
        } else if (strcmp(k, "avx512") == 0) {
            p->kernel = ANUBIS_KERNEL_AVX512;
        } else if (strcmp(k, "avx2") == 0) {
            p->kernel = ANUBIS_KERNEL_AVX2;
        }
    }
    return EFI_SUCCESS;
}

EFI_STATUS caps_save(EFI_HANDLE image, const anubis_profile_t *p) {
    EFI_FILE *root = NULL, *f = NULL;
    EFI_STATUS st;
    char buf[CAPS_MAX];
    int n;

    st = open_root(image, &root);
    if (EFI_ERROR(st)) {
        return st;
    }
    st = root->Open(root, &f, CAPS_NAME,
                    EFI_FILE_MODE_CREATE | EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE,
                    0);
    if (EFI_ERROR(st)) {
        return st;
    }
    n = snprintf(buf, sizeof buf,
                 "{\"version\":1,\"model\":\"%s\",\"sha_ni\":%d,\"avx2\":%d,"
                 "\"avx512f\":%d,\"avx512vl\":%d,\"aes\":%d,\"kernel\":\"%s\","
                 "\"solved\":%llu}\n",
                 p->cpu.model, p->cpu.sha_ni, p->cpu.avx2, p->cpu.avx512f,
                 p->cpu.avx512vl, p->cpu.aes, anubis_kernel_name(p->kernel),
                 p->solved);
    if (n < 0) {
        f->Close(f);
        return EFI_DEVICE_ERROR;
    }
    {
        UINTN sz = (UINTN)n;
        f->SetPosition(f, 0);
        st = f->Write(f, &sz, buf);
        if (EFI_ERROR(st)) {
            /* some FAT drivers keep the old length on a rewrite */
            f->SetPosition(f, 0);
            sz = (UINTN)n;
            st = f->Write(f, &sz, buf);
        }
    }
    f->Flush(f);
    f->Close(f);
    return st;
}
