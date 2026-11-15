# Standalone build: the PJRT C API client only needs libc and libdl, because
# the TPU plugin itself (libtpu.so) is dlopen()ed at runtime - the same shape
# as the Vulkan and OpenCL backends in this repository.
CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu17
BINS    := pjrt_probe tensor_pjrt sha256_pjrt

all: $(BINS)

pjrt_probe: pjrt_probe.c pjrt_c_api.h
	$(CC) $(CFLAGS) -I. -o $@ $< -ldl

tensor_pjrt: tensor_pjrt.c pjrt_c_api.h
	$(CC) $(CFLAGS) -I. -o $@ $< -ldl

sha256_pjrt: sha256_pjrt.c pjrt_c_api.h pjrt_common.h
	$(CC) $(CFLAGS) -I. -o $@ $< -ldl

clean:
	rm -f $(BINS)

.PHONY: all clean
