# Standalone build: the PJRT C API client only needs libc and libdl, because
# the TPU plugin itself (libtpu.so) is dlopen()ed at runtime - the same shape
# as the Vulkan and OpenCL backends in this repository.
#
# tpu_backend.o is the one artifact the universal libanubis_solver.so links:
# it must stay -fPIC and must not depend on libtpu being present.
CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu17
BINS    := pjrt_probe tensor_pjrt sha256_pjrt
LIBOBJ  := tpu_backend.o
SELFTEST := tpu_backend_selftest

all: $(BINS)

pjrt_probe: pjrt_probe.c pjrt_c_api.h
	$(CC) $(CFLAGS) -I. -o $@ $< -ldl

tensor_pjrt: tensor_pjrt.c pjrt_c_api.h
	$(CC) $(CFLAGS) -I. -o $@ $< -ldl

sha256_pjrt: sha256_pjrt.c pjrt_c_api.h pjrt_common.h
	$(CC) $(CFLAGS) -I. -o $@ $< -ldl

# Library object for libanubis_solver.so. -fPIC is not optional here.
$(LIBOBJ): tpu_backend.c tpu_backend.h pjrt_c_api.h pjrt_common.h
	$(CC) $(CFLAGS) -fPIC -I. -c -o $@ tpu_backend.c

$(SELFTEST): tpu_backend_selftest.c $(LIBOBJ) tpu_backend.h pjrt_common.h
	$(CC) $(CFLAGS) -I. -o $@ tpu_backend_selftest.c $(LIBOBJ) -ldl -lpthread

selftest: $(SELFTEST)

clean:
	rm -f $(BINS) $(LIBOBJ) $(SELFTEST)

.PHONY: all clean selftest
