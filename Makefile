CC      ?= cc
PY      ?= python3
CFLAGS  ?= -O3 -fPIC -Wall -Wextra -std=gnu17
NATIVE  ?= -march=native
LDFLAGS ?= -shared -pthread
LDLIBS  ?=
INC     := -Iinclude -Isrc -Isrc/core -Isrc/hashx -Isrc/cpu -Isrc/cuda \
           -Isrc/levelzero -Isrc/vulkan -Isrc/opencl -Ithird_party/argon2

# ---- build configuration --------------------------------------------------
# `make config` answers three questions and writes config.mk:
#   which GPU architecture (a T4 is sm_75), which CPU architecture (x86_64 or
#   aarch64), and what that CPU supports (AVX2 / AVX-512 / SHA-NI, or NEON).
# config.mk is generated, per-machine and git-ignored; overrides on the command
# line win, e.g. make config CUDA_ARCH=sm_86 CPU_SIMD=avx2
CONFIG_ARGS ?=
-include config.mk
ifeq ($(wildcard config.mk),)
$(info [anubis] no config.mk yet - run 'make config' to detect GPU arch, CPU arch and SIMD)
endif
NVCC_ARCH := $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),)

.PHONY: config
config:
	$(PY) tools/configure.py $(CONFIG_ARGS)

# Optional CUDA backend (NVIDIA). Built when nvcc is available; the library
# then links libcudart and prefers CUDA over Vulkan at runtime.
CUDA ?= auto
NVCC ?= nvcc
CUDA_SRC :=
ifneq ($(CUDA),0)
ifneq ($(shell command -v $(NVCC) 2>/dev/null),)
CUDA_SRC := src/cuda/cuda_sha256.cu src/cuda/cuda_argon2.cu src/cuda/cuda_hashx.cu \
            src/cuda/cuda_tensor.cu
CFLAGS   += -DANUBIS_HAVE_CUDA
# nvcc is the linker here; it rejects bare -pthread, so pass it to the host
# compiler explicitly.
LDFLAGS  := $(filter-out -pthread,$(LDFLAGS)) -Xcompiler -pthread -lcudart
NVCC_LINK_ARCH := $(NVCC_ARCH)
LINK      = $(NVCC)
endif
endif
LINK     ?= $(CC)

LIB      := libanubis_solver.so
BUILD    := build
CORE_SRC := src/core/anubis_solver.c src/hashx/hashx.c

# Optional Vulkan backend (SHA-256 challenges). Auto-detects the headers;
# the library dlopen()s libvulkan.so.1 at runtime, so there is no hard
# dependency on a Vulkan loader being present.
VULKAN ?= auto
ifneq ($(VULKAN),0)
ifneq ($(wildcard /usr/include/vulkan/vulkan.h),)
CORE_SRC += src/vulkan/vulkan.c
CFLAGS  += -DANUBIS_HAVE_VULKAN
LDFLAGS += -ldl
endif
endif
# Optional OpenCL backend (any GPU with an ICD: Intel, AMD, NVIDIA, Mesa).
# dlopen()s libOpenCL.so.1, so there is no hard link and no -lOpenCL here.
# Vulkan is still the preferred path where it works; this one is used when no
# other GPU backend answers, or when ANUBIS_OPENCL=1 asks for it by name.
OPENCL ?= auto
ifneq ($(OPENCL),0)
ifneq ($(wildcard /usr/include/CL/cl.h),)
CORE_SRC += src/opencl/opencl_backend.c
CFLAGS   += -DANUBIS_HAVE_OPENCL
LDFLAGS  += -ldl
endif
endif
# Optional oneAPI Level Zero backend (Intel GPUs). Needs the Level Zero
# headers; the loader is dlopen()ed at runtime, so there is no hard link.
L0 ?= auto
L0_INC ?= $(if $(wildcard /usr/include/level_zero/ze_api.h),/usr/include,\
            $(if $(wildcard /usr/local/include/level_zero/ze_api.h),/usr/local/include,))
ifneq ($(L0),0)
ifneq ($(wildcard $(L0_INC)/level_zero/ze_api.h),)
CORE_SRC += src/levelzero/levelzero_sha256.c src/levelzero/levelzero_argon2.c \
             src/levelzero/levelzero_hashx.c
CFLAGS   += -DANUBIS_HAVE_LEVELZERO -I$(L0_INC)
LDFLAGS  += -ldl
endif
endif

# The architecture of what we are building, not of what we are building on:
#   make ARCH=aarch64 CC=aarch64-linux-gnu-gcc NATIVE=
# is how the released ARM binaries are produced on an x86 box.
ARCH     ?= $(shell uname -m)

# Assembly kernels: x86-64 (AVX2 + SHA-NI) or AArch64 (NEON + SHA-2).
# The portable C fallbacks are always compiled.
ifeq ($(ARCH),x86_64)
ASM_SRC  := src/cpu/avx2.s src/cpu/shani.s src/cpu/avx512.s
else ifeq ($(ARCH),aarch64)
ASM_SRC  := src/cpu/neon.s
CFLAGS   += -DANUBIS_HAVE_NEON_ASM
else
ASM_SRC  :=
endif
# The C fetch layer: follow an Anubis challenge and solve it, over HTTPS.
# It lives in the library so the services and every .so consumer (Python, the
# language bindings) share one client. It needs OpenSSL headers; the
# libcurl-impersonate browser fingerprint is dlopen()ed at runtime, so there is
# no build-time dependency on it. `make FETCH=0` drops the layer, and the .so
# then has no OpenSSL dependency at all.
FETCH_SRC := src/fetch/anubis_fetch.c src/fetch/fetch_impersonate.c
FETCH ?= auto
HAVE_FETCH :=
ifneq ($(FETCH),0)
ifneq ($(ARCH),none)
ifneq ($(shell printf '#include <openssl/ssl.h>\n' | $(CC) -x c -E - >/dev/null 2>&1 && echo yes),)
CORE_SRC += $(FETCH_SRC)
CFLAGS   += -I$(CURDIR)/src/fetch -DANUBIS_HAVE_FETCH
LDLIBS   += -lssl -lcrypto -ldl
HAVE_FETCH := 1
endif
endif
endif

ARGON2_ALL := $(wildcard third_party/argon2/*.c third_party/argon2/blake2/*.c)
# opt.c needs x86 SSE/AVX intrinsics; ref.c is the portable implementation.
ifeq ($(ARCH),x86_64)
ARGON2_SRC := $(filter-out third_party/argon2/ref.c,$(ARGON2_ALL))
else
ARGON2_SRC := $(filter-out third_party/argon2/opt.c,$(ARGON2_ALL))
endif

OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(CORE_SRC)) \
       $(patsubst src/%.s,$(BUILD)/%.o,$(ASM_SRC)) \
       $(patsubst src/%.cu,$(BUILD)/%.o,$(CUDA_SRC)) \
       $(patsubst third_party/argon2/%.c,$(BUILD)/argon2/%.o,$(ARGON2_SRC))

DEP := $(OBJ:.o=.d)
-include $(DEP)

.DEFAULT_GOAL := all

.PHONY: all test clean shader windows services test-api test-mcp test-impersonate impersonate-lib

# The Windows pair and the release zip/msi: a mingw-w64 cross build, no Windows
# needed. `make windows WIN_ARGS=--smoke` also runs a challenge/pass round trip
# under wine. See betterAnusbis/win32/build-windows.sh and RELEASING.md.
windows:
	./betterAnusbis/win32/build-windows.sh $(WIN_ARGS)

all: $(LIB)

$(BUILD):
	mkdir -p $(BUILD)/argon2/blake2

$(BUILD)/%.o: src/%.c | $(BUILD)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -pthread -MMD -MP -c $< -o $@

$(BUILD)/%.o: src/%.cu | $(BUILD)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 $(NVCC_ARCH) -Xcompiler -fPIC $(INC) -c $< -o $@

$(BUILD)/%.o: src/%.s | $(BUILD)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/argon2/%.o: third_party/argon2/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -pthread -MMD -MP -c $< -o $@

$(LIB): $(OBJ)
	$(LINK) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)

# ---------------------------------------------------------------------------
# The C services (2.0.1 "shaoogle"): an MCP server and an HTTP API service,
# both thin layers over libanubis_solver.so plus the shared fetch client. They
# are separate binaries on purpose - a gate operator should not have to install
# a JSON-RPC server, and an agent should not have to run a gate.
# ---------------------------------------------------------------------------
SERVICE_LIBS := -L. -lanubis_solver -Wl,-rpath,'$$ORIGIN' -lpthread

# The fetch layer lives in the library (see the FETCH block above), so the
# services are one translation unit plus the .so. Building it into the library
# as well means .so consumers (Python, the bindings) get the same client.
ifeq ($(HAVE_FETCH),1)
SERVICES := anubis_api anubis_mcp
else
SERVICES :=
endif

anubis_api: src/api/anubis_api.c $(LIB)
	$(CC) $(CFLAGS) -Isrc/fetch -Iinclude -o $@ src/api/anubis_api.c $(SERVICE_LIBS)

anubis_mcp: src/mcp/anubis_mcp.c $(LIB)
	$(CC) $(CFLAGS) -Isrc/fetch -Iinclude -o $@ src/mcp/anubis_mcp.c $(SERVICE_LIBS)

services: $(SERVICES)
ifeq ($(HAVE_FETCH),1)
	@echo "services: anubis_api and anubis_mcp are up to date"
else
	@echo "services need OpenSSL headers (libssl-dev) and the fetch layer;" >&2
	@echo "this build ran with FETCH=0 or a compiler that has no <openssl/ssl.h>" >&2
	@false
endif

# The browser impersonation profile is a runtime library: copy it into the
# layout the fetch layer searches. It is not linked, and not needed to build.
# Point CURL_IMPERSONATE_LIB at a libcurl-impersonate-chrome.so.4 (from a
# curl-impersonate build, or a curl_cffi wheel's lib64/ directory).
CURL_IMPERSONATE_LIB ?=
impersonate-lib:
	@if [ -z "$(CURL_IMPERSONATE_LIB)" ]; then \
	    echo "usage: make impersonate-lib CURL_IMPERSONATE_LIB=/path/to/libcurl-impersonate-chrome.so.4"; \
	    false; \
	fi
	@if [ ! -f "$(CURL_IMPERSONATE_LIB)" ]; then \
	    echo "not a file: $(CURL_IMPERSONATE_LIB)"; false; \
	fi
	@mkdir -p 3rd_party/curl-impersonate/lib
	cp -a "$(CURL_IMPERSONATE_LIB)" 3rd_party/curl-impersonate/lib/
	@echo "installed 3rd_party/curl-impersonate/lib/$$(basename $(CURL_IMPERSONATE_LIB))"
	@echo "run: ./anubis_mcp   (or anubis_api) and look for \"curl-impersonate\" in the fetch result"

test-api: anubis_api
	tests/api_test.sh

test-mcp: anubis_mcp
	tests/mcp_test.sh

test-impersonate: $(LIB) anubis_mcp anubis_api
	tests/impersonate_test.sh

test: $(LIB)
	python3 tests/test_core.py

shader:
	bash tools/build_shader.sh

# Assemble the hand-written Intel Xe listing. iga64 is Intel's assembler; it
# comes from libigc2-tools, or from a built IGC tree (see src/intel/xe/README.md).
#
#   make xe-asm        -> build/intelXe_sha256.krn (the machine code)
#   make xe-loader     -> build/xe-loader, which runs it through OpenCL
IGA64 ?= $(shell command -v iga64 2>/dev/null)
.PHONY: xe-asm
xe-asm: | $(BUILD)
ifneq ($(IGA64),)
	$(IGA64) -p 12p1 -a src/intel/xe/intelXe_sha256.xe -o $(BUILD)/intelXe_sha256.krn
	@echo "assembled $(BUILD)/intelXe_sha256.krn: $$(wc -c < $(BUILD)/intelXe_sha256.krn) bytes, $$(( $$(wc -c < $(BUILD)/intelXe_sha256.krn) / 16 )) instructions"
else
	@echo "xe-asm needs iga64: apt install libigc2-tools, or build visa/iga from IGC"
	@false
endif

# HashX as Intel Xe assembly.  The HashX program is generated per challenge, so
# the listing is generated too: the keys and the 512 instructions are baked in
# as immediates and the whole kernel becomes straight-line code.
#
#   make xe-hashx        -> regenerate, check the digests, assemble
#   python3 tools/gen_xe_hashx.py --selftest    -> just the 64-bit primitives
#   python3 tools/gen_xe_hashx.py --stages      -> the register file vs the C
.PHONY: xe-hashx
xe-hashx:
	python3 tools/gen_xe_hashx.py --assemble

# Intel Xe assembly loader: patches a listing assembled by IGA into an IGC Zebin
# and loads it through OpenCL, which is the only way to execute hand-written gen
# instructions on this machine (Vulkan and OpenCL both take SPIR-V otherwise).
# Opt-in, because it is a research tool, not part of the library.
#
#   make xe-loader
#   build/xe-loader --text /tmp/ours.krn --count 65536
#
# Needs OpenCL headers; the driver is loaded at runtime.
XE_CL_HDR ?= $(wildcard /usr/include/CL/cl.h)
.PHONY: xe-loader
xe-loader: | $(BUILD)
ifneq ($(XE_CL_HDR),)
	$(CC) -O2 -DCL_TARGET_OPENCL_VERSION=300 -o $(BUILD)/xe-loader tools/xe_load.c -lOpenCL
	@echo "built $(BUILD)/xe-loader"
else
	@echo "xe-loader needs OpenCL headers: apt install ocl-icd-opencl-dev opencl-headers"
	@false
endif

clean:
	rm -rf $(BUILD) $(LIB) anubis_api anubis_mcp
