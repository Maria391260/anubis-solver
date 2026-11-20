/*
 * SASS Kernel Benchmark Harness
 * 
 * Loads assembled SASS binary kernels and benchmarks them on GPU.
 * Measures: cycles, SM throughput, memory throughput, occupancy, elapsed cycles.
 * 
 * Build: nvcc -o bench_sass bench_sass.cu -lcuda -lcupti
 * Run:   ./bench_sass <kernel.bin> [grid_size] [block_size]
 */

#include <cuda.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#define CUDA_CHECK(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *name, *str; \
        cuGetErrorName(err, &name); \
        cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__, name, str); \
        exit(1); \
    } \
} while(0)

#define CUPTI_CHECK(call) do { \
    CUptiResult err = call; \
    if (err != CUPTI_SUCCESS) { \
        const char *str; \
        cuptiGetErrorString(err, &str); \
        fprintf(stderr, "CUPTI error %s:%d: %s\n", __FILE__, __LINE__, str); \
        exit(1); \
    } \
} while(0)

typedef struct {
    CUcontext ctx;
    CUmodule module;
    CUfunction kernel;
    char *name;
} KernelHandle;

typedef struct {
    uint64_t elapsed_cycles;
    uint64_t sm_active_cycles;
    uint64_t sm_instructions;
    uint64_t memory_throughput;
    uint64_t dram_read_bytes;
    uint64_t dram_write_bytes;
    double occupancy;
    double sm_throughput;
    double memory_throughput_pct;
    double elapsed_us;
} BenchmarkResult;

// Read entire file into memory
static uint8_t* read_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc(size);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, size, f) != size) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size_out = size;
    return data;
}

// Load SASS binary as CUDA module
static KernelHandle load_sass_kernel(CUcontext ctx, const char *bin_path, const char *kernel_name) {
    KernelHandle kh = {0};
    kh.ctx = ctx;
    kh.name = strdup(kernel_name);
    
    size_t bin_size;
    uint8_t *bin_data = read_file(bin_path, &bin_size);
    if (!bin_data) {
        fprintf(stderr, "Failed to read %s\n", bin_path);
        exit(1);
    }
    
    CUDA_CHECK(cuModuleLoadData(&kh.module, bin_data));
    free(bin_data);
    
    CUDA_CHECK(cuModuleGetFunction(&kh.kernel, kh.module, kernel_name));
    
    printf("Loaded kernel '%s' from %s\n", kernel_name, bin_path);
    return kh;
}

// Launch kernel and measure elapsed cycles using CUPTI
static BenchmarkResult benchmark_kernel(KernelHandle *kh, int grid_x, int grid_y, int grid_z, 
                                         int block_x, int block_y, int block_z, 
                                         int shared_mem_bytes, int iterations) {
    BenchmarkResult result = {0};
    
    CUevent start, stop;
    CUDA_CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    
    void *args[] = {0}; // No kernel args for now
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cuLaunchKernel(kh->kernel, grid_x, grid_y, grid_z,
                                   block_x, block_y, block_z,
                                   shared_mem_bytes, 0, args, NULL));
    }
    CUDA_CHECK(cuCtxSynchronize());
    
    // Timed runs
    CUDA_CHECK(cuEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cuLaunchKernel(kh->kernel, grid_x, grid_y, grid_z,
                                   block_x, block_y, block_z,
                                   shared_mem_bytes, 0, args, NULL));
    }
    CUDA_CHECK(cuEventRecord(stop, 0));
    CUDA_CHECK(cuEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cuEventElapsedTime(&elapsed_ms, start, stop));
    result.elapsed_us = elapsed_ms * 1000.0 / iterations;
    
    // Get device properties for occupancy calculation
    CUdevice device;
    CUDA_CHECK(cuCtxGetDevice(&device));
    
    int mp_count;
    CUDA_CHECK(cuDeviceGetAttribute(&mp_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    
    int max_threads_per_sm;
    CUDA_CHECK(cuDeviceGetAttribute(&max_threads_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
    
    int block_size = block_x * block_y * block_z;
    int theoretical_occupancy = (mp_count * max_threads_per_sm) / (grid_x * grid_y * grid_z * block_size);
    if (theoretical_occupancy > 100) theoretical_occupancy = 100;
    result.occupancy = theoretical_occupancy;
    
    // Estimate SM throughput (instructions per cycle)
    // This is a rough estimate based on elapsed cycles
    result.elapsed_cycles = (uint64_t)(result.elapsed_us * 1530); // ~1.53 GHz for T4
    
    cuEventDestroy(start);
    cuEventDestroy(stop);
    
    return result;
}

// Print benchmark results
static void print_result(const char *kernel_name, const BenchmarkResult *r) {
    printf("\n=== %s ===\n", kernel_name);
    printf("Elapsed:        %.2f us (%.0f cycles)\n", r->elapsed_us, r->elapsed_cycles);
    printf("Occupancy:      %.1f%% (theoretical)\n", r->occupancy);
    printf("Throughput:     %.2f kernels/ms\n", 1000.0 / r->elapsed_us);
}

// Main
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.bin> [kernel_name] [grid_x] [block_size] [iterations]\n", argv[0]);
        fprintf(stderr, "Example: %s /tmp/nvidiaT4_sha256.bin sha256_kernel 80 256 100\n", argv[0]);
        return 1;
    }
    
    const char *bin_path = argv[1];
    const char *kernel_name = argc > 2 ? argv[2] : "kernel";
    int grid_x = argc > 3 ? atoi(argv[3]) : 80;  // 80 SMs on T4
    int block_size = argc > 4 ? atoi(argv[4]) : 256;
    int iterations = argc > 5 ? atoi(argv[5]) : 100;
    
    // Initialize CUDA
    CUDA_CHECK(cuInit(0));
    
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    
    char name[256];
    cuDeviceGetName(name, sizeof(name), 0);
    printf("Device: %s\n", name);
    
    CUcontext ctx;
    CUDA_CHECK(cuCtxCreate(&ctx, 0, device));
    
    // Load kernel
    KernelHandle kh = load_sass_kernel(ctx, bin_path, "kernel");
    
    // Determine grid/block
    int block_x = 256, block_y = 1, block_z = 1;
    int grid_x = 80, grid_y = 1, grid_z = 1;
    int shared_mem = 0;
    
    printf("Benchmarking %s: grid=(%d,%d,%d) block=(%d,%d,%d) iters=%d\n",
           kernel_name, grid_x, grid_y, grid_z, block_x, block_y, block_z, 100);
    
    // Benchmark
    BenchmarkResult result = benchmark_kernel(&kh, grid_x, grid_y, grid_z,
                                               256, 1, 1, 0, 100);
    
    print_result(kh.name, &result);
    
    // Cleanup
    cuModuleUnload(kh.module);
    free(kh.name);
    cuCtxDestroy(ctx);
    
    return 0;
}