/*
 * cuda_min.h -- the handful of CUDA driver API declarations ptx_loader.c
 * needs, for build machines that ship ptxas but not the driver headers
 * (Kaggle images are one). Used with -DANUBIS_MIN_CUDA_H; when a real
 * cuda.h is available it is used instead.
 *
 * Only the entry points below are declared. They are called exactly as
 * libcuda exports them (the _v2 spellings that cuda.h's macros resolve to),
 * so the ABI matches.
 */
#ifndef ANUBIS_CUDA_MIN_H
#define ANUBIS_CUDA_MIN_H

#include <stddef.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUevent_st *CUevent;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CU_EVENT_DEFAULT 0u

CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxSynchronize(void);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount);
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX,
                        unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, void *hStream,
                        void **kernelParams, void **extra);
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult cuEventDestroy_v2(CUevent hEvent);

/* same aliases cuda.h provides */
#define cuMemAlloc cuMemAlloc_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemsetD8 cuMemsetD8_v2
#define cuEventDestroy cuEventDestroy_v2

#endif /* ANUBIS_CUDA_MIN_H */
