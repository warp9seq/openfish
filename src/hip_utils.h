#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <hip/hip_runtime.h>
#include <openfish/openfish_error.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIP_CHECK(error) { gpuAssert2((error),__FILE__, __LINE__); }

static inline void gpuAssert2(hipError_t error, const char *file, int line) {
     if (error != hipSuccess) { \
        fprintf(stderr,"HIP error: %s, in file: %s, line number: %d\n", hipGetErrorString(error), file, line); \
        exit(1); \
     } \
}

#define checkHipError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	hipError_t code = hipGetLastError();
	if (code != hipSuccess) {
        fprintf(stderr, "Hip error: %s \n in file: %s, line number: %d\n", hipGetErrorString(code), file, line);
        exit(1);
   }
}

// Post-launch kernel error check. checkHipError() (= hipGetLastError()) is sync-free and catches
// launch-config errors immediately; async execution errors are sticky and surface on the next HIP
// call. In DEBUG builds we also hipDeviceSynchronize() so such an async error is pinpointed to THIS
// kernel. Release builds never sync here (no per-kernel serialization; the kernels run on the
// default stream, so their ordering is preserved and the final device->host copy synchronizes).
#ifdef DEBUG
#define checkKernel() { checkHipError(); HIP_CHECK(hipDeviceSynchronize()); }
#else
#define checkKernel() { checkHipError(); }
#endif

// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ static float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#ifdef __cplusplus
}
#endif

#endif // HIP_UTILS_H