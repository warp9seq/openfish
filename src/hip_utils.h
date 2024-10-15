#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <hip/hip_runtime.h>
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

const int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = (condition);                                                 \
        if(error != hipSuccess)                                                             \
        {                \
            ERROR("%s", hipGetErrorString(error));                               \
            exit(error_exit_code);                                                          \
        }                                                                                   \
    }

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