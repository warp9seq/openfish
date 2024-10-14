#ifndef SCAN_CUDA_H
#define SCAN_CUDA_H

#include <stdint.h>
#include "decode.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void bwd_scan(
	const scan_args_t args,
	float *out
);

__global__ void fwd_post_scan(
    const scan_args_t args,
    const float *bwd,
    float *out
);

#ifdef __cplusplus
}
#endif

#endif // SCAN_CUDA_H