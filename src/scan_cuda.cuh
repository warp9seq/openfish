#ifndef SCAN_CUDA_H
#define SCAN_CUDA_H

#include "decode.h"

#include <cuda_fp16.h>

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

__global__ void rotary(
	float *_x0,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seqlen,
    const uint64_t stride_head,
    const uint64_t rotary_half
);

__global__ void rotary_f16(
	half *x,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seq,
    const uint64_t stride_head,
    const uint64_t rotary_half
);

#ifdef __cplusplus
}
#endif

#endif // SCAN_CUDA_H