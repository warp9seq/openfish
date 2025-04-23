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
    float *_o0,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seqlen,
    const uint64_t stride_c,
    const uint64_t stride_head,
    const uint64_t stride_head_dim,
    const uint64_t stride_rotary
);

// __global__ void rotary(
// 	half *_OUT,
//     half *_X,
//     half *_COS,
//     half *_SIN,
//     const uint64_t seqlen_offt,
//     const uint64_t seqlen,
//     const uint64_t rotary_dim,
//     const uint64_t seqlen_ro,
//     const uint64_t stride_out_batch,
//     const uint64_t stride_out_seqlen,
//     const uint64_t stride_out_nheads,
//     const uint64_t stride_out_headdim,
//     const uint64_t block_k,
//     const uint64_t stride_x_batch,
//     const uint64_t stride_x_seqlen,
//     const uint64_t stride_x_nheads,
//     const uint64_t stride_x_headdim,
//     const uint64_t block_m
// );

#ifdef __cplusplus
}
#endif

#endif // SCAN_CUDA_H