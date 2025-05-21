#ifndef ROTARY_EMB_H
#define ROTARY_EMB_H

#include <math.h>
#include <float.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void rotary_emb(
	half *x,
    half *o,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seqlen,
    const uint64_t stride_c,
    const uint64_t stride_head,
    const uint64_t stride_head_dim,
    const uint64_t stride_rotary
) {
    const uint64_t batch = blockIdx.x;
    const uint64_t head = blockIdx.y;
    const uint64_t rot = blockIdx.z;
    const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const uint64_t nthreads = blockDim.x * blockDim.y;
    const uint64_t q = 0;
    const uint64_t k = 1;

    if (tid >= seqlen) return;

    for (int seq = tid; seq < seqlen; seq += nthreads) {
        float x0 = __half2float(*(x + (batch * stride_batch) + (seq * stride_seqlen) + (q * stride_c) + (head * stride_head) + (rot * stride_head_dim)));
        float x1 = __half2float(*(x + (batch * stride_batch) + (seq * stride_seqlen) + (q * stride_c) + (head * stride_head) + stride_rotary + (rot * stride_head_dim)));

        half *o0 = o + (batch * stride_batch) + (seq * stride_seqlen) + (q * stride_c) + (head * stride_head) + (rot * stride_head_dim);
        half *o1 = o + (batch * stride_batch) + (seq * stride_seqlen) + (q * stride_c) + (head * stride_head) + stride_rotary + (rot * stride_head_dim);

        float cos = *(_cos + (seq * stride_rotary) + rot);
        float sin = *(_sin + (seq * stride_rotary) + rot);

        *o0 = __float2half(x0 * cos - x1 * sin);
        *o1 = __float2half(x0 * sin + x1 * cos);
    }

    for (int seq = tid; seq < seqlen; seq += nthreads) {
        float x0 = __half2float(*(x + (batch * stride_batch) + (seq * stride_seqlen) + (k * stride_c) + (head * stride_head) + (rot * stride_head_dim)));
        float x1 = __half2float(*(x + (batch * stride_batch) + (seq * stride_seqlen) + (k * stride_c) + (head * stride_head) + stride_rotary + (rot * stride_head_dim)));

        half *o0 = o + (batch * stride_batch) + (seq * stride_seqlen) + (k * stride_c) + (head * stride_head) + (rot * stride_head_dim);
        half *o1 = o + (batch * stride_batch) + (seq * stride_seqlen) + (k * stride_c) + (head * stride_head) + stride_rotary + (rot * stride_head_dim);

        float cos = *(_cos + (seq * stride_rotary) + rot);
        float sin = *(_sin + (seq * stride_rotary) + rot);

        *o0 = __float2half(x0 * cos - x1 * sin);
        *o1 = __float2half(x0 * sin + x1 * cos);
    }
}

#ifdef __cplusplus
}
#endif

#endif // ROTARY_EMB_H