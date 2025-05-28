// The MIT License (MIT)

// Copyright (c) 2025 Bonson Wong

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROTARY_EMB_HIP_H
#define ROTARY_EMB_HIP_H

#include <math.h>
#include <float.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void rotary_emb(
	half *x,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seq,
    const uint64_t stride_head,
    const uint64_t rotary_half
) {
    const uint64_t batch = blockIdx.x;
    const uint64_t head = blockIdx.y;
    const uint64_t rot = threadIdx.x;
    const uint64_t tid = threadIdx.y;
    const uint64_t nthreads = blockDim.y;

    if (tid >= seqlen) return;

    half *_o0 = x + (batch * stride_batch) + (head * stride_head) + rot;
    half *_o1 = x + (batch * stride_batch) + (head * stride_head) + rotary_half + rot;

    for (int seq = tid; seq < seqlen; seq += nthreads) {
        float cos = *(_cos + (seq * rotary_half) + rot);
        float sin = *(_sin + (seq * rotary_half) + rot);

        half *o0 = _o0 + (seq * stride_seq);
        half *o1 = _o1 + (seq * stride_seq);

        float x0 = __half2float(*o0);
        float x1 = __half2float(*o1);

        *o0 = __float2half(x0 * cos - x1 * sin);
        *o1 = __float2half(x0 * sin + x1 * cos);
    }
}

#ifdef __cplusplus
}
#endif

#endif // ROTARY_EMB_HIP_H