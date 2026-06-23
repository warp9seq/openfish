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

#ifndef NN_KERNEL_CUDA_H
#define NN_KERNEL_CUDA_H

#include <math.h>
#include <float.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static __global__ void rotary_emb(
	half *x,
    const float *_cos,
    const float *_sin,
    const uint64_t seq_len,
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

    if (tid >= seq_len) return;

    half *_o0 = x + (batch * stride_batch) + (head * stride_head) + rot;
    half *_o1 = x + (batch * stride_batch) + (head * stride_head) + rotary_half + rot;

    for (int seq = tid; seq < seq_len; seq += nthreads) {
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

static __global__ void silu_mul(
	const half *x_gpu,
	half *o_gpu,
    const uint64_t hidden_dim,
    const uint64_t n_tokens
) {
    uint64_t j = blockIdx.x;

    for (uint64_t k = threadIdx.x; k < hidden_dim; k += blockDim.x) {
        uint64_t i = k + j * (hidden_dim * 2);

        half y = x_gpu[i];
        half gate = x_gpu[i + hidden_dim];

        float g = __half2float(gate);
        float silu = g / (1.0f + __expf(-g));

        o_gpu[k + j * hidden_dim] = __float2half(silu * __half2float(y));
    }
}

static __global__ void rmsnorm(
    const half* input,
    const half* residual,
    const half* weight,
    half* output,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    int row = blockIdx.x;  // Which sequence/batch element
    
    if (row >= n_tokens) return;
    
    const half* x = input + row * hidden_dim;
    const half* res = residual + row * hidden_dim;
    half* y = output + row * hidden_dim;
    
    // Step 1: Compute sum of squares using shared memory reduction
    __shared__ float shared_sum[32];  // For warp reduction
    
    float thread_sum = 0.0f;
    float x_new; // if this for loop happens more than once it will break, in this case we need to cache more than one x
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]) + (__half2float(res[i]) * alpha);
        x_new = val;
        thread_sum += val * val;
    }
    
    // Warp-level reduction
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    
    // Reduce within warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    float sum_sq = 0.0f;
    if (threadIdx.x < warpSize) {
        int num_warps = (blockDim.x + (warpSize-1)) / warpSize;
        sum_sq = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
        
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    // Broadcast RMS to all threads
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / hidden_dim;
        rms_shared = rsqrtf(mean_sq + eps);  // 1 / sqrt(mean_sq + eps)
    }
    __syncthreads();
    
    float rms_inv = rms_shared;
    
    // Step 2: Normalize and apply weight
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float w = __half2float(weight[i]);
        y[i] = __float2half(x_new * rms_inv * w);
    }
}

static __global__ void rmsnorm_quant(
    const half* input,
    const half* weight,
    int8_t* residual,
    float* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    int row = blockIdx.x;  // Which sequence/batch element
    int idx = threadIdx.x;
    
    if (row >= n_tokens) return;
    
    const half* inp = input + row * hidden_dim;
    int8_t* res = residual + row * hidden_dim;
    float* res_scale = residual_scale + row;
    float w = __half2float(weight[idx]);
    
    // Step 1: Compute sum of squares using shared memory reduction
    __shared__ float shared_sum[32];  // For warp reduction
    
    float thread_sum = 0.0f;
    float val = __half2float(inp[idx]) + (((float)res[idx] * (*res_scale)) * alpha);
    thread_sum += val * val;
    
    // Warp-level reduction
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    
    // Reduce within warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    float sum_sq = 0.0f;
    if (threadIdx.x < warpSize) {
        int num_warps = (blockDim.x + (warpSize-1)) / warpSize;
        sum_sq = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
        
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    // Broadcast RMS to all threads
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / hidden_dim;
        rms_shared = rsqrtf(mean_sq + eps);  // 1 / sqrt(mean_sq + eps)
    }
    __syncthreads();
    
    float rms_inv = rms_shared;

    // Step 2: Find max absolute value for output quantization
    __shared__ float shared_max[32];
    
    float thread_max = 0.0f;
    float normalized = val * rms_inv * w;
    thread_max = fmaxf(thread_max, fabsf(normalized));
    
    // Reduce to find max
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    float abs_max = 0.0f;
    if (threadIdx.x < warpSize) {
        int num_warps = (blockDim.x + (warpSize-1)) / warpSize;
        abs_max = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : 0.0f;
        
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            abs_max = fmaxf(abs_max, __shfl_down_sync(0xffffffff, abs_max, offset));
        }
    }
    
    // write to quant scale
    __shared__ float quant_scale_shared;
    if (threadIdx.x == 0) {
        quant_scale_shared = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
        *res_scale = 1.0f / quant_scale_shared;
    }
    __syncthreads();
    
    
    // clamp and write quantized norm
    float quant_scale = quant_scale_shared;
    int quantized = __float2int_rn(normalized * quant_scale);
    quantized = max(-127, min(127, quantized));
    res[idx] = (int8_t)quantized;
}

// FLSTM epilogue: fuse bias-add, gate activations, cell update, and hidden state output.
// scratch: (N, 4*C) — result of addmm(up_bias_hh_, hh[t], W_hh_fused) (bias already included)
// ih_t:    (N, 4*C) — precomputed input-hidden contribution for this timestep
// c:       (N, C)   — cell state, updated in-place
// hh_next: (N, C)   — output hidden state h[t+1]
// Launch: flstm_step<<<N, min(C, 1024)>>>(...)
static __global__ void flstm_step(
    const half* scratch,
    const half* ih_t,
    half* c,
    half* hh_next,
    int C4, int C
) {
    int n = blockIdx.x;
    for (int ch = threadIdx.x; ch < C; ch += blockDim.x) {
        int base = n * C4 + ch;
        float gi = __half2float(scratch[base + 0*C]) + __half2float(ih_t[base + 0*C]);
        float gf = __half2float(scratch[base + 1*C]) + __half2float(ih_t[base + 1*C]);
        float gg = __half2float(scratch[base + 2*C]) + __half2float(ih_t[base + 2*C]);
        float go = __half2float(scratch[base + 3*C]) + __half2float(ih_t[base + 3*C]);
        float i_g = fmaxf(0.f, fminf(1.f, gi * 0.2f + 0.5f));
        float f_g = fmaxf(0.f, fminf(1.f, gf * 0.2f + 0.5f));
        float g_g = fmaxf(-1.f, fminf(1.f, gg));
        float o_g = fmaxf(0.f, fminf(1.f, go * 0.2f + 0.5f));
        float c_new = f_g * __half2float(c[n * C + ch]) + i_g * g_g;
        c[n * C + ch]       = __float2half(c_new);
        hh_next[n * C + ch] = __float2half(o_g * tanhf(c_new));
    }
}

#ifdef __cplusplus
}
#endif

#endif // NN_KERNEL_CUDA_H