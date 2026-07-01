#include <openfish/openfish.h>
#include "error.h"
#include "cuda_utils.h"
#include "nn_kernel_cuda.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>
#include <stdlib.h>

void openfish_rmsnorm_quant_int8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);
    ASSERT(hidden_dim % 2 == 0);  // kernel is half2-vectorized: one thread per adjacent pair

    int threads = hidden_dim / 2;
    int blocks = n_tokens;
    
    rmsnorm_quant_int8<<<blocks, threads>>>(
        (half *)in, (half *)weight, (int8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void openfish_rmsnorm_quant_fp8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    (void)in; (void)weight; (void)residual; (void)residual_scale;
    (void)n_tokens; (void)hidden_dim; (void)alpha; (void)eps;
    OPENFISH_ERROR("%s", "fp8 fused rmsnorm not implemented for CUDA");
    exit(EXIT_FAILURE);
}

void openfish_rmsnorm_gpu(
    const void* in,
    const void* residual,
    const void* weight,
    void* out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);
    ASSERT(hidden_dim % 2 == 0);  // kernel is half2-vectorized: one thread per adjacent pair

    int threads = hidden_dim / 2;
    int blocks = n_tokens;

    rmsnorm<<<blocks, threads>>>(
        (half *)in, (half *)residual, (half *)weight, (half *)out, n_tokens, hidden_dim, alpha, eps
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void openfish_silu_mul_gpu(
    const void *in,
    void *out,
    int n_tokens,
    int hidden_dim
) {
    int threads = 1024;
    int blocks = n_tokens;

    silu_mul<<<blocks, threads>>>(
        (const half *)in,
        (half *)out,
        hidden_dim,
        n_tokens
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void openfish_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size, int hidden_dim
) {
    int threads = (hidden_dim < 1024) ? hidden_dim : 1024;
    flstm_step<<<batch_size, threads>>>(
        (const half*)scratch, (const half*)ih_t,
        (half*)cell, (half*)hh_next,
        4 * hidden_dim, hidden_dim
    );
    checkCudaError();
}

void openfish_rotary_emb_gpu(
    void *x,
    const void *sin_gpu,
    const void *cos_gpu,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int sincos_width,
    int stride_batch,
    int stride_seq,
    int stride_head
) {
    int thread_h = 32;
    dim3 block_size(sincos_width, thread_h, 1);
	dim3 grid_size(batch_size, n_heads, 1);

    rotary_emb<<<grid_size, block_size>>>(
        (half *)x,
        (const float *)cos_gpu,
        (const float *)sin_gpu,
        seq_len,
        stride_batch,
        stride_seq,
        stride_head,
        sincos_width
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}
