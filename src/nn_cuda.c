#include <openfish/openfish.h>
#include "error.h"
#include "cuda_utils.h"
#include "nn_kernel_cuda.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>
#include <stdlib.h>

void openfish_rmsnorm_quant_gpu(
    const void* input,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;
    size_t shared_mem_bytes = static_cast<size_t>(threads) * 2 * sizeof(float);

    rmsnorm_quant<<<blocks, threads, shared_mem_bytes>>>(
        (half *)input, (half *)weight, (int8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void openfish_rmsnorm_quant_fp8_gpu(
    const void* input,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    (void)input; (void)weight; (void)residual; (void)residual_scale;
    (void)n_tokens; (void)hidden_dim; (void)alpha; (void)eps;
    OPENFISH_ERROR("%s", "fp8 fused rmsnorm not implemented for CUDA");
    exit(EXIT_FAILURE);
}

void openfish_rmsnorm_gpu(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;
    size_t shared_mem_bytes = static_cast<size_t>(threads) * sizeof(float);

    rmsnorm<<<blocks, threads, shared_mem_bytes>>>(
        (half *)input, (half *)residual, (half *)weight, (half *)output, n_tokens, hidden_dim, alpha, eps
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void openfish_silu_mul_gpu(
    const void *x_gpu,
    void *o_gpu,
    uint64_t n_tokens,
    uint64_t hidden_dim
) {
    int threads = 1024;
    int blocks = (int)n_tokens;

    silu_mul<<<blocks, threads>>>(
        (const half *)x_gpu,
        (half *)o_gpu,
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
    void* c,
    void* hh_next,
    int N, int C
) {
    int threads = (C < 1024) ? C : 1024;
    flstm_step<<<N, threads>>>(
        (const half*)scratch, (const half*)ih_t,
        (half*)c, (half*)hh_next,
        4 * C, C
    );
    checkCudaError();
}

void openfish_rotary_emb_gpu(
    void *x_gpu,
    const void *sin_gpu,
    const void *cos_gpu,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head
) {
    int thread_h = 32;
    dim3 block_size(rotary_half, thread_h, 1);
	dim3 grid_size(batch_size, n_heads, 1);

    rotary_emb<<<grid_size, block_size>>>(
        (half *)x_gpu,
        (const float *)cos_gpu,
        (const float *)sin_gpu,
        seq_len,
        stride_batch,
        stride_seq,
        stride_head,
        rotary_half
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}
