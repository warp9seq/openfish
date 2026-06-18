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
    int MN,
    int K,
    float alpha,
    float eps
) {
    ASSERT(K <= 1024);

    int threads = K;
    int blocks = MN;
    size_t shared_mem_bytes = static_cast<size_t>(threads) * 2 * sizeof(float);

    rmsnorm_quant<<<blocks, threads, shared_mem_bytes>>>(
        (half *)input, (half *)weight, (int8_t *)residual, (float *)residual_scale, MN, K, alpha, eps
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
    int MN,
    int K,
    float alpha,
    float eps
) {
    (void)input; (void)weight; (void)residual; (void)residual_scale;
    (void)MN; (void)K; (void)alpha; (void)eps;
    OPENFISH_ERROR("%s", "fp8 fused rmsnorm not implemented for CUDA");
    exit(EXIT_FAILURE);
}

void openfish_rmsnorm_gpu(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
) {
    ASSERT(K <= 1024);

    int threads = K;
    int blocks = MN;
    size_t shared_mem_bytes = static_cast<size_t>(threads) * sizeof(float);

    rmsnorm<<<blocks, threads, shared_mem_bytes>>>(
        (half *)input, (half *)residual, (half *)weight, (half *)output, MN, K, alpha, eps
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void openfish_silu_mul_gpu(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
) {
    int threads = 1024;
    int blocks = (int)MN;

    silu_mul<<<blocks, threads>>>(
        (half *)x_gpu,
        (half *)o_gpu,
        K,
        MN
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
    void *sin_gpu,
    void *cos_gpu,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head
) {
    int thread_h = 32;
    dim3 block_size(rotary_half, thread_h, 1);
	dim3 grid_size(batch_size, nheads, 1);

    rotary_emb<<<grid_size, block_size>>>(
        (half *)x_gpu,
        (float *)cos_gpu,
        (float *)sin_gpu,
        seqlen,
        stride_batch,
        stride_seq,
        stride_head,
        rotary_half
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}
