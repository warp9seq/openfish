#include "nn_cuda.h"
#include "error.h"
#include "cuda_utils.cuh"
#include "nn_kernel_cuda.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

void rmsnorm_quant_cuda(
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

void rmsnorm_cuda(
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

void silu_mul_cuda(
    void *x_gpu,
    void *o_gpu,
    int MN,
    int K
) {
    int threads = 1024;
    int blocks = MN;

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

void rotary_emb_cuda(
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


void flstm_step_cuda(
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