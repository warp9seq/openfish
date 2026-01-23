#include "nn_cuda.h"
#include "error.h"
#include "cuda_utils.cuh"
#include "nn_cuda.cuh"

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
    
    rmsnorm_quant<<<blocks, threads>>>(
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
    
    rmsnorm<<<blocks, threads>>>(
        (half *)input, (half *)residual, (half *)weight, (half *)output, MN, K, alpha, eps
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

void silu_mul_cuda(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
) {
    dim3 block(32, 32);
	dim3 grid(
        (K + block.x - 1) / block.x,
        (MN + block.y - 1) / block.y
    );

    silu_mul<<<grid, block>>>(
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