#include "nn_hip.h"
#include "error.h"
#include "nn_kernel_hip.h"
#include "hip_utils.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

void rmsnorm_quant_hip(
    const void* input,
    const void* weight,
    void* residual,
    void* residual_scale,
    int MN,
    int K,
    float alpha,
    float eps
) {
    hipError_t ret;
    ASSERT(K <= 1024);
    
    int threads = K;
    int blocks = MN;
    
    rmsnorm_quant<<<blocks, threads>>>(
        (half *)input, (half *)weight, (int8_t *)residual, (float *)residual_scale, MN, K, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void rmsnorm_hip(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
) {
    hipError_t ret;
    ASSERT(K <= 1024);
    
    int threads = K;
    int blocks = MN;
    
    rmsnorm<<<blocks, threads>>>(
        (half *)input, (half *)residual, (half *)weight, (half *)output, MN, K, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void silu_mul_hip(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
) {
    hipError_t ret;

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
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void rotary_emb_hip(
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
    hipError_t ret;

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
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}