#include <openfish/openfish.h>
#include "error.h"
#include "nn_kernel_hip.h"
#include "hip_utils.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
    hipError_t ret;
    ASSERT(K <= 1024);

    int threads = K;
    int blocks = MN;

    rmsnorm_quant_fp8<<<blocks, threads>>>(
        (half *)input, (half *)weight, (uint8_t *)residual, (float *)residual_scale, MN, K, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
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

void openfish_silu_mul_gpu(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
) {
    hipError_t ret;

    int threads = 1024;
    int blocks = (int)MN;

    silu_mul<<<blocks, threads>>>(
        (half *)x_gpu,
        (half *)o_gpu,
        K,
        MN
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void openfish_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* c,
    void* hh_next,
    int N, int C
) {
    hipError_t ret;
    int threads = (C < 1024) ? C : 1024;
    flstm_step<<<N, threads>>>(
        (const half*)scratch, (const half*)ih_t,
        (half*)c, (half*)hh_next,
        4 * C, C
    );
    checkHipError(); HIP_CHECK(ret);
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
