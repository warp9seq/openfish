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
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;

    rmsnorm_quant<<<blocks, threads>>>(
        (half *)input, (half *)weight, (int8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
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
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;

    rmsnorm_quant_fp8<<<blocks, threads>>>(
        (half *)input, (half *)weight, (uint8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void openfish_quant_fp8_gpu(
    const void* x,
    void*       x_fp8,
    void*       scale,
    int         n_tokens,
    int         hidden_dim
) {
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    quant_fp8<<<n_tokens, hidden_dim>>>(
        (const half *)x, (uint8_t *)x_fp8, (float *)scale, n_tokens, hidden_dim
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void openfish_dequant_fp8_transpose_gpu(
    const void* in,    /* fp8  [n_timesteps, batch_size, n_channels] */
    void*       out,   /* f16  [batch_size, n_timesteps, n_channels] */
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
) {
    hipError_t ret;
    ASSERT(n_channels <= 1024);

    dequant_fp8_transpose<<<n_timesteps * batch_size, n_channels>>>(
        (const uint8_t *)in, (half *)out, n_timesteps, batch_size, n_channels, scale
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
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;

    rmsnorm<<<blocks, threads>>>(
        (half *)input, (half *)residual, (half *)weight, (half *)output, n_tokens, hidden_dim, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void openfish_silu_mul_gpu(
    const void *x_gpu,
    void *o_gpu,
    uint64_t n_tokens,
    uint64_t hidden_dim
) {
    hipError_t ret;

    int threads = 1024;
    int blocks = (int)n_tokens;

    silu_mul<<<blocks, threads>>>(
        (const half *)x_gpu,
        (half *)o_gpu,
        hidden_dim,
        n_tokens
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
    int threads = (C < 1024) ? C : 1024;
    flstm_step<<<N, threads>>>(
        (const half*)scratch, (const half*)ih_t,
        (half*)c, (half*)hh_next,
        4 * C, C
    );
    checkHipError();
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
    hipError_t ret;

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
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}
