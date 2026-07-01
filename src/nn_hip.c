#include <openfish/openfish.h>
#include "error.h"
#include "nn_kernel_hip.h"
#include "hip_utils.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;

    rmsnorm_quant_int8<<<blocks, threads>>>(
        (half *)in, (half *)weight, (int8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
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
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    int threads = hidden_dim;
    int blocks = n_tokens;

    rmsnorm_quant_fp8<<<blocks, threads>>>(
        (half *)in, (half *)weight, (uint8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void openfish_quant_fp8_gpu(
    const void* in,
    void*       out,
    void*       scale,
    int         n_tokens,
    int         hidden_dim
) {
    hipError_t ret;
    ASSERT(hidden_dim <= 1024);

    quant_fp8<<<n_tokens, hidden_dim>>>(
        (const half *)in, (uint8_t *)out, (float *)scale, n_tokens, hidden_dim
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
    const void* in,
    const void* residual,
    const void* weight,
    void* out,
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
        (half *)in, (half *)residual, (half *)weight, (half *)out, n_tokens, hidden_dim, alpha, eps
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}

void openfish_silu_mul_gpu(
    const void *in,
    void *out,
    int n_tokens,
    int hidden_dim
) {
    hipError_t ret;

    int threads = 1024;
    int blocks = n_tokens;

    silu_mul<<<blocks, threads>>>(
        (const half *)in,
        (half *)out,
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
    checkHipError();
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
    hipError_t ret;

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
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
}
