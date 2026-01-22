#ifndef NN_CUDA_H
#define NN_CUDA_H

#include <openfish/openfish.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
);

void swiglu_cuda(
    void *x,
    void *w0,
    void *w1,
    void *d0,
    void *d1,
    void *d2, // result
    int64_t B,
    int64_t I,
    int64_t H
);

void silu_mul_cuda(
    void *x_gpu,
    void *o_gpu,
    uint64_t M,
    uint64_t K
);

void quant_gemm_cuda(
    void *a_quant,
    void *b_quant,
    void *a_scale,
    void *b_scale,
    void *o_gpu,
    int M,
    int N,
    int K
);

void rmsnorm_cuda(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
);

void rmsnorm_quant_cuda(
    const void* input,
    const void* weight,
    void* residual,
    void* residual_scale,
    int MN,
    int K,
    float alpha,
    float eps
);

#ifdef __cplusplus
}
#endif

#endif // NN_CUDA_H