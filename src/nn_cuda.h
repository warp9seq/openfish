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

void silu_mul_cuda(
    void *x_gpu,
    void *o_gpu,
    int MN,
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

void flstm_step_cuda(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size, int hidden_dim
);

#ifdef __cplusplus
}
#endif

#endif // NN_CUDA_H