#ifndef NN_HIP_H
#define NN_HIP_H

#include <openfish/openfish.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
);

void silu_mul_hip(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
);

void rmsnorm_hip(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
);

void rmsnorm_quant_hip(
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

#endif // NN_HIP_H