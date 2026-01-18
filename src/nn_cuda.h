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

void dual_gemm_lhs_activation_and_mul_cuda(
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

#ifdef __cplusplus
}
#endif

#endif // NN_CUDA_H