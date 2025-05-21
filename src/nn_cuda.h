#ifndef NN_CUDA_H
#define NN_CUDA_H

#include <openfish/openfish.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void rotary_emb_cuda(
    void *x0_gpu,
    void *o0_gpu,
    void *sin_gpu,
    void *cos_gpu,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_dim,
    int stride_batch,
    int stride_seq,
    int stride_c,
    int stride_head,
    int stride_head_dim,
    int stride_rotary
);

#ifdef __cplusplus
}
#endif

#endif // NN_CUDA_H