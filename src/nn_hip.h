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

#ifdef __cplusplus
}
#endif

#endif // NN_HIP_H