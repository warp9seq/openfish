#ifndef DECODE_CUDA_H
#define DECODE_CUDA_H

#include <openfish/openfish.h>
#include <stdint.h>
#include "decode.h"

#ifdef __cplusplus
extern "C" {
#endif

openfish_gpubuf_t *gpubuf_init_cuda(
    const int T,
    const int N,
    const int state_len
);

void gpubuf_free_cuda(
    openfish_gpubuf_t *gpubuf
);

void decode_cuda(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

void set_device_cuda(
    int device
);

void *upload_scores_to_cuda(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
);

void free_scores_cuda(
    void *scores_TNC_gpu
);

void write_gpubuf_cuda(
    const uint64_t T,
    const uint64_t N,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
);

void flash_fwd(
    void *qkv_gpu,
    void *o_gpu,
    int batch_size,
    int seqlen,
    int num_heads,
    int head_dim,
    int batch_stride,
    int row_stride,
    int head_stride,
    int win_upper,
    int win_lower
);

void run_rotary(
    void *x0,
    void *x1,
    void **o0,
    void **o1,
    void *sin,
    void *cos
);

// void run_flash(
//     void *q,
//     void *k,
//     void *v,
//     void **o
// );

#ifdef __cplusplus
}
#endif

#endif // DECODE_CUDA_H