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

#ifdef __cplusplus
}
#endif

#endif // DECODE_CUDA_H