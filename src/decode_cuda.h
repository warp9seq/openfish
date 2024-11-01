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

#ifdef __cplusplus
}
#endif

#endif // DECODE_CUDA_H