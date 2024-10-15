#ifndef DECODE_CUDA_H
#define DECODE_CUDA_H

#include <openfish/openfish.h>
#include <stdint.h>
#include "decode.h"

#ifdef __cplusplus
extern "C" {
#endif

void decode_cuda(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const decoder_opts_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

#ifdef __cplusplus
}
#endif

#endif // DECODE_CUDA_H