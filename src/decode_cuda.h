#ifndef DECODE_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "decode.h"

void decode_cuda(
    const int T,
    const int N,
    const int C,
    float *scores_TNC,
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