#ifndef DECODE_CPU_H
#define DECODE_CPU_H

#include <openfish/openfish.h>

#include <stdint.h>
#include "decode.h"

#ifdef __cplusplus
extern "C" {
#endif

void decode_cpu(
    const int T,
    const int N,
    const int C,
    const int target_threads,
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

#endif // DECODE_CPU_H