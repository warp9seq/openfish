#pragma once

#include <stdint.h>
#include "decode.h"

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