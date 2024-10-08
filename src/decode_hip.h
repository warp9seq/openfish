#pragma once

#include "decode.h"

void decode_hip(
    const int T,
    const int N,
    const int C,
    float *scores_TNC,
    const int state_len,
    const DecoderOptions *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);