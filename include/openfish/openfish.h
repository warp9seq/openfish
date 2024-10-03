#pragma once

#include "decode_cpu.h"

#ifdef HAVE_CUDA
#include "decode_cuda.cuh"
#endif

void decode(
    const int T,
    const int N,
    const int C,
    const int target_threads,
    float *scores_TNC,
    const int state_len,
    const DecoderOptions *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);