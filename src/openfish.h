#pragma once

#include "decode_cpu.h"

#ifdef HAVE_CUDA
#include "decode_gpu.cuh"
#endif

void decode(
    const int T,
    const int N,
    const int C,
    const int target_threads,
    float *scores_TNC,
    std::vector<DecodedChunk>& chunk_results,
    const int state_len,
    const DecoderOptions *options
);