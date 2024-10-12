#ifndef OPENFISH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "decode_cpu.h"

#ifdef HAVE_CUDA
#include "decode_cuda.h"
#elif HAVE_HIP
#include "decode_hip.h"
#endif

void decode(
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

#endif // OPENFISH_H