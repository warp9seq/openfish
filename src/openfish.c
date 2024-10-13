#include <openfish/openfish.h>

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
) {
#ifdef HAVE_CUDA
    decode_cuda(T, N, C, scores_TNC, state_len, options, moves, sequence, qstring);
#elif HAVE_HIP
    decode_hip(T, N, C, scores_TNC, state_len, options, moves, sequence, qstring);
#else
    decode_cpu(T, N, C, target_threads, scores_TNC, state_len, options, moves, sequence, qstring);
#endif
}