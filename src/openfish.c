#include <openfish/openfish.h>

#include <openfish/openfish_error.h>

#ifdef HAVE_CUDA
#include "decode_cuda.h"
#endif

#ifdef HAVE_HIP
#include "decode_hip.h"
#endif

void decode_gpu(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
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
    OPENFISH_ERROR("%s", "not compiled for gpu");
#endif
}