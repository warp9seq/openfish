#include <openfish/openfish.h>

#include <openfish/openfish_error.h>

#ifdef HAVE_CUDA
#include "decode_cuda.h"
#endif

#ifdef HAVE_HIP
#include "decode_hip.h"
#endif

openfish_gpubuf_t *openfish_gpubuf_init(
    const int T,
    const int N,
    const int state_len
) {
#ifdef HAVE_CUDA
    return gpubuf_init_cuda(T, N, state_len);
#elif HAVE_HIP
    return gpubuf_init_hip(T, N, state_len);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
) {
#ifdef HAVE_CUDA
    gpubuf_free_cuda(gpubuf);
#elif HAVE_HIP
    gpubuf_free_hip(gpubuf);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_decode_gpu(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
#ifdef HAVE_CUDA
    decode_cuda(T, N, C, scores_TNC, state_len, options, gpubuf, moves, sequence, qstring);
#elif HAVE_HIP
    decode_hip(T, N, C, scores_TNC, state_len, options, gpubuf, moves, sequence, qstring);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}