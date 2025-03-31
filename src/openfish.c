#include <openfish/openfish.h>

#include <openfish/openfish_error.h>

#ifdef HAVE_CUDA
#include "decode_cuda.h"
#endif

#ifdef HAVE_ROCM
#include "decode_hip.h"
#endif

openfish_gpubuf_t *openfish_gpubuf_init(
    const int T,
    const int N,
    const int state_len
) {
#ifdef HAVE_CUDA
    return gpubuf_init_cuda(T, N, state_len);
#elif HAVE_ROCM
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
#elif HAVE_ROCM
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
#elif HAVE_ROCM
    decode_hip(T, N, C, scores_TNC, state_len, options, gpubuf, moves, sequence, qstring);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_flash_fwd(
    void *q_gpu,
    void *k_gpu,
    void *v_gpu,
    void *o_gpu,
    int batch_size,
    int seqlen,
    int num_heads,
    int head_dim,
    int win_upper,
    int win_lower
) {
    openfish_flash_fwd(
        q_gpu,
        k_gpu,
        v_gpu,
        o_gpu,
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        win_upper,
        win_lower
    );
}