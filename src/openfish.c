#include <openfish/openfish.h>

#include <openfish/openfish_error.h>

#ifdef HAVE_CUDA
#include "decode_cuda.h"
#include "nn_cuda.h"
#endif

#ifdef HAVE_ROCM
#include "decode_hip.h"
#include "nn_hip.h"
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

void openfish_rotary_emb_gpu(
    void *x_gpu,
    void *sin_gpu,
    void *cos_gpu,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head
) {
#ifdef HAVE_CUDA
    rotary_emb_cuda(
        x_gpu,
        sin_gpu,
        cos_gpu,
        batch_size,
        seqlen,
        nheads,
        head_dim,
        rotary_half,
        stride_batch,
        stride_seq,
        stride_head
    );
#elif HAVE_ROCM
    rotary_emb_hip(
        x_gpu,
        sin_gpu,
        cos_gpu,
        batch_size,
        seqlen,
        nheads,
        head_dim,
        rotary_half,
        stride_batch,
        stride_seq,
        stride_head
    );
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}