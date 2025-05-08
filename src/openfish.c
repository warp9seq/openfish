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
    void *qkv_gpu,
    void *o_gpu,
    int batch_size,
    int seqlen,
    int num_heads,
    int head_dim,
    int batch_stride,
    int row_stride,
    int head_stride,
    int win_upper,
    int win_lower
) {
    flash_fwd(
        qkv_gpu,
        o_gpu,
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        batch_stride,
        row_stride,
        head_stride,
        win_upper,
        win_lower
    );
}

void openfish_rotary_f16(
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
    rotary_f16_cuda(
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
}

void openfish_swiglu(
    void *x,
    void *w0,
    void *w1,
    void *d0,
    void *d1,
    void *d2,
    int64_t B,
    int64_t I,
    int64_t H
) {
    swiglu(
        x,
        w0,
        w1,
        d0,
        d1,
        d2,
        B,
        I,
        H
    );
}