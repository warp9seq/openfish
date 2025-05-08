#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "openfish_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECODER_INIT {32, 100.0, 2.0, 0.0, 1.0, 1.0, false}

typedef struct openfish_gpubuf {
    float *bwd_NTC;
    float *post_NTC;
    uint8_t *moves;
    char *sequence;
    char *qstring;
    void *beam_vector;
    void *states;
    float *qual_data;
    float *base_probs;
    float *total_probs;
} openfish_gpubuf_t;

typedef struct openfish_opt {
    size_t beam_width;
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
    float temperature;
    bool move_pad;
} openfish_opt_t;

void openfish_decode_cpu(
    const int T,
    const int N,
    const int C,
    int nthreads,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

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
);

openfish_gpubuf_t *openfish_gpubuf_init(
    const int T,
    const int N,
    const int state_len
);

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
);

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
);

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
);

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
);

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H