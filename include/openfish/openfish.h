#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>

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

void openfish_rotary_emb_cpu(
    void *x,
    void *sin_buf,
    void *cos_buf,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head,
    int nthreads
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
);

#ifdef HAVE_CUDA
class ForwardPass;
class FlashRNNFuncFused
{
private:
    std::unique_ptr<ForwardPass> fw;
    // BackwardPass bw;
    // BackwardPassCut bwc;

public:
    FlashRNNFuncFused(const bool training, const int batch_size, const int hidden_size, const int num_heads);
    
    void forward(
        bool training,
        void *x, // W_ih * x + b_ih
        void *s0,
        void *recurrent_kernel, // W_hh
        void *bias, // b_hh
        void *states,
        void *gate_cache_r,
        void *gate_cache_i,
        void *gate_buffer,
        int seqlen,
        int batch_size,
        int nheads,
        int head_dim,
        void *blas_handle
    );
};
#endif

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H