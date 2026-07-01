#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "openfish_error.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
} openfish_opt_t;

openfish_opt_t openfish_decoder_default_opts(void);

void openfish_decode_cpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    int n_threads,
    const void *scores_TNC,
    int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

// Rotary position embedding (rotate-half convention).
//
// `sincos_width` is the width of the sin/cos tables: sin_buf and cos_buf MUST be
// laid out as [seq_len, sincos_width]. The kernel rotates 2*sincos_width elements
// of each head, pairing element i with element i+sincos_width:
//     out_i               = x_i*cos_j - x_{i+sincos_width}*sin_j
//     out_{i+sincos_width} = x_i*sin_j + x_{i+sincos_width}*cos_j   (j = i, table col)
// so it rotates 2*sincos_width dims (require 2*sincos_width <= head_dim).
//
// NOTE: fused GEMM+rotary kernels elsewhere may instead expect FULL-width tables
// [seq_len, 2*sincos_width] (each column duplicated). Do not confuse the two — the
// table's second dimension must always equal the width the consumer documents.
void openfish_rotary_emb_cpu(
    void *x,
    const void *sin_buf,
    const void *cos_buf,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int sincos_width,   // width of sin/cos tables: [seq_len, sincos_width]; rotates 2*sincos_width dims
    int stride_batch,
    int stride_seq,
    int stride_head,
    int n_threads
);

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
);

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_TNC,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
);

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
);

// See openfish_rotary_emb_cpu: sin_gpu/cos_gpu are [seq_len, sincos_width] (rotate-half),
// rotating 2*sincos_width dims per head (require 2*sincos_width <= head_dim).
void openfish_rotary_emb_gpu(
    void *x,
    const void *sin_gpu,
    const void *cos_gpu,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int sincos_width,   // width of sin/cos tables: [seq_len, sincos_width]; rotates 2*sincos_width dims
    int stride_batch,
    int stride_seq,
    int stride_head
);

void openfish_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size,
    int hidden_dim
);

void openfish_silu_mul_gpu(
    const void *in,
    void *out,
    int n_tokens,
    int hidden_dim
);

void openfish_rmsnorm_gpu(
    const void* in,
    const void* residual,
    const void* weight,
    void* out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

void openfish_rmsnorm_quant_int8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

void openfish_rmsnorm_quant_fp8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

void openfish_quant_fp8_gpu(
    const void* in,     /* f16  [n_tokens, hidden_dim] input  */
    void*       out,    /* uint8[n_tokens, hidden_dim] fp8 E4M3FN output */
    void*       scale,  /* f32  [n_tokens]             per-token scale output */
    int         n_tokens,
    int         hidden_dim
);

void openfish_dequant_fp8_transpose_gpu(
    const void* in,     /* fp8  [n_timesteps, batch_size, n_channels] in  */
    void*       out,    /* f16  [batch_size, n_timesteps, n_channels] out (dequant × scale, transposed) */
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
);

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H