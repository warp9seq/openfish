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

// Emission-score element type of the input scores tensor.
//   OPENFISH_SCORE_F16 - native float scores (fp16 on the GPU path, fp32 on the CPU path);
//                        use score_scale = 1.0f for the unquantized pipeline.
//   OPENFISH_SCORE_I8  - int8 quantized scores in [-127, 127] the raw value is
//                        dequantized on read as (float)s * score_scale (e.g. 5.0f/127.0f).
typedef enum {
    OPENFISH_SCORE_F16 = 0,
    OPENFISH_SCORE_I8  = 1
} openfish_score_dtype_t;

openfish_opt_t openfish_decoder_default_opts(void);

void openfish_decode_cpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    int n_threads,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

// CPU beam search over PRE-COMPUTED posteriors. Runs only the beam_search + sequence-generation
// half of the decoder (openfish_decode_cpu does the whole thing); the forward/backward scan must
// already have filled gpubuf->bwd_NTC / gpubuf->post_NTC (e.g. via openfish_decode_gpu_scan on a
// unified-memory GPU, so the CPU reads them with no copy). scores_NTC is the same raw score buffer
// the scan consumed (int8 or fp16); the beam reads it sparsely.
void openfish_decode_cpu_beam(
    int n_timesteps,
    int batch_size,
    int n_channels,
    int n_threads,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
);

#if defined(HAVE_CUDA) || defined(HAVE_ROCM) || defined(HAVE_METAL)

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

// GPU forward/backward scan only: fills gpubuf->bwd_NTC and gpubuf->post_NTC from scores_NTC and
// returns (no beam search, no host result buffers). Pair with openfish_decode_cpu_beam.
void openfish_decode_gpu_scan(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf
);

openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
);

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
);

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM) || defined(HAVE_METAL)

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H