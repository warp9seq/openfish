#ifndef BEAMSEARCH_CPU_H
#define BEAMSEARCH_CPU_H

#include "decode.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void openfish_generate_sequence_cpu(
    const uint8_t *moves,
    const state_t *states,
    const float *qual_data,
    float shift,
    float scale,
    size_t n_timesteps,
    size_t seq_len,
    float *base_probs,
    float *total_probs,
    char *sequence,
    char *qstring
);

void openfish_beam_search_cpu(
    const float *scores,
    size_t scores_block_stride,
    const float *back_guide,
    const float *posts,
    int num_state_bits,
    size_t n_timesteps,
    float beam_cut,
    float fixed_stay_score,
    state_t *states,
    uint8_t *moves,
    float *qual_data,
    float score_scale,
    float posts_scale,
    beam_element_t *beam_vector
);

#ifdef __cplusplus
}
#endif

#endif // BEAMSEARCH_CPU_H