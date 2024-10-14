#ifndef BEAMSEARCH_CPU_H
#define BEAMSEARCH_CPU_H

#include "decode.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void generate_sequence(
    const uint8_t *moves,
    const state_t *states,
    const float *qual_data,
    const float shift,
    const float scale,
    const size_t num_ts,
    const size_t seq_len,
    float *base_probs,
    float *total_probs,
    char *sequence,
    char *qstring
);

void beam_search(
    const float *const scores,
    size_t scores_block_stride,
    const float *const back_guide,
    const float *const posts,
    const int num_state_bits,
    const size_t num_ts,
    const float beam_cut,
    const float fixed_stay_score,
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