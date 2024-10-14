#ifndef BEAMSEARCH_HIP_H
#define BEAMSEARCH_HIP_H

#include "decode.h"

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct beam_element {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
} beam_element_t;

typedef struct beam_args {
    float *scores_TNC;
    float *bwd_NTC;
    float *post_NTC;
    size_t T;
    size_t N;
    size_t C;
    int num_state_bits;
} beam_args_t;

__global__ void generate_sequence(
    const beam_args_t args,
    const uint8_t *_moves,
    const state_t *_states,
    const float *_qual_data,
    float *_base_probs,
    float *_total_probs,
    char *_sequence,
    char *_qstring,
    const float shift,
    const float scale
);

__global__ void beam_search(
    const beam_args_t beam_args,
    state_t *_states,
    uint8_t *_moves,
    beam_element_t *_beam_vector,
    const float beam_cut,
    const float fixed_stay_score,
    const float score_scale
);

__global__ void compute_qual_data(
    const beam_args_t beam_args,
    state_t *_states,
    float *_qual_data,
    const float posts_scale
);

#ifdef __cplusplus
}
#endif

#endif // BEAMSEARCH_HIP_H