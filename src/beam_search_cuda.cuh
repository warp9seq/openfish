#pragma once

#include <cstddef>
#include <cstdint>

// 16 bit state supports 7-mers with 4 bases.
typedef int16_t state_t;

typedef struct {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
} beam_element_t;

__global__ void generate_sequence_cuda(
    const uint8_t *moves,
    const state_t *states,
    const float *qual_data,
    float *base_probs,
    float *total_probs,
    char *sequence,
    char *qstring,
    const float shift,
    const float scale,
    const size_t seq_len,
    const size_t T,
    const size_t N
);

__global__ void beam_search_cuda(
    const float *const _scores_TNC,
    const float *const _bwd_NTC,
    const float *const _post_NTC,
    state_t *_states,
    uint8_t *_moves,
    float *_qual_data,
    beam_element_t *_beam_vector,
    const int num_state_bits,
    const float beam_cut,
    const float fixed_stay_score,
    const float score_scale,
    const float posts_scale,
    const uint64_t T,
    const uint64_t N,
    const uint64_t C
);