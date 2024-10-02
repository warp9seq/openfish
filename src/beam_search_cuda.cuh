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
    const int32_t *states,
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

__global__ void beam_search_cuda(
    const float *const scores,
    size_t scores_block_stride,
    const float *const back_guide,
    const float *const posts,
    const int num_state_bits,
    const size_t num_ts,
    const float beam_cut,
    const float fixed_stay_score,
    int32_t *states,
    uint8_t *moves,
    float *qual_data,
    float score_scale,
    float posts_scale,
    beam_element_t *beam_vector
);