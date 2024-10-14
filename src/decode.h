#ifndef DECODE_H
#define DECODE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t state_t;

typedef struct beam_element {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
} beam_element_t;

typedef struct beam_front_element {
    uint32_t hash;
    state_t state;
    uint8_t prev_element_index;
    bool stay;
} beam_front_element_t;

typedef struct beam_args {
    float *scores_TNC;
    float *bwd_NTC;
    float *post_NTC;
    size_t T;
    size_t N;
    size_t C;
    int num_state_bits;
} beam_args_t;

typedef struct scan_args {
    float *scores_in;
    uint64_t num_states;
    uint64_t T;
    uint64_t N;
    uint64_t C;
    float fixed_stay_score;
} scan_args_t;

#define NUM_BASE_BITS (2)
#define NUM_BASES (4)
#define NUM_TRANSITIONS (NUM_BASES + 1)
#define MAX_BEAM_WIDTH (32)
#define HASH_PRESENT_BITS (4096)
#define HASH_PRESENT_MASK (HASH_PRESENT_BITS - 1)
#define MAX_STATES (1024)
#define MAX_BEAM_CANDIDATES (NUM_TRANSITIONS * MAX_BEAM_WIDTH)
#define CRC_SEED (0x12345678u)

#ifdef __cplusplus
}
#endif

#endif // DECODE_H