#pragma once
#include <stdbool.h>
#include <stdlib.h>

#define NUM_BASE_BITS (2)
#define NUM_BASES (1 << NUM_BASE_BITS)
#define NUM_TRANSITIONS (NUM_BASES + 1)
#define MAX_BEAM_WIDTH (32)
#define HASH_PRESENT_BITS (4096)
#define HASH_PRESENT_MASK (HASH_PRESENT_BITS - 1)
#define MAX_STATES (1024)
#define MAX_BEAM_CANDIDATES (NUM_TRANSITIONS * MAX_BEAM_WIDTH)
#define CRC_SEED (0x12345678u)

typedef struct decoder_opts {
    size_t beam_width;
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
    float temperature;
    bool move_pad;
} decoder_opts_t;

#define DECODER_INIT {32, 100.0, 2.0, 0.0, 1.0, 1.0, false}