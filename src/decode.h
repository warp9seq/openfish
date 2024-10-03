#pragma once

#include <string>
#include <vector>
#include <cstdint>

#define DTYPE_CPU float
#define DTYPE_GPU float

constexpr int NUM_BASE_BITS = 2;
constexpr int NUM_BASES = 1 << NUM_BASE_BITS;
constexpr size_t MAX_BEAM_WIDTH = 32;

struct DecoderOptions {
    size_t beam_width = 32;
    float beam_cut = 100.0;
    float blank_score = 2.0;
    float q_shift = 0.0;
    float q_scale = 1.0;
    float temperature = 1.0;
    bool move_pad = false;
};
