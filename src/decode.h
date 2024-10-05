#pragma once

#include <string>
#include <vector>
#include <cstdint>

#define DTYPE_CPU float
#define DTYPE_GPU float

constexpr int NUM_BASE_BITS = 2;
constexpr int NUM_BASES = 1 << NUM_BASE_BITS;
constexpr size_t MAX_BEAM_WIDTH = 32;

constexpr uint32_t HASH_PRESENT_BITS = 4096;
constexpr uint32_t HASH_PRESENT_MASK = HASH_PRESENT_BITS - 1;
constexpr uint32_t MAX_STATES = 1024;

constexpr uint32_t CRC_SEED = 0x12345678u;

struct DecoderOptions {
    size_t beam_width = 32;
    float beam_cut = 100.0;
    float blank_score = 2.0;
    float q_shift = 0.0;
    float q_scale = 1.0;
    float temperature = 1.0;
    bool move_pad = false;
};
