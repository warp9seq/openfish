#pragma once

#include <string>
#include <vector>

#define DTYPE_CPU float
#define DTYPE_GPU float

struct DecodedChunk {
    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;
};

struct DecoderOptions {
    size_t beam_width = 32;
    float beam_cut = 100.0;
    float blank_score = 2.0;
    float q_shift = 0.0;
    float q_scale = 1.0;
    float temperature = 1.0;
    bool move_pad = false;
};
