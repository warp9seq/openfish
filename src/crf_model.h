#pragma once

#include <string>

struct CRFModelConfig {
    float qscale;
    float qbias;
    int conv;
    int insize;
    int stride;
    bool bias;
    bool clamp;
    // If there is a decomposition of the linear layer, this is the bottleneck feature size.
    bool decomposition;
    int out_features;
    int state_len;
    // Output feature size of the linear layer.  Dictated by state_len and whether
    // blank scores are explicitly stored in the linear layer output.
    int outsize;
    float blank_score;
    float scale;
    int num_features;
};

CRFModelConfig load_crf_model_config(const std::string& path);