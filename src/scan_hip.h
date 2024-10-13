#pragma once

#include "decode.h"

typedef struct scan_args {
    float *scores_in;
    uint64_t num_states;
    uint64_t T;
    uint64_t N;
    uint64_t C;
    float fixed_stay_score;
} scan_args_t;

__global__ void bwd_scan(
	const scan_args_t args,
	float *out
);

__global__ void fwd_post_scan(
    const scan_args_t args,
    const float *bwd,
    float *out
);