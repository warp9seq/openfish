#pragma once

#include "decode.h"

typedef struct scan_args {
    DTYPE_GPU *scores_in;
    uint64_t num_states;
    uint64_t T;
    uint64_t N;
    uint64_t C;
    DTYPE_GPU fixed_stay_score;
} scan_args_t;

__global__ void bwd_scan(
	const scan_args_t args,
	DTYPE_GPU *out
);

__global__ void fwd_post_scan(
    const scan_args_t args,
    const DTYPE_GPU *bwd,
    DTYPE_GPU *out
);