#include "scan_hip.h"
#include "hip_utils.hpp"

#include <math.h>
#include <float.h>

__global__ void bwd_scan(
	const scan_args_t args,
	DTYPE_GPU *out
) {
	const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
	const uint64_t nthreads = blockDim.x * blockDim.y;

    const DTYPE_GPU *scores_in = args.scores_in;
    const uint64_t num_states = args.num_states;
    const uint64_t T = args.T;
    const uint64_t N = args.N;

	if (chunk >= args.N || tid >= num_states) {
		return;
	}

    const uint64_t ntransitions = NUM_BASES + 1;
    const DTYPE_GPU fixed_stay_score = args.fixed_stay_score;

    const uint64_t ts_states = num_states * NUM_BASES;

    const DTYPE_GPU* const chunk_in = scores_in + chunk * ts_states;
    DTYPE_GPU* const chunk_out = out + chunk * (T+1) * num_states;
    DTYPE_GPU* const alpha_init = chunk_out + num_states * T;
    for (uint64_t state = tid; state < num_states; state += nthreads) {
        alpha_init[state] = 0.0f;
    }

    for (uint64_t ts = 0; ts < T; ++ts) {
        __syncthreads();
        const DTYPE_GPU* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        DTYPE_GPU* const ts_alpha_in = alpha_init - num_states * ts;
        DTYPE_GPU* const ts_alpha_out = ts_alpha_in - num_states;

        for (uint64_t state = tid; state < num_states; state += nthreads) {
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = (state * NUM_BASES) % num_states;
            const uint64_t step_trans_idx_a = step_state_idx_a * NUM_BASES +
                ((state * NUM_BASES) / num_states);

            DTYPE_GPU vals[ntransitions];
            vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
            DTYPE_GPU max_val = vals[0];
            for (uint64_t base = 0; base < NUM_BASES; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    ts_in[step_trans_idx_a + base * NUM_BASES];
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            DTYPE_GPU sum = 0.0f;
            for (uint64_t i = 0; i < ntransitions; ++i) {
                sum += __expf(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + __logf(sum);
        }
    }
}

__global__ void fwd_post_scan(
    const scan_args_t args,
    const DTYPE_GPU *bwd,
    DTYPE_GPU *out
) {
    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const uint64_t nthreads = blockDim.x * blockDim.y;

    const DTYPE_GPU *scores_in = args.scores_in;
    const uint64_t num_states = args.num_states;
    const uint64_t _T = args.T;
    const uint64_t T = args.T + 1;
    const uint64_t N = args.N;

	if (chunk >= N || tid >= num_states) {
		return;
	}

    constexpr uint64_t ntransitions = NUM_BASES + 1;
    const DTYPE_GPU fixed_stay_score = args.fixed_stay_score;
    
    const uint64_t msb = num_states / NUM_BASES;
    const uint64_t ts_states = num_states * NUM_BASES;

    __shared__ DTYPE_GPU fwd_vals[MAX_STATES];
    __shared__ DTYPE_GPU exp_vals[MAX_STATES];
    // __shared__ DTYPE_GPU exp_sum;
    __shared__ DTYPE_GPU max_val;
    max_val = -FLT_MAX;

    //scores for this batch
    const DTYPE_GPU* const chunk_scores = scores_in + chunk * ts_states;

    // alternating forward guide buffers used for successive time steps
    __shared__ DTYPE_GPU ts_fwd[2][MAX_STATES];

    // the forward guide input for the first step is 0
    for (uint64_t state = tid; state < num_states; state += nthreads) {
        ts_fwd[0][state] = 0.0f;
    }
    __syncthreads();

    for (uint64_t ts = 0; ts < T; ++ts) {
        // we read forward guide values written to TG memory in the previous step as inputs to this step
        // however, there has already been a TG barrier since they were written
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        // this time step's scores
        const DTYPE_GPU* const ts_scores = chunk_scores + N * ts_states * ts;

        // alternating TG buffer twiddling
        const DTYPE_GPU* const ts_alpha_in = ts_fwd[ts & 1];
        DTYPE_GPU* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // calculate the next time step's forward guide from this time step's scores and forward guide
        // it's written to threadgroup memory for use in the next iteration
        for (uint64_t state = tid; state < num_states; state += nthreads) {
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = state / NUM_BASES;
            const uint64_t step_trans_idx_a = state * NUM_BASES;
            DTYPE_GPU vals[ntransitions];
            DTYPE_GPU fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
            for (uint64_t base = 0; base < NUM_BASES; ++base) {
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                DTYPE_GPU ts_score = ts < _T ? ts_scores[step_trans_idx_a + base] : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * msb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            DTYPE_GPU fwd_sum = 0.0f;
            for (uint64_t i = 0; i < ntransitions; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + __logf(fwd_sum);

            // Load the forward guide value calculated in the last time step for use n this time step's posterior probability calculation
            const DTYPE_GPU fwd_val = ts_alpha_in[state];

            // calculate fwd/bwd guide product in log space
            const DTYPE_GPU val = fwd_val + bwd[ts_idx + state];

            fwd_vals[state] = val;
            atomicMaxFloat(&max_val, val);
        }
        __syncthreads();

        // enter exp vals
        for (uint64_t state = tid; state < num_states; state += nthreads) {
            exp_vals[state] = __expf(fwd_vals[state] - max_val);
            // atomicAdd(&exp_sum, exp_vals[state]); // this does not give us deterministic results, turn on for production
        }
        __syncthreads();

        // get max exp val
        DTYPE_GPU exp_sum = 0.0f;
        for (uint64_t state = 0; state < num_states; ++state) {
            exp_sum += exp_vals[state];
        }
        
        // calculate posterior probability
        for (uint64_t state = tid; state < num_states; state += nthreads) {
            out[ts_idx + state] = exp_vals[state] / exp_sum;
        }
        max_val = -FLT_MAX;
        __syncthreads();
    }
}