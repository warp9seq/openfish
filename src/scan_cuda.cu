#include "scan_cuda.cuh"
#include "cuda_utils.cuh"

#include <math.h>
#include <float.h>

#define BLOCK_M_MAX (8)
#define BLOCK_K_MAX (256)

__global__ void bwd_scan(
	const scan_args_t args,
	float *out
) {
	const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const uint64_t state = tid;

    const half *scores_in = (half *)args.scores_in;
    const uint64_t num_states = args.num_states;
    const uint64_t T = args.T;
    const uint64_t N = args.N;

	if (chunk >= args.N || tid >= num_states) {
		return;
	}

    const float fixed_stay_score = args.fixed_stay_score;

    const uint64_t ts_states = num_states * NUM_BASES;

    const half *const chunk_in = scores_in + chunk * ts_states;
    float* const chunk_out = out + chunk * (T+1) * num_states;
    float* const alpha_init = chunk_out + num_states * T;
    alpha_init[state] = 0.0f;

    for (uint64_t ts = 0; ts < T; ++ts) {
        __syncthreads();
        const half *const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        float* const ts_alpha_in = alpha_init - num_states * ts;
        float* const ts_alpha_out = ts_alpha_in - num_states;

        const uint64_t stay_state_idx = state;
        const uint64_t step_state_idx_a = (state * NUM_BASES) % num_states;
        const uint64_t step_trans_idx_a = step_state_idx_a * NUM_BASES + ((state * NUM_BASES) / num_states);

        float vals[NUM_TRANSITIONS];
        vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
        float max_val = vals[0];
        for (uint64_t base = 0; base < NUM_BASES; ++base) {
            vals[base + 1] = ts_alpha_in[step_state_idx_a + base] + __half2float(ts_in[step_trans_idx_a + base * NUM_BASES]);
            max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
        }
        float sum = 0.0f;
        for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
            sum += __expf(vals[i] - max_val);
        }
        ts_alpha_out[state] = max_val + __logf(sum);
    }
}

__global__ void fwd_post_scan(
    const scan_args_t args,
    const float *bwd,
    float *out
) {
    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const uint64_t nthreads = blockDim.x * blockDim.y;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const unsigned mask = 0xFFFFFFFFU;
    (void)mask;
    const uint64_t state = tid;

    const half *scores_in = (half *)args.scores_in;
    const uint64_t num_states = args.num_states;
    const uint64_t _T = args.T;
    const uint64_t T = args.T + 1;
    const uint64_t N = args.N;

	if (chunk >= N || tid >= num_states) {
		return;
	}

    const float fixed_stay_score = args.fixed_stay_score;
    
    const uint64_t msb = num_states / NUM_BASES;
    const uint64_t ts_states = num_states * NUM_BASES;

    __shared__ float fwd_vals[MAX_STATES];
    __shared__ float fwd_maxs[32]; // threadblock max stored in [0]
    __shared__ float exp_vals[MAX_STATES];
    __shared__ float exp_sums[32]; // threadblock sum stored in [0]
    float warp_max;

    // scores for this batch
    const half *const chunk_scores = scores_in + chunk * ts_states;

    // alternating forward guide buffers used for successive time steps
    __shared__ float ts_fwd[2][MAX_STATES];

    // the forward guide input for the first step is 0
    for (uint64_t state = tid; state < num_states; state += nthreads) {
        ts_fwd[0][state] = 0.0f;
    }
    __syncthreads();

    for (uint64_t ts = 0; ts < T; ++ts) {
        warp_max = -FLT_MAX;
        // we read forward guide values written to TG memory in the previous step as inputs to this step
        // however, there has already been a TG barrier since they were written
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        // this time step's scores
        const half *const ts_scores = chunk_scores + N * ts_states * ts;

        // alternating TG buffer twiddling
        const float* const ts_alpha_in = ts_fwd[ts & 1];
        float* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // calculate the next time step's forward guide from this time step's scores and forward guide
        // it's written to threadgroup memory for use in the next iteration
        const uint64_t stay_state_idx = state;
        const uint64_t step_state_idx_a = state / NUM_BASES;
        const uint64_t step_trans_idx_a = state * NUM_BASES;
        float vals[NUM_TRANSITIONS];
        float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
        for (uint64_t base = 0; base < NUM_BASES; ++base) {
            // todo: this is a bandaid for indexing past the actual T dimension of scores
            // need to verify with actual MetalTxCaller impl output,
            // otherwise output remains exactly the same for this impl whether it indexes past or not
            float ts_score = ts < _T ? __half2float(ts_scores[step_trans_idx_a + base]) : 0.0f;

            vals[base + 1] = ts_alpha_in[step_state_idx_a + base * msb] + ts_score;
            fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
        }
        float fwd_sum = 0.0f;
        for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
            fwd_sum += __expf(vals[i] - fwd_max_val);
        }
        ts_alpha_out[state] = fwd_max_val + __logf(fwd_sum);

        // load the forward guide value calculated in the last time step for use n this time step's posterior probability calculation
        const float fwd_val = ts_alpha_in[state];

        // calculate fwd/bwd guide product in log space
        const float val = fwd_val + bwd[ts_idx + state];

        fwd_vals[state] = val;
        warp_max = max(warp_max, val);
        __syncthreads();

        // find max fwd val in warp
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            warp_max = max(warp_max, __shfl_down_sync(mask, warp_max, offset));
        }
        if (lane_id == 0) fwd_maxs[warp_id] = warp_max;
        __syncthreads();

        // set max fwd vals in all warps
        if (warp_id == 0) {
            warp_max = (tid < num_states/warpSize) ? fwd_maxs[lane_id] : 0;

            for (int offset = warpSize/2; offset > 0; offset >>= 1) {
                warp_max = max(warp_max, __shfl_down_sync(mask, warp_max, offset));
            }
            
            if (tid == 0) fwd_maxs[0] = warp_max;
        }
        __syncthreads();
        
        // enter exp vals
        float warp_sum = 0.0f;
        exp_vals[state] = __expf(fwd_vals[state] - fwd_maxs[0]);
        warp_sum += exp_vals[state];
        __syncthreads();
        
        // sum exp vals in warp
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }
        if (lane_id == 0) exp_sums[warp_id] = warp_sum;
        __syncthreads();

        // sum exp vals in all warps
        if (warp_id == 0) {
            warp_sum = (tid < num_states/warpSize) ? exp_sums[lane_id] : 0;

            for (int offset = warpSize/2; offset > 0; offset >>= 1) {
                warp_sum += __shfl_down_sync(mask, warp_sum, offset);
            }
            
            if (tid == 0) exp_sums[0] = warp_sum;
        }
        __syncthreads();
        
        // calculate posterior probability
        out[ts_idx + state] = exp_vals[state] / exp_sums[0];
        __syncthreads();
    }
}

__global__ void rotary(
	float *x,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seq,
    const uint64_t stride_head,
    const uint64_t rotary_half
) {
    const uint64_t batch = blockIdx.x;
    const uint64_t head = blockIdx.y;
    const uint64_t rot = threadIdx.x;
    const uint64_t tid = threadIdx.y;
    const uint64_t nthreads = blockDim.y;

    if (tid >= seqlen) return;

    float *_o0 = x + (batch * stride_batch) + (head * stride_head) + rot;
    float *_o1 = x + (batch * stride_batch) + (head * stride_head) + rotary_half + rot;

    for (int seq = tid; seq < seqlen; seq += nthreads) {
        float cos = *(_cos + (seq * rotary_half) + rot);
        float sin = *(_sin + (seq * rotary_half) + rot);

        float *o0 = _o0 + (seq * stride_seq);
        float *o1 = _o1 + (seq * stride_seq);

        float x0 = *o0;
        float x1 = *o1;

        *o0 = (x0 * cos - x1 * sin);
        *o1 = (x0 * sin + x1 * cos);
    }
}
