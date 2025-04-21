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
	float *_x0,
    float *_x1,
    float *_o0,
    float *_o1,
    float *_cos,
    float *_sin,
    const uint64_t rotary_dim,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seqlen,
    const uint64_t stride_nheads,
    const uint64_t stride_headdim
) {
    const uint64_t pid_batch = blockIdx.x;
    const uint64_t pid_head = blockIdx.y;
    const uint64_t pid_k = blockIdx.z;
    const uint64_t pid_m = threadIdx.x;

    if (pid_m >= seqlen) return;
    if (pid_head >= 8) return;
    if (pid_k >= rotary_dim) return;

    float *x0 = _x0 + (pid_batch * stride_batch) + (pid_m * stride_seqlen) + (pid_head * stride_nheads) + (pid_k * stride_headdim);
    float *o0 = _o0 + (pid_batch * stride_batch) + (pid_m * stride_seqlen) + (pid_head * stride_nheads) + (pid_k * stride_headdim);

    float *x1 = _x1 + (pid_batch * stride_batch) + (pid_m * stride_seqlen) + (pid_head * stride_nheads) + (pid_k * stride_headdim);
    float *o1 = _o1 + (pid_batch * stride_batch) + (pid_m * stride_seqlen) + (pid_head * stride_nheads) + (pid_k * stride_headdim);

    float *cos = _cos + (pid_m * rotary_dim) + pid_k;
    float *sin = _sin + (pid_m * rotary_dim) + pid_k;

    *o0 = (*x0) + (*cos) - (*x1) * (*sin);
    *o1 = (*x0) + (*sin) + (*x1) * (*cos);
}

__global__ void rotary_2(
	half *_OUT,
    half *_X,
    half *_COS,
    half *_SIN,
    const uint64_t seqlen_offt,
    const uint64_t seqlen,
    const uint64_t rotary_dim,
    const uint64_t seqlen_ro,
    const uint64_t stride_out_batch,
    const uint64_t stride_out_seqlen,
    const uint64_t stride_out_nheads,
    const uint64_t stride_out_headdim,
    const uint64_t block_k,
    const uint64_t stride_x_batch,
    const uint64_t stride_x_seqlen,
    const uint64_t stride_x_nheads,
    const uint64_t stride_x_headdim,
    const uint64_t block_m
) {
	const uint64_t pid_m = blockIdx.x;
    const uint64_t pid_head = blockIdx.y;
    const uint64_t pid_batch = blockIdx.z;
    const uint64_t rotary_dim_half = rotary_dim / 2;
    const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const uint64_t k = threadIdx.x;
    const uint64_t m = threadIdx.y;
    const uint64_t nthreads = blockDim.x * blockDim.y;

    if (pid_m * block_m >= seqlen) { return; }
    if (k >= (block_k / 2)) { return; }
    if (m >= block_m) { return; }

    half *X = _X + (pid_batch * stride_x_batch) + (pid_head * stride_x_nheads);
    half *OUT = _OUT + (pid_batch * stride_out_batch) + (pid_head * stride_out_nheads);

    __shared__ uint64_t rm[BLOCK_M_MAX];
    __shared__ uint64_t rm_cs[BLOCK_M_MAX];
    __shared__ uint64_t rk_half[BLOCK_K_MAX / 2];

    for (uint64_t i = tid; i < block_m; i += nthreads) {
        uint64_t val = (pid_m * block_m) + i;
        rm[i] = val;
        rm_cs[i] = val + seqlen_offt;
    }
    for (uint64_t i = tid; i < (block_k / 2); i += nthreads) {
        rk_half[i] = i;
    }
    __syncthreads();

    // Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
    X = X + (rm[m] * stride_x_seqlen + rk_half[k] * stride_x_headdim);
    half *COS = _COS + (rm_cs[m] * rotary_dim_half + rk_half[k]);
    half *SIN = _SIN + (rm_cs[m] * rotary_dim_half + rk_half[k]);

    float cos = __half2float(*COS);
    float sin = __half2float(*SIN);
    if (!(rm_cs[m] < seqlen_ro && rk_half[k] < rotary_dim_half)) {
        cos = 1.0;
        sin = 0.0;
    }

    float x0 = __half2float(*X);
    float x1 = __half2float(*(X + (rotary_dim_half * stride_x_headdim)));
    if (!(rm[m] < seqlen && rk_half[k] < rotary_dim_half)) {
        x0 = 0.0;
        x1 = 0.0;
    }

    half o0 = __float2half(x0 * cos - x1 * sin);
    half o1 = __float2half(x0 * sin + x1 * cos);

    // write back result
    OUT = OUT + (rm[m] * stride_out_seqlen + rk_half[k] * stride_out_headdim);
    if (rm[m] < seqlen && rk_half[k] < rotary_dim_half) {
        *OUT = o0;
        *(OUT + rotary_dim_half * stride_out_headdim) = o1;
    }
}
