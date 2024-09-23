#include "decode_gpu.cuh"
#include "error.h"
#include "error.cuh"
#include "misc.h"

#include <math.h>
#include <vector>

__global__ void bwd_scan(
	const DTYPE_GPU *scores_in,
	DTYPE_GPU *out,
	const int T,
	const int N,
	const int num_states,
	const int num_states_per_thread
) {
	uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	uint64_t thread_idx = threadIdx.x + (threadIdx.y * blockDim.x);
	uint64_t state_begin = thread_idx * num_states_per_thread;
	uint64_t state_end = state_begin + num_states_per_thread;

	if (chunk >= N || state_begin >= num_states) {
		return;
	}

	const uint64_t kNumBases = 4;
    const uint64_t kNumTransitions = kNumBases + 1;
    const DTYPE_GPU kFixedStayScore = 2.0f;

    const uint64_t ts_states = num_states * kNumBases;

    const DTYPE_GPU* const chunk_in = scores_in + chunk * ts_states; // should be half DTYPE_GPU (for GPU impl)
    DTYPE_GPU* const chunk_out = out + chunk * (T+1) * num_states;
    DTYPE_GPU* const alpha_init = chunk_out + num_states * T;
    for (uint64_t state = state_begin; state < state_end; ++state) {
        alpha_init[state] = 0.0f;
    }

    for (uint64_t ts = 0; ts < T; ++ts) {
        __syncthreads();
        const DTYPE_GPU* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        DTYPE_GPU* const ts_alpha_in = alpha_init - num_states * ts;
        DTYPE_GPU* const ts_alpha_out = ts_alpha_in - num_states;

        for (uint64_t state = state_begin; state < state_end; ++state) {
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = (state * kNumBases) % num_states;
            const uint64_t step_trans_idx_a = step_state_idx_a * kNumBases +
                ((state * kNumBases) / num_states);

            DTYPE_GPU vals[kNumTransitions];
            vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            DTYPE_GPU max_val = vals[0];
            for (uint64_t base = 0; base < kNumBases; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    ts_in[step_trans_idx_a + base * kNumBases];
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            DTYPE_GPU sum = 0.0f;
            for (uint64_t i = 0; i < kNumTransitions; ++i) {
                sum += exp(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + log(sum);
        }
    }
}

__global__ void fwd_post_scan(
    const DTYPE_GPU *scores_in,
    const DTYPE_GPU *bwd,
    DTYPE_GPU *out,
    const uint64_t _T,
    const uint64_t N,
    const uint64_t num_states,
    const int num_states_per_thread
) {
    uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	uint64_t thread_idx = threadIdx.x + (threadIdx.y * blockDim.x);
	uint64_t state_begin = thread_idx * num_states_per_thread;
	uint64_t state_end = state_begin + num_states_per_thread;

	if (chunk >= N || state_begin >= num_states) {
		return;
	}

    const uint64_t T = _T+1; 
    constexpr uint64_t kNumBases = 4;
    constexpr uint64_t kNumTransitions = kNumBases + 1;
    constexpr DTYPE_GPU kFixedStayScore = 2.0f;
    
    const uint64_t kMsb = num_states / kNumBases;
    const uint64_t ts_states = num_states * kNumBases;

    constexpr uint64_t max_threads_per_block = 1024;
    __shared__ DTYPE_GPU fwd_vals[max_threads_per_block];
    __shared__ DTYPE_GPU exp_vals[max_threads_per_block];

    // This batch element's scores.
    const DTYPE_GPU* const chunk_scores = scores_in + chunk * ts_states;

    // Alternating forward guide buffers used for successive time steps.
    constexpr uint64_t kMaxStates = 1024;
    __shared__ DTYPE_GPU ts_fwd[2][kMaxStates]; // threadgroup

    // The forward guide input for the first step is 0.
    for (uint64_t state = state_begin; state < state_end; ++state) {
        ts_fwd[0][state] = 0.0f;
    }
    __syncthreads();

    for (uint64_t ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        // This time step's scores.
        const DTYPE_GPU* const ts_scores = chunk_scores + N * ts_states * ts;

        // Alternating TG buffer twiddling.
        const DTYPE_GPU* const ts_alpha_in = ts_fwd[ts & 1];
        DTYPE_GPU* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        for (uint64_t state = state_begin; state < state_end; ++state) {
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = state / kNumBases;
            const uint64_t step_trans_idx_a = state * kNumBases;
            DTYPE_GPU vals[kNumTransitions];
            DTYPE_GPU fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            for (uint64_t base = 0; base < kNumBases; ++base) {
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                DTYPE_GPU ts_score = ts < _T ? ts_scores[step_trans_idx_a + base] : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            DTYPE_GPU fwd_sum = 0.0f;
            for (uint64_t i = 0; i < kNumTransitions; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + log(fwd_sum);

            // Load the forward guide value calculated in the last time step for use
            // in this time step's posterior probability calculation.
            const DTYPE_GPU fwd_val = ts_alpha_in[state];

            // Calculate fwd/bwd guide product in log space.
            const DTYPE_GPU val = fwd_val + bwd[ts_idx + state];

            fwd_vals[state] = val;
        }
        __syncthreads();

        // find max fwd val in this timestep
        float max_val = fwd_vals[0];
        for (uint64_t state = 1; state < num_states; ++state) {
            if (fwd_vals[state] > max_val) {
                max_val = fwd_vals[state];
            }
        }
        __syncthreads();

        // enter exp vals
        for (uint64_t state = state_begin; state < state_end; ++state) {
            exp_vals[state] = exp(fwd_vals[state] - max_val);
        }
        __syncthreads();

        // get max exp val
        DTYPE_GPU exp_sum = 0.0;
        for (uint64_t state = 0; state < num_states; ++state) {
            exp_sum += exp_vals[state];
        }
        
        // calculate posterior probability
        for (uint64_t state = state_begin; state < state_end; ++state) {
            out[ts_idx + state] = exp_vals[state] / exp_sum;
        }
        __syncthreads();
    }
}

void decode_gpu(
    const int T,
    const int N,
    const int C,
    const int target_threads,
    float *scores_TNC,
    std::vector<DecodedChunk>& chunk_results,
    const int state_len,
    const DecoderOptions* options
) {
    float t0, t1, elapsed;
    dim3 threads_per_block(32, 32, 1);
	dim3 num_blocks(N, 1, 1);

    // expect input already transformed
    // scores_TNC = scores_TNC.to(torch::kCPU).to(DTYPE_GPU).transpose(0, 1).contiguous();
    
    const int n_base = 4;
    const int num_states = std::pow(n_base, state_len);
    const int states_per_thread = std::max(1, num_states / 1024);
    const uint64_t num_scan_elem = N * (T + 1) * num_states;

    LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    DTYPE_GPU *bwd_NTC = (DTYPE_GPU *)malloc(num_scan_elem * sizeof(DTYPE_GPU));
    DTYPE_GPU *post_NTC = (DTYPE_GPU *)malloc(num_scan_elem * sizeof(DTYPE_GPU));

    DTYPE_GPU *scores_TNC_cuda;
    DTYPE_GPU *bwd_NTC_cuda;
    DTYPE_GPU *post_NTC_cuda;

    // copy score tensor over
    cudaMalloc((void **)&scores_TNC_cuda, sizeof(DTYPE_GPU) * T * N * C);
	checkCudaError();
	cudaMemcpy(scores_TNC_cuda, scores_TNC, sizeof(DTYPE_GPU) * T * N * C, cudaMemcpyHostToDevice);
	checkCudaError();

    // init scan tensors
    cudaMalloc((void **)&bwd_NTC_cuda, sizeof(DTYPE_GPU) * num_scan_elem);
	checkCudaError();
    cudaMalloc((void **)&post_NTC_cuda, sizeof(DTYPE_GPU) * num_scan_elem);
	checkCudaError();

    // bwd scan
	t0 = (float)clock()/CLOCKS_PER_SEC;
	bwd_scan<<<num_blocks,threads_per_block>>>(scores_TNC_cuda, bwd_NTC_cuda, T, N, num_states, states_per_thread);
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = (float)clock()/CLOCKS_PER_SEC;
    elapsed = t1 - t0;
    fprintf(stderr, "bwd scan completed in %f secs\n", elapsed);
    
    // fwd + post scan
	t0 = (float)clock()/CLOCKS_PER_SEC;
	fwd_post_scan<<<num_blocks,threads_per_block>>>(scores_TNC_cuda, bwd_NTC_cuda, post_NTC_cuda, T, N, num_states, states_per_thread);
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = (float)clock()/CLOCKS_PER_SEC;
    elapsed = t1 - t0;
    fprintf(stderr, "fwd scan completed in %f secs\n", elapsed);

	// copy results
    cudaMemcpy(bwd_NTC, bwd_NTC_cuda, sizeof(DTYPE_GPU) * num_scan_elem, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(post_NTC, post_NTC_cuda, sizeof(DTYPE_GPU) * num_scan_elem, cudaMemcpyDeviceToHost);
    checkCudaError();

    // write tensors
    FILE *fp;
    fp = fopen("scores_TNC.blob", "w");
    fwrite(scores_TNC, sizeof(DTYPE_GPU), T * N * C, fp);
    fclose(fp);

    fp = fopen("bwd_NTC.blob", "w");
    fwrite(bwd_NTC, sizeof(DTYPE_GPU), num_scan_elem, fp);
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    fwrite(post_NTC, sizeof(DTYPE_GPU), num_scan_elem, fp);
    fclose(fp);

    // cleanup
    free(bwd_NTC);
    free(post_NTC);
    
    cudaFree(scores_TNC_cuda);
    cudaFree(bwd_NTC_cuda);
    cudaFree(post_NTC_cuda);
}
