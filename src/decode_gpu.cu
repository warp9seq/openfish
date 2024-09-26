#include "decode_gpu.cuh"
#include "error.h"
#include "error.cuh"
#include "misc.h"

#include <math.h>
#include <vector>
#include <float.h>

// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ static float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

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

	const uint64_t k_num_bases = 4;
    const uint64_t k_num_transitions = k_num_bases + 1;
    const DTYPE_GPU k_fixed_stay_score = 2.0f;

    const uint64_t ts_states = num_states * k_num_bases;

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
            const uint64_t step_state_idx_a = (state * k_num_bases) % num_states;
            const uint64_t step_trans_idx_a = step_state_idx_a * k_num_bases +
                ((state * k_num_bases) / num_states);

            DTYPE_GPU vals[k_num_transitions];
            vals[0] = ts_alpha_in[stay_state_idx] + k_fixed_stay_score;
            DTYPE_GPU max_val = vals[0];
            for (uint64_t base = 0; base < k_num_bases; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    ts_in[step_trans_idx_a + base * k_num_bases];
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            DTYPE_GPU sum = 0.0f;
            for (uint64_t i = 0; i < k_num_transitions; ++i) {
                sum += __expf(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + __logf(sum);
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
    constexpr uint64_t k_num_bases = 4;
    constexpr uint64_t k_num_transitions = k_num_bases + 1;
    constexpr DTYPE_GPU k_fixed_stay_score = 2.0f;
    
    const uint64_t kMsb = num_states / k_num_bases;
    const uint64_t ts_states = num_states * k_num_bases;

    constexpr uint64_t k_max_states = 1024;
    __shared__ DTYPE_GPU fwd_vals[k_max_states];
    __shared__ DTYPE_GPU exp_vals[k_max_states];
    __shared__ DTYPE_GPU exp_sum;
    __shared__ DTYPE_GPU max_val;
    max_val = FLT_MIN;

    // This batch element's scores.
    const DTYPE_GPU* const chunk_scores = scores_in + chunk * ts_states;

    // Alternating forward guide buffers used for successive time steps.
    __shared__ DTYPE_GPU ts_fwd[2][k_max_states]; // threadgroup

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
            const uint64_t step_state_idx_a = state / k_num_bases;
            const uint64_t step_trans_idx_a = state * k_num_bases;
            DTYPE_GPU vals[k_num_transitions];
            DTYPE_GPU fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + k_fixed_stay_score;
            for (uint64_t base = 0; base < k_num_bases; ++base) {
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                DTYPE_GPU ts_score = ts < _T ? ts_scores[step_trans_idx_a + base] : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            DTYPE_GPU fwd_sum = 0.0f;
            for (uint64_t i = 0; i < k_num_transitions; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + __logf(fwd_sum);

            // Load the forward guide value calculated in the last time step for use
            // in this time step's posterior probability calculation.
            const DTYPE_GPU fwd_val = ts_alpha_in[state];

            // Calculate fwd/bwd guide product in log space.
            const DTYPE_GPU val = fwd_val + bwd[ts_idx + state];

            fwd_vals[state] = val;
            atomicMaxFloat(&max_val, val);
        }
        exp_sum = 0.0;
        __syncthreads();

        // enter exp vals
        for (uint64_t state = state_begin; state < state_end; ++state) {
            DTYPE_GPU exp_val = __expf(fwd_vals[state] - max_val);
            exp_vals[state] = exp_val;
            atomicAdd(&exp_sum, exp_val);
        }
        __syncthreads();

        // calculate posterior probability
        for (uint64_t state = state_begin; state < state_end; ++state) {
            out[ts_idx + state] = exp_vals[state] / exp_sum;
        }
        max_val = FLT_MIN;
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
    const int n_base = 4;
    const int num_states = std::pow(n_base, state_len);

    // calculate grid / block dims
    const int target_block_width = (int)ceil(sqrt((float)num_states));
    int block_width = 2;
    int grid_len = 2;
    while (block_width < target_block_width) {
        block_width *= 2;
    }
    while (grid_len < N) {
        grid_len *= 2;
    }

    fprintf(stderr, "chosen block_dims: %d x %d for num_states %d\n", block_width, block_width, num_states);
    fprintf(stderr, "chosen grid_len: %d for batch size %d\n", grid_len, N);

    double t0, t1, elapsed;
    dim3 block_size(block_width, block_width, 1);
	dim3 grid_size(grid_len, 1, 1);

    // expect input already transformed
    // scores_TNC = scores_TNC.to(torch::kCPU).to(DTYPE_GPU).transpose(0, 1).contiguous();
    
    
    const int states_per_thread = std::max(1, num_states / (block_width * block_width));
    const uint64_t num_scan_elem = N * (T + 1) * num_states;

    LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    DTYPE_GPU *bwd_NTC = (DTYPE_GPU *)malloc(num_scan_elem * sizeof(DTYPE_GPU));
    MALLOC_CHK(bwd_NTC);
    DTYPE_GPU *post_NTC = (DTYPE_GPU *)malloc(num_scan_elem * sizeof(DTYPE_GPU));
    MALLOC_CHK(post_NTC);

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

#ifdef BENCH
    int n_batch = 140; // simulate 20k reads
    if (num_states == 64) n_batch = 140; // fast
    else if (num_states == 256) n_batch = 345; // hac
    else if (num_states == 1024) n_batch = 685; // sup
    fprintf(stderr, "simulating %d batches...\n", n_batch);
#endif

    // bwd scan
	t0 = realtime();
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
        bwd_scan<<<grid_size,block_size>>>(scores_TNC_cuda, bwd_NTC_cuda, T, N, num_states, states_per_thread);
        cudaDeviceSynchronize();
        checkCudaError();
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    fprintf(stderr, "bwd scan completed in %f secs\n", elapsed);
    
    // fwd + post scan
	t0 = realtime();
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
        fwd_post_scan<<<grid_size,block_size>>>(scores_TNC_cuda, bwd_NTC_cuda, post_NTC_cuda, T, N, num_states, states_per_thread);
        cudaDeviceSynchronize();
        checkCudaError();
    }
	// end timing
	t1 = realtime();
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
