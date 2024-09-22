#include "decode_gpu.cuh"
#include "error.h"
#include "error.cuh"
#include "misc.h"

#include <math.h>
#include <vector>

__global__ void bwd_scan(
	const float *scores_in,
	float *out,
	const int T,
	const int N,
	const int num_states,
	const int num_states_per_thread
) {
	int chunk = blockIdx.x + (blockIdx.y * gridDim.x);
	int thread_idx = threadIdx.x + (threadIdx.y * blockDim.x);
	int state_begin = thread_idx * num_states_per_thread;
	int state_end = state_begin + num_states_per_thread;

	if (chunk >= N || state_begin >= num_states) {
		return;
	}

	const int kNumBases = 4;
    const int kNumTransitions = kNumBases + 1;
    const float kFixedStayScore = 2.0f;

    const int ts_states = num_states * kNumBases;

    const float* const chunk_in = scores_in + chunk * ts_states; // should be half float (for GPU impl)
    float* const chunk_out = out + chunk * (T+1) * num_states;
    float* const alpha_init = chunk_out + num_states * T;
    for (int state = 0; state < num_states; ++state) {
        alpha_init[state] = 0.0f;
    }

    for (int ts = 0; ts < T; ++ts) {
        __syncthreads();
        const float* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        float* const ts_alpha_in = alpha_init - num_states * ts;
        float* const ts_alpha_out = ts_alpha_in - num_states;

        for (int state = state_begin; state < state_end; ++state) {
            const int stay_state_idx = state;
            const int step_state_idx_a = (state * kNumBases) % num_states;
            const int step_trans_idx_a = step_state_idx_a * kNumBases +
                ((state * kNumBases) / num_states);

            float vals[kNumTransitions];
            vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            float max_val = vals[0];
            for (int base = 0; base < kNumBases; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    ts_in[step_trans_idx_a + base * kNumBases];
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            float sum = 0.0f;
            for (int i = 0; i < kNumTransitions; ++i) {
                sum += exp(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + log(sum);
        }
    }
}

// static void backward_scan(const float *scores_in, float *out, const uint64_t chunk, const uint64_t T, const uint64_t N, const uint64_t num_states) {
//     const uint64_t kNumBases = 4;
//     const uint64_t kNumTransitions = kNumBases + 1;
//     const float kFixedStayScore = 2.0f;

//     const uint64_t ts_states = num_states * kNumBases;

//     const float* const chunk_in = scores_in + chunk * ts_states; // should be half float (for GPU impl)
//     float* const chunk_out = out + chunk * (T+1) * num_states;
//     float* const alpha_init = chunk_out + num_states * T;
//     for (uint64_t state = 0; state < num_states; ++state) { // (for GPU impl) its 1 thread per state, but below we iterate through all the states on 1 thread
//         alpha_init[state] = 0.0f;
//     }

//     for (uint64_t ts = 0; ts < T; ++ts) {
//         // threadgroup_barrier(mem_flags::medevice); // synchronize all threads before next time step (for GPU impl)
//         const float* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
//         float* const ts_alpha_in = alpha_init - num_states * ts;
//         float* const ts_alpha_out = ts_alpha_in - num_states;

//         for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
//             const uint64_t stay_state_idx = state;
//             const uint64_t step_state_idx_a = (state * kNumBases) % num_states;
//             const uint64_t step_trans_idx_a = step_state_idx_a * kNumBases +
//                 ((state * kNumBases) / num_states);

//             float vals[kNumTransitions];
//             vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
//             float max_val = vals[0];
//             for (uint64_t base = 0; base < kNumBases; ++base) {
//                 vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
//                     ts_in[step_trans_idx_a + base * kNumBases];
//                 max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
//             }
//             float sum = 0.0f;
//             for (uint64_t i = 0; i < kNumTransitions; ++i) {
//                 sum += exp(vals[i] - max_val);
//             }
//             ts_alpha_out[state] = max_val + log(sum);
//         }
//     }
// }

static void forward_scan(const float *scores_in, const float *bwd, float *out, const uint64_t chunk, const uint64_t _T, const uint64_t N, const uint64_t num_states) {
    const uint64_t T = _T+1; 
    constexpr uint64_t kNumBases = 4;
    constexpr uint64_t kNumTransitions = kNumBases + 1;
    constexpr float kFixedStayScore = 2.0f;
    
    const uint64_t kMsb = num_states / kNumBases;
    const uint64_t ts_states = num_states * kNumBases;

    // This batch element's scores.
    const float* const chunk_scores = scores_in + chunk * ts_states;

    // Alternating forward guide buffers used for successive time steps.
    constexpr uint64_t kMaxStates = 1024;
    float ts_fwd[2][kMaxStates]; // threadgroup

    // The forward guide input for the first step is 0.
    for (uint64_t state = 0; state < num_states; ++state) {
        ts_fwd[0][state] = 0.0f;
    }
    // threadgroup_barrier(mem_flags::mem_threadgroup); // ------------------------------------------------------------------

    for (uint64_t ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        // This time step's scores.
        const float* const ts_scores = chunk_scores + N * ts_states * ts;

        // Alternating TG buffer twiddling.
        const float* const ts_alpha_in = ts_fwd[ts & 1];
        float* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = state / kNumBases;
            const uint64_t step_trans_idx_a = state * kNumBases;
            float vals[kNumTransitions];
            float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            for (uint64_t base = 0; base < kNumBases; ++base) {
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                float ts_score = ts < _T ? ts_scores[step_trans_idx_a + base] : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            float fwd_sum = 0.0f;
            for (uint64_t i = 0; i < kNumTransitions; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + log(fwd_sum);

            // Load the forward guide value calculated in the last time step for use
            // in this time step's posterior probability calculation.
            const float fwd_val = ts_alpha_in[state];

            // Calculate fwd/bwd guide product in log space.
            const float val = fwd_val + bwd[ts_idx + state];
            out[ts_idx + state] = val;
        }
    }
}

static void softmax(const float *fwd, float *out, const uint64_t chunk, const uint64_t _T, const uint64_t num_states) {
    const uint64_t T = _T+1; 
    for (uint64_t ts = 0; ts < T; ++ts) {
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        float max_val = fwd[ts_idx];
        for (uint64_t state = 0; state < num_states; ++state) {
            max_val = max_val > fwd[ts_idx + state] ? max_val : fwd[ts_idx + state];
        }

        float exp_sum = 0;
        float exp_vals[num_states];
        for (uint64_t state = 0; state < num_states; ++state) {
            const float val = fwd[ts_idx + state];
            const float exp_val = exp(val - max_val);
            exp_vals[state] = exp_val;
            exp_sum += exp_val;
        }

        for (uint64_t state = 0; state < num_states; ++state) {
            const float exp_val = exp_vals[state];

            // Write out the posterior probability 
            out[ts_idx + state] = (float)(exp_val / exp_sum);
        }
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
    float t0, t1;
    // expect input already transformed
    // scores_TNC = scores_TNC.to(torch::kCPU).to(DTYPE_CPU).transpose(0, 1).contiguous();
    
    const int n_base = 4;
    const int num_states = std::pow(n_base, state_len);
    const int states_per_thread = std::max(1, num_states / 1024);
    fprintf(stderr, "num_states: %d\n", num_states);
    fprintf(stderr, "states per thread: %d\n", states_per_thread);

    LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    float *bwd_NTC = (float *)calloc(N * (T + 1) * num_states, sizeof(DTYPE_CPU));
    // float *fwd_NTC = (float *)calloc(N * (T + 1) * num_states, sizeof(DTYPE_CPU));
    // float *post_NTC = (float *)calloc(N * (T + 1) * num_states, sizeof(DTYPE_CPU));

    float *scores_TNC_cuda;
    float *bwd_NTC_cuda;
    // float *fwd_NTC_cuda;
    // float *post_NTC_cuda;
    // init scan tensors
	cudaMalloc((void **)&bwd_NTC_cuda, sizeof(DTYPE_CPU) * N * (T + 1) * num_states);
	checkCudaError();
    cudaMemcpy(bwd_NTC_cuda, bwd_NTC, sizeof(DTYPE_CPU) * N * (T + 1) * num_states, cudaMemcpyHostToDevice);
	checkCudaError();

	// copy score tensor over
    cudaMalloc((void **)&scores_TNC_cuda, sizeof(DTYPE_CPU) * T * N * C);
	checkCudaError();
	cudaMemcpy(scores_TNC_cuda, scores_TNC, sizeof(DTYPE_CPU) * T * N * C, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	checkCudaError();

    fprintf(stderr, "copied scores over to cuda device\n");

    dim3 threads_per_block(32, 32, 1);
	dim3 num_blocks(N, 1, 1);

    // start measuring time for cuda kernel only
    float elapsedtimekernel;
	t0 = (float)clock()/CLOCKS_PER_SEC;

	bwd_scan<<<num_blocks,threads_per_block>>>(scores_TNC_cuda, bwd_NTC_cuda, T, 1, num_states, states_per_thread);
	cudaDeviceSynchronize();
    checkCudaError();

	// end measuring time for cuda kernel
	t1 = (float)clock()/CLOCKS_PER_SEC;
    elapsedtimekernel = t1 - t0;

    fprintf(stderr, "bwd scan completed in %f secs\n", elapsedtimekernel);

	// copy the answer back from cuda ro ram
	// start measuring time
	cudaMemcpy(bwd_NTC, bwd_NTC_cuda, sizeof(DTYPE_CPU) * N * (T + 1) * num_states, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaError();

    fprintf(stderr, "%s\n", "copied bwd over to host");

    // write tensors
    FILE *fp;
    fp = fopen("scores_TNC.blob", "w");
    fwrite(scores_TNC, sizeof(float), T * N * C, fp);
    fclose(fp);

    fp = fopen("bwd_NTC.blob", "w");
    fwrite(bwd_NTC, sizeof(DTYPE_CPU), N * (T + 1) * num_states, fp);
    fclose(fp);

    // fp = fopen("fwd_NTC.blob", "w");
    // fwrite(fwd_NTC, sizeof(float), N * (T + 1) * m_states, fp);
    // fclose(fp);

    // fp = fopen("post_NTC.blob", "w");
    // fwrite(post_NTC, sizeof(float), N * (T + 1) * m_states, fp);
    // fclose(fp);

    free(bwd_NTC);
    // free(fwd_NTC);
    // free(post_NTC);
    
    cudaFree(scores_TNC_cuda);
    cudaFree(bwd_NTC_cuda);
    // cudaFree(fwd_NTC_cuda);
    // cudaFree(post_NTC_cuda);
}
