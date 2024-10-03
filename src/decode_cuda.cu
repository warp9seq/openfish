#include "decode_cuda.cuh"
#include "beam_search_cuda.cuh"
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
    // __shared__ DTYPE_GPU exp_sum;
    __shared__ DTYPE_GPU max_val;
    max_val = -FLT_MAX;

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
        __syncthreads();

        // enter exp vals
        for (uint64_t state = state_begin; state < state_end; ++state) {
            exp_vals[state] = __expf(fwd_vals[state] - max_val);
            // atomicAdd(&exp_sum, exp_vals[state]); // for some reason this is not synchronized
        }
        __syncthreads();

        // get max exp val
        DTYPE_GPU exp_sum = 0.0f;
        for (uint64_t state = 0; state < num_states; ++state) {
            exp_sum += exp_vals[state];
        }
        __syncthreads();
        
        // calculate posterior probability
        for (uint64_t state = state_begin; state < state_end; ++state) {
            out[ts_idx + state] = exp_vals[state] / exp_sum;
        }
        max_val = -FLT_MAX;
        __syncthreads();
    }
}

void decode_cuda(
    const int T,
    const int N,
    const int C,
    const int target_threads,
    float *scores_TNC,
    std::vector<DecodedChunk>& chunk_results,
    const int state_len,
    const DecoderOptions *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
    const int num_states = std::pow(NUM_BASES, state_len);

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
    dim3 block_size_beam(1, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(grid_len, 1, 1);

    // expect input already transformed
    // scores_TNC = scores_TNC.to(torch::kCPU).to(DTYPE_GPU).transpose(0, 1).contiguous();
    
    const int states_per_thread = std::max(1, num_states / (block_width * block_width));
    const uint64_t num_scan_elem = N * (T + 1) * num_states;

    LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

#ifdef DEBUG
    DTYPE_GPU *bwd_NTC = (DTYPE_GPU *)malloc(num_scan_elem * sizeof(DTYPE_GPU));
    MALLOC_CHK(bwd_NTC);

    DTYPE_GPU *post_NTC = (DTYPE_GPU *)malloc(num_scan_elem * sizeof(DTYPE_GPU));
    MALLOC_CHK(post_NTC);
#endif

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

    // beam search

    // results
    *moves = (uint8_t *)calloc(N * T, sizeof(uint8_t));
    MALLOC_CHK(moves);

    *sequence = (char *)calloc(N * T, sizeof(char));
    MALLOC_CHK(sequence);

    *qstring = (char *)calloc(N * T, sizeof(char));
    MALLOC_CHK(qstring);

#ifdef DEBUG
    state_t *states = (state_t *)calloc(N * T, sizeof(state_t));
    MALLOC_CHK(states);

    float *qual_data = (float *)calloc(N * T * NUM_BASES, sizeof(float));
    MALLOC_CHK(qual_data);

    float *base_probs = (float *)calloc(N * T, sizeof(float));
    MALLOC_CHK(base_probs);

    float *total_probs = (float *)calloc(N * T, sizeof(float));
    MALLOC_CHK(total_probs);
#endif

    // intermediate results
    uint8_t *moves_cuda;
    char *sequence_cuda;
    char *qstring_cuda;

    cudaMalloc((void **)&moves_cuda, sizeof(uint8_t) * N * T);
    checkCudaError();
    cudaMemset(moves_cuda, 0, sizeof(uint8_t) * N * T);
	checkCudaError();

    cudaMalloc((void **)&sequence_cuda, sizeof(char) * N * T);
    checkCudaError();
    cudaMemset(sequence_cuda, 0, sizeof(char) * N * T);
	checkCudaError();

    cudaMalloc((void **)&qstring_cuda, sizeof(char) * N * T);
    checkCudaError();
    cudaMemset(qstring_cuda, 0, sizeof(char) * N * T);
	checkCudaError();
    
    // intermediate
    beam_element_t *beam_vector_cuda;
    state_t *states_cuda;
    float *qual_data_cuda;
    float *base_probs_cuda;
    float *total_probs_cuda;

    cudaMalloc((void **)&beam_vector_cuda, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
    checkCudaError();
    cudaMemset(beam_vector_cuda, 0, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
	checkCudaError();

    cudaMalloc((void **)&states_cuda, sizeof(state_t) * N * T);
    checkCudaError();
    cudaMemset(states_cuda, 0, sizeof(state_t) * N * T);
	checkCudaError();

    cudaMalloc((void **)&qual_data_cuda, sizeof(float) * N * T * NUM_BASES);
    checkCudaError();
    cudaMemset(qual_data_cuda, 0, sizeof(float) * N * T * NUM_BASES);
	checkCudaError();

    cudaMalloc((void **)&base_probs_cuda, sizeof(float) * N * T);
    checkCudaError();
    cudaMemset(base_probs_cuda, 0, sizeof(float) * N * T);
	checkCudaError();

    cudaMalloc((void **)&total_probs_cuda, sizeof(float) * N * T);
    checkCudaError();
    cudaMemset(total_probs_cuda, 0, sizeof(float) * N * T);
	checkCudaError();

    const int num_state_bits = static_cast<int>(log2(num_states));
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    t0 = realtime();
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
        beam_search_cuda<<<grid_size,block_size_beam>>>(
            scores_TNC_cuda,
            bwd_NTC_cuda,
            post_NTC_cuda,
            states_cuda,
            moves_cuda,
            qual_data_cuda,
            beam_vector_cuda,
            num_state_bits,
            beam_cut,
            fixed_stay_score,
            1.0f,
            1.0f,
            T,
            N,
            C
        );
        cudaDeviceSynchronize();
        checkCudaError();
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    fprintf(stderr, "beam search completed in %f secs\n", elapsed);

    t0 = realtime();
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
        generate_sequence_cuda<<<grid_size,block_size_gen>>>(
            moves_cuda,
            states_cuda,
            qual_data_cuda,
            base_probs_cuda,
            total_probs_cuda,
            sequence_cuda,
            qstring_cuda,
            q_shift,
            q_scale,
            T,
            N
        );
        cudaDeviceSynchronize();
        checkCudaError();
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    fprintf(stderr, "generate sequence completed in %f secs\n", elapsed);

    // copy beam_search results
    cudaMemcpy(*moves, moves_cuda, sizeof(uint8_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(*sequence, sequence_cuda, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(*qstring, qstring_cuda, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

#ifdef DEBUG
    // copy scan results
    cudaMemcpy(bwd_NTC, bwd_NTC_cuda, sizeof(DTYPE_GPU) * num_scan_elem, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(post_NTC, post_NTC_cuda, sizeof(DTYPE_GPU) * num_scan_elem, cudaMemcpyDeviceToHost);
    checkCudaError();

    // copy intermediate
    cudaMemcpy(states, states_cuda, sizeof(state_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(total_probs, total_probs_cuda, sizeof(float) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(qual_data, qual_data_cuda, sizeof(float) * N * T * NUM_BASES, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(base_probs, base_probs_cuda, sizeof(float) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    // write results
    FILE *fp;

    fp = fopen("states.blob", "w");
    fwrite(states, sizeof(state_t), N * T, fp);
    fclose(fp);

    fp = fopen("qual_data.blob", "w");
    fwrite(qual_data, sizeof(float), N * T * NUM_BASES, fp);
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    fwrite(base_probs, sizeof(float), N * T, fp);
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    fwrite(total_probs, sizeof(float), N * T, fp);
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
#endif
    cudaFree(scores_TNC_cuda);
    cudaFree(bwd_NTC_cuda);
    cudaFree(post_NTC_cuda);

    cudaFree(moves_cuda);
    cudaFree(sequence_cuda);
    cudaFree(qstring_cuda);

    cudaFree(beam_vector_cuda);
    cudaFree(states_cuda);
    cudaFree(qual_data_cuda);
    cudaFree(base_probs_cuda);
    cudaFree(total_probs_cuda);
}
