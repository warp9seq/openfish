#include "decode_cuda.h"
#include "scan_cuda.cuh"
#include "beam_search_cuda.cuh"
#include "error.h"
#include "cuda_utils.cuh"
#include "misc.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

void decode_cuda(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const decoder_opts_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
    const int num_states = pow(NUM_BASES, state_len);

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

    OPENFISH_LOG_DEBUG("chosen block_dims: %d x %d for num_states %d", block_width, block_width, num_states);
    OPENFISH_LOG_DEBUG("chosen grid_len: %d for batch size %d", grid_len, N);

    double t0, t1, elapsed;
    dim3 block_size(block_width, block_width, 1);
    dim3 block_size_beam(MAX_BEAM_WIDTH, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(grid_len, 1, 1);
    
    const uint64_t num_scan_elem = N * (T + 1) * num_states;

    OPENFISH_LOG_DEBUG("scores tensor dim: %d, %d, %d", T, N, C);
    
    float *bwd_NTC_gpu;
    float *post_NTC_gpu;

    // init scan tensors
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(float) * num_scan_elem);
    cudaMalloc((void **)&bwd_NTC_gpu, sizeof(float) * num_scan_elem);
	checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(float) * num_scan_elem);
    cudaMalloc((void **)&post_NTC_gpu, sizeof(float) * num_scan_elem);
	checkCudaError();

    scan_args_t scan_args = {0};
    scan_args.scores_in = scores_TNC;
    scan_args.T = T;
    scan_args.N = N;
    scan_args.C = C;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;

    // bwd scan
	t0 = realtime();
    bwd_scan<<<grid_size,block_size>>>(scan_args, bwd_NTC_gpu);
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("bwd scan completed in %f secs", elapsed);
    
    // fwd + post scan
	t0 = realtime();
    fwd_post_scan<<<grid_size,block_size>>>(scan_args, bwd_NTC_gpu, post_NTC_gpu);
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("fwd scan completed in %f secs", elapsed);

    // beam search

    // init results
    *moves = (uint8_t *)malloc(N * T * sizeof(uint8_t));
    MALLOC_CHK(*moves);

    *sequence = (char *)malloc(N * T * sizeof(char));
    MALLOC_CHK(*sequence);

    *qstring = (char *)malloc(N * T * sizeof(char));
    MALLOC_CHK(*qstring);

    // intermediate results
    uint8_t *moves_gpu;
    char *sequence_gpu;
    char *qstring_gpu;

    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(uint8_t) * N * T);
    cudaMalloc((void **)&moves_gpu, sizeof(uint8_t) * N * T);
    checkCudaError();
    cudaMemset(moves_gpu, 0, sizeof(uint8_t) * N * T);
	checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(uint8_t) * N * T);
    cudaMalloc((void **)&sequence_gpu, sizeof(uint8_t) * N * T);
    checkCudaError();
    cudaMemset(sequence_gpu, 0, sizeof(char) * N * T);
	checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(char) * N * T);
    cudaMalloc((void **)&qstring_gpu, sizeof(char) * N * T);
    checkCudaError();
    cudaMemset(qstring_gpu, 0, sizeof(char) * N * T);
	checkCudaError();
    
    // intermediate
    beam_element_t *beam_vector_gpu;
    state_t *states_gpu;
    float *qual_data_gpu;
    float *base_probs_gpu;
    float *total_probs_gpu;

    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
    cudaMalloc((void **)&beam_vector_gpu, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
    checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(state_t) * N * T);
    cudaMalloc((void **)&states_gpu, sizeof(state_t) * N * T);
    checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(float) * N * T * NUM_BASES);
    cudaMalloc((void **)&qual_data_gpu, sizeof(float) * N * T * NUM_BASES);
    checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(float) * N * T);
    cudaMalloc((void **)&base_probs_gpu, sizeof(float) * N * T);
    checkCudaError();
    OPENFISH_LOG_DEBUG("allocing %zu bytes on the GPU", sizeof(float) * N * T);
    cudaMalloc((void **)&total_probs_gpu, sizeof(float) * N * T);
    checkCudaError();

    const int num_state_bits = (int)log2(num_states);
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    beam_args_t beam_args = {0};
    beam_args.scores_TNC = scores_TNC_gpu;
    beam_args.bwd_NTC = bwd_NTC_gpu;
    beam_args.post_NTC = post_NTC_gpu;
    beam_args.T = T;
    beam_args.N = N;
    beam_args.C = C;
    beam_args.num_state_bits = num_state_bits;

    t0 = realtime();
    beam_search<<<grid_size,block_size_beam>>>(
        beam_args,
        states_gpu,
        moves_gpu,
        beam_vector_gpu,
        beam_cut,
        fixed_stay_score,
        1.0f
    );
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("beam search completed in %f secs", elapsed);

    t0 = realtime();
    compute_qual_data<<<grid_size,block_size_gen>>>(
        beam_args,
        states_gpu,
        qual_data_gpu,
        1.0f
    );
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("compute quality data completed in %f secs", elapsed);

    t0 = realtime();
    generate_sequence<<<grid_size,block_size_gen>>>(
        beam_args,
        moves_gpu,
        states_gpu,
        qual_data_gpu,
        base_probs_gpu,
        total_probs_gpu,
        sequence_gpu,
        qstring_gpu,
        q_shift,
        q_scale
    );
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("generate sequence completed in %f secs", elapsed);

    // copy beam_search results
    cudaMemcpy(*moves, moves_gpu, sizeof(uint8_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(*sequence, sequence_gpu, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(*qstring, qstring_gpu, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaFree(scores_TNC_gpu);
    checkCudaError();
    cudaFree(bwd_NTC_gpu);
    checkCudaError();
    cudaFree(post_NTC_gpu);
    checkCudaError();

    cudaFree(moves_gpu);
    checkCudaError();
    cudaFree(sequence_gpu);
    checkCudaError();
    cudaFree(qstring_gpu);
    checkCudaError();

    cudaFree(beam_vector_gpu);
    checkCudaError();
    cudaFree(states_gpu);
    checkCudaError();
    cudaFree(qual_data_gpu);
    checkCudaError();
    cudaFree(base_probs_gpu);
    checkCudaError();
    cudaFree(total_probs_gpu);
    checkCudaError();
}
