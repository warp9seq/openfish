#include "decode_cuda.h"
#include "scan_cuda.cuh"
#include "beam_search_cuda.cuh"
#include "error.h"
#include "cuda_utils.cuh"
#include "misc.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

void *upload_scores_to_cuda(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
) {
    void *scores_TNC_gpu;

    cudaMalloc((void **)&scores_TNC_gpu, sizeof(half) * T * N * C);
	checkCudaError();

	cudaMemcpy(scores_TNC_gpu, scores_TNC, sizeof(half) * T * N * C, cudaMemcpyHostToDevice);
	checkCudaError();

    return scores_TNC_gpu;
}

void free_scores_cuda(
    void *scores_TNC_gpu
) {
    cudaFree(scores_TNC_gpu);
	checkCudaError();
}

openfish_gpubuf_t *gpubuf_init_cuda(
    const int T,
    const int N,
    const int state_len
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    cudaMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) *  N * (T + 1) * num_states);
	checkCudaError();
    cudaMalloc((void **)&gpubuf->post_NTC, sizeof(float) *  N * (T + 1) * num_states);
	checkCudaError();

    // return buffers
    cudaMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->sequence, sizeof(char) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qstring, sizeof(char) * N * T);
    checkCudaError();

    // beamsearch buffers
    cudaMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
    checkCudaError();
    cudaMalloc((void **)&gpubuf->states, sizeof(state_t) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->base_probs, sizeof(float) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->total_probs, sizeof(float) * N * T);
    checkCudaError();

    return gpubuf;
}

void gpubuf_free_cuda(
    openfish_gpubuf_t *gpubuf
) {
    cudaFree(gpubuf->bwd_NTC);
    checkCudaError();
    cudaFree(gpubuf->post_NTC);
    checkCudaError();

    cudaFree(gpubuf->moves);
    checkCudaError();
    cudaFree(gpubuf->sequence);
    checkCudaError();
    cudaFree(gpubuf->qstring);
    checkCudaError();

    cudaFree(gpubuf->beam_vector);
    checkCudaError();
    cudaFree(gpubuf->states);
    checkCudaError();
    cudaFree(gpubuf->qual_data);
    checkCudaError();
    cudaFree(gpubuf->base_probs);
    checkCudaError();
    cudaFree(gpubuf->total_probs);
    checkCudaError();

    free(gpubuf);
}

void decode_cuda(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
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

    OPENFISH_LOG_DEBUG("scores tensor dim: %d, %d, %d", T, N, C);

    scan_args_t scan_args = {0};
    scan_args.scores_in = scores_TNC;
    scan_args.T = T;
    scan_args.N = N;
    scan_args.C = C;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;

    // bwd scan
	t0 = realtime();
    bwd_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC);
    cudaDeviceSynchronize();
    checkCudaError();
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("bwd scan completed in %f secs", elapsed);
    
    // fwd + post scan
	t0 = realtime();
    fwd_post_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC);
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

    cudaMemset(gpubuf->moves, 0, sizeof(uint8_t) * N * T);
	checkCudaError();
    cudaMemset(gpubuf->sequence, 0, sizeof(char) * N * T);
	checkCudaError();
    cudaMemset(gpubuf->qstring, 0, sizeof(char) * N * T);
	checkCudaError();

    const int num_state_bits = (int)log2(num_states);
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    beam_args_t beam_args = {0};
    beam_args.scores_TNC = (half *)scores_TNC;
    beam_args.bwd_NTC = gpubuf->bwd_NTC;
    beam_args.post_NTC = gpubuf->post_NTC;
    beam_args.T = T;
    beam_args.N = N;
    beam_args.C = C;
    beam_args.num_state_bits = num_state_bits;

    t0 = realtime();
    beam_search<<<grid_size,block_size_beam>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->moves,
        (beam_element_t *)gpubuf->beam_vector,
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
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
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
        gpubuf->moves,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        gpubuf->base_probs,
        gpubuf->total_probs,
        gpubuf->sequence,
        gpubuf->qstring,
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
    cudaMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(*sequence, gpubuf->sequence, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(*qstring, gpubuf->qstring, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
}
