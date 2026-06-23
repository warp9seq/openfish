#include <openfish/openfish.h>
#include <stdint.h>
#include "decode.h"
#include "scan_cuda.h"
#include "beam_search_cuda.h"
#include "error.h"
#include "cuda_utils.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    cudaMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states);
	checkCudaError();
    cudaMalloc((void **)&gpubuf->post_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states);
	checkCudaError();

    // return buffers
    cudaMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->sequence, sizeof(char) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qstring, sizeof(char) * batch_size * n_timesteps);
    checkCudaError();

    // beamsearch buffers
    cudaMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1));
    checkCudaError();
    cudaMalloc((void **)&gpubuf->states, sizeof(state_t) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qual_data, sizeof(float) * batch_size * n_timesteps * NUM_BASES);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->base_probs, sizeof(float) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->total_probs, sizeof(float) * batch_size * n_timesteps);
    checkCudaError();

    return gpubuf;
}

void openfish_gpubuf_free(
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

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_TNC,
    int state_len,
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
    while (block_width < target_block_width) {
        block_width *= 2;
    }

    OPENFISH_LOG_TRACE("chosen block_dims: %d x %d for num_states %d", block_width, block_width, num_states);
    
    dim3 block_size(block_width, block_width, 1);
    dim3 block_size_beam(MAX_BEAM_WIDTH * NUM_BASES, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(batch_size, 1, 1);

    OPENFISH_LOG_TRACE("scores tensor dim: %d, %d, %d", n_timesteps, batch_size, n_channels);

    scan_args_t scan_args = {0};
    scan_args.scores_in = scores_TNC;
    scan_args.n_timesteps = n_timesteps;
    scan_args.batch_size = batch_size;
    scan_args.n_channels = n_channels;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;

    // init results
    *moves = (uint8_t *)malloc(batch_size * n_timesteps * sizeof(uint8_t));
    MALLOC_CHK(*moves);
    *sequence = (char *)malloc(batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*sequence);
    *qstring = (char *)malloc(batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*qstring);

    cudaMemset(gpubuf->moves, 0, sizeof(uint8_t) * batch_size * n_timesteps);
	checkCudaError();
    cudaMemset(gpubuf->sequence, 0, sizeof(char) * batch_size * n_timesteps);
	checkCudaError();
    cudaMemset(gpubuf->qstring, 0, sizeof(char) * batch_size * n_timesteps);
	checkCudaError();

    const int num_state_bits = (int)log2((double)num_states);
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;
    
    beam_args_t beam_args = {0};
    beam_args.scores_TNC = (const half *)scores_TNC;
    beam_args.bwd_NTC = gpubuf->bwd_NTC;
    beam_args.post_NTC = gpubuf->post_NTC;
    beam_args.n_timesteps = n_timesteps;
    beam_args.batch_size = batch_size;
    beam_args.n_channels = n_channels;
    beam_args.num_state_bits = num_state_bits;

    // bwd scan
    // fwd + post scan
    // beam search

    OPENFISH_LOG_TRACE("%s", "bwd scan...");
    bwd_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    OPENFISH_LOG_TRACE("%s", "beam search...");
    // dynamic shared memory holds the back-guide sort scratch (num_states floats)
    beam_search<<<grid_size,block_size_beam,num_states*sizeof(float)>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->moves,
        (beam_element_t *)gpubuf->beam_vector,
        beam_cut,
        fixed_stay_score,
        1.0f
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    OPENFISH_LOG_TRACE("%s", "fwd + post scan...");
    fwd_post_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    OPENFISH_LOG_TRACE("%s", "compute qual data...");
    compute_qual_data<<<grid_size,block_size_gen>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        1.0f
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
    
    OPENFISH_LOG_TRACE("%s", "gen sequence...");
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
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    // copy beam_search results
    cudaMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(*sequence, gpubuf->sequence, sizeof(char) * batch_size * n_timesteps, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(*qstring, gpubuf->qstring, sizeof(char) * batch_size * n_timesteps, cudaMemcpyDeviceToHost);
    checkCudaError();
}

