#include "decode_hip.h"
#include "scan_hip.h"
#include "beam_search_hip.h"
#include "error.h"
#include "hip_utils.h"
#include "misc.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

openfish_gpubuf_t *gpubuf_init_cuda(
    const int T,
    const int N,
    const int state_len
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    cudaMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states);
	checkCudaError();
    cudaMalloc((void **)&gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states);
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
    double t0, t1, elapsed;
    t0 = realtime();

    const int num_states = pow(NUM_BASES, state_len);
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    checkCudaError();
    cudaStreamCreate(&stream2);
    checkCudaError();

    // calculate grid / block dims
    const int target_block_width = (int)ceil(sqrt((float)num_states));
    int block_width = 2;
    while (block_width < target_block_width) {
        block_width *= 2;
    }

    OPENFISH_LOG_DEBUG("chosen block_dims: %d x %d for num_states %d", block_width, block_width, num_states);
    
    dim3 block_size(block_width, block_width, 1);
    dim3 block_size_beam(MAX_BEAM_WIDTH * NUM_BASES, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(N, 1, 1);

    OPENFISH_LOG_DEBUG("scores tensor dim: %d, %d, %d", T, N, C);

    scan_args_t scan_args = {0};
    scan_args.scores_in = scores_TNC;
    scan_args.T = T;
    scan_args.N = N;
    scan_args.C = C;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;

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

    const int num_state_bits = (int)log2((double)num_states);
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

#ifdef BENCH
    int n_batch = 140; // simulate 20k reads
    if (num_states == 64) n_batch = 140; // fast
    else if (num_states == 256) n_batch = 345; // hac
    else if (num_states == 1024) n_batch = 685; // sup
    OPENFISH_LOG_DEBUG("simulating %d batches...", n_batch);
#endif

    // bwd scan
    // fwd + post scan
    // beam search
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
        bwd_scan<<<grid_size,block_size,0,stream1>>>(scan_args, gpubuf->bwd_NTC);
        checkCudaError();
        cudaStreamSynchronize(stream1);
        checkCudaError();
        beam_search<<<grid_size,block_size_beam,0,stream2>>>(
            beam_args,
            (state_t *)gpubuf->states,
            gpubuf->moves,
            (beam_element_t *)gpubuf->beam_vector,
            beam_cut,
            fixed_stay_score,
            1.0f
        );
        checkCudaError();
        fwd_post_scan<<<grid_size,block_size,0,stream1>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC);
        checkCudaError();
        cudaStreamSynchronize(stream2);
        checkCudaError();
        cudaStreamSynchronize(stream1);
        checkCudaError();
        compute_qual_data<<<grid_size,block_size_gen,0,stream2>>>(
            beam_args,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            1.0f
        );
        checkCudaError();
        generate_sequence<<<grid_size,block_size_gen,0,stream2>>>(
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
        cudaStreamSynchronize(stream2);
        checkCudaError();
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("decode completed in %f secs", elapsed);

    // copy beam_search results
    cudaMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(*sequence, gpubuf->sequence, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(*qstring, gpubuf->qstring, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaStreamDestroy(stream1);
    checkCudaError();
    cudaStreamDestroy(stream2);
    checkCudaError();
}

// misc stuff for testing //////////////////////////////////////////////////////
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

void write_gpubuf_cuda(
    const uint64_t T,
    const uint64_t N,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
) {
    const int num_states = pow(NUM_BASES, state_len);

    float *bwd_NTC = (float *)malloc(N * (T + 1) * num_states * sizeof(float));
    MALLOC_CHK(bwd_NTC);
    float *post_NTC = (float *)malloc(N * (T + 1) * num_states * sizeof(float));
    MALLOC_CHK(post_NTC);
    state_t *states = (state_t *)malloc(N * T * sizeof(state_t));
    MALLOC_CHK(states);
    float *qual_data = (float *)malloc(N * T * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);
    float *base_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(base_probs);
    float *total_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(total_probs);

    // copy scan results
    cudaMemcpy(bwd_NTC, gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(post_NTC, gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states, cudaMemcpyDeviceToHost);
    checkCudaError();

    // copy intermediate
    cudaMemcpy(states, gpubuf->states, sizeof(state_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(total_probs, gpubuf->total_probs, sizeof(float) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(qual_data, gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(base_probs, gpubuf->base_probs, sizeof(float) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    // write results
    FILE *fp;

    fp = fopen("bwd_NTC.blob", "w");
    F_CHK(fp, "bwd_NTC.blob");
    if (fwrite(bwd_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    F_CHK(fp, "post_NTC.blob");
    if (fwrite(post_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    // write beam results
    fp = fopen("qual_data.blob", "w");
    F_CHK(fp, "qual_data.blob");
    if (fwrite(qual_data, sizeof(float), N * T * NUM_BASES, fp) != N * T * NUM_BASES) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    F_CHK(fp, "base_probs.blob");
    if (fwrite(base_probs, sizeof(float), N * T, fp) != N * T) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    F_CHK(fp, "total_probs.blob");
    if (fwrite(total_probs, sizeof(float), N * T, fp) != N * T) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    // cleanup
    free(bwd_NTC);
    free(post_NTC);
    free(states);
    free(qual_data);
    free(base_probs);
    free(total_probs);
}
////////////////////////////////////////////////////////////////////////////////