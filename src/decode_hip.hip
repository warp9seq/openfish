#include "decode_hip.h"
#include "scan_hip.h"
#include "beam_search_hip.h"
#include "error.h"
#include "hip_utils.hpp"
#include "misc.h"

#include <hip/hip_runtime.h>

void decode_hip(
    const int T,
    const int N,
    const int C,
    float *scores_TNC,
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

    fprintf(stderr, "chosen block_dims: %d x %d for num_states %d\n", block_width, block_width, num_states);
    fprintf(stderr, "chosen grid_len: %d for batch size %d\n", grid_len, N);

    double t0, t1, elapsed;
    dim3 block_size(block_width, block_width, 1);
    dim3 block_size_beam(MAX_BEAM_WIDTH, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(grid_len, 1, 1);

    // expect input already transformed
    // scores_TNC = scores_TNC.to(torch::kCPU).to(float).transpose(0, 1).contiguous();
    
    const uint64_t num_scan_elem = N * (T + 1) * num_states;

    LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

#ifdef DEBUG
    float *bwd_NTC = (float *)malloc(num_scan_elem * sizeof(float));
    MALLOC_CHK(bwd_NTC);

    float *post_NTC = (float *)malloc(num_scan_elem * sizeof(float));
    MALLOC_CHK(post_NTC);
#endif

    float *scores_TNC_gpu;
    float *bwd_NTC_gpu;
    float *post_NTC_gpu;

    // copy score tensor over
    HIP_CHECK(hipMalloc((void **)&scores_TNC_gpu, sizeof(float) * T * N * C));

	HIP_CHECK(hipMemcpy(scores_TNC_gpu, scores_TNC, sizeof(float) * T * N * C, hipMemcpyHostToDevice));

    // init scan tensors
    HIP_CHECK(hipMalloc((void **)&bwd_NTC_gpu, sizeof(float) * num_scan_elem));
    HIP_CHECK(hipMalloc((void **)&post_NTC_gpu, sizeof(float) * num_scan_elem));

    scan_args_t scan_args = {0};
    scan_args.scores_in = scores_TNC_gpu;
    scan_args.T = T;
    scan_args.N = N;
    scan_args.C = C;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;

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
        bwd_scan<<<grid_size,block_size>>>(scan_args, bwd_NTC_gpu);
        HIP_CHECK(hipDeviceSynchronize());
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
        fwd_post_scan<<<grid_size,block_size>>>(scan_args, bwd_NTC_gpu, post_NTC_gpu);
        HIP_CHECK(hipDeviceSynchronize());
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    fprintf(stderr, "fwd scan completed in %f secs\n", elapsed);

    // beam search

    // init results
    *moves = (uint8_t *)malloc(N * T * sizeof(uint8_t));
    MALLOC_CHK(*moves);

    *sequence = (char *)malloc(N * T * sizeof(char));
    MALLOC_CHK(*sequence);

    *qstring = (char *)malloc(N * T * sizeof(char));
    MALLOC_CHK(*qstring);

#ifdef DEBUG
    state_t *states = (state_t *)malloc(N * T * sizeof(state_t));
    MALLOC_CHK(states);

    float *qual_data = (float *)malloc(N * T * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);

    float *base_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(base_probs);

    float *total_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(total_probs);
#endif

    // intermediate results
    uint8_t *moves_gpu;
    char *sequence_gpu;
    char *qstring_gpu;

    HIP_CHECK(hipMalloc((void **)&moves_gpu, sizeof(uint8_t) * N * T));
    HIP_CHECK(hipMemset(moves_gpu, 0, sizeof(uint8_t) * N * T));
    HIP_CHECK(hipMalloc((void **)&sequence_gpu, sizeof(char) * N * T));
    HIP_CHECK(hipMemset(sequence_gpu, 0, sizeof(char) * N * T));
    HIP_CHECK(hipMalloc((void **)&qstring_gpu, sizeof(char) * N * T));
    HIP_CHECK(hipMemset(qstring_gpu, 0, sizeof(char) * N * T));
    
    // intermediate
    beam_element_t *beam_vector_gpu;
    state_t *states_gpu;
    float *qual_data_gpu;
    float *base_probs_gpu;
    float *total_probs_gpu;

    HIP_CHECK(hipMalloc((void **)&beam_vector_gpu, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1)));
    HIP_CHECK(hipMalloc((void **)&states_gpu, sizeof(state_t) * N * T));
    HIP_CHECK(hipMalloc((void **)&qual_data_gpu, sizeof(float) * N * T * NUM_BASES));
    HIP_CHECK(hipMalloc((void **)&base_probs_gpu, sizeof(float) * N * T));
    HIP_CHECK(hipMalloc((void **)&total_probs_gpu, sizeof(float) * N * T));

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
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
        beam_search<<<grid_size,block_size_beam>>>(
            beam_args,
            states_gpu,
            moves_gpu,
            beam_vector_gpu,
            beam_cut,
            fixed_stay_score,
            1.0f
        );
        HIP_CHECK(hipDeviceSynchronize());
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
        compute_qual_data<<<grid_size,block_size_gen>>>(
            beam_args,
            states_gpu,
            qual_data_gpu,
            1.0f
        );
        HIP_CHECK(hipDeviceSynchronize());
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    fprintf(stderr, "compute quality data completed in %f secs\n", elapsed);

    t0 = realtime();
#ifdef BENCH
    for (int i = 0; i < n_batch; ++i)
#endif
    {
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
        HIP_CHECK(hipDeviceSynchronize());
    }
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    fprintf(stderr, "generate sequence completed in %f secs\n", elapsed);

    // copy beam_search results
    HIP_CHECK(hipMemcpy(*moves, moves_gpu, sizeof(uint8_t) * N * T, hipMemcpyDeviceToHost));
	HIP_CHECK(hipMemcpy(*sequence, sequence_gpu, sizeof(char) * N * T, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(*qstring, qstring_gpu, sizeof(char) * N * T, hipMemcpyDeviceToHost));

#ifdef DEBUG
    // copy scan results
    HIP_CHECK(hipMemcpy(bwd_NTC, bwd_NTC_gpu, sizeof(float) * num_scan_elem, hipMemcpyDeviceToHost));
	HIP_CHECK(hipMemcpy(post_NTC, post_NTC_gpu, sizeof(float) * num_scan_elem, hipMemcpyDeviceToHost));

    // copy intermediate
    HIP_CHECK(hipMemcpy(states, states_gpu, sizeof(state_t) * N * T, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(total_probs, total_probs_gpu, sizeof(float) * N * T, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(qual_data, qual_data_gpu, sizeof(float) * N * T * NUM_BASES, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(base_probs, base_probs_gpu, sizeof(float) * N * T, hipMemcpyDeviceToHost));

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
    fwrite(bwd_NTC, sizeof(float), num_scan_elem, fp);
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    fwrite(post_NTC, sizeof(float), num_scan_elem, fp);
    fclose(fp);

    // cleanup
    free(bwd_NTC);
    free(post_NTC);
#endif
    HIP_CHECK(hipFree(scores_TNC_gpu));
    HIP_CHECK(hipFree(bwd_NTC_gpu));
    HIP_CHECK(hipFree(post_NTC_gpu));
    HIP_CHECK(hipFree(moves_gpu));
    HIP_CHECK(hipFree(sequence_gpu));
    HIP_CHECK(hipFree(qstring_gpu));
    HIP_CHECK(hipFree(beam_vector_gpu));
    HIP_CHECK(hipFree(states_gpu));
    HIP_CHECK(hipFree(qual_data_gpu));
    HIP_CHECK(hipFree(base_probs_gpu));
    HIP_CHECK(hipFree(total_probs_gpu));
}
