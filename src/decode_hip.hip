#include "decode_hip.h"
#include "scan_hip.h"
#include "beam_search_hip.h"
#include "error.h"
#include "hip_utils.h"
#include "misc.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

openfish_gpubuf_t *gpubuf_init_hip(
    const int T,
    const int N,
    const int state_len
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    HIP_CHECK(hipMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states));
    HIP_CHECK(hipMalloc((void **)&gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states));

    // return buffers
    HIP_CHECK(hipMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * N * T));
    HIP_CHECK(hipMalloc((void **)&gpubuf->sequence, sizeof(char) * N * T));
    HIP_CHECK(hipMalloc((void **)&gpubuf->qstring, sizeof(char) * N * T));
    // beamsearch buffers
    HIP_CHECK(hipMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1)));
    HIP_CHECK(hipMalloc((void **)&gpubuf->states, sizeof(state_t) * N * T));
    HIP_CHECK(hipMalloc((void **)&gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES));
    HIP_CHECK(hipMalloc((void **)&gpubuf->base_probs, sizeof(float) * N * T));
    HIP_CHECK(hipMalloc((void **)&gpubuf->total_probs, sizeof(float) * N * T));

    return gpubuf;
}

void gpubuf_free_hip(
    openfish_gpubuf_t *gpubuf
) {
    HIP_CHECK(hipFree(gpubuf->bwd_NTC));
    HIP_CHECK(hipFree(gpubuf->post_NTC));

    HIP_CHECK(hipFree(gpubuf->moves));
    HIP_CHECK(hipFree(gpubuf->sequence));
    HIP_CHECK(hipFree(gpubuf->qstring));

    HIP_CHECK(hipFree(gpubuf->beam_vector));
    HIP_CHECK(hipFree(gpubuf->states));
    HIP_CHECK(hipFree(gpubuf->qual_data));
    HIP_CHECK(hipFree(gpubuf->base_probs));
    HIP_CHECK(hipFree(gpubuf->total_probs));

    free(gpubuf);
}

void decode_hip(
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
    HIP_CHECK(hipDeviceSynchronize());
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("bwd scan completed in %f secs", elapsed);

    // fwd + post scan
	t0 = realtime();
    fwd_post_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC);
    HIP_CHECK(hipDeviceSynchronize());
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

    HIP_CHECK(hipMemset(gpubuf->moves, 0, sizeof(uint8_t) * N * T));
    HIP_CHECK(hipMemset(gpubuf->sequence, 0, sizeof(char) * N * T));
    HIP_CHECK(hipMemset(gpubuf->qstring, 0, sizeof(char) * N * T));

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
    HIP_CHECK(hipDeviceSynchronize());
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
    HIP_CHECK(hipDeviceSynchronize());
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
    HIP_CHECK(hipDeviceSynchronize());
	// end timing
	t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("generate sequence completed in %f secs", elapsed);

    // copy beam_search results
    HIP_CHECK(hipMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * N * T, hipMemcpyDeviceToHost));
	HIP_CHECK(hipMemcpy(*sequence, gpubuf->sequence, sizeof(char) * N * T, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(*qstring, gpubuf->qstring, sizeof(char) * N * T, hipMemcpyDeviceToHost));
}

// misc stuff for testing //////////////////////////////////////////////////////
void *upload_scores_to_hip(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
) {
    void *scores_TNC_gpu;

    HIP_CHECK(hipMalloc((void **)&scores_TNC_gpu, sizeof(half) * T * N * C));
	HIP_CHECK(hipMemcpy(scores_TNC_gpu, scores_TNC, sizeof(half) * T * N * C, hipMemcpyHostToDevice));

    return scores_TNC_gpu;
}

void free_scores_hip(
    void *scores_TNC_gpu
) {
    HIP_CHECK(hipFree(scores_TNC_gpu));
}

void write_gpubuf_hip(
    const int T,
    const int N,
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
    HIP_CHECK(hipMemcpy(bwd_NTC, gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states, hipMemcpyDeviceToHost));
	HIP_CHECK(hipMemcpy(post_NTC, gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states, hipMemcpyDeviceToHost));

    // copy intermediate
    HIP_CHECK(hipMemcpy(states, gpubuf->states, sizeof(state_t) * N * T, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(total_probs, gpubuf->total_probs, sizeof(float) * N * T, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(qual_data, gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(base_probs, gpubuf->base_probs, sizeof(float) * N * T, hipMemcpyDeviceToHost));

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
    fwrite(bwd_NTC, sizeof(float), N * (T + 1) * num_states, fp);
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    fwrite(post_NTC, sizeof(float), N * (T + 1) * num_states, fp);
    fclose(fp);

    // cleanup
    free(states);
    free(qual_data);
    free(base_probs);
    free(total_probs);

    free(bwd_NTC);
    free(post_NTC);
}
////////////////////////////////////////////////////////////////////////////////