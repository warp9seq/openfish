#include "decode_hip.h"
#include "scan_hip.h"
#include "beam_search_hip.h"
#include "error.h"
#include "hip_utils.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


openfish_gpubuf_t *gpubuf_init_hip(
    const int T,
    const int N,
    const int state_len
) {
    hipError_t ret;
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    ret = hipMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states);
	checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states);
	checkHipError(); HIP_CHECK(ret);

    // return buffers
    ret = hipMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * N * T);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->sequence, sizeof(char) * N * T);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->qstring, sizeof(char) * N * T);
    checkHipError(); HIP_CHECK(ret);

    // beamsearch buffers
    ret = hipMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->states, sizeof(state_t) * N * T);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->base_probs, sizeof(float) * N * T);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->total_probs, sizeof(float) * N * T);
    checkHipError(); HIP_CHECK(ret);

    return gpubuf;
}

void gpubuf_free_hip(
    openfish_gpubuf_t *gpubuf
) {
    hipError_t ret;
    ret = hipFree(gpubuf->bwd_NTC);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->post_NTC);
    checkHipError(); HIP_CHECK(ret);

    ret = hipFree(gpubuf->moves);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->sequence);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->qstring);
    checkHipError(); HIP_CHECK(ret);

    ret = hipFree(gpubuf->beam_vector);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->states);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->qual_data);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->base_probs);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->total_probs);
    checkHipError(); HIP_CHECK(ret);

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
    hipError_t ret;
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
	dim3 grid_size(N, 1, 1);

    OPENFISH_LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

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

    ret = hipMemset(gpubuf->moves, 0, sizeof(uint8_t) * N * T);
	checkHipError(); HIP_CHECK(ret);
    ret = hipMemset(gpubuf->sequence, 0, sizeof(char) * N * T);
	checkHipError(); HIP_CHECK(ret);
    ret = hipMemset(gpubuf->qstring, 0, sizeof(char) * N * T);
	checkHipError(); HIP_CHECK(ret);

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

    // bwd scan
    // fwd + post scan
    // beam search

    OPENFISH_LOG_TRACE("%s", "bwd scan...");
    bwd_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC);
    checkHipError(); 
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);

    OPENFISH_LOG_TRACE("%s", "beam search...");
    beam_search<<<grid_size,block_size_beam>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->moves,
        (beam_element_t *)gpubuf->beam_vector,
        beam_cut,
        fixed_stay_score,
        1.0f
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);

    OPENFISH_LOG_TRACE("%s", "fwd + post scan...");
    fwd_post_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC);
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);

    OPENFISH_LOG_TRACE("%s", "compute qual data...");
    compute_qual_data<<<grid_size,block_size_gen>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        1.0f
    );
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);
    
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
    checkHipError();
    ret = hipDeviceSynchronize();
    checkHipError(); HIP_CHECK(ret);

    // copy beam_search results
    ret = hipMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
	ret = hipMemcpy(*sequence, gpubuf->sequence, sizeof(char) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(*qstring, gpubuf->qstring, sizeof(char) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
}

// misc stuff for testing //////////////////////////////////////////////////////
void set_device_hip(
    int device
) {
    hipError_t ret;
    ret = hipSetDevice(device);
	checkHipError(); HIP_CHECK(ret);
}

void *upload_scores_to_hip(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
) {
    hipError_t ret;
    void *scores_TNC_gpu;

    ret = hipMalloc((void **)&scores_TNC_gpu, sizeof(half) * T * N * C);
	checkHipError(); HIP_CHECK(ret);

	ret = hipMemcpy(scores_TNC_gpu, scores_TNC, sizeof(half) * T * N * C, hipMemcpyHostToDevice);
	checkHipError(); HIP_CHECK(ret);

    return scores_TNC_gpu;
}

void free_scores_hip(
    void *scores_TNC_gpu
) {
    hipError_t ret;
    ret = hipFree(scores_TNC_gpu);
	checkHipError(); HIP_CHECK(ret);
}

void write_gpubuf_hip(
    const uint64_t T,
    const uint64_t N,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
) {
    hipError_t ret;
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
    ret = hipMemcpy(bwd_NTC, gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
	ret = hipMemcpy(post_NTC, gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);

    // copy intermediate
    ret = hipMemcpy(states, gpubuf->states, sizeof(state_t) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);

    ret = hipMemcpy(total_probs, gpubuf->total_probs, sizeof(float) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);

    ret = hipMemcpy(qual_data, gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);

    ret = hipMemcpy(base_probs, gpubuf->base_probs, sizeof(float) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);

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