#include <openfish/openfish.h>
#include "decode.h"
#include "scan_hip.h"
#include "beam_search_hip.h"
#include "error.h"
#include "hip_utils.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


openfish_gpubuf_t *openfish_gpubuf_init(
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

void openfish_gpubuf_free(
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

void openfish_decode_gpu(
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

#ifdef BENCH
    // per-kernel timing breakdown, accumulated across all decode calls (bench builds only)
    static double t_bwd = 0, t_beam = 0, t_fwd = 0, t_qual = 0, t_gen = 0;
    static int n_calls = 0;
    hipEvent_t ev0, ev1;
    ret = hipEventCreate(&ev0); checkHipError(); HIP_CHECK(ret);
    ret = hipEventCreate(&ev1); checkHipError(); HIP_CHECK(ret);
    float ev_ms = 0;
    ++n_calls;
    #define TIME_KERNEL(acc, launch) do { \
        ret = hipEventRecord(ev0, 0); checkHipError(); HIP_CHECK(ret); launch; checkHipError(); \
        ret = hipEventRecord(ev1, 0); checkHipError(); HIP_CHECK(ret); \
        ret = hipEventSynchronize(ev1); checkHipError(); HIP_CHECK(ret); \
        ret = hipEventElapsedTime(&ev_ms, ev0, ev1); checkHipError(); HIP_CHECK(ret); (acc) += ev_ms; \
    } while (0)
#else
    #define TIME_KERNEL(acc, launch) do { launch; checkHipError(); \
        ret = hipDeviceSynchronize(); checkHipError(); HIP_CHECK(ret); } while (0)
#endif

    OPENFISH_LOG_TRACE("%s", "bwd scan...");
    TIME_KERNEL(t_bwd, (bwd_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC)));

    OPENFISH_LOG_TRACE("%s", "beam search...");
    // dynamic shared memory holds the back-guide sort scratch (num_states floats)
    TIME_KERNEL(t_beam, (beam_search<<<grid_size,block_size_beam,num_states*sizeof(float)>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->moves,
        (beam_element_t *)gpubuf->beam_vector,
        beam_cut,
        fixed_stay_score,
        1.0f
    )));

    OPENFISH_LOG_TRACE("%s", "fwd + post scan...");
    TIME_KERNEL(t_fwd, (fwd_post_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC)));

    OPENFISH_LOG_TRACE("%s", "compute qual data...");
    TIME_KERNEL(t_qual, (compute_qual_data<<<grid_size,block_size_gen>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        1.0f
    )));

    OPENFISH_LOG_TRACE("%s", "gen sequence...");
    TIME_KERNEL(t_gen, (generate_sequence<<<grid_size,block_size_gen>>>(
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
    )));

#ifdef BENCH
    ret = hipEventDestroy(ev0); checkHipError(); HIP_CHECK(ret);
    ret = hipEventDestroy(ev1); checkHipError(); HIP_CHECK(ret);
    OPENFISH_LOG_DEBUG("kernel ms totals after %d calls: bwd=%.1f beam=%.1f fwd_post=%.1f qual=%.1f gen=%.1f",
        n_calls, t_bwd, t_beam, t_fwd, t_qual, t_gen);
#endif
    #undef TIME_KERNEL

    // copy beam_search results
    ret = hipMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
	ret = hipMemcpy(*sequence, gpubuf->sequence, sizeof(char) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(*qstring, gpubuf->qstring, sizeof(char) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
}

