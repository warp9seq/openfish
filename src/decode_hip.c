#include <openfish/openfish.h>
#include "openfish_defs.h"
#include "kernels_hip.h"
#include "error.h"
#include "hip_utils.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


// host_visible=1 is the scan-only variant for the GPU-scan + CPU-beam split
// (openfish_decode_gpu_scan -> openfish_decode_cpu_beam): it allocates ONLY the scan output tensors
// (bwd_NTC/post_NTC) in managed/unified memory so the CPU beam reads them with no copy, and leaves
// every GPU-beam buffer NULL -- the CPU beam allocates its own beam/qual/output scratch, so those
// would just be wasted device memory (beam_vector alone is batch*32*(T+1)*sizeof(beam_element_t)).
// host_visible=0 is the full fused-decode buffer: device-only, all buffers allocated.
static openfish_gpubuf_t *gpubuf_init_ex(
    int n_timesteps,
    int batch_size,
    int state_len,
    int host_visible
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(calloc(1, sizeof(openfish_gpubuf_t))); // unused ptrs -> NULL
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);
    const size_t scan_bytes = sizeof(float) * batch_size * (n_timesteps + 1) * num_states;

    // scan tensors (always needed)
    if (host_visible) {
        HIP_CHECK(hipMallocManaged((void **)&gpubuf->bwd_NTC, scan_bytes));
        HIP_CHECK(hipMallocManaged((void **)&gpubuf->post_NTC, scan_bytes));
        // Dedicated stream so openfish_decode_gpu_scan syncs only its own work (hipStreamSynchronize)
        // rather than the whole device -- it runs concurrently with the caller's inference stream.
        hipStream_t s;
        HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        gpubuf->stream = (void *)s;
        return gpubuf;  // scan-only: CPU beam owns its own beam/qual/output buffers
    }
    HIP_CHECK(hipMalloc((void **)&gpubuf->bwd_NTC, scan_bytes));
    HIP_CHECK(hipMalloc((void **)&gpubuf->post_NTC, scan_bytes));

    // return buffers
    HIP_CHECK(hipMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps));
    HIP_CHECK(hipMalloc((void **)&gpubuf->sequence, sizeof(char) * batch_size * n_timesteps));
    HIP_CHECK(hipMalloc((void **)&gpubuf->qstring, sizeof(char) * batch_size * n_timesteps));

    // beamsearch buffers
    HIP_CHECK(hipMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1)));
    HIP_CHECK(hipMalloc((void **)&gpubuf->states, sizeof(state_t) * batch_size * n_timesteps));
    HIP_CHECK(hipMalloc((void **)&gpubuf->qual_data, sizeof(float) * batch_size * n_timesteps * NUM_BASES));
    HIP_CHECK(hipMalloc((void **)&gpubuf->base_probs, sizeof(float) * batch_size * n_timesteps));
    HIP_CHECK(hipMalloc((void **)&gpubuf->total_probs, sizeof(float) * batch_size * n_timesteps));

    return gpubuf;
}

openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    return gpubuf_init_ex(n_timesteps, batch_size, state_len, 0);
}

openfish_gpubuf_t *openfish_gpubuf_init_hostvis(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    return gpubuf_init_ex(n_timesteps, batch_size, state_len, 1);
}

void openfish_gpubuf_free(
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

    if (gpubuf->stream) HIP_CHECK(hipStreamDestroy((hipStream_t)gpubuf->stream));

    free(gpubuf);
}

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
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

    OPENFISH_LOG_TRACE("scores tensor dim (NTC): %d, %d, %d", batch_size, n_timesteps, n_channels);

    scan_params_t scan_args = {0};
    scan_args.n_timesteps = n_timesteps;
    scan_args.batch_size = batch_size;
    scan_args.n_channels = n_channels;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;
    scan_args.score_scale = score_scale;

    // init results
    *moves = (uint8_t *)malloc(batch_size * n_timesteps * sizeof(uint8_t));
    MALLOC_CHK(*moves);
    *sequence = (char *)malloc(batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*sequence);
    *qstring = (char *)malloc(batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*qstring);

    HIP_CHECK(hipMemset(gpubuf->moves, 0, sizeof(uint8_t) * batch_size * n_timesteps));
    HIP_CHECK(hipMemset(gpubuf->sequence, 0, sizeof(char) * batch_size * n_timesteps));
    HIP_CHECK(hipMemset(gpubuf->qstring, 0, sizeof(char) * batch_size * n_timesteps));

    const int num_state_bits = (int)log2((double)num_states);
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    beam_params_t beam_args = {0};
    beam_args.n_timesteps = n_timesteps;
    beam_args.batch_size = batch_size;
    beam_args.n_channels = n_channels;
    beam_args.num_state_bits = num_state_bits;

    // bwd scan
    // fwd + post scan
    // beam search

    // the compact_offsets prefix-sum view overlays cand_scratch in the beam_search kernel,
    // so the int offsets must fit within the bool bloom-filter storage.
    ASSERT(MAX_BEAM_CANDIDATES * sizeof(int) <= HASH_PRESENT_BITS * sizeof(bool));

    // scores are read (and dequantized via score_scale) in three kernels; instantiate each on the
    // score element type. the f16 path passes score_scale = 1.0 and is numerically unchanged.
    OPENFISH_LOG_TRACE("bwd scan / beam search / fwd + post scan (score_dtype=%d)...", (int)score_dtype);
    if (score_dtype == OPENFISH_SCORE_I8) {
        bwd_scan<int8_t><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
        checkKernel();

        beam_search<int8_t><<<grid_size,block_size_beam,num_states*sizeof(float)>>>(
            beam_args, scores_NTC, gpubuf->bwd_NTC,
            (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
            beam_cut, fixed_stay_score, score_scale
        );
        checkKernel();

        fwd_post_scan<int8_t><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
        checkKernel();
    } else {
        bwd_scan<half><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
        checkKernel();

        beam_search<half><<<grid_size,block_size_beam,num_states*sizeof(float)>>>(
            beam_args, scores_NTC, gpubuf->bwd_NTC,
            (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
            beam_cut, fixed_stay_score, score_scale
        );
        checkKernel();

        fwd_post_scan<half><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
        checkKernel();
    }

    OPENFISH_LOG_TRACE("%s", "compute qual data...");
    compute_qual_data<<<grid_size,block_size_gen>>>(
        beam_args,
        gpubuf->post_NTC,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        1.0f
    );
    checkKernel();

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
    checkKernel();

    // copy beam_search results
    HIP_CHECK(hipMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(*sequence, gpubuf->sequence, sizeof(char) * batch_size * n_timesteps, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(*qstring, gpubuf->qstring, sizeof(char) * batch_size * n_timesteps, hipMemcpyDeviceToHost));
}

// GPU scan only (bwd_scan + fwd_post_scan): fills gpubuf->bwd_NTC and gpubuf->post_NTC, no beam
// and no host result buffers. Same kernels as openfish_decode_gpu (beam_search is skipped -- it
// only reads bwd_NTC, so dropping it leaves both scan outputs correct). Pair with
// openfish_decode_cpu_beam on a unified-memory GPU (the CPU reads the posteriors with no copy).
void openfish_decode_gpu_scan(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf
) {
    const int num_states = pow(NUM_BASES, state_len);

    // calculate grid / block dims (mirrors openfish_decode_gpu)
    const int target_block_width = (int)ceil(sqrt((float)num_states));
    int block_width = 2;
    while (block_width < target_block_width) {
        block_width *= 2;
    }

    dim3 block_size(block_width, block_width, 1);
    dim3 grid_size(batch_size, 1, 1);

    scan_params_t scan_args = {0};
    scan_args.n_timesteps = n_timesteps;
    scan_args.batch_size = batch_size;
    scan_args.n_channels = n_channels;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;
    scan_args.score_scale = score_scale;

    hipStream_t stream = gpubuf->stream ? (hipStream_t)gpubuf->stream : (hipStream_t)0;

    OPENFISH_LOG_TRACE("gpu scan-only: bwd scan / fwd + post scan (score_dtype=%d)...", (int)score_dtype);
    if (score_dtype == OPENFISH_SCORE_I8) {
        bwd_scan<int8_t><<<grid_size,block_size,0,stream>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
        checkKernel();

        fwd_post_scan<int8_t><<<grid_size,block_size,0,stream>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
        checkKernel();
    } else {
        bwd_scan<half><<<grid_size,block_size,0,stream>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
        checkKernel();

        fwd_post_scan<half><<<grid_size,block_size,0,stream>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
        checkKernel();
    }

    // No trailing device->host copy here (unlike openfish_decode_gpu). Sync only this scan's stream so
    // the caller's inference stream keeps running; the caller then reads gpubuf->post_NTC / bwd_NTC.
    HIP_CHECK(hipStreamSynchronize(stream));
}

