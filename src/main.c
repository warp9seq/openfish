#include "error.h"
#include "misc.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined HAVE_CUDA
#include "test_utils_cuda.h"
#endif

#if defined HAVE_ROCM
#include "test_utils_hip.h"
#endif

#if defined HAVE_METAL
#include "test_utils_metal.h"
#endif

#if defined(HAVE_CUDA) || defined(HAVE_ROCM) || defined(HAVE_METAL)
#define HAVE_GPU 1
#endif

int main(int argc, char* argv[]) {
#if defined DEBUG
    if (argc < 4) {
        fprintf(stderr,"Usage: %s <scores.blob> <BATCH_SIZE> <STATE_LEN>\n", argv[0]);
        fprintf(stderr,"e.g. %s test/blobs/fast_1000c_scores_TNC.blob models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 1000 3\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    set_openfish_log_level(OPENFISH_LOG_DBUG);

    const int device = 0;
#if defined HAVE_CUDA
    set_device_cuda(device);
#elif defined HAVE_ROCM
    set_device_hip(device);
#elif defined HAVE_METAL
    set_device_metal(device);
#endif

    OPENFISH_LOG_DEBUG("simulating batches on device %d", device);

    const int n_timesteps = 1666;
    const int batch_size = strtol(argv[2], NULL, 10);
    ASSERT(batch_size > 0);
    const int state_len = strtol(argv[3], NULL, 10);
    ASSERT(state_len > 0);
    const int n_channels = pow(4, state_len) * 4;

    // optional score dtype selector: 0 = native float (f16 GPU / f32 CPU), 1 = int8 (GPU only).
    // int8 mode reuses the native score blob and quantizes it on the host, then decodes with the
    // dequant multiplier (5/127) so the output can be compared against the native decode.
    const int req_dtype = (argc > 4) ? (int)strtol(argv[4], NULL, 10) : 0;
    // optional score_scale override (arg 6): defaults to 1.0 (f16) or 5/127 (i8) when < 0. test-only.
    const float req_scale = (argc > 5) ? (float)strtod(argv[5], NULL) : -1.0f;

    // read scores from file
    size_t scores_len = n_timesteps * batch_size * n_channels;
#if defined HAVE_CUDA || defined HAVE_ROCM || defined HAVE_METAL
    const int elem_size = sizeof(uint16_t);
#else
    const int elem_size = sizeof(float);
#endif
    void *scores = calloc(scores_len, elem_size);
    MALLOC_CHK(scores);

    FILE *fp = fopen(argv[1], "rb");
    F_CHK(fp, argv[1]);

    size_t result = fread(scores, elem_size, scores_len, fp);
    if (result != scores_len) {
        OPENFISH_ERROR("%s: %s", "error reading score file", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    // The test blobs are stored TNC (timestep-major); the library now consumes NTC
    // (batch-major). Transpose once here so the harness matches the new contract.
    // In production, slorado provides scores already in NTC and skips this step.
    void *scores_ntc = calloc(scores_len, elem_size);
    MALLOC_CHK(scores_ntc);
    for (int t = 0; t < n_timesteps; ++t) {
        for (int n = 0; n < batch_size; ++n) {
            memcpy((char *)scores_ntc + ((size_t)(n * n_timesteps + t) * n_channels) * elem_size,
                   (char *)scores     + ((size_t)(t * batch_size + n) * n_channels) * elem_size,
                   (size_t)n_channels * elem_size);
        }
    }
    free(scores);
    scores = scores_ntc;

    // choose the score dtype passed to the decoder. default: native float, unscaled.
    openfish_score_dtype_t of_dtype = OPENFISH_SCORE_F16;
    float of_scale = (req_scale >= 0.0f) ? req_scale : 1.0f;
    int up_elem_size = elem_size;

    if (req_dtype == 1) {
#if defined HAVE_METAL
        OPENFISH_ERROR("%s", "int8 score dtype is not supported on the Metal path");
        exit(EXIT_FAILURE);
#else
        // quantize the native scores to int8: q = round(clamp(tanh(x)*5, -5, 5) * 127/5).
        // the stored value is already tanh(x)*5 (fp16 on the GPU path, fp32 on the CPU path),
        // so we just rescale and round.
        int8_t *q = (int8_t *)malloc(scores_len);
        MALLOC_CHK(q);
        for (size_t i = 0; i < scores_len; ++i) {
#if defined HAVE_CUDA || defined HAVE_ROCM
            float f = __half2float(((const half *)scores)[i]) * (127.0f / 5.0f);
#else
            float f = ((const float *)scores)[i] * (127.0f / 5.0f);
#endif
            f = roundf(f);
            if (f > 127.0f) f = 127.0f;
            if (f < -127.0f) f = -127.0f;
            q[i] = (int8_t)f;
        }
        free(scores);
        scores = q;
        of_dtype = OPENFISH_SCORE_I8;
        of_scale = (req_scale >= 0.0f) ? req_scale : 5.0f / 127.0f;
        up_elem_size = sizeof(int8_t);
        OPENFISH_LOG_DEBUG("quantized scores to int8 (score_scale = %g)", of_scale);
#endif
    }

#if !defined HAVE_GPU
    (void)up_elem_size; // only consumed by the GPU upload helpers
#endif

    // upload scores to gpu
#if defined HAVE_CUDA
    void *scores_gpu = upload_scores_to_cuda(n_timesteps, batch_size, n_channels, scores, up_elem_size);
#elif defined HAVE_ROCM
    void *scores_gpu = upload_scores_to_hip(n_timesteps, batch_size, n_channels, scores, up_elem_size);
#elif defined HAVE_METAL
    void *scores_gpu = upload_scores_to_metal(n_timesteps, batch_size, n_channels, scores, up_elem_size);
#endif

#if defined HAVE_GPU
    openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(n_timesteps, batch_size, state_len);
#endif
    openfish_opt_t options = openfish_decoder_default_opts();

    // config mods from 4.2.0 models
    if (state_len == 3) { // fast
        options.q_scale = 0.97;
        options.q_shift = -1.8;
    } else if (state_len == 4) { // hac
        options.q_scale = 0.95;
        options.q_shift = -0.2;
    } else if (state_len == 5) { // sup
        options.q_scale = 0.95;
        options.q_shift = 0.5;
    }

    uint8_t *moves;
    char *sequence;
    char *qstring;

    double t0, t1, elapsed;
    t0 = openfish_realtime();
    
#ifdef BENCH
    int n_batch = 140; // simulate 20k reads
    if (state_len == 3)      n_batch = 700; // fast
    else if (state_len == 4) n_batch = 1725; // hac
    else if (state_len == 5) n_batch = 3425; // sup
    OPENFISH_LOG_DEBUG("simulating %d batches...", n_batch);
    for (int i = 0; i < n_batch; ++i) {
#endif

    // decode scores
#if defined HAVE_GPU
        openfish_decode_gpu(n_timesteps, batch_size, n_channels, scores_gpu, of_dtype, of_scale, state_len, &options, gpubuf, &moves, &sequence, &qstring);
#else
        int n_threads = 8;
        openfish_decode_cpu(n_timesteps, batch_size, n_channels, n_threads, scores, of_dtype, of_scale, state_len, &options, &moves, &sequence, &qstring);
#endif

#ifdef BENCH
        if (i + 1 != n_batch) {
            free(moves);
            free(sequence);
            free(qstring);
        }
    }
#endif

    // end timing
    t1 = openfish_realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("decode completed in %f secs", elapsed);

    // write results to file
    fp = fopen("moves.blob", "w");
    F_CHK(fp, "moves.blob");
    if (fwrite(moves, sizeof(uint8_t), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing moves file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("sequence.blob", "w");
    F_CHK(fp, "sequence.blob");
    if (fwrite(sequence, sizeof(char), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("qstring.blob", "w");
    F_CHK(fp, "qstring.blob");
    if (fwrite(qstring, sizeof(char), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing qstring file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    free(moves);
    free(sequence);
    free(qstring);

    free(scores);

#if defined DEBUG && defined HAVE_CUDA
    write_gpubuf_cuda(n_timesteps, batch_size, state_len, gpubuf);
#endif

#if defined DEBUG && defined HAVE_ROCM
    write_gpubuf_hip(n_timesteps, batch_size, state_len, gpubuf);
#endif

#if defined DEBUG && defined HAVE_METAL
    write_gpubuf_metal(n_timesteps, batch_size, state_len, gpubuf);
#endif

#if defined HAVE_GPU
    openfish_gpubuf_free(gpubuf);
#endif

#if defined HAVE_CUDA
    free_scores_cuda(scores_gpu);
#elif defined HAVE_ROCM
    free_scores_hip(scores_gpu);
#elif defined HAVE_METAL
    free_scores_metal(scores_gpu);
#endif
#endif
    return 0;
}