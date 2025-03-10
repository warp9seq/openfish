#include "error.h"
#include "misc.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#if defined HAVE_CUDA
#include "decode_cuda.h"
#endif

#if defined HAVE_ROCM
#include "decode_hip.h"
#endif

int main(int argc, char* argv[]) {
#if defined DEBUG
#pragma omp parallel
{
    if (argc < 4) {
        fprintf(stderr,"Usage: %s <scores.blob> <BATCH_SIZE> <STATE_LEN>\n", argv[0]);
        fprintf(stderr,"e.g. %s test/blobs/fast_1000c_scores_TNC.blob models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 1000 3\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    set_openfish_log_level(OPENFISH_LOG_DBUG);

    const int device = omp_get_thread_num();
#if defined HAVE_CUDA
    set_device_cuda(device);
#elif defined HAVE_ROCM
    set_device_hip(device);
#endif

    OPENFISH_LOG_DEBUG("simulating batches on device %d", device);

    const int T = 1666;
    const int N = strtol(argv[2], NULL, 10);
    assert(N > 0);
    const int state_len = strtol(argv[3], NULL, 10);
    assert(state_len > 0);
    const int C = pow(4, state_len) * 4;

    // read scores from file
    size_t scores_len = T * N * C;
#if defined HAVE_CUDA || defined HAVE_ROCM
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

    // upload scores to gpu
#if defined HAVE_CUDA
    void *scores_gpu = upload_scores_to_cuda(T, N, C, scores);
#elif defined HAVE_ROCM
    void *scores_gpu = upload_scores_to_hip(T, N, C, scores);
#endif

#if defined HAVE_CUDA || defined HAVE_ROCM
    openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(T, N, state_len);
#endif
    openfish_opt_t options = DECODER_INIT;

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
    t0 = realtime();
    
#ifdef BENCH
    int n_batch = 140; // simulate 20k reads
    if (state_len == 3)      n_batch = 14000; // fast
    else if (state_len == 4) n_batch = 34500; // hac
    else if (state_len == 5) n_batch = 68500; // sup
    OPENFISH_LOG_DEBUG("simulating %d batches...", n_batch);
    for (int i = 0; i < n_batch; ++i) {
#endif

    // decode scores
#if defined HAVE_CUDA || defined HAVE_ROCM
        openfish_decode_gpu(T, N, C, scores_gpu, state_len, &options, gpubuf, &moves, &sequence, &qstring);
#else
        int nthreads = 8;
        openfish_decode_cpu(T, N, C, nthreads, scores, state_len, &options, &moves, &sequence, &qstring);
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
    t1 = realtime();
    elapsed = t1 - t0;
    OPENFISH_LOG_DEBUG("decode completed in %f secs", elapsed);

    // write results to file
    fp = fopen("moves.blob", "w");
    F_CHK(fp, "moves.blob");
    if (fwrite(moves, sizeof(uint8_t), N * T, fp) != N * T) {
        fprintf(stderr, "error writing moves file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("sequence.blob", "w");
    F_CHK(fp, "sequence.blob");
    if (fwrite(sequence, sizeof(char), N * T, fp) != N * T) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("qstring.blob", "w");
    F_CHK(fp, "qstring.blob");
    if (fwrite(qstring, sizeof(char), N * T, fp) != N * T) {
        fprintf(stderr, "error writing qstring file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    free(moves);
    free(sequence);
    free(qstring);

    free(scores);

#if defined DEBUG && defined HAVE_CUDA
    write_gpubuf_cuda(T, N, state_len, gpubuf);
#endif

#if defined DEBUG && defined HAVE_ROCM
    write_gpubuf_hip(T, N, state_len, gpubuf);
#endif

#if defined HAVE_CUDA || defined HAVE_ROCM
    openfish_gpubuf_free(gpubuf);
#endif

#if defined HAVE_CUDA
    free_scores_cuda(scores_gpu);
#elif defined HAVE_ROCM
    free_scores_hip(scores_gpu);
#endif
}
#endif
    return 0;
}
