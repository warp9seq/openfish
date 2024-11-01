#include "error.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    const int T = 1666;
    const int N = strtol(argv[3], NULL, 10);
    const int state_len = strtol(argv[4], NULL, 10);
    const int C = pow(4, state_len) * 4;

    set_openfish_log_level(OPENFISH_LOG_DBUG);
    
    size_t scores_len = T * N * C;
#if defined HAVE_CUDA || defined HAVE_HIP
    const int elem_size = sizeof(uint16_t);
    openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(T, N, state_len);
#else
    const int elem_size = sizeof(float);
#endif
    void *scores = calloc(scores_len, elem_size);
    MALLOC_CHK(scores);

    FILE *fp = fopen(argv[1], "rb");

    size_t result = fread(scores, elem_size, scores_len, fp);
    if (result != scores_len) {
        OPENFISH_ERROR("%s", "error reading score file");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    
    
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

#if defined HAVE_CUDA || defined HAVE_HIP
    openfish_decode_gpu(T, N, C, scores, state_len, &options, gpubuf, &moves, &sequence, &qstring);
#else
    int nthreads = 40;
    openfish_decode_cpu(T, N, C, nthreads, scores, state_len, &options, &moves, &sequence, &qstring);
#endif

    fp = fopen("moves.blob", "w");
    fwrite(moves, sizeof(uint8_t), N * T, fp);
    fclose(fp);

    fp = fopen("sequence.blob", "w");
    fwrite(sequence, sizeof(char), N * T, fp);
    fclose(fp);

    fp = fopen("qstring.blob", "w");
    fwrite(qstring, sizeof(char), N * T, fp);
    fclose(fp);

    free(moves);
    free(sequence);
    free(qstring);
    
    free(scores);

    openfish_gpubuf_free(gpubuf);

    return 0;
}
