#include <openfish/openfish.h>

#include "error.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    const int T = 1666;
    const int N = strtol(argv[3], NULL, 10);
    const int state_len = strtol(argv[4], NULL, 10);
    const int C = pow(4, state_len) * 4;
    
    size_t scores_len = T * N * C;
    float *scores = (float *)calloc(scores_len, sizeof(float));
    MALLOC_CHK(scores);

    FILE *fp = fopen(argv[1], "rb");

    size_t result = fread(scores, sizeof(float), scores_len, fp);
    if (result != scores_len) {
        ERROR("%s", "error reading score file");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    
    
    decoder_opts_t options = DECODER_INIT;

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
    
    const int target_threads = 40;

    uint8_t *moves;
    char *sequence;
    char *qstring;
    decode(T, N, C, target_threads, scores, state_len, &options, &moves, &sequence, &qstring);

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

    return 0;
}