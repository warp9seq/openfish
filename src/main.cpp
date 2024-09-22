#include "decode_cpu.h"
#include "decode_gpu.cuh"
#include "error.h"

#include <math.h>

int main(int argc, char* argv[]) {
    const int T = 1666;
    const int N = strtol(argv[3], NULL, 10);
    const int state_len = strtol(argv[4], NULL, 10);
    const int C = std::pow(4, state_len) * 4;
    
    size_t scores_len = T * N * C;
    float *scores = (float *)calloc(scores_len, sizeof(DTYPE_CPU));

    FILE *fp = fopen(argv[1], "rb");

    size_t result = fread(scores, sizeof(DTYPE_CPU), scores_len, fp);
    if (result != scores_len) {
        ERROR("%s", "error reading score file");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    
    const int target_threads = 40;
    const DecoderOptions options = DecoderOptions();
    std::vector<DecodedChunk> chunk_results = {};
    decode_gpu(T, N, C, target_threads, scores, chunk_results, state_len, &options);

    free(scores);

    return 0;
}