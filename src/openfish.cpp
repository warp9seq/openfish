#include "openfish.h"

void decode(
    const int T,
    const int N,
    const int C,
    const int target_threads,
    float *scores_TNC,
    std::vector<DecodedChunk>& chunk_results,
    const int state_len,
    const DecoderOptions *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
#ifdef HAVE_CUDA
    decode_cuda(T, N, C, target_threads, scores_TNC, chunk_results, state_len, options, moves, sequence, qstring);
#else
    decode_cpu(T, N, C, target_threads, scores_TNC, chunk_results, state_len, options, moves, sequence, qstring);
#endif
}