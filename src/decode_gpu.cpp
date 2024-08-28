#include "decode_gpu.h"
#include "decode.h"
#include "error.h"

void decode_gpu(const int target_threads, const torch::Tensor& scores, std::vector<DecodedChunk>& chunk_results, const int num_chunks, const CRFModelConfig* config, const DecoderOptions* options, const int runner_idx) {
    ERROR("%s", "not implemented yet");
    exit(EXIT_FAILURE);
}
