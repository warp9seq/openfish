#include "decode_cpu.h"
#include "crf_model.h"

int main(int argc, char* argv[]) {
    torch::Tensor scores;
    torch::load(scores, argv[1]);
    const CRFModelConfig config = load_crf_model_config(argv[2]);
    const int num_chunks = strtol(argv[3], NULL, 10);
    
    const int target_threads = 40;
    const DecoderOptions options = DecoderOptions();
    std::vector<DecodedChunk> chunk_results = {};
    decode_cpu(target_threads, scores, chunk_results, num_chunks, &config, &options);

    return 0;
}