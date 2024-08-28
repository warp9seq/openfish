#pragma once

#include "decode.h"
#include "crf_model.h"

#include <torch/torch.h>

void decode_gpu(const int target_threads, const torch::Tensor& scores, std::vector<DecodedChunk>& chunk_results, const int num_chunks, const CRFModelConfig* config, const DecoderOptions* options, const int runner_idx);