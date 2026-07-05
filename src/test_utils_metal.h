#pragma once

// Test-harness helpers for the Metal backend, mirroring test_utils_cuda.h / test_utils_hip.h.
// These are implemented in decode_metal.mm and exposed with C linkage so main.c (compiled as C)
// can call them.

#include <openfish/openfish.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void set_device_metal(int device);

// scores_TNC is host float16 [T,N,C] (same as the CUDA/ROCm GPU path).
// returns an opaque handle (an MTLBuffer) to pass as scores_TNC into openfish_decode_gpu.
void *upload_scores_to_metal(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_TNC
);

void free_scores_metal(void *scores_gpu);

#ifdef DEBUG
void write_gpubuf_metal(
    uint64_t n_timesteps,
    uint64_t batch_size,
    int state_len,
    const openfish_gpubuf_t *gpubuf
);
#endif

#ifdef __cplusplus
}
#endif
