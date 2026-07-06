#pragma once

#include <openfish/openfish.h>
#include "openfish_defs.h"
#include "error.h"
#include "cuda_utils.h"

#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static inline void set_device_cuda(int device) {
    cudaSetDevice(device);
    checkCudaError();
}

static inline void *upload_scores_to_cuda(
    const int n_timesteps,
    const int batch_size,
    const int n_channels,
    const void *scores_TNC
) {
    void *scores_TNC_gpu;
    cudaMalloc((void **)&scores_TNC_gpu, sizeof(half) * n_timesteps * batch_size * n_channels);
    checkCudaError();
    cudaMemcpy(scores_TNC_gpu, scores_TNC, sizeof(half) * n_timesteps * batch_size * n_channels, cudaMemcpyHostToDevice);
    checkCudaError();
    return scores_TNC_gpu;
}

static inline void free_scores_cuda(void *scores_TNC_gpu) {
    cudaFree(scores_TNC_gpu);
    checkCudaError();
}

static inline void write_gpubuf_cuda(
    const uint64_t n_timesteps,
    const uint64_t batch_size,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
) {
    const int num_states = pow(NUM_BASES, state_len);

    float *bwd_NTC = (float *)malloc(batch_size * (n_timesteps + 1) * num_states * sizeof(float));
    MALLOC_CHK(bwd_NTC);
    float *post_NTC = (float *)malloc(batch_size * (n_timesteps + 1) * num_states * sizeof(float));
    MALLOC_CHK(post_NTC);
    state_t *states = (state_t *)malloc(batch_size * n_timesteps * sizeof(state_t));
    MALLOC_CHK(states);
    float *qual_data = (float *)malloc(batch_size * n_timesteps * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);
    float *base_probs = (float *)malloc(batch_size * n_timesteps * sizeof(float));
    MALLOC_CHK(base_probs);
    float *total_probs = (float *)malloc(batch_size * n_timesteps * sizeof(float));
    MALLOC_CHK(total_probs);

    cudaMemcpy(bwd_NTC, gpubuf->bwd_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(post_NTC, gpubuf->post_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(states, gpubuf->states, sizeof(state_t) * batch_size * n_timesteps, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(total_probs, gpubuf->total_probs, sizeof(float) * batch_size * n_timesteps, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(qual_data, gpubuf->qual_data, sizeof(float) * batch_size * n_timesteps * NUM_BASES, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(base_probs, gpubuf->base_probs, sizeof(float) * batch_size * n_timesteps, cudaMemcpyDeviceToHost);
    checkCudaError();

    FILE *fp;

    fp = fopen("bwd_NTC.blob", "w");
    F_CHK(fp, "bwd_NTC.blob");
    if (fwrite(bwd_NTC, sizeof(float), batch_size * (n_timesteps + 1) * num_states, fp) != batch_size * (n_timesteps + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    F_CHK(fp, "post_NTC.blob");
    if (fwrite(post_NTC, sizeof(float), batch_size * (n_timesteps + 1) * num_states, fp) != batch_size * (n_timesteps + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("qual_data.blob", "w");
    F_CHK(fp, "qual_data.blob");
    if (fwrite(qual_data, sizeof(float), batch_size * n_timesteps * NUM_BASES, fp) != batch_size * n_timesteps * NUM_BASES) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    F_CHK(fp, "base_probs.blob");
    if (fwrite(base_probs, sizeof(float), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    F_CHK(fp, "total_probs.blob");
    if (fwrite(total_probs, sizeof(float), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    free(bwd_NTC);
    free(post_NTC);
    free(states);
    free(qual_data);
    free(base_probs);
    free(total_probs);
}
