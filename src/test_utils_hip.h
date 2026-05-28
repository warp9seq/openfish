#pragma once

#include <openfish/openfish.h>
#include "decode.h"
#include "error.h"
#include "hip_utils.h"

#include <hip/hip_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static inline void set_device_hip(int device) {
    hipError_t ret;
    ret = hipSetDevice(device);
    checkHipError(); HIP_CHECK(ret);
}

static inline void *upload_scores_to_hip(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
) {
    hipError_t ret;
    void *scores_TNC_gpu;
    ret = hipMalloc((void **)&scores_TNC_gpu, sizeof(half) * T * N * C);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(scores_TNC_gpu, scores_TNC, sizeof(half) * T * N * C, hipMemcpyHostToDevice);
    checkHipError(); HIP_CHECK(ret);
    return scores_TNC_gpu;
}

static inline void free_scores_hip(void *scores_TNC_gpu) {
    hipError_t ret;
    ret = hipFree(scores_TNC_gpu);
    checkHipError(); HIP_CHECK(ret);
}

static inline void write_gpubuf_hip(
    const uint64_t T,
    const uint64_t N,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
) {
    hipError_t ret;
    const int num_states = pow(NUM_BASES, state_len);

    float *bwd_NTC = (float *)malloc(N * (T + 1) * num_states * sizeof(float));
    MALLOC_CHK(bwd_NTC);
    float *post_NTC = (float *)malloc(N * (T + 1) * num_states * sizeof(float));
    MALLOC_CHK(post_NTC);
    state_t *states = (state_t *)malloc(N * T * sizeof(state_t));
    MALLOC_CHK(states);
    float *qual_data = (float *)malloc(N * T * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);
    float *base_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(base_probs);
    float *total_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(total_probs);

    ret = hipMemcpy(bwd_NTC, gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(post_NTC, gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(states, gpubuf->states, sizeof(state_t) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(total_probs, gpubuf->total_probs, sizeof(float) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(qual_data, gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMemcpy(base_probs, gpubuf->base_probs, sizeof(float) * N * T, hipMemcpyDeviceToHost);
    checkHipError(); HIP_CHECK(ret);

    FILE *fp;

    fp = fopen("bwd_NTC.blob", "w");
    F_CHK(fp, "bwd_NTC.blob");
    if (fwrite(bwd_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    F_CHK(fp, "post_NTC.blob");
    if (fwrite(post_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("qual_data.blob", "w");
    F_CHK(fp, "qual_data.blob");
    if (fwrite(qual_data, sizeof(float), N * T * NUM_BASES, fp) != N * T * NUM_BASES) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    F_CHK(fp, "base_probs.blob");
    if (fwrite(base_probs, sizeof(float), N * T, fp) != N * T) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    F_CHK(fp, "total_probs.blob");
    if (fwrite(total_probs, sizeof(float), N * T, fp) != N * T) {
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
