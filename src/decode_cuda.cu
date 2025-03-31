#include "decode_cuda.h"
#include "scan_cuda.cuh"
#include "beam_search_cuda.cuh"
#include "error.h"
#include "cuda_utils.cuh"

#include <openfish/openfish_error.h>

#include <flash.h>

void run_flash(
    void *q,
    void *k,
    void *v,
    void **o
) {
    int batch_stride = 426496;
    int head_stride = 53312;
    int row_stride = 64;
    size_t batch_size = 512;
    size_t seqlen = 833;
    size_t num_heads = 8;
    size_t head_dim = 64;
    size_t numel = batch_size * seqlen * num_heads * head_dim;

    cutlass::half_t *q_gpu;
    cutlass::half_t *k_gpu;
    cutlass::half_t *v_gpu;
    cutlass::half_t *o_gpu;
    cudaMalloc((void **)&q_gpu, sizeof(cutlass::half_t) * numel);
	checkCudaError();
    cudaMalloc((void **)&k_gpu, sizeof(cutlass::half_t) * numel);
	checkCudaError();
    cudaMalloc((void **)&v_gpu, sizeof(cutlass::half_t) * numel);
	checkCudaError();

    cudaMemcpy(q_gpu, q, sizeof(cutlass::half_t) * numel, cudaMemcpyHostToDevice);
    checkCudaError();
    cudaMemcpy(k_gpu, k, sizeof(cutlass::half_t) * numel, cudaMemcpyHostToDevice);
    checkCudaError();
    cudaMemcpy(v_gpu, v, sizeof(cutlass::half_t) * numel, cudaMemcpyHostToDevice);
    checkCudaError();
    
    *o = (uint8_t *)malloc(sizeof(cutlass::half_t) * numel);
    MALLOC_CHK(*o);

    cudaMalloc((void **)&o_gpu, sizeof(cutlass::half_t) * numel);
	checkCudaError();

    int seqlen_q = seqlen;
    int seqlen_k = seqlen;
    int num_heads_k = num_heads;
    
    int q_batch_stride = batch_stride;
    int k_batch_stride = batch_stride;
    int v_batch_stride = batch_stride;
    int o_batch_stride = batch_stride;
    int q_head_stride = head_stride;
    int k_head_stride = head_stride;
    int v_head_stride = head_stride;
    int o_head_stride = head_stride;
    int q_row_stride = row_stride;
    int k_row_stride = row_stride;
    int v_row_stride = row_stride;
    int o_row_stride = row_stride;
    float softmax_scale = 1.0 / std::sqrt(num_heads);
    int window_size_left = 127;
    int window_size_right = 128;
    bool casual = false;

    // upload qkv
    flash_attn::flash_attention_forward(
        q_gpu,
        k_gpu,
        v_gpu,
        o_gpu,
        batch_size,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_heads_k,
        head_dim,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_head_stride,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        o_row_stride,
        softmax_scale,
        casual,
        window_size_left,
        window_size_right
    );

    cudaMemcpy(*o, o_gpu, sizeof(cutlass::half_t) * numel, cudaMemcpyDeviceToHost);
    checkCudaError();
}

openfish_gpubuf_t *gpubuf_init_cuda(
    const int T,
    const int N,
    const int state_len
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    cudaMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states);
	checkCudaError();
    cudaMalloc((void **)&gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states);
	checkCudaError();

    // return buffers
    cudaMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->sequence, sizeof(char) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qstring, sizeof(char) * N * T);
    checkCudaError();

    // beamsearch buffers
    cudaMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * N * MAX_BEAM_WIDTH * (T + 1));
    checkCudaError();
    cudaMalloc((void **)&gpubuf->states, sizeof(state_t) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->base_probs, sizeof(float) * N * T);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->total_probs, sizeof(float) * N * T);
    checkCudaError();

    return gpubuf;
}

void gpubuf_free_cuda(
    openfish_gpubuf_t *gpubuf
) {
    cudaFree(gpubuf->bwd_NTC);
    checkCudaError();
    cudaFree(gpubuf->post_NTC);
    checkCudaError();

    cudaFree(gpubuf->moves);
    checkCudaError();
    cudaFree(gpubuf->sequence);
    checkCudaError();
    cudaFree(gpubuf->qstring);
    checkCudaError();

    cudaFree(gpubuf->beam_vector);
    checkCudaError();
    cudaFree(gpubuf->states);
    checkCudaError();
    cudaFree(gpubuf->qual_data);
    checkCudaError();
    cudaFree(gpubuf->base_probs);
    checkCudaError();
    cudaFree(gpubuf->total_probs);
    checkCudaError();

    free(gpubuf);
}

void decode_cuda(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
    const int num_states = pow(NUM_BASES, state_len);

    // calculate grid / block dims
    const int target_block_width = (int)ceil(sqrt((float)num_states));
    int block_width = 2;
    while (block_width < target_block_width) {
        block_width *= 2;
    }

    OPENFISH_LOG_TRACE("chosen block_dims: %d x %d for num_states %d", block_width, block_width, num_states);
    
    dim3 block_size(block_width, block_width, 1);
    dim3 block_size_beam(MAX_BEAM_WIDTH * NUM_BASES, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(N, 1, 1);

    OPENFISH_LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    scan_args_t scan_args = {0};
    scan_args.scores_in = scores_TNC;
    scan_args.T = T;
    scan_args.N = N;
    scan_args.C = C;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;

    // init results
    *moves = (uint8_t *)malloc(N * T * sizeof(uint8_t));
    MALLOC_CHK(*moves);
    *sequence = (char *)malloc(N * T * sizeof(char));
    MALLOC_CHK(*sequence);
    *qstring = (char *)malloc(N * T * sizeof(char));
    MALLOC_CHK(*qstring);

    cudaMemset(gpubuf->moves, 0, sizeof(uint8_t) * N * T);
	checkCudaError();
    cudaMemset(gpubuf->sequence, 0, sizeof(char) * N * T);
	checkCudaError();
    cudaMemset(gpubuf->qstring, 0, sizeof(char) * N * T);
	checkCudaError();

    const int num_state_bits = (int)log2((double)num_states);
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    beam_args_t beam_args = {0};
    beam_args.scores_TNC = (half *)scores_TNC;
    beam_args.bwd_NTC = gpubuf->bwd_NTC;
    beam_args.post_NTC = gpubuf->post_NTC;
    beam_args.T = T;
    beam_args.N = N;
    beam_args.C = C;
    beam_args.num_state_bits = num_state_bits;

    // bwd scan
    // fwd + post scan
    // beam search
    bwd_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    beam_search<<<grid_size,block_size_beam>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->moves,
        (beam_element_t *)gpubuf->beam_vector,
        beam_cut,
        fixed_stay_score,
        1.0f
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    fwd_post_scan<<<grid_size,block_size>>>(scan_args, gpubuf->bwd_NTC, gpubuf->post_NTC);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    compute_qual_data<<<grid_size,block_size_gen>>>(
        beam_args,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        1.0f
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
    
    generate_sequence<<<grid_size,block_size_gen>>>(
        beam_args,
        gpubuf->moves,
        (state_t *)gpubuf->states,
        gpubuf->qual_data,
        gpubuf->base_probs,
        gpubuf->total_probs,
        gpubuf->sequence,
        gpubuf->qstring,
        q_shift,
        q_scale
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    // copy beam_search results
    cudaMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(*sequence, gpubuf->sequence, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
    cudaMemcpy(*qstring, gpubuf->qstring, sizeof(char) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();
}

// misc stuff for testing //////////////////////////////////////////////////////
void set_device_cuda(
    int device
) {
    cudaSetDevice(device);
	checkCudaError();
}

void *upload_scores_to_cuda(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
) {
    void *scores_TNC_gpu;

    cudaMalloc((void **)&scores_TNC_gpu, sizeof(half) * T * N * C);
	checkCudaError();

	cudaMemcpy(scores_TNC_gpu, scores_TNC, sizeof(half) * T * N * C, cudaMemcpyHostToDevice);
	checkCudaError();

    return scores_TNC_gpu;
}

void free_scores_cuda(
    void *scores_TNC_gpu
) {
    cudaFree(scores_TNC_gpu);
	checkCudaError();
}

void write_gpubuf_cuda(
    const uint64_t T,
    const uint64_t N,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
) {
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

    // copy scan results
    cudaMemcpy(bwd_NTC, gpubuf->bwd_NTC, sizeof(float) * N * (T + 1) * num_states, cudaMemcpyDeviceToHost);
    checkCudaError();
	cudaMemcpy(post_NTC, gpubuf->post_NTC, sizeof(float) * N * (T + 1) * num_states, cudaMemcpyDeviceToHost);
    checkCudaError();

    // copy intermediate
    cudaMemcpy(states, gpubuf->states, sizeof(state_t) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(total_probs, gpubuf->total_probs, sizeof(float) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(qual_data, gpubuf->qual_data, sizeof(float) * N * T * NUM_BASES, cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaMemcpy(base_probs, gpubuf->base_probs, sizeof(float) * N * T, cudaMemcpyDeviceToHost);
    checkCudaError();

    // write results
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

    // write beam results
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

    // cleanup
    free(bwd_NTC);
    free(post_NTC);
    free(states);
    free(qual_data);
    free(base_probs);
    free(total_probs);
}
////////////////////////////////////////////////////////////////////////////////