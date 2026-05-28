#include <openfish/openfish.h>

#include <openfish/openfish_error.h>
#include "decode.h"

#ifdef HAVE_CUDA
#include "decode_cuda.h"
#include "nn_cuda.h"
#endif

#ifdef HAVE_ROCM
#include "decode_hip.h"
#include "nn_hip.h"
#endif

openfish_opt_t openfish_decoder_default_opts(void) {
    openfish_opt_t opt = {32, 100.0f, 2.0f, 0.0f, 1.0f, 1.0f, false};
    return opt;
}

openfish_gpubuf_t *openfish_gpubuf_init(
    const int T,
    const int N,
    const int state_len
) {
#ifdef HAVE_CUDA
    return gpubuf_init_cuda(T, N, state_len);
#elif HAVE_ROCM
    return gpubuf_init_hip(T, N, state_len);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

size_t openfish_gpubuf_size(
    const int T,
    const int N,
    const int state_len
) {
    const size_t num_states = (size_t)1 << (2 * state_len);
    return
        sizeof(float) * (size_t)N * (T + 1) * num_states +          // bwd_NTC
        sizeof(float) * (size_t)N * (T + 1) * num_states +          // post_NTC
        sizeof(uint8_t) * (size_t)N * T +                            // moves
        sizeof(char) * (size_t)N * T +                               // sequence
        sizeof(char) * (size_t)N * T +                               // qstring
        sizeof(beam_element_t) * (size_t)N * MAX_BEAM_WIDTH * (T + 1) + // beam_vector
        sizeof(state_t) * (size_t)N * T +                            // states
        sizeof(float) * (size_t)N * T * NUM_BASES +                  // qual_data
        sizeof(float) * (size_t)N * T +                              // base_probs
        sizeof(float) * (size_t)N * T;                               // total_probs
}

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
) {
#ifdef HAVE_CUDA
    gpubuf_free_cuda(gpubuf);
#elif HAVE_ROCM
    gpubuf_free_hip(gpubuf);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_decode_gpu(
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
#ifdef HAVE_CUDA
    decode_cuda(T, N, C, scores_TNC, state_len, options, gpubuf, moves, sequence, qstring);
#elif HAVE_ROCM
    decode_hip(T, N, C, scores_TNC, state_len, options, gpubuf, moves, sequence, qstring);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_rotary_emb_gpu(
    void *x_gpu,
    void *sin_gpu,
    void *cos_gpu,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head
) {
#ifdef HAVE_CUDA
    rotary_emb_cuda(
        x_gpu,
        sin_gpu,
        cos_gpu,
        batch_size,
        seqlen,
        nheads,
        head_dim,
        rotary_half,
        stride_batch,
        stride_seq,
        stride_head
    );
#elif HAVE_ROCM
    rotary_emb_hip(
        x_gpu,
        sin_gpu,
        cos_gpu,
        batch_size,
        seqlen,
        nheads,
        head_dim,
        rotary_half,
        stride_batch,
        stride_seq,
        stride_head
    );
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* c,
    void* hh_next,
    int N, int C
) {
#ifdef HAVE_CUDA
    flstm_step_cuda(scratch, ih_t, c, hh_next, N, C);
#elif HAVE_ROCM
    flstm_step_hip(scratch, ih_t, c, hh_next, N, C);
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_silu_mul_gpu(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
) {
#ifdef HAVE_CUDA
    silu_mul_cuda(
        x_gpu,
        o_gpu,
        MN,
        K
    );
#elif HAVE_ROCM
    silu_mul_hip(
        x_gpu,
        o_gpu,
        MN,
        K
    );
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_rmsnorm_gpu(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
) {
#ifdef HAVE_CUDA
    rmsnorm_cuda(
        input,
        residual,
        weight,
        output,
        MN,
        K,
        alpha,
        eps
    );
#elif HAVE_ROCM
    rmsnorm_hip(
        input,
        residual,
        weight,
        output,
        MN,
        K,
        alpha,
        eps
    );
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}

void openfish_rmsnorm_quant_gpu(
    const void* input,
    const void* weight,
    void* residual,
    void* residual_scale,
    int MN,
    int K,
    float alpha,
    float eps
) {
#ifdef HAVE_CUDA
    rmsnorm_quant_cuda(
        input,
        weight,
        residual,
        residual_scale,
        MN,
        K,
        alpha,
        eps
    );
#elif HAVE_ROCM
    rmsnorm_quant_hip(
        input,
        weight,
        residual,
        residual_scale,
        MN,
        K,
        alpha,
        eps
    );
#else
    OPENFISH_ERROR("%s", "not compiled for gpu");
    exit(EXIT_FAILURE);
#endif
}