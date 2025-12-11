#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>

#include "openfish_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECODER_INIT {32, 100.0, 2.0, 0.0, 1.0, 1.0, false}

// flash rnn h /////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FLASHRNN_NUM_GATES_R
// all needed definitions from external
#define FLASHRNN_NUM_HEADS 1
#define FLASHRNN_HIDDEN_SIZE 96
#define FLASHRNN_NUM_GATES_R 4
#define FLASHRNN_NUM_GATES_W 4
#define FLASHRNN_NUM_GATES_I 4
#define FLASHRNN_NUM_GATES_T 4
#define FLASHRNN_NUM_STATES 2
#define FLASHRNN_DTYPE __nv_bfloat16
#define FLASHRNN_USE_DTYPE_BFLOAT16
#define FLASHRNN_DTYPE_R __nv_bfloat16
#define FLASHRNN_DTYPE_B __nv_bfloat16
#define FLASHRNN_DTYPE_W __nv_bfloat16
#define FLASHRNN_DTYPE_G __nv_bfloat16
#define FLASHRNN_DTYPE_S __nv_bfloat16
#define FLASHRNN_DTYPE_A __nv_bfloat16

// fused forward
// optimized for hidden size 1024
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN 1 // Rtch 16?
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE 12  // Rtcg 1024 best 64
#define FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH 1      // Btcb
// means extra warps for threads
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH 1 // Wtcb
// means each warp loops over batches stored in additional shared memory
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH 1      // Wlcp
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE 1       // Wlcg
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN 4      // Wtch 1024 best 8
#define FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN 16 // Wrch 1024 best 8

#define FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_FORWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH 8   // Wtdb
#define FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN 16 // Wtdg
#define FLASHRNN_FORWARD_WARP_TILING_DIM_GATE 32   // Wtdg

// fused backward
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN 32 // Rtch 16?
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE 1    // Rtcg
#define FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH 1       // Btcb
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH 1        // Wtcb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH 1       // Wtlb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN 1      // Wlch
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE 8         // Wtcg
#define FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE 32 // Wrcg optimal for 1024

#define FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH 8   // Wtdb
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE 16   // Wtdh
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN 32 // Wtdh

// defines whether g = Wx + Ry + b for every gate, enables half the cache for
// backward
#define FLASHRNN_SIMPLE_AGG true
#endif

#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define FLASHRNN_ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define FLASHRNN_ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define FLASHRNN_ACC_DTYPE float
#endif

// flash rnn fused foward cu //////////////////////////////////////////////////////////////////////////////////////////////////////

// gate order: i f z o
// FLASHRNN_NUM_GATES_R:     1
//             1 - - -
// FLASHRNN_NUM_GATES_I:     3
//             - 1 1 1

// dimensions
// G: # gates
// FLASHRNN_NUM_GATES_R: # recurrent gates per hidden dimensions (1 for lstmhin,
// 4 for slstm) FLASHRNN_NUM_GATES_I: # gates from input FLASHRNN_NUM_GATES_T: #
// total gates S: # states T: # time steps B: # batch dim H: # hidden dim I: #
// input dim

// General naming convention: dim = real size in memory, count = number along
// axis -> high level dim = count * dim
// -> tile dim = total dim / tile count

#ifndef FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE

// optimized for hidden size 1024
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN 1 // FRTCH 16?
#define FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH 1      // Btcb
// means extra warps for threads
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH 1 // Wtcb
// means each warp loops over batches stored in additional shared memory
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH 1      // Wlcp
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE 1       // FWLCG
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN 4      // FWTCH 1024 best 8
#define FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN 16 // FWRCH 1024 best 8

#define FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_FORWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH 8 // FWTDB
#define FLASHRNN_FORWARD_WARP_TILING_DIM_GATE 32 // FWTDG

#endif

#define FRTCH FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN
#define FRTCG FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
#define FBTCB FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
#define FWTCB FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH
#define FWLCB FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH
#define FWLCG FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE
#define FWTCH FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN
#define FWRCH FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN
#define FMTC FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT
#define FSMP FLASHRNN_FORWARD_SHARED_MEMORY_PADDING
#define FWTDB FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH
#define FWTDG FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
#define FWTDH FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN

#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define MAT_DTYPE wmma::precision::tf32
#define DTYPE float
#define ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define MAT_DTYPE __half
#define DTYPE __half
#define ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define MAT_DTYPE __nv_bfloat16
#define DTYPE __nv_bfloat16
#define ACC_DTYPE float
#endif

#define HS FLASHRNN_HIDDEN_SIZE
#define NH FLASHRNN_NUM_HEADS

#define WARP_SIZE 32

typedef struct openfish_gpubuf {
    float *bwd_NTC;
    float *post_NTC;
    uint8_t *moves;
    char *sequence;
    char *qstring;
    void *beam_vector;
    void *states;
    float *qual_data;
    float *base_probs;
    float *total_probs;
} openfish_gpubuf_t;

typedef struct openfish_opt {
    size_t beam_width;
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
    float temperature;
    bool move_pad;
} openfish_opt_t;

void openfish_decode_cpu(
    const int T,
    const int N,
    const int C,
    int nthreads,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

void openfish_rotary_emb_cpu(
    void *x,
    void *sin_buf,
    void *cos_buf,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head,
    int nthreads
);

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
);

openfish_gpubuf_t *openfish_gpubuf_init(
    const int T,
    const int N,
    const int state_len
);

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
);

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
);

#ifdef HAVE_CUDA
class ForwardPass;
class FlashRNNFuncFused
{
private:
    ForwardPass *fw;
    // BackwardPass bw;
    // BackwardPassCut bwc;

public:
    FlashRNNFuncFused(const bool training, const int batch_size, const int hidden_size, const int num_heads);
    ~FlashRNNFuncFused();
    
    void forward(
        bool training,
        void *x, // W_ih * x + b_ih
        void *s0,
        void *recurrent_kernel, // W_hh
        void *bias, // b_hh
        void *states,
        void *gate_cache_r,
        void *gate_cache_i,
        void *gate_buffer,
        int seqlen,
        int batch_size,
        int nheads,
        int head_dim,
        void *blas_handle
    );
};
#endif

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H