#include "nn_cuda.h"
#include "error.h"
#include "cuda_utils.cuh"
#include "rotary_emb_cuda.cuh"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

void rotary_emb_cuda(
    void *x0_gpu,
    void *o0_gpu,
    void *sin_gpu,
    void *cos_gpu,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_dim,
    int stride_batch,
    int stride_seq,
    int stride_c,
    int stride_head,
    int stride_head_dim,
    int stride_rotary
) {
    int block_width = 32;
    dim3 block_size(block_width, block_width, 1);
	dim3 grid_size(batch_size, nheads, rotary_dim);

    rotary_emb<<<grid_size, block_size>>>(
        (half *)x0_gpu,
        (half *)o0_gpu,
        (float *)cos_gpu,
        (float *)sin_gpu,
        seqlen,
        stride_batch,
        stride_seq,
        stride_c,
        stride_head,
        stride_head_dim,
        stride_rotary
    );
    checkCudaError();
}