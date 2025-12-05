#include "nn_cuda.h"
#include "error.h"
#include "cuda_utils.cuh"
#include "rotary_emb_cuda.cuh"
#include "flashrnn.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

void rotary_emb_cuda(
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
    int thread_h = 32;
    dim3 block_size(rotary_half, thread_h, 1);
	dim3 grid_size(batch_size, nheads, 1);

    rotary_emb<<<grid_size, block_size>>>(
        (half *)x_gpu,
        (float *)cos_gpu,
        (float *)sin_gpu,
        seqlen,
        stride_batch,
        stride_seq,
        stride_head,
        rotary_half
    );
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}

FlashRNNFuncFused::~FlashRNNFuncFused() {
    delete fw;
}

FlashRNNFuncFused::FlashRNNFuncFused(const bool training, const int batch_size, const int hidden_size, const int num_heads) {
    fw = new ForwardPass(training, batch_size, hidden_size, num_heads, 0, 0);
}

void FlashRNNFuncFused::forward(
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
) {
    const auto time_steps = seqlen;
    // const auto batch_size = x.size(1);
    const auto num_heads = nheads;
    // const auto head_dim = recurrent_kernel.size(1);
    const auto hidden_size = head_dim * num_heads;

    // make sure its contiguious and on gpu
    // CHECK_INPUT(x);
    // CHECK_INPUT(s0);
    // CHECK_INPUT(recurrent_kernel);
    // CHECK_INPUT(bias);

    // TORCH_CHECK(x.scalar_type() == typeToTorchDtype<FLASHRNN_DTYPE_W>(),
    //             "Bad input type");
    // TORCH_CHECK(s0.scalar_type() == typeToTorchDtype<FLASHRNN_DTYPE_S>(),
    //             "Bad input type");
    // TORCH_CHECK(recurrent_kernel.scalar_type() ==
    //                 typeToTorchDtype<FLASHRNN_DTYPE_R>(),
    //             "Bad input type");
    // TORCH_CHECK(bias.scalar_type() == typeToTorchDtype<FLASHRNN_DTYPE_B>(),
    //             "Bad input type");

    // const auto options = x.options();
    // const at::cuda::CUDAGuard guard(options.device_index());
    int res = 1;
//         Tensor states = torch::empty(
//             {FLASHRNN_NUM_STATES, time_steps + 1, batch_size, num_heads, head_dim},
//             options.dtype(typeToTorchDtype<FLASHRNN_DTYPE_S>()));
//         Tensor gate_cache_r;
//         Tensor gate_cache_i;
//         // if (training) {
//         gate_cache_r = torch::empty(
//             {time_steps, batch_size, num_heads, head_dim, FLASHRNN_NUM_GATES_R},
//             options.dtype(typeToTorchDtype<FLASHRNN_DTYPE_G>()));
// #if FLASHRNN_SIMPLE_AGG
//         gate_cache_i =
//             torch::empty({}, options.dtype(typeToTorchDtype<FLASHRNN_DTYPE_G>()));
// #else
//         // this is for both recurrent and input base caches, without additional
//         // bias-only gates
//         gate_cache_i = torch::empty(
//             {time_steps, batch_size, num_heads, head_dim, FLASHRNN_NUM_GATES_I},
//             options.dtype(typeToTorchDtype<FLASHRNN_DTYPE_G>()));
// #endif
//         Tensor gate_buffer =
//             torch::ones({FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN *
//                              FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH *
//                              FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH *
//                              FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH *
//                              FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH,
//                          FLASHRNN_NUM_GATES_R * hidden_size},
//                         options.dtype(torch::kFloat32));
    // for (uint i = 0; i < FLASHRNN_NUM_STATES; i++) {
    //     states[i][0] = s0[i];
    // }

    fw->Set(training, batch_size, hidden_size, num_heads, (cublasHandle_t &)blas_handle, 0); // current stream default 0
    res = fw->Run(
        time_steps,
        reinterpret_cast<FLASHRNN_DTYPE_R *>(recurrent_kernel),
        reinterpret_cast<FLASHRNN_DTYPE_B *>(bias),
        reinterpret_cast<FLASHRNN_DTYPE_W *>(x),
        reinterpret_cast<FLASHRNN_DTYPE_S *>(states),
        reinterpret_cast<FLASHRNN_DTYPE_G *>(gate_cache_r),
        reinterpret_cast<FLASHRNN_DTYPE_G *>(gate_cache_i),
        reinterpret_cast<FLASHRNN_ACC_DTYPE *>(gate_buffer)
    );

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
    //     x.scalar_type(), "FlashRNNFuncFused.forward", ([&]
    //                                                    {
    //   fw.Set(training, batch_size, hidden_size, num_heads, blas_handle, 0); // current stream default 0
    //   res = fw.Run(
    //       time_steps,
    //       reinterpret_cast<FLASHRNN_DTYPE_R *>(recurrent_kernel.data_ptr()),
    //       reinterpret_cast<FLASHRNN_DTYPE_B *>(bias.data_ptr()),
    //       reinterpret_cast<FLASHRNN_DTYPE_W *>(x.data_ptr()),
    //       reinterpret_cast<FLASHRNN_DTYPE_S *>(states.data_ptr()),
    //       reinterpret_cast<FLASHRNN_DTYPE_G *>(gate_cache_r.data_ptr()),
    //       reinterpret_cast<FLASHRNN_DTYPE_G *>(gate_cache_i.data_ptr()),
    //       reinterpret_cast<FLASHRNN_ACC_DTYPE *>(gate_buffer.data_ptr())); }));
    if (res != 0) {
        exit(1);
    }
    // return {states, gate_cache_r, gate_cache_i};
}
