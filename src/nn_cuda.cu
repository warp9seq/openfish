#include "nn_cuda.h"
#include "error.h"
#include "cuda_utils.cuh"
#include "rotary_emb_cuda.cuh"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

#include "../cutlass/examples/45_dual_gemm/device/dual_gemm.h"
#include "swiglu_kernel.h"

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

template <typename T>
using SiLu = cutlass::epilogue::thread::SiLu<T>;

template <typename scalar_t, template <typename> typename ActivationFn>
void dual_gemm_lhs_activation_and_mul_cuda(
    void *x,
    void *w0,
    void *w1,
    void *d0,
    void *d1,
    void *d2, // result
    int64_t B,
    int64_t I,
    int64_t H
) {
    int d_stride_0 = H;
    int x_stride_0 = I;
    int w_stride_0 = I;

    // templati-ze the cutlass kernel
    cutlass::gemm::GemmCoord problem_size(B, H, I);

    constexpr int kStages = 3;
    constexpr bool kSplitKSerial = false;

    using ElementOutput = scalar_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
    using EpilogueOutputOp2 = EpilogueLHSActivationAndMul<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ActivationFn,
        ElementOutput,
        ElementCompute>;

    const ElementCompute alpha0 = ElementCompute(1);
    const ElementCompute beta0 = ElementCompute(0);
    const ElementCompute alpha1 = ElementCompute(1);
    const ElementCompute beta1 = ElementCompute(0);

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // Optionally, we might not need intermediate GEMM outputs
    constexpr bool kStoreD0 = true;
    constexpr bool kStoreD1 = true;
    using ArchTag = cutlass::arch::Sm80;

    using DualGemm = cutlass::gemm::device::DualGemm<
        scalar_t,
        cutlass::layout::RowMajor,
        scalar_t,
        cutlass::layout::ColumnMajor,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp01,
        EpilogueOutputOp01,
        EpilogueOutputOp2,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
        kStages,
        kStoreD0,
        kStoreD1,
        kSplitKSerial>;
    // {
    //     cudaDeviceProp *p = getDeviceProperties(x.device().index());
    //     ASSERT(p->major * 10 + p->minor >= ArchTag::kMinComputeCapability)
    // }

    int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
    using RefA = typename cutlass::TensorRef<typename DualGemm::ElementA, typename DualGemm::LayoutA>;
    using RefB0 = typename cutlass::TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB0>;
    using RefB1 = typename cutlass::TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB1>;
    using RefC = typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>;
    RefC ref_b0, ref_b1;

    typename DualGemm::Arguments arguments{
        cutlass::gemm::DualGemmMode::kGemm,
        problem_size,
        RefA{
            (scalar_t *)x,
            typename DualGemm::LayoutA::Stride(x_stride_0)},
        RefB0{
            (scalar_t *)w0,
            typename DualGemm::LayoutB0::Stride(w_stride_0)},
        ref_b0,
        RefC{
            (scalar_t *)d0,
            typename DualGemm::LayoutC::Stride(d_stride_0)},
        RefB1{
            (scalar_t *)w1,
            typename DualGemm::LayoutB1::Stride(w_stride_0)},
        ref_b1,
        RefC{
            (scalar_t *)d1,
            typename DualGemm::LayoutC::Stride(d_stride_0)},
        RefC{
            (scalar_t *)d2,
            typename DualGemm::LayoutC::Stride(d_stride_0)},
        typename DualGemm::EpilogueOutputOp0::Params{alpha0, beta0},
        typename DualGemm::EpilogueOutputOp1::Params{alpha1, beta1},
        typename DualGemm::EpilogueOutputOp2::Params{},
        split_k_slices};

    DualGemm dual_gemm;

    uint8_t *workspace;
    cudaMalloc((void **)&workspace, sizeof(uint8_t) * dual_gemm.get_workspace_size(arguments));
	checkCudaError();

    cutlass::Status status = dual_gemm.can_implement(arguments);
    ASSERT(status == cutlass::Status::kSuccess);
    checkCudaError();

    status = dual_gemm.initialize(arguments, workspace);
    ASSERT(status == cutlass::Status::kSuccess);
    checkCudaError();

    status = dual_gemm();
    ASSERT(status == cutlass::Status::kSuccess);
    checkCudaError();

    cudaDeviceSynchronize();
    checkCudaError();

    cudaFree(workspace);
	checkCudaError();
}

// void swiglu_test(
//     void *x,
//     void *w0,
//     void *w1,
//     void **o
// ) {
//     // x: 500 833 512 | 3
//     // w: 4096 512 | 2
//     // t: 500 833 2048 | 3

//     int64_t B = 1 * 833;
//     int64_t I = 512;
//     int64_t H = 2048;

//     void *x_gpu;

//     cudaMalloc((void **)&x_gpu, sizeof(cutlass::half_t) * B * I);
// 	checkCudaError();
//     cudaMemcpy(x_gpu, x, sizeof(cutlass::half_t) * B * I, cudaMemcpyHostToDevice);
//     checkCudaError();

//     cutlass::half_t *w0_gpu;
//     cutlass::half_t *w1_gpu;

//     cudaMalloc((void **)&w0_gpu, sizeof(cutlass::half_t) * H * I);
// 	checkCudaError();
//     cudaMemcpy(w0_gpu, w0, sizeof(cutlass::half_t) * H * I, cudaMemcpyHostToDevice);
//     checkCudaError();

//     cudaMalloc((void **)&w1_gpu, sizeof(cutlass::half_t) * H * I);
// 	checkCudaError();
//     cudaMemcpy(w1_gpu, w1, sizeof(cutlass::half_t) * H * I, cudaMemcpyHostToDevice);
//     checkCudaError();

//     cutlass::half_t *d0;
//     cutlass::half_t *d1;
//     cutlass::half_t *d2;
//     float *o_gpu;

//     cudaMalloc((void **)&d0, sizeof(cutlass::half_t) * B * H);
// 	checkCudaError();

//     cudaMalloc((void **)&d1, sizeof(cutlass::half_t) * B * H);
// 	checkCudaError();

//     cudaMalloc((void **)&d2, sizeof(cutlass::half_t) * B * H);
// 	checkCudaError();

//     cudaMalloc((void **)&o_gpu, sizeof(float) * B * H);
// 	checkCudaError();

//     dual_gemm_lhs_activation_and_mul_cuda<cutlass::half_t, SiLu>(x_gpu, w0_gpu, w1_gpu, d0, d1, d2, B, I, H);
    
//     dim3 block_size(32, 32, 1);
// 	dim3 grid_size(32, 32, 1);

//     half2float_vec_cpy<<<grid_size,block_size>>>(
//         (half *)d2,
//         o_gpu,
//         B * H
//     );
//     checkCudaError();
//     cudaDeviceSynchronize();
//     checkCudaError();

//     cudaMemcpy(*o, o_gpu, sizeof(float) * B * H, cudaMemcpyDeviceToHost);
//     checkCudaError();

//     cudaFree(d0);
// 	checkCudaError();
//     cudaFree(d1);
// 	checkCudaError();
//     cudaFree(d2);
// 	checkCudaError();
//     cudaFree(x_gpu);
// 	checkCudaError();
//     cudaFree(w0_gpu);
// 	checkCudaError();
//     cudaFree(w1_gpu);
// 	checkCudaError();
// }