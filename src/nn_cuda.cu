#include "nn_cuda.h"
#include "error.h"
#include "cuda_utils.cuh"
#include "rotary_emb_cuda.cuh"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

#include "../cutlass/examples/45_dual_gemm/device/dual_gemm.h"
#include "swiglu_kernel.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include "cutlass_ext/gemm_universal_base_compat.h"
#include "cutlass_ext/epilogue_per_row_per_col.h"
#include "cutlass_ext/gemm_with_epilogue_visitor.h"
#include <cuda_runtime.h>

void silu_mul_cuda(
    void *x_gpu,
    void *o_gpu,
    uint64_t M,
    uint64_t K
) {
    cudaError_t result;

    dim3 block(32, 32);
	dim3 grid(
        (K + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    silu_mul<<<grid, block>>>(
        (half *)x_gpu,
        (half *)o_gpu,
        K,
        M
    );

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "cuda kernel failed: " << std::endl;
        exit(1);
    }
}

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
static void dual_gemm_lhs_activation_and_mul_cuda(
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

void swiglu_cuda(
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
    dual_gemm_lhs_activation_and_mul_cuda<cutlass::half_t, SiLu>(x, w0, w1, d0, d1, d2, B, I, H);
}

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // data type of epilogue operations
using ElementInput = int8_t;                        // data type of elements in input matrix
using ElementOutput = cutlass::half_t;              // data type of elements in output matrix D
using ElementCompute = float;                       // data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using OperatorClass = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

void quant_gemm_cuda(
  void *a_quant,
  void *b_quant,
  void *a_scale,
  void *b_scale,
  void *o_gpu,
  int M,
  int N,
  int K
) {
  constexpr int Stages = 3;
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  using DefaultGemmConf = typename cutlass::gemm::device::DefaultGemmConfiguration<
    OperatorClass,
    SmArch,
    ElementInput,
    ElementInput,
    ElementOutput,
    ElementCompute>;
  using GemmOp = typename DefaultGemmConf::Operator;
  using EpilogueOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInput, cutlass::layout::RowMajor,
    DefaultGemmConf::kAlignmentA, ElementInput, cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, OperatorClass, SmArch, ThreadblockShape, WarpShape,
    InstructionShape, EpilogueOp, ThreadblockSwizzle, Stages, true, GemmOp>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<ThreadblockShape,
    GemmKernel_::kThreadCount, AlphaColTileIterator, typename GemmKernel_::Epilogue::OutputTileIterator,
    ElementAccumulator, ElementCompute, EpilogueOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor,
    typename GemmKernel_::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  // // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = { M, N, K };

  // // Initialize alpha and beta for dot product computation
  // ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0);
  // ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  typename EpilogueOp::Params linearScalingParams; // TODO: right now it's unused (scaling is done in visitor, no activation needed)
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
    {reinterpret_cast<ElementInput *>(a_quant), K},
    {reinterpret_cast<ElementInput *>(b_quant), K},
    {reinterpret_cast<ElementCompute *>(b_scale), 0},
    {reinterpret_cast<ElementCompute *>(a_scale), 0}, {nullptr, 0},
    {reinterpret_cast<ElementOutput *>(o_gpu), N}, 0, 0,
    typename EpilogueVisitor::Arguments(linearScalingParams, 0, 0, 0)
  };

  Gemm gemm_op;
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  uint8_t *workspace = NULL; // we dont need extra space apparently
//   cudaMalloc((void **)&workspace, sizeof(uint8_t) * Gemm::get_workspace_size(arguments));
//   checkCudaError();

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);

//   cudaDeviceSynchronize();
//   checkCudaError();

//   cudaFree(workspace);
//   checkCudaError();
}
