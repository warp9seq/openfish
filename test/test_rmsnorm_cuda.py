"""Build-and-run test for the CUDA RMSNorm kernels.

Single script: it JIT-compiles the real CUDA kernels (src/nn_cuda.c) plus a tiny
torch binding with ninja (torch.utils.cpp_extension), runs openfish_rmsnorm_gpu
and openfish_rmsnorm_quant_int8_gpu on the GPU, and diffs the result against a
torch reference -> PASS/FAIL. Also benchmarks each kernel with CUDA events.

    ./pyvenv/bin/python test/test_rmsnorm_cuda.py

Requires a CUDA-enabled torch (see test/requirements.txt) and a CUDA toolkit.
(The fp8 fused kernel is not implemented on CUDA, so it is not covered here.)
"""

import os
import sys

# CUDA_HOME must be set before importing torch.utils.cpp_extension (it is
# resolved at import time). The ninja and nvcc binaries also need to be on PATH
# for the subprocess build (the venv's bin holds the pip-installed ninja).
CUDA_HOME = os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = os.pathsep.join([
    os.path.dirname(sys.executable),
    os.path.join(CUDA_HOME, "bin"),
    os.environ.get("PATH", ""),
])
# A100 = 8.0; override via env for other GPUs.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

import torch
from torch.utils.cpp_extension import load_inline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# C++ binding (compiled by the host compiler). It calls the real
# openfish_rmsnorm_gpu / openfish_rmsnorm_quant_int8_gpu launch wrappers.
CPP_SRC = r"""
#include <torch/extension.h>
#include <openfish/openfish.h>
#include <cuda_runtime.h>

// out = rmsnorm(in + alpha*residual) * weight   (fp32 math, fp16 out)
void rmsnorm(torch::Tensor in, torch::Tensor residual, torch::Tensor weight,
             torch::Tensor out, double alpha, double eps) {
    TORCH_CHECK(in.is_cuda() && residual.is_cuda() && weight.is_cuda() && out.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(in.scalar_type() == torch::kHalf, "in must be fp16");
    const int n_tokens   = in.size(0);
    const int hidden_dim = in.size(1);
    openfish_rmsnorm_gpu(in.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                         out.data_ptr(), n_tokens, hidden_dim, (float)alpha, (float)eps);
}

// In-place: residual (int8) and residual_scale (f32, per-token) are read as the
// previous quantized residual and overwritten with the new quantized rmsnorm.
void rmsnorm_quant_int8(torch::Tensor in, torch::Tensor weight,
                        torch::Tensor residual, torch::Tensor residual_scale,
                        double alpha, double eps) {
    TORCH_CHECK(in.is_cuda() && weight.is_cuda() && residual.is_cuda() && residual_scale.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(in.scalar_type() == torch::kHalf, "in must be fp16");
    TORCH_CHECK(residual.scalar_type() == torch::kChar, "residual must be int8");
    const int n_tokens   = in.size(0);
    const int hidden_dim = in.size(1);
    openfish_rmsnorm_quant_int8_gpu(in.data_ptr(), weight.data_ptr(), residual.data_ptr(),
                                    residual_scale.data_ptr(), n_tokens, hidden_dim,
                                    (float)alpha, (float)eps);
}

double rmsnorm_bench(torch::Tensor in, torch::Tensor residual, torch::Tensor weight,
                     torch::Tensor out, double alpha, double eps,
                     int64_t warmup, int64_t iters) {
    const int n_tokens   = in.size(0);
    const int hidden_dim = in.size(1);
    for (int i = 0; i < warmup; ++i)
        openfish_rmsnorm_gpu(in.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                             out.data_ptr(), n_tokens, hidden_dim, (float)alpha, (float)eps);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        openfish_rmsnorm_gpu(in.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                             out.data_ptr(), n_tokens, hidden_dim, (float)alpha, (float)eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return (double)ms / (double)iters;
}

double rmsnorm_quant_int8_bench(torch::Tensor in, torch::Tensor weight,
                                torch::Tensor residual, torch::Tensor residual_scale,
                                double alpha, double eps, int64_t warmup, int64_t iters) {
    const int n_tokens   = in.size(0);
    const int hidden_dim = in.size(1);
    for (int i = 0; i < warmup; ++i)
        openfish_rmsnorm_quant_int8_gpu(in.data_ptr(), weight.data_ptr(), residual.data_ptr(),
                                        residual_scale.data_ptr(), n_tokens, hidden_dim,
                                        (float)alpha, (float)eps);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        openfish_rmsnorm_quant_int8_gpu(in.data_ptr(), weight.data_ptr(), residual.data_ptr(),
                                        residual_scale.data_ptr(), n_tokens, hidden_dim,
                                        (float)alpha, (float)eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return (double)ms / (double)iters;
}
"""

# Compile the real kernel translation unit straight into the extension (nvcc,
# -fPIC) instead of linking lib/libopenfish.a, whose objects are built without
# -fPIC and so cannot go into a shared object. This still exercises the actual
# launch wrappers from src/nn_cuda.c.
CUDA_SRC = r"""
#include "error.c"
#include "nn_cuda.c"
"""


def ref_rmsnorm(in_, residual, weight, alpha, eps):
    val = in_.to(torch.float32) + residual.to(torch.float32) * alpha
    rms = torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (val * rms * weight.to(torch.float32)).to(torch.float16)


def ref_rmsnorm_quant_int8(in_, weight, residual, residual_scale, alpha, eps):
    val = in_.to(torch.float32) + (residual.to(torch.float32) * residual_scale[:, None]) * alpha
    rms = torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = val * rms * weight.to(torch.float32)
    amax = normalized.abs().amax(dim=-1, keepdim=True)
    quant_scale = torch.where(amax > 0, 127.0 / amax, torch.ones_like(amax))
    q = torch.clamp(torch.round(normalized * quant_scale), -127, 127).to(torch.int8)
    new_scale = (1.0 / quant_scale).squeeze(-1)
    return q, new_scale, normalized


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")

    print(">> JIT-compiling CUDA kernels + binding with ninja")
    mod = load_inline(
        name="openfish_rmsnorm_test",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["rmsnorm", "rmsnorm_quant_int8", "rmsnorm_bench", "rmsnorm_quant_int8_bench"],
        extra_include_paths=[os.path.join(ROOT, "include"), os.path.join(ROOT, "src")],
        extra_cflags=["-DHAVE_CUDA=1", "-O2"],
        extra_cuda_cflags=["-DHAVE_CUDA=1", "-O2", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=False,
    )

    dev = "cuda"
    torch.manual_seed(0)
    alpha, eps = 2.0, 1e-5
    ok = True

    # ---- correctness: rmsnorm ----
    n_tokens, hidden_dim = 512, 512
    in_ = torch.randn(n_tokens, hidden_dim, device=dev, dtype=torch.float16)
    residual = torch.randn(n_tokens, hidden_dim, device=dev, dtype=torch.float16)
    weight = torch.randn(hidden_dim, device=dev, dtype=torch.float16)
    out = torch.empty_like(in_)
    mod.rmsnorm(in_, residual, weight, out, alpha, eps)
    torch.cuda.synchronize()
    expected = ref_rmsnorm(in_, residual, weight, alpha, eps)
    max_diff = (out.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    tol = 5e-3
    print(f"[rmsnorm]            max abs diff: {max_diff:.3e} (tol {tol:.1e})")
    ok &= max_diff < tol

    # ---- correctness: rmsnorm_quant_int8 ----
    in8 = torch.randn(n_tokens, hidden_dim, device=dev, dtype=torch.float16)
    weight8 = torch.randn(hidden_dim, device=dev, dtype=torch.float16)
    residual8 = torch.randint(-127, 128, (n_tokens, hidden_dim), device=dev, dtype=torch.int8)
    res_scale = torch.rand(n_tokens, device=dev, dtype=torch.float32) * 0.02 + 0.001
    q_ref, scale_ref, norm_ref = ref_rmsnorm_quant_int8(
        in8, weight8, residual8.clone(), res_scale.clone(), alpha, eps)
    q_got = residual8.clone()
    scale_got = res_scale.clone()
    mod.rmsnorm_quant_int8(in8, weight8, q_got, scale_got, alpha, eps)
    torch.cuda.synchronize()
    # Compare in dequantized space (a single int8 level near boundaries can differ
    # by 1 due to fp rounding order); scale should match closely.
    scale_diff = (scale_got - scale_ref).abs().max().item()
    dq_got = q_got.to(torch.float32) * scale_got[:, None]
    dq_ref = norm_ref
    dq_diff = (dq_got - dq_ref).abs().max().item()
    lvl_mismatch = (q_got.to(torch.int32) - q_ref.to(torch.int32)).abs()
    lvl_bad = (lvl_mismatch > 1).float().mean().item()
    # Correctness: per-token scale must match, and no quantized level may differ by
    # more than 1 (a single-level delta is just fp round-to-nearest tie-breaking at a
    # bin boundary; dq_diff is therefore ~one quant step by construction, not an error).
    scale_tol = 1e-4
    print(f"[rmsnorm_quant_int8] scale max diff: {scale_diff:.3e} (tol {scale_tol:.1e})  "
          f"dequant max diff: {dq_diff:.3e} (~1 quant step, informational)  levels>1 off: {lvl_bad*100:.3f}%")
    ok &= (scale_diff < scale_tol) and (lvl_bad == 0.0)

    print("PASS" if ok else "FAIL")

    # ---- timing ----
    print("\n>> timing (mean over 100 iters, after 20 warmup)")
    print(f"{'kernel':>20} {'tokens':>8} {'hidden':>7} {'ms/call':>10} {'GB/s':>8}")
    for n_tokens, hidden_dim in [(512, 512), (4096, 768), (8192, 1024)]:
        in_ = torch.randn(n_tokens, hidden_dim, device=dev, dtype=torch.float16)
        residual = torch.randn(n_tokens, hidden_dim, device=dev, dtype=torch.float16)
        weight = torch.randn(hidden_dim, device=dev, dtype=torch.float16)
        out = torch.empty_like(in_)
        ms = mod.rmsnorm_bench(in_, residual, weight, out, alpha, eps, 20, 100)
        # in + residual read (2B each) + out write (2B) + weight read (2B)
        bytes_moved = n_tokens * hidden_dim * (2 + 2 + 2) + hidden_dim * 2
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(f"{'rmsnorm':>20} {n_tokens:>8} {hidden_dim:>7} {ms:>10.4f} {gbps:>8.1f}")

        residual8 = torch.randint(-127, 128, (n_tokens, hidden_dim), device=dev, dtype=torch.int8)
        res_scale = torch.rand(n_tokens, device=dev, dtype=torch.float32) * 0.02 + 0.001
        ms = mod.rmsnorm_quant_int8_bench(in_, weight, residual8, res_scale, alpha, eps, 20, 100)
        # in read (2B) + residual read+write (1B each) + weight read (2B)
        bytes_moved = n_tokens * hidden_dim * (2 + 1 + 1) + hidden_dim * 2
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(f"{'rmsnorm_quant_int8':>20} {n_tokens:>8} {hidden_dim:>7} {ms:>10.4f} {gbps:>8.1f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
