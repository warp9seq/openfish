"""Build-and-run test for the CUDA rotary embedding kernel.

Single script: it JIT-compiles the real CUDA kernel (src/nn_cuda.c) plus a tiny
torch binding with ninja (torch.utils.cpp_extension), runs openfish_rotary_emb_gpu
on the GPU, and diffs the result against a torch reference -> PASS/FAIL.

    ./pyvenv/bin/python test/test_rotary_cuda.py

Requires a CUDA-enabled torch (see test/requirements.txt) and a CUDA toolkit.
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

# TxModel.cpp constants (MultiHeadAttentionImpl / RotaryEmbeddingImpl).
THETA = 10000.0
MAX_SEQ_LEN = 2048
ROTARY_DIM = 32

# C++ binding (compiled by the host compiler). It calls the real
# openfish_rotary_emb_gpu for q (chunk 0) and k (chunk 1), passing the *full
# qkv* strides exactly as TxModel.cpp does. v (chunk 2) is untouched.
CPP_SRC = r"""
#include <torch/extension.h>
#include <openfish/openfish.h>
#include <cuda_runtime.h>

// Run the kernel on q (chunk 0) and k (chunk 1) of qkv, passing the *full qkv*
// strides exactly as TxModel.cpp does. v (chunk 2) is untouched.
// qkv: [N, seq_len, 3, n_heads, head_dim], fp16, CUDA, contiguous.
// sin/cos: [max_seq_len, head_dim/2], fp32, CUDA.
static void run_rotary(torch::Tensor &qkv, torch::Tensor &sin, torch::Tensor &cos, int rotary_dim) {
    const int N        = qkv.size(0);
    const int seq_len  = qkv.size(1);
    const int n_heads  = qkv.size(3);
    const int head_dim = qkv.size(4);
    const int sb = qkv.stride(0);
    const int ss = qkv.stride(1);
    const int sh = qkv.stride(3);
    const int chunk_stride = qkv.stride(2);  // = n_heads * head_dim when contiguous

    char *base = static_cast<char *>(qkv.data_ptr());
    const size_t elem = qkv.element_size();

    // API arg order is (x, sin, cos).
    for (int chunk = 0; chunk < 2; ++chunk) {  // q, k
        openfish_rotary_emb_gpu(
            base + (size_t)chunk * chunk_stride * elem,
            sin.data_ptr(), cos.data_ptr(),
            N, seq_len, n_heads, head_dim, rotary_dim, sb, ss, sh);
    }
}

void rotary_emb(torch::Tensor qkv, torch::Tensor sin, torch::Tensor cos, int64_t rotary_dim) {
    TORCH_CHECK(qkv.is_cuda() && sin.is_cuda() && cos.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qkv.scalar_type() == torch::kHalf, "qkv must be fp16");
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(2) == 3, "qkv must be [N,seq,3,nhead,head_dim]");
    run_rotary(qkv, sin, cos, (int)rotary_dim);
}

// Returns mean milliseconds per rotary_emb call (q+k), timed with CUDA events.
double rotary_emb_bench(torch::Tensor qkv, torch::Tensor sin, torch::Tensor cos,
                        int64_t rotary_dim, int64_t warmup, int64_t iters) {
    for (int i = 0; i < warmup; ++i) run_rotary(qkv, sin, cos, (int)rotary_dim);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) run_rotary(qkv, sin, cos, (int)rotary_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / (double)iters;
}
"""

# Compile the real kernel translation unit straight into the extension (nvcc,
# -fPIC) instead of linking lib/libopenfish.a, whose objects are built without
# -fPIC and so cannot go into a shared object. This still exercises the actual
# openfish_rotary_emb_gpu launch wrapper from src/nn_cuda.c.
CUDA_SRC = r"""
#include "error.c"
#include "nn_cuda.c"
"""


def build_cos_sin(head_dim, device):
    inv_freq = torch.pow(THETA, torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim).reciprocal()
    freqs = torch.outer(torch.arange(MAX_SEQ_LEN, dtype=torch.float64), inv_freq)
    cos = torch.cos(freqs).to(torch.float32).contiguous().to(device)
    sin = torch.sin(freqs).to(torch.float32).contiguous().to(device)
    return cos, sin


def reference(qkv, cos, sin, rotary_dim):
    """Rotate-half RoPE on q and k chunks; fp32 math, fp16 output (matches kernel)."""
    out = qkv.clone()
    seq_len = qkv.size(1)
    c = cos[:seq_len, :rotary_dim].view(1, seq_len, 1, rotary_dim)
    s = sin[:seq_len, :rotary_dim].view(1, seq_len, 1, rotary_dim)
    for chunk in (0, 1):
        x = out[:, :, chunk, :, :]
        x0 = x[..., :rotary_dim].to(torch.float32)
        x1 = x[..., rotary_dim:2 * rotary_dim].to(torch.float32)
        x[..., :rotary_dim] = (x0 * c - x1 * s).to(torch.float16)
        x[..., rotary_dim:2 * rotary_dim] = (x0 * s + x1 * c).to(torch.float16)
    return out


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")

    print(">> JIT-compiling CUDA kernel + binding with ninja")
    mod = load_inline(
        name="openfish_rotary_test",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["rotary_emb", "rotary_emb_bench"],
        extra_include_paths=[os.path.join(ROOT, "include"), os.path.join(ROOT, "src")],
        extra_cflags=["-DHAVE_CUDA=1", "-O2"],
        extra_cuda_cflags=["-DHAVE_CUDA=1", "-O2", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=False,
    )

    dev = "cuda"
    head_dim = 2 * ROTARY_DIM  # = 64, matching the model
    cos, sin = build_cos_sin(head_dim, dev)

    # ---- correctness ----
    torch.manual_seed(0)
    batch, seq_len, n_heads = 4, 128, 8
    qkv = torch.randn(batch, seq_len, 3, n_heads, head_dim, device=dev).to(torch.float16).contiguous()
    expected = reference(qkv, cos, sin, ROTARY_DIM)

    got = qkv.clone()
    mod.rotary_emb(got, sin, cos, ROTARY_DIM)
    torch.cuda.synchronize()

    # v chunk must be untouched.
    assert torch.equal(got[:, :, 2], qkv[:, :, 2]), "v chunk was modified"

    max_diff = (got.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    # fp16 rounding: a single fp16 ulp near magnitude ~4 is ~4e-3, and GPU
    # FMA/rounding order can differ from torch by ~1 ulp, so allow a small margin.
    tol = 5e-3
    print(f"kernel vs reference max abs diff: {max_diff:.3e} (tol {tol:.1e})")
    ok = max_diff < tol
    print("PASS" if ok else "FAIL")

    # ---- timing ----
    # bytes moved per call: q+k each read+write 2*rotary_dim halves per (N,seq,head).
    print("\n>> timing (mean over 100 iters, after 20 warmup)")
    print(f"{'N':>5} {'seq':>6} {'heads':>6} {'ms/call':>10} {'GB/s':>8}")
    for batch, seq_len, n_heads in [(4, 128, 8), (64, 512, 8), (256, 1666, 8), (512, 2048, 8)]:
        qkv = torch.randn(batch, seq_len, 3, n_heads, head_dim, device=dev).to(torch.float16).contiguous()
        ms = mod.rotary_emb_bench(qkv, sin, cos, ROTARY_DIM, 20, 100)
        # 2 chunks * (read+write) * 2*rotary_dim halves(2B) + cos/sin reads (2*rotary_dim*4B)
        elems = batch * seq_len * n_heads
        bytes_moved = 2 * elems * (2 * (2 * ROTARY_DIM) * 2 + (2 * ROTARY_DIM) * 4)
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(f"{batch:>5} {seq_len:>6} {n_heads:>6} {ms:>10.4f} {gbps:>8.1f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
