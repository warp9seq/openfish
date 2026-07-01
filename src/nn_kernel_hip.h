// The MIT License (MIT)

// Copyright (c) 2025 Bonson Wong

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef NN_KERNEL_HUI
#define NN_KERNEL_HUI

#include <math.h>
#include <float.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

static __global__ void rotary_emb(
	half *x,
    const float *_cos,
    const float *_sin,
    const uint64_t seq_len,
    const uint64_t stride_batch,
    const uint64_t stride_seq,
    const uint64_t stride_head,
    const uint64_t sincos_width
) {
    const uint64_t batch = blockIdx.x;
    const uint64_t head = blockIdx.y;
    const uint64_t rot = threadIdx.x;
    const uint64_t tid = threadIdx.y;
    const uint64_t n_threads = blockDim.y;

    if (tid >= seq_len) return;

    half *_o0 = x + (batch * stride_batch) + (head * stride_head) + rot;
    half *_o1 = x + (batch * stride_batch) + (head * stride_head) + sincos_width + rot;

    for (int seq = tid; seq < seq_len; seq += n_threads) {
        float cos = *(_cos + (seq * sincos_width) + rot);
        float sin = *(_sin + (seq * sincos_width) + rot);

        half *o0 = _o0 + (seq * stride_seq);
        half *o1 = _o1 + (seq * stride_seq);

        float x0 = __half2float(*o0);
        float x1 = __half2float(*o1);

        *o0 = __float2half(x0 * cos - x1 * sin);
        *o1 = __float2half(x0 * sin + x1 * cos);
    }
}

static __global__ void silu_mul(
	const half *in,
	half *out,
    const uint64_t hidden_dim,
    const uint64_t n_tokens
) {
    uint64_t j = blockIdx.x;

    for (uint64_t k = threadIdx.x; k < hidden_dim; k += blockDim.x) {
        uint64_t i = k + j * (hidden_dim * 2);

        half y = in[i];
        half gate = in[i + hidden_dim];

        float g = __half2float(gate);
        float silu = g / (1.0f + __expf(-g));

        out[k + j * hidden_dim] = __float2half(silu * __half2float(y));
    }
}

// ── Block-wide reductions ─────────────────────────────────────────────────────
// Warp-level primitives, then a block-level reduction that broadcasts the result
// to all threads via a caller-supplied shared scratch buffer (>= blockDim.x/warpSize
// floats). The trailing double-sync leaves `shared` safe to reuse for a second
// reduction in the same kernel. blockDim.x <= 1024 => num_warps <= 16 (warpSize 64),
// so the final reduction fits in a single warp.
static __device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}
static __device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down(v, offset));
    return v;
}
static __device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    v = warp_reduce_sum(v);
    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float r = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) r = warp_reduce_sum(r);
    if (threadIdx.x == 0) shared[0] = r;
    __syncthreads();
    float result = shared[0];
    __syncthreads();
    return result;
}
static __device__ __forceinline__ float block_reduce_max(float v, float* shared) {
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    v = warp_reduce_max(v);
    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float r = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) r = warp_reduce_max(r);
    if (threadIdx.x == 0) shared[0] = r;
    __syncthreads();
    float result = shared[0];
    __syncthreads();
    return result;
}

static __global__ void rmsnorm(
    const half* in,
    const half* residual,
    const half* weight,
    half* out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    int row = blockIdx.x;  // Which sequence/batch element

    if (row >= n_tokens) return;

    // Vectorized half2: each thread owns one adjacent pair. blockDim.x == hidden_dim/2.
    const half2* x = reinterpret_cast<const half2*>(in + row * hidden_dim);
    const half2* res = reinterpret_cast<const half2*>(residual + row * hidden_dim);
    const half2* w2 = reinterpret_cast<const half2*>(weight);
    half2* y = reinterpret_cast<half2*>(out + row * hidden_dim);
    int hd2 = hidden_dim >> 1;

    __shared__ float shared[64];

    float thread_sum = 0.0f;
    float2 v_new; // valid because blockDim.x == hidden_dim/2, so this loop runs exactly once per thread
    for (int i = threadIdx.x; i < hd2; i += blockDim.x) {
        float2 xf = __half22float2(x[i]);
        float2 rf = __half22float2(res[i]);
        float2 val = make_float2(xf.x + rf.x * alpha, xf.y + rf.y * alpha);
        v_new = val;
        thread_sum += val.x * val.x + val.y * val.y;
    }

    float sum_sq = block_reduce_sum(thread_sum, shared);
    float rms_inv = rsqrtf(sum_sq / hidden_dim + eps);

    for (int i = threadIdx.x; i < hd2; i += blockDim.x) {
        float2 wf = __half22float2(w2[i]);
        y[i] = __float22half2_rn(make_float2(v_new.x * rms_inv * wf.x,
                                             v_new.y * rms_inv * wf.y));
    }
}

// Convert E4M3FN fp8 byte (IEEE, PyTorch kFloat8_e4m3fn) to float32.
// NaN encodings 0x7F/0xFF -> 0.0f.
static __device__ __forceinline__ float e4m3fn_to_float(uint8_t b) {
    if ((b & 0x7F) == 0x7F) return 0.0f;
    if ((b & 0x7F) == 0) return 0.0f;
    uint8_t  sign = b >> 7;
    uint8_t  exp  = (b >> 3) & 0xF;
    uint8_t  mant = b & 0x7;
    uint32_t f32_bits;
    if (exp == 0) {
        // Denormal: (-1)^s * mant * 2^(-9)
        float val = (float)mant * 1.953125e-3f;
        f32_bits = __float_as_uint(val) | ((uint32_t)sign << 31);
    } else {
        // Normal: bias 7->127 means +120 to f32 exponent; top-align 3-bit mantissa
        f32_bits = ((uint32_t)sign << 31)
                 | ((uint32_t)(exp + 120) << 23)
                 | ((uint32_t)mant << 20);
    }
    return __uint_as_float(f32_bits);
}

// Convert float32 to E4M3FN fp8 byte (IEEE, PyTorch kFloat8_e4m3fn).
// Values outside [-448, 448] are saturated; NaN is never produced.
static __device__ __forceinline__ uint8_t float_to_e4m3fn(float f) {
    uint32_t bits     = __float_as_uint(f);
    uint32_t sign     = bits >> 31;
    uint32_t f32_exp  = (bits >> 23) & 0xFF;
    uint32_t f32_mant = bits & 0x7FFFFF;

    if (f32_exp == 0) return (uint8_t)(sign << 7);  // zero / float denormal -> fp8 zero

    int e4m3_exp = (int)f32_exp - 120;  // unbiased_f32 = f32_exp-127, E4M3FN biased = +7

    if (e4m3_exp >= 16) return (uint8_t)((sign << 7) | 0x7E);  // saturate to +-448

    if (e4m3_exp > 0) {
        // Normal E4M3FN: round 23-bit float mantissa down to 3 bits
        uint32_t mant3 = (f32_mant + (1U << 19)) >> 20;
        if (mant3 >= 8) { mant3 = 0; ++e4m3_exp; }
        if (e4m3_exp >= 16) return (uint8_t)((sign << 7) | 0x7E);
        if (e4m3_exp == 15 && mant3 == 7) mant3 = 6;  // avoid NaN encoding
        return (uint8_t)((sign << 7) | ((uint32_t)e4m3_exp << 3) | mant3);
    } else {
        // Denormal E4M3FN: value = mant * 2^(-9)
        if (e4m3_exp <= -4) return (uint8_t)(sign << 7);  // underflow to zero
        int shift = 21 - e4m3_exp;  // 21..24
        uint32_t full = (1U << 23) | f32_mant;
        uint32_t mant3 = (full + (1U << (shift - 1))) >> shift;
        if (mant3 >= 8) return (uint8_t)((sign << 7) | (1U << 3));  // round up to normal exp=1
        return (uint8_t)((sign << 7) | mant3);
    }
}

// Combined rmsnorm + fp8 E4M3FN quantization kernel.
// Fuses: dequant fp8 residual -> deepnorm add -> rmsnorm -> quant fp8 residual.
// half2-vectorized: each thread owns one adjacent pair; one block per row (n_tokens rows total).
static __global__ void rmsnorm_quant_fp8(
    const half*  in,
    const half*  weight,
    uint8_t*     residual,
    float*       residual_scale,
    int          n_tokens,
    int          hidden_dim,
    float        alpha,
    float        eps
) {
    int row = blockIdx.x;
    int idx = threadIdx.x;  // owns adjacent pair (2*idx, 2*idx+1); blockDim.x == hidden_dim/2

    if (row >= n_tokens) return;

    const half2* inp      = reinterpret_cast<const half2*>(in + (int64_t)row * hidden_dim);
    const half2* w2       = reinterpret_cast<const half2*>(weight);
    uchar2*      res      = reinterpret_cast<uchar2*>(residual + (int64_t)row * hidden_dim);
    float*       res_scale = residual_scale + row;

    __shared__ float shared[64];

    float2 wf = __half22float2(w2[idx]);

    // Dequantize fp8 residual, add scaled to new in (DeepNorm: in + alpha*residual)
    float rs = *res_scale;
    uchar2 rq = res[idx];
    float2 xf = __half22float2(inp[idx]);
    float2 val = make_float2(xf.x + e4m3fn_to_float(rq.x) * rs * alpha,
                             xf.y + e4m3fn_to_float(rq.y) * rs * alpha);

    // ── Reduce sum-of-squares for RMS ─────────────────────────────────────────
    float sum_sq = block_reduce_sum(val.x * val.x + val.y * val.y, shared);
    float rms_inv = rsqrtf(sum_sq / hidden_dim + eps);

    float2 normalized = make_float2(val.x * rms_inv * wf.x, val.y * rms_inv * wf.y);

    // ── Reduce amax for fp8 quantization scale ────────────────────────────────
    float abs_max = block_reduce_max(fmaxf(fabsf(normalized.x), fabsf(normalized.y)), shared);

    float fp8_scale = fmaxf(abs_max, 1e-12f) / 448.0f;
    if (idx == 0) *res_scale = fp8_scale;  // all threads already read old *res_scale into val

    // ── Write quantized fp8 residual ──────────────────────────────────────────
    uchar2 out;
    out.x = float_to_e4m3fn(fmaxf(-448.0f, fminf(448.0f, normalized.x / fp8_scale)));
    out.y = float_to_e4m3fn(fmaxf(-448.0f, fminf(448.0f, normalized.y / fp8_scale)));
    res[idx] = out;
}

// Standalone f16 → fp8 E4M3FN per-token quantize.
// One block per row, hidden_dim threads per block.
// scale[row] = max(amax/448, 1e-12);  out[row] = clamp(x/scale, -448, 448) as fp8.
static __global__ void quant_fp8(
    const half*  in,       // [n_tokens, hidden_dim] f16 input (row-major)
    uint8_t*     out,      // [n_tokens, hidden_dim] fp8 output
    float*       scale,    // [n_tokens]             per-token scale output
    int          n_tokens,
    int          hidden_dim
) {
    int row = blockIdx.x;
    int idx = threadIdx.x;

    if (row >= n_tokens || idx >= hidden_dim) return;

    float val = __half2float(in[(int64_t)row * hidden_dim + idx]);

    // ── Per-row amax reduce ───────────────────────────────────────────────────
    __shared__ float shared[64];
    float abs_max = block_reduce_max(fabsf(val), shared);

    float fp8_scale = fmaxf(abs_max / 448.0f, 1e-12f);
    if (idx == 0) scale[row] = fp8_scale;

    float fp8_val = fmaxf(-448.0f, fminf(448.0f, val / fp8_scale));
    out[(int64_t)row * hidden_dim + idx] = float_to_e4m3fn(fp8_val);
}

// Dequantize fp8 [n_timesteps,batch_size,n_channels] (× scalar scale) AND transpose to
// f16 [batch_size,n_timesteps,n_channels], in one pass.
//   out[n,t,c] = e4m3fn_to_float(in[t,n,c]) * scale
// One block per (t,n) row, n_channels threads. Reads/writes are coalesced along c (the contiguous
// innermost of both layouts); the transpose lives only in the block→(n,t) index map.
// Replaces torch's .to(kHalf).transpose(0,1).contiguous() (3 ops + intermediates).
static __global__ void dequant_fp8_transpose(
    const uint8_t* in,    // [n_timesteps, batch_size, n_channels] fp8 (row-major, contiguous)
    half*          out,   // [batch_size, n_timesteps, n_channels] f16
    int n_timesteps, int batch_size, int n_channels, float scale
) {
    int tn = blockIdx.x;           // 0 .. n_timesteps*batch_size-1
    int c  = threadIdx.x;          // 0 .. n_channels-1
    if (c >= n_channels) return;
    int t = tn / batch_size;
    int n = tn - t * batch_size;
    float v = e4m3fn_to_float(in[(int64_t)tn * n_channels + c]) * scale;
    out[(((int64_t)n * n_timesteps + t) * n_channels) + c] = __float2half(v);
}

static __global__ void rmsnorm_quant_int8( // need to verify if works on rocm
    const half* in,
    const half* weight,
    int8_t* residual,
    float* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    int row = blockIdx.x;  // Which sequence/batch element
    int idx = threadIdx.x;  // owns adjacent pair (2*idx, 2*idx+1); blockDim.x == hidden_dim/2

    if (row >= n_tokens) return;

    // Vectorized half2 in / weight, char2 int8 residual.
    const half2* inp = reinterpret_cast<const half2*>(in + row * hidden_dim);
    const half2* w2 = reinterpret_cast<const half2*>(weight);
    char2* res = reinterpret_cast<char2*>(residual + row * hidden_dim);
    float* res_scale = residual_scale + row;

    __shared__ float shared[64];

    float2 wf = __half22float2(w2[idx]);

    // Step 1: RMS over (in + alpha * dequant(residual))
    float2 xf = __half22float2(inp[idx]);
    char2 rq = res[idx];
    float rs = *res_scale;
    float2 val = make_float2(xf.x + ((float)rq.x * rs) * alpha,
                             xf.y + ((float)rq.y * rs) * alpha);
    float sum_sq = block_reduce_sum(val.x * val.x + val.y * val.y, shared);
    float rms_inv = rsqrtf(sum_sq / hidden_dim + eps);

    // Step 2: amax of the normalized values for the output quant scale
    float2 normalized = make_float2(val.x * rms_inv * wf.x, val.y * rms_inv * wf.y);
    float abs_max = block_reduce_max(fmaxf(fabsf(normalized.x), fabsf(normalized.y)), shared);

    float quant_scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
    if (idx == 0) *res_scale = 1.0f / quant_scale;  // all threads already read old *res_scale into val

    // clamp and write quantized norm
    char2 q;
    q.x = (int8_t)max(-127, min(127, __float2int_rn(normalized.x * quant_scale)));
    q.y = (int8_t)max(-127, min(127, __float2int_rn(normalized.y * quant_scale)));
    res[idx] = q;
}

static __global__ void flstm_step(
    const half* scratch,
    const half* ih_t,
    half* cell,
    half* hh_next,
    int gate_dim, int hidden_dim
) {
    int n = blockIdx.x;
    for (int ch = threadIdx.x; ch < hidden_dim; ch += blockDim.x) {
        int base = n * gate_dim + ch;
        float gi = __half2float(scratch[base + 0*hidden_dim]) + __half2float(ih_t[base + 0*hidden_dim]);
        float gf = __half2float(scratch[base + 1*hidden_dim]) + __half2float(ih_t[base + 1*hidden_dim]);
        float gg = __half2float(scratch[base + 2*hidden_dim]) + __half2float(ih_t[base + 2*hidden_dim]);
        float go = __half2float(scratch[base + 3*hidden_dim]) + __half2float(ih_t[base + 3*hidden_dim]);
        float i_g = fmaxf(0.f, fminf(1.f, gi * 0.2f + 0.5f));
        float f_g = fmaxf(0.f, fminf(1.f, gf * 0.2f + 0.5f));
        float g_g = fmaxf(-1.f, fminf(1.f, gg));
        float o_g = fmaxf(0.f, fminf(1.f, go * 0.2f + 0.5f));
        float c_new = f_g * __half2float(cell[n * hidden_dim + ch]) + i_g * g_g;
        cell[n * hidden_dim + ch]    = __float2half(c_new);
        hh_next[n * hidden_dim + ch] = __float2half(o_g * tanhf(c_new));
    }
}

#ifdef __cplusplus
}
#endif

#endif // NN_KERNEL_HUI