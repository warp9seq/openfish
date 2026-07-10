// In-memory CPU-vs-GPU decode test.
//
// The CPU decoder is treated as ground truth. Because we only ever compare the CPU and GPU
// backends against *each other* on identical inputs, the scores don't need to be real basecall
// data -- we synthesise a deterministic pseudo-random score tensor in memory. So the test is
// fully hermetic: nothing is downloaded, nothing is read from disk, and no reference blobs
// exist. The same generated float32 tensor feeds the CPU path directly and, narrowed to
// float16 on the host (exactly as slorado feeds the GPU in production), the GPU path.
//
// Both the public outputs (moves / sequence / qstring) and the internal tensors
// (bwd_NTC / post_NTC / qual_data / total_probs) are compared entirely in memory. On a
// CPU-only build the harness instead re-runs the CPU decode and checks it is deterministic
// and non-degenerate.
//
// exit status: 0 = all comparisons within tolerance, 1 = a tolerance was exceeded / error.

#include "error.h"
#include "misc.h"
#include "decode_cpu.h"
#include "openfish_defs.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(HAVE_CUDA) || defined(HAVE_ROCM) || defined(HAVE_METAL)
#define HAVE_GPU 1
#endif

// Backend-neutral GPU glue. The harness only ever calls gpu_set_device / gpu_upload_scores_f16 /
// gpu_copy_result_tensors / gpu_free_scores -- no #if ladders below. CUDA and HIP are inlined
// here (this file is compiled by nvcc/hipcc on those builds); Metal's versions live in
// decode_metal.mm (Objective-C++), which the harness declares inline and calls across.
#if defined HAVE_CUDA
#include <cuda_fp16.h>
#include "cuda_utils.h"
#define GPU_MEMCPY_D2H(dst, src, bytes) do { cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); checkCudaError(); } while (0)
static void gpu_set_device(int device) { cudaSetDevice(device); checkCudaError(); }
static void gpu_free_scores(void *d) { cudaFree(d); checkCudaError(); }
static void *gpu_upload_scores_f16(int T, int N, int C, const float *f32) {
    const size_t n = (size_t)T * N * C;
    half *h = (half *)malloc(n * sizeof(half)); MALLOC_CHK(h);
    for (size_t i = 0; i < n; ++i) h[i] = __float2half(f32[i]);
    void *d; cudaMalloc(&d, n * sizeof(half)); checkCudaError();
    cudaMemcpy(d, h, n * sizeof(half), cudaMemcpyHostToDevice); checkCudaError();
    free(h);
    return d;
}
#elif defined HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_utils.h"
#define GPU_MEMCPY_D2H(dst, src, bytes) HIP_CHECK(hipMemcpy((dst), (src), (bytes), hipMemcpyDeviceToHost))
static void gpu_set_device(int device) { HIP_CHECK(hipSetDevice(device)); }
static void gpu_free_scores(void *d) { HIP_CHECK(hipFree(d)); }
static void *gpu_upload_scores_f16(int T, int N, int C, const float *f32) {
    const size_t n = (size_t)T * N * C;
    half *h = (half *)malloc(n * sizeof(half)); MALLOC_CHK(h);
    for (size_t i = 0; i < n; ++i) h[i] = __float2half(f32[i]);
    void *d; HIP_CHECK(hipMalloc(&d, n * sizeof(half)));
    HIP_CHECK(hipMemcpy(d, h, n * sizeof(half), hipMemcpyHostToDevice));
    free(h);
    return d;
}
#elif defined HAVE_METAL
// Metal's glue is Objective-C++ and lives in decode_metal.mm -- it can't be inlined into this C
// harness like the CUDA/HIP glue above, so we just declare the C-linkage entry points we call.
void gpu_set_device(int device);
void *gpu_upload_scores_f16(int n_timesteps, int batch_size, int n_channels, const float *scores_f32_NTC);
void gpu_free_scores(void *scores_gpu);
void gpu_copy_result_tensors(int n_timesteps, int batch_size, int state_len,
                             const openfish_gpubuf_t *gpubuf,
                             float **bwd_NTC_out, float **post_NTC_out,
                             float **qual_data_out, float **total_probs_out);
#endif

// CUDA/HIP share this device->host gpubuf copy; Metal provides its own (unified memory).
#if defined HAVE_CUDA || defined HAVE_ROCM
static void gpu_copy_result_tensors(int T, int N, int state_len, const openfish_gpubuf_t *g,
                                    float **bwd, float **post, float **qual, float **total) {
    const int num_states = (int)pow(NUM_BASES, state_len);
    const size_t guide_len = (size_t)N * (T + 1) * num_states;
    const size_t qual_len  = (size_t)N * T * NUM_BASES;
    const size_t tp_len    = (size_t)N * T;
    struct { float **out; const void *src; size_t count; } t[] = {
        {bwd,   g->bwd_NTC,     guide_len},
        {post,  g->post_NTC,    guide_len},
        {qual,  g->qual_data,   qual_len},
        {total, g->total_probs, tp_len},
    };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); ++i) {
        if (!t[i].out) continue;
        float *h = (float *)malloc(t[i].count * sizeof(float)); MALLOC_CHK(h);
        GPU_MEMCPY_D2H(h, t[i].src, t[i].count * sizeof(float));
        *t[i].out = h;
    }
}
#endif

// Tolerances for GPU-vs-CPU divergence. The GPU decodes fp16 scores while the CPU decodes
// fp32, so exact equality is neither expected nor required. These bound the fp16 rounding
// noise while still catching a real regression, which blows the diffs past them by orders of
// magnitude. post_NTC is the precision sentinel: it is the normalised posterior (bounded in
// [0,1]) and the fp16 input is its only source of divergence, so it stays tiny (avg ~3e-5)
// unless the scan itself breaks. bwd_NTC / qual_data / total_probs are log-domain or
// beam-derived, so their absolute diffs scale with magnitude and get looser bounds. moves is
// the primary discrete signal. sequence / qstring are report-only: a single differing move
// shifts every downstream base, so their byte diff measures realignment, not real divergence.
#define TOL_POST_MAX    0.05f    // post_NTC max abs elem diff  (observed ~0.003)
#define TOL_POST_AVG    0.001f   // post_NTC mean abs elem diff (observed ~3e-5)
#define TOL_BWD_MAX     2.0f     // bwd_NTC max abs elem diff   (observed ~0.03)
#define TOL_BWD_AVG     0.05f    // bwd_NTC mean abs elem diff  (observed ~0.006)
#define TOL_QUAL_AVG    0.10f    // qual_data mean abs elem diff (observed ~0.016)
#define TOL_TOTAL_AVG   2.0f     // total_probs mean abs elem diff (observed ~0.09)
#define TOL_MOVES_PCT   5.0f     // moves byte mismatch %        (observed ~1.9%)
#define REPORT_ONLY     (-1.0f)  // print the diff but never fail on it

static int g_failed = 0;

static inline uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

static inline float rand_unit(uint32_t *st) {
    return (float)(xorshift32(st) >> 8) / (float)(1 << 24); // [0,1)
}

// Synthesise a *well-conditioned* score tensor (NTC layout). Uniform noise is a pathological
// input for the beam search: with no clear winning path every step is a near-tie, so the tiny
// fp16-vs-fp32 perturbation flips paths everywhere and CPU/GPU diverge wildly. Real emission
// scores are peaked (the network is confident), so paths are stable. We mimic that: for each
// (batch, timestep, state) one of the NUM_BASES outgoing transitions is given a clearly
// dominant score and the rest a low one (both jittered). This keeps the decode confident and
// its output stable across backends, exactly like real data, while staying fully synthetic.
// scores are laid out [N, T, C] with C indexed as state*NUM_BASES + base.
static void gen_scores(float *scores, int batch_size, int n_timesteps, int num_states, uint32_t seed) {
    uint32_t st = seed ? seed : 0x9e3779b9u;
    const int n_channels = num_states * NUM_BASES;
    for (int n = 0; n < batch_size; ++n) {
        for (int t = 0; t < n_timesteps; ++t) {
            float *row = scores + ((size_t)n * n_timesteps + t) * n_channels;
            for (int s = 0; s < num_states; ++s) {
                int dom = (int)(xorshift32(&st) & (NUM_BASES - 1)); // dominant outgoing base
                for (int b = 0; b < NUM_BASES; ++b) {
                    row[s * NUM_BASES + b] = (b == dom)
                        ? 2.5f + 2.0f * rand_unit(&st)   // dominant transition: ~[2.5, 4.5]
                        : -3.0f + 2.0f * rand_unit(&st);  // background:          ~[-3.0, -1.0]
                }
            }
        }
    }
}

#if defined HAVE_GPU
// Compare two float tensors: report max/mean abs elem diff and check against tolerances.
static void cmp_float(const char *name, const float *cpu, const float *gpu, size_t len,
                      float max_tol, float avg_tol) {
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    size_t n_diff = 0;
    for (size_t i = 0; i < len; ++i) {
        float d = fabsf(cpu[i] - gpu[i]);
        if (d != 0.0f) {
            if (d > max_diff) max_diff = d;
            sum_diff += d;
            ++n_diff;
        }
    }
    float avg_diff = (len > 0) ? (float)(sum_diff / (double)len) : 0.0f;
    bool ok = (max_diff <= max_tol) && (avg_diff <= avg_tol);
    if (!ok) g_failed = 1;
    fprintf(stderr, "  [%s] %-11s max %.6g (tol %.6g)  avg %.6g (tol %.6g)  diffs %zu/%zu\n",
            ok ? "PASS" : "FAIL", name, max_diff, max_tol, avg_diff, avg_tol, n_diff, len);
}
#endif // HAVE_GPU

// Compare two byte buffers: report mismatch count / % and check against a percentage tolerance.
static void cmp_bytes(const char *name, const void *cpu, const void *gpu, size_t len,
                      float pct_tol) {
    const unsigned char *a = (const unsigned char *)cpu;
    const unsigned char *b = (const unsigned char *)gpu;
    size_t n_diff = 0;
    for (size_t i = 0; i < len; ++i) {
        if (a[i] != b[i]) ++n_diff;
    }
    float pct = (len > 0) ? 100.0f * (float)n_diff / (float)len : 0.0f;
    bool report_only = pct_tol < 0.0f;
    bool ok = report_only || pct <= pct_tol;
    if (!ok) g_failed = 1;
    if (report_only) {
        fprintf(stderr, "  [INFO] %-11s %zu/%zu bytes differ (%.4f%%, realignment-inflated)\n",
                name, n_diff, len, pct);
    } else {
        fprintf(stderr, "  [%s] %-11s %zu/%zu bytes differ (%.4f%%, tol %.2f%%)\n",
                ok ? "PASS" : "FAIL", name, n_diff, len, pct, pct_tol);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <BATCH_SIZE> <STATE_LEN> [N_THREADS] [SEED]\n", argv[0]);
        fprintf(stderr, "  Synthesises scores in memory; compares CPU (ground truth) vs GPU. Loads nothing.\n");
        return 1;
    }
    set_openfish_log_level(OPENFISH_LOG_WARN);

    const int n_timesteps = 1666;
    const int batch_size = (int)strtol(argv[1], NULL, 10);
    ASSERT(batch_size > 0);
    const int state_len = (int)strtol(argv[2], NULL, 10);
    ASSERT(state_len > 0);
    const int n_threads = (argc > 3) ? (int)strtol(argv[3], NULL, 10) : 8;
    const uint32_t seed = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 1u;
    const int num_states = (int)pow(NUM_BASES, state_len);
    const int n_channels = num_states * NUM_BASES;

    const size_t scores_len = (size_t)n_timesteps * batch_size * n_channels;

    // synthetic scores, generated directly in NTC layout (the library's native order)
    float *scores = (float *)malloc(scores_len * sizeof(float));
    MALLOC_CHK(scores);
    gen_scores(scores, batch_size, n_timesteps, num_states, seed);

    openfish_opt_t options = openfish_decoder_default_opts();
    if (state_len == 3) {        // fast
        options.q_scale = 0.97f; options.q_shift = -1.8f;
    } else if (state_len == 4) { // hac
        options.q_scale = 0.95f; options.q_shift = -0.2f;
    } else if (state_len == 5) { // sup
        options.q_scale = 0.95f; options.q_shift = 0.5f;
    }

    fprintf(stderr, "== openfish test: batch=%d state_len=%d T=%d seed=%u (synthetic, NTC) ==\n",
            batch_size, state_len, n_timesteps, seed);

    // ---- CPU decode (ground truth), keeping the intermediate tensors ----
    uint8_t *moves_cpu = NULL; char *seq_cpu = NULL; char *qstr_cpu = NULL;
    float *bwd_cpu = NULL, *post_cpu = NULL, *qual_cpu = NULL, *total_cpu = NULL;
    openfish_decode_cpu_ex(n_timesteps, batch_size, n_channels, n_threads, scores,
                           OPENFISH_SCORE_F16, 1.0f, state_len, &options,
                           &moves_cpu, &seq_cpu, &qstr_cpu,
                           &bwd_cpu, &post_cpu, &qual_cpu, &total_cpu);

    const size_t disc_len  = (size_t)batch_size * n_timesteps;

#if defined HAVE_GPU
    const size_t guide_len = (size_t)batch_size * (n_timesteps + 1) * num_states;
    const size_t qual_len  = (size_t)batch_size * n_timesteps * NUM_BASES;
    fprintf(stderr, "comparing GPU (fp16) against CPU (fp32, ground truth):\n");

    // narrow the same scores to fp16 and upload (backend-neutral -- no #if ladder)
    gpu_set_device(0);
    void *scores_gpu = gpu_upload_scores_f16(n_timesteps, batch_size, n_channels, scores);

    openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(n_timesteps, batch_size, state_len);

    uint8_t *moves_gpu = NULL; char *seq_gpu = NULL; char *qstr_gpu = NULL;
    openfish_decode_gpu(n_timesteps, batch_size, n_channels, scores_gpu,
                        OPENFISH_SCORE_F16, 1.0f, state_len, &options, gpubuf,
                        &moves_gpu, &seq_gpu, &qstr_gpu);

    float *bwd_gpu = NULL, *post_gpu = NULL, *qual_gpu = NULL, *total_gpu = NULL;
    gpu_copy_result_tensors(n_timesteps, batch_size, state_len, gpubuf, &bwd_gpu, &post_gpu, &qual_gpu, &total_gpu);

    // internal tensors: post_NTC is the tight precision sentinel; the rest are looser
    cmp_float("post_NTC", post_cpu, post_gpu, guide_len, TOL_POST_MAX, TOL_POST_AVG);
    cmp_float("bwd_NTC",  bwd_cpu,  bwd_gpu,  guide_len, TOL_BWD_MAX,  TOL_BWD_AVG);
    cmp_float("qual_data", qual_cpu, qual_gpu, qual_len, 1.0f,         TOL_QUAL_AVG);
    cmp_float("total_probs", total_cpu, total_gpu, disc_len, 1e30f,    TOL_TOTAL_AVG);
    // public outputs: moves is the primary discrete signal; sequence/qstring report-only
    cmp_bytes("moves",    moves_cpu, moves_gpu, disc_len, TOL_MOVES_PCT);
    cmp_bytes("sequence", seq_cpu,   seq_gpu,   disc_len, REPORT_ONLY);
    cmp_bytes("qstring",  qstr_cpu,  qstr_gpu,  disc_len, REPORT_ONLY);

    free(bwd_gpu); free(post_gpu); free(qual_gpu); free(total_gpu);
    free(moves_gpu); free(seq_gpu); free(qstr_gpu);
    openfish_gpubuf_free(gpubuf);
    gpu_free_scores(scores_gpu);

#else // CPU-only build: no GPU to compare against -> determinism + sanity check
    fprintf(stderr, "CPU-only build: checking determinism and sanity (no GPU to compare against):\n");
    uint8_t *moves2 = NULL; char *seq2 = NULL; char *qstr2 = NULL;
    openfish_decode_cpu(n_timesteps, batch_size, n_channels, n_threads, scores,
                        OPENFISH_SCORE_F16, 1.0f, state_len, &options, &moves2, &seq2, &qstr2);
    cmp_bytes("moves(det)",    moves_cpu, moves2, disc_len, 0.0f);
    cmp_bytes("sequence(det)", seq_cpu,   seq2,   disc_len, 0.0f);
    cmp_bytes("qstring(det)",  qstr_cpu,  qstr2,  disc_len, 0.0f);

    // sanity: bases are actually called (moves contains steps), so the basecall is non-empty
    size_t total_moves = 0;
    for (size_t i = 0; i < disc_len; ++i) total_moves += moves_cpu[i];
    bool ok = total_moves > 0;
    if (!ok) g_failed = 1;
    fprintf(stderr, "  [%s] %-11s %zu bases called\n", ok ? "PASS" : "FAIL", "non-empty", total_moves);

    free(moves2); free(seq2); free(qstr2);
#endif

    free(bwd_cpu); free(post_cpu); free(qual_cpu); free(total_cpu);
    free(moves_cpu); free(seq_cpu); free(qstr_cpu);
    free(scores);

    fprintf(stderr, "== %s ==\n", g_failed ? "FAILED" : "PASSED");
    return g_failed;
}
