// Metal (Apple Silicon) GPU backend for CRF-CTC decoding.
//
// Mirrors decode_cuda.c / decode_hip.c: one threadgroup per chunk, five compute kernels
// dispatched in sequence (bwd_scan -> beam_search -> fwd_post_scan -> compute_qual_data ->
// generate_sequence). The MSL kernels live in openfish_metal.metal and are compiled at
// runtime with newLibraryWithSource: (no offline metal toolchain required).
//
// On Apple Silicon all buffers use MTLResourceStorageModeShared (unified memory), so there
// are no explicit host<->device copies: gpubuf's public raw pointers alias the shared buffer
// contents directly, and results are read straight out of those buffers.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern "C" {
#include <openfish/openfish.h>
#include <openfish/openfish_error.h>
#include "error.h"
}

#include <stdint.h>
#include <string.h>
#include <math.h>

// constants, state_t, beam structs and the scan_params_t / beam_params_t argument blocks,
// shared verbatim with the shader (see openfish_defs.h).
#include "openfish_defs.h"

// MSL source string generated from openfish_metal.metal at build time (see Makefile).
#include "openfish_metal_src.h"

// internal handle: the public struct MUST be the first member so a openfish_gpubuf_t*
// can be cast back to metal_gpubuf*.
struct metal_gpubuf {
    openfish_gpubuf_t pub;
    id<MTLBuffer> bwd_NTC;
    id<MTLBuffer> post_NTC;
    id<MTLBuffer> moves;
    id<MTLBuffer> sequence;
    id<MTLBuffer> qstring;
    id<MTLBuffer> beam_vector;
    id<MTLBuffer> states;
    id<MTLBuffer> qual_data;
    id<MTLBuffer> base_probs;
    id<MTLBuffer> total_probs;
    int n_timesteps;
    int batch_size;
    int state_len;
};

// -------------------------------------------------------------- global metal state

static id<MTLDevice>              g_device = nil;
static id<MTLCommandQueue>        g_queue  = nil;
static id<MTLComputePipelineState> g_pso_bwd  = nil;
static id<MTLComputePipelineState> g_pso_fwd  = nil;
static id<MTLComputePipelineState> g_pso_beam = nil;
static id<MTLComputePipelineState> g_pso_qual = nil;
static id<MTLComputePipelineState> g_pso_gen  = nil;
static bool g_init = false;

static id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, const char *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        OPENFISH_ERROR("metal: kernel function '%s' not found", name);
        exit(EXIT_FAILURE);
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        OPENFISH_ERROR("metal: pipeline for '%s' failed: %s", name, [[err localizedDescription] UTF8String]);
        exit(EXIT_FAILURE);
    }
    return pso;
}

static void ensure_metal_init(void) {
    if (g_init) return;

    // sanity: the persistent buffers store these structs, so device layout must match host.
    static_assert(sizeof(beam_element_t) == 8, "beam_element_t layout must match MSL");
    // setBytes: payloads must match the shader's scan_params_t / beam_params_t.
    static_assert(sizeof(scan_params_t) == 40, "scan_params_t layout must match MSL");
    static_assert(sizeof(beam_params_t) == 32, "beam_params_t layout must match MSL");

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        OPENFISH_ERROR("%s", "metal: no system default GPU device");
        exit(EXIT_FAILURE);
    }
    g_queue = [g_device newCommandQueue];

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:OPENFISH_METAL_SRC];
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [g_device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        OPENFISH_ERROR("metal: shader compile failed: %s", [[err localizedDescription] UTF8String]);
        exit(EXIT_FAILURE);
    }

    g_pso_bwd  = make_pso(lib, "bwd_scan");
    g_pso_fwd  = make_pso(lib, "fwd_post_scan");
    g_pso_beam = make_pso(lib, "beam_search");
    g_pso_qual = make_pso(lib, "compute_qual_data");
    g_pso_gen  = make_pso(lib, "generate_sequence");

    OPENFISH_LOG_TRACE("metal: initialised on device %s", [[g_device name] UTF8String]);
    g_init = true;
}

static id<MTLBuffer> new_shared_buffer(size_t bytes) {
    id<MTLBuffer> b = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) {
        OPENFISH_ERROR("metal: failed to allocate %zu byte buffer", bytes);
        exit(EXIT_FAILURE);
    }
    return b;
}

// -------------------------------------------------------------- gpubuf lifecycle

extern "C" openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    ensure_metal_init();

    metal_gpubuf *mg = new metal_gpubuf();
    mg->n_timesteps = n_timesteps;
    mg->batch_size = batch_size;
    mg->state_len = state_len;

    const int num_states = (int)pow(NUM_BASES, state_len);

    mg->bwd_NTC     = new_shared_buffer(sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states);
    mg->post_NTC    = new_shared_buffer(sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states);
    mg->moves       = new_shared_buffer(sizeof(uint8_t) * (size_t)batch_size * n_timesteps);
    mg->sequence    = new_shared_buffer(sizeof(char) * (size_t)batch_size * n_timesteps);
    mg->qstring     = new_shared_buffer(sizeof(char) * (size_t)batch_size * n_timesteps);
    mg->beam_vector = new_shared_buffer(sizeof(beam_element_t) * (size_t)batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1));
    mg->states      = new_shared_buffer(sizeof(state_t) * (size_t)batch_size * n_timesteps);
    mg->qual_data   = new_shared_buffer(sizeof(float) * (size_t)batch_size * n_timesteps * NUM_BASES);
    mg->base_probs  = new_shared_buffer(sizeof(float) * (size_t)batch_size * n_timesteps);
    mg->total_probs = new_shared_buffer(sizeof(float) * (size_t)batch_size * n_timesteps);

    // public raw pointers alias the shared buffer contents (unified memory).
    mg->pub.bwd_NTC     = (float *)[mg->bwd_NTC contents];
    mg->pub.post_NTC    = (float *)[mg->post_NTC contents];
    mg->pub.moves       = (uint8_t *)[mg->moves contents];
    mg->pub.sequence    = (char *)[mg->sequence contents];
    mg->pub.qstring     = (char *)[mg->qstring contents];
    mg->pub.beam_vector = [mg->beam_vector contents];
    mg->pub.states      = [mg->states contents];
    mg->pub.qual_data   = (float *)[mg->qual_data contents];
    mg->pub.base_probs  = (float *)[mg->base_probs contents];
    mg->pub.total_probs = (float *)[mg->total_probs contents];

    return &mg->pub;
}

extern "C" void openfish_gpubuf_free(openfish_gpubuf_t *gpubuf) {
    if (!gpubuf) return;
    metal_gpubuf *mg = (metal_gpubuf *)gpubuf;
    delete mg; // ARC releases the id<MTLBuffer> members
}

// -------------------------------------------------------------- decode

extern "C" void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_TNC,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
    ensure_metal_init();

    metal_gpubuf *mg = (metal_gpubuf *)gpubuf;
    id<MTLBuffer> scores = (__bridge id<MTLBuffer>)scores_TNC;

    const int num_states = (int)pow(NUM_BASES, state_len);
    const int num_state_bits = (int)log2((double)num_states);

    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;
    const float score_scale = 1.0f;
    const float posts_scale = 1.0f;

    OPENFISH_LOG_TRACE("scores tensor dim: %d, %d, %d", n_timesteps, batch_size, n_channels);

    // host result buffers (freed by the caller, matching the CUDA/CPU API contract)
    *moves    = (uint8_t *)malloc((size_t)batch_size * n_timesteps * sizeof(uint8_t));
    MALLOC_CHK(*moves);
    *sequence = (char *)malloc((size_t)batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*sequence);
    *qstring  = (char *)malloc((size_t)batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*qstring);

    // zero the output buffers (trailing positions past seq_len must stay 0)
    memset([mg->moves contents], 0, (size_t)batch_size * n_timesteps * sizeof(uint8_t));
    memset([mg->sequence contents], 0, (size_t)batch_size * n_timesteps * sizeof(char));
    memset([mg->qstring contents], 0, (size_t)batch_size * n_timesteps * sizeof(char));

    scan_params_t scan_args = {0};
    scan_args.num_states = num_states;
    scan_args.n_timesteps = n_timesteps;
    scan_args.batch_size = batch_size;
    scan_args.n_channels = n_channels;
    scan_args.fixed_stay_score = fixed_stay_score;

    beam_params_t beam_args = {0};
    beam_args.n_timesteps = n_timesteps;
    beam_args.batch_size = batch_size;
    beam_args.n_channels = n_channels;
    beam_args.num_state_bits = num_state_bits;

    // the compact_offsets prefix-sum view overlays cand_scratch in the beam_search kernel,
    // so the int offsets must fit within the bool bloom-filter storage.
    ASSERT(MAX_BEAM_CANDIDATES * sizeof(int) <= HASH_PRESENT_BITS * sizeof(bool));

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        const MTLSize grid = MTLSizeMake(batch_size, 1, 1);

        // ---- bwd scan (1 thread per state) ----
        [enc setComputePipelineState:g_pso_bwd];
        [enc setBytes:&scan_args length:sizeof(scan_args) atIndex:0];
        [enc setBuffer:scores offset:0 atIndex:1];
        [enc setBuffer:mg->bwd_NTC offset:0 atIndex:2];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(num_states, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- beam search (MAX_BEAM_WIDTH * NUM_BASES threads) ----
        [enc setComputePipelineState:g_pso_beam];
        [enc setBytes:&beam_args length:sizeof(beam_args) atIndex:0];
        [enc setBuffer:scores offset:0 atIndex:1];
        [enc setBuffer:mg->bwd_NTC offset:0 atIndex:2];
        [enc setBuffer:mg->states offset:0 atIndex:3];
        [enc setBuffer:mg->moves offset:0 atIndex:4];
        [enc setBuffer:mg->beam_vector offset:0 atIndex:5];
        [enc setBytes:&beam_cut length:sizeof(float) atIndex:6];
        [enc setBytes:&fixed_stay_score length:sizeof(float) atIndex:7];
        [enc setBytes:&score_scale length:sizeof(float) atIndex:8];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(MAX_BEAM_WIDTH * NUM_BASES, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- fwd + post scan (1 thread per state) ----
        [enc setComputePipelineState:g_pso_fwd];
        [enc setBytes:&scan_args length:sizeof(scan_args) atIndex:0];
        [enc setBuffer:scores offset:0 atIndex:1];
        [enc setBuffer:mg->bwd_NTC offset:0 atIndex:2];
        [enc setBuffer:mg->post_NTC offset:0 atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(num_states, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- compute qual data (1 thread per chunk) ----
        [enc setComputePipelineState:g_pso_qual];
        [enc setBytes:&beam_args length:sizeof(beam_args) atIndex:0];
        [enc setBuffer:mg->post_NTC offset:0 atIndex:1];
        [enc setBuffer:mg->states offset:0 atIndex:2];
        [enc setBuffer:mg->qual_data offset:0 atIndex:3];
        [enc setBytes:&posts_scale length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- generate sequence (1 thread per chunk) ----
        [enc setComputePipelineState:g_pso_gen];
        [enc setBytes:&beam_args length:sizeof(beam_args) atIndex:0];
        [enc setBuffer:mg->moves offset:0 atIndex:1];
        [enc setBuffer:mg->states offset:0 atIndex:2];
        [enc setBuffer:mg->qual_data offset:0 atIndex:3];
        [enc setBuffer:mg->base_probs offset:0 atIndex:4];
        [enc setBuffer:mg->total_probs offset:0 atIndex:5];
        [enc setBuffer:mg->sequence offset:0 atIndex:6];
        [enc setBuffer:mg->qstring offset:0 atIndex:7];
        [enc setBytes:&q_shift length:sizeof(float) atIndex:8];
        [enc setBytes:&q_scale length:sizeof(float) atIndex:9];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            OPENFISH_ERROR("metal: command buffer error: %s", [[cb.error localizedDescription] UTF8String]);
            exit(EXIT_FAILURE);
        }
    }

    // copy results out of the shared buffers into the caller-owned arrays
    memcpy(*moves,    [mg->moves contents],    (size_t)batch_size * n_timesteps * sizeof(uint8_t));
    memcpy(*sequence, [mg->sequence contents], (size_t)batch_size * n_timesteps * sizeof(char));
    memcpy(*qstring,  [mg->qstring contents],  (size_t)batch_size * n_timesteps * sizeof(char));
}

// -------------------------------------------------------------- test-harness helpers (test_utils_metal.h)

extern "C" void set_device_metal(int device) {
    (void)device; // Metal uses the system default device; no per-index selection
    ensure_metal_init();
}

// scores_TNC is host float16 [T,N,C] (matching the CUDA/ROCm GPU path); upload into a shared
// buffer. returns a +1 retained MTLBuffer bridged to void* (release with free_scores_metal).
extern "C" void *upload_scores_to_metal(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_TNC
) {
    ensure_metal_init();
    const size_t bytes = (size_t)n_timesteps * batch_size * n_channels * sizeof(uint16_t);
    id<MTLBuffer> buf = new_shared_buffer(bytes);
    memcpy([buf contents], scores_TNC, bytes);
    return (void *)CFBridgingRetain(buf);
}

extern "C" void free_scores_metal(void *scores_gpu) {
    if (scores_gpu) {
        CFBridgingRelease(scores_gpu);
    }
}

#ifdef DEBUG
extern "C" void write_gpubuf_metal(
    uint64_t n_timesteps,
    uint64_t batch_size,
    int state_len,
    const openfish_gpubuf_t *gpubuf
) {
    const int num_states = (int)pow(NUM_BASES, state_len);
    const size_t tens = (size_t)batch_size * (n_timesteps + 1) * num_states;
    const size_t intens = (size_t)batch_size * n_timesteps;

    struct { const char *name; const void *data; size_t count; size_t elem; } blobs[] = {
        {"bwd_NTC.blob",    gpubuf->bwd_NTC,    tens,               sizeof(float)},
        {"post_NTC.blob",   gpubuf->post_NTC,   tens,               sizeof(float)},
        {"qual_data.blob",  gpubuf->qual_data,  intens * NUM_BASES, sizeof(float)},
        {"base_probs.blob", gpubuf->base_probs, intens,             sizeof(float)},
        {"total_probs.blob",gpubuf->total_probs,intens,             sizeof(float)},
    };
    for (size_t b = 0; b < sizeof(blobs) / sizeof(blobs[0]); ++b) {
        FILE *fp = fopen(blobs[b].name, "w");
        F_CHK(fp, blobs[b].name);
        if (fwrite(blobs[b].data, blobs[b].elem, blobs[b].count, fp) != blobs[b].count) {
            fprintf(stderr, "error writing %s: %s\n", blobs[b].name, strerror(errno));
            exit(EXIT_FAILURE);
        }
        fclose(fp);
    }
}
#endif
