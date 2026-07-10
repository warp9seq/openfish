#ifndef OPENFISH_DEFS_H
#define OPENFISH_DEFS_H

// Single source of truth for the decoding constants, index/state types, beam structs, and the
// kernel-argument blocks. Shared by every backend:
//   - CPU / CUDA / HIP  #include this header directly
//   - Metal host glue    (decode_metal.mm) includes this header directly
//   - Metal shader       (kernels_metal.metal) receives it by build-time concatenation into the
//                         runtime shader source (newLibraryWithSource: cannot resolve local #includes)
//
// It must therefore stay valid as C, C++, CUDA, HIP and MSL. The fixed-width integer names below are
// native to all of them (MSL provides uint64_t/int32_t/... too); only the standard-library includes
// are gated on #ifdef __METAL_VERSION__ (MSL has no <stdint.h>).

#ifndef __METAL_VERSION__
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>   // size_t
#endif

#define NUM_BASE_BITS (2)
#define NUM_BASES (4)
#define NUM_TRANSITIONS (NUM_BASES + 1)
#define MAX_BEAM_WIDTH (32)
#define HASH_PRESENT_BITS (1024)
#define HASH_PRESENT_MASK (HASH_PRESENT_BITS - 1)
#define MAX_STATES (1024)
#define MAX_BEAM_CANDIDATES (NUM_TRANSITIONS * MAX_BEAM_WIDTH)
#define CRC_SEED (0x12345678u)

typedef int32_t state_t;

typedef struct beam_element {
    state_t state;
    uint8_t prev_element_index;
    bool    stay;
} beam_element_t;

typedef struct beam_front_element {
    uint32_t hash;
    state_t  state;
    uint8_t  prev_element_index;
    bool     stay;
} beam_front_element_t;

// Scalar-only kernel parameter blocks shared by every backend. Device pointers (the score/guide tensors)
// are passed to the kernels separately — as explicit kernel arguments on CUDA/HIP, and bound as separate
// MTLBuffers on Metal — rather than embedded here, so a single definition stays valid across all of them.
// On Metal these are filled host-side via setBytes:, and read device-side as `constant &`.
typedef struct scan_params {
    uint64_t num_states;
    uint64_t n_timesteps;
    uint64_t batch_size;
    uint64_t n_channels;
    float    fixed_stay_score;
    float    score_scale;      // emission-score dequant multiplier (1.0 for the native float path)
} scan_params_t;

typedef struct beam_params {
    uint64_t n_timesteps;
    uint64_t batch_size;
    uint64_t n_channels;
    int32_t  num_state_bits;
} beam_params_t;

#endif // OPENFISH_DEFS_H
