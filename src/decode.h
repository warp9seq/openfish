#ifndef DECODE_H
#define DECODE_H

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_BASE_BITS (2)
#define NUM_BASES (4)
#define NUM_TRANSITIONS (NUM_BASES + 1)
#define MAX_BEAM_WIDTH (32)
#define HASH_PRESENT_BITS (4096)
#define HASH_PRESENT_MASK (HASH_PRESENT_BITS - 1)
#define MAX_STATES (1024)
#define MAX_BEAM_CANDIDATES (NUM_TRANSITIONS * MAX_BEAM_WIDTH)
#define CRC_SEED (0x12345678u)

#ifdef __cplusplus
}
#endif

#endif // DECODE_H