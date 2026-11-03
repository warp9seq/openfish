#ifndef SCORES_H
#define SCORES_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

// ieee fp16 to fp32, exact for all inputs (including subnormals, inf and nan)
// uses the hardware conversion where the compiler knows it is available (aarch64, or x86 built with -mf16c)
#if defined(__FLT16_MANT_DIG__) && (defined(__aarch64__) || defined(__F16C__))
static inline float half_to_float(uint16_t h) {
    _Float16 f;
    memcpy(&f, &h, sizeof(f));
    return (float)f;
}
#else
static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;

    uint32_t bits;
    if (exp == 0x1f) {
        // inf or nan
        bits = sign | 0x7f800000 | (mant << 13);
    } else if (exp != 0) {
        // normal, rebias exponent from 15 to 127
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    } else {
        // zero or subnormal, value is mant * 2^-24
        const float f = (float)mant * 5.9604644775390625e-8f;
        memcpy(&bits, &f, sizeof(bits));
        bits |= sign;
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
#endif

// element type of a scores tensor
typedef enum score_type {
    SCORES_F32,
    SCORES_F16,
} score_type_t;

static inline size_t score_elem_size(score_type_t type) {
    return type == SCORES_F16 ? sizeof(uint16_t) : sizeof(float);
}

// pointer to the score at element offset i
static inline const void *score_ptr(const void *scores, score_type_t type, size_t i) {
    return (const char *)scores + i * score_elem_size(type);
}

// load the score at element offset i as fp32
static inline float load_score(const void *scores, score_type_t type, size_t i) {
    if (type == SCORES_F16) {
        return half_to_float(((const uint16_t *)scores)[i]);
    }
    return ((const float *)scores)[i];
}

#endif // SCORES_H
