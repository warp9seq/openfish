#ifndef OPENFISH_SIMD_H
#define OPENFISH_SIMD_H

// portable 8-wide float vectors built on gcc/clang vector extensions
// the same code lowers to avx2 or sse on x86 and to neon on arm

#include <stdint.h>
#include <string.h>

#define SIMD_WIDTH (8)

typedef float vfloat_t __attribute__((vector_size(SIMD_WIDTH * sizeof(float))));
typedef int32_t vint_t __attribute__((vector_size(SIMD_WIDTH * sizeof(int32_t))));

// on x86-64 linux, build an avx2 and a baseline clone of each simd function and pick one at load time
// this keeps the binary portable without needing -march flags
// arm doesn't need this as neon is always available on aarch64
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && defined(__linux__)
#define SIMD_TARGET __attribute__((target_clones("avx2", "default")))
#else
#define SIMD_TARGET
#endif

// helpers are always inlined so they are compiled for the target of the calling clone
#define SIMD_INLINE static inline __attribute__((always_inline))

// vectors are never passed through a real call, so the abi warning gcc gives for the baseline clone doesn't apply
// it is emitted wherever the helpers are inlined, so this is left disabled for files that include this header
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

SIMD_INLINE vfloat_t vf_load(const float *ptr) {
    vfloat_t v;
    memcpy(&v, ptr, sizeof(v));
    return v;
}

SIMD_INLINE void vf_store(float *ptr, vfloat_t v) {
    memcpy(ptr, &v, sizeof(v));
}

SIMD_INLINE vfloat_t vf_select(vint_t mask, vfloat_t a, vfloat_t b) {
    return (vfloat_t)((mask & (vint_t)a) | (~mask & (vint_t)b));
}

SIMD_INLINE vfloat_t vf_max(vfloat_t a, vfloat_t b) {
    return vf_select(a > b, a, b);
}

SIMD_INLINE float vf_hmax(vfloat_t v) {
    float m = v[0];
    for (int i = 1; i < SIMD_WIDTH; ++i) {
        m = m > v[i] ? m : v[i];
    }
    return m;
}

SIMD_INLINE float vf_hsum(vfloat_t v) {
    float s = 0.0f;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        s += v[i];
    }
    return s;
}

// cephes expf, max error ~2 ulp
// inputs are clamped to [-87.3, 88.3] so results stay finite and normal
// note: relies on (x + magic) - magic rounding, so this must not be built with -ffast-math
SIMD_INLINE vfloat_t vf_exp(vfloat_t x) {
    const vfloat_t round_magic = (vfloat_t){0} + 12582912.0f; // 1.5 * 2^23

    x = vf_max(x, (vfloat_t){0} - 87.3f);
    x = vf_select(x > 88.3f, (vfloat_t){0} + 88.3f, x);

    // x = n * ln2 + r, with |r| <= ln2 / 2
    vfloat_t n = (x * 1.44269504088896341f + round_magic) - round_magic;
    vfloat_t r = x - n * 0.693359375f;
    r = r + n * 2.12194440e-4f;

    vfloat_t y = (vfloat_t){0} + 1.9875691500e-4f;
    y = y * r + 1.3981999507e-3f;
    y = y * r + 8.3334519073e-3f;
    y = y * r + 4.1665795894e-2f;
    y = y * r + 1.6666665459e-1f;
    y = y * r + 5.0000001201e-1f;
    y = y * (r * r) + r + 1.0f;

    // scale by 2^n
    const vint_t pow2n = (__builtin_convertvector(n, vint_t) + 127) << 23;
    return y * (vfloat_t)pow2n;
}

// cephes logf, max error ~1 ulp
// inputs must be positive, finite and normal
SIMD_INLINE vfloat_t vf_log(vfloat_t x) {
    const vint_t bits = (vint_t)x;

    // x = m * 2^e, with m in [0.5, 1)
    vint_t e = ((bits >> 23) & 0xff) - 126;
    vfloat_t m = (vfloat_t)((bits & 0x007fffff) | 0x3f000000);

    // shift m into [sqrt(0.5), sqrt(2)) and take m - 1
    const vint_t small = m < 0.707106781186547524f;
    e = e + small; // mask is -1 where true
    m = m - 1.0f + vf_select(small, m, (vfloat_t){0});

    const vfloat_t z = m * m;
    vfloat_t y = (vfloat_t){0} + 7.0376836292e-2f;
    y = y * m - 1.1514610310e-1f;
    y = y * m + 1.1676998740e-1f;
    y = y * m - 1.2420140846e-1f;
    y = y * m + 1.4249322787e-1f;
    y = y * m - 1.6668057665e-1f;
    y = y * m + 2.0000714765e-1f;
    y = y * m - 2.4999993993e-1f;
    y = y * m + 3.3333331174e-1f;
    y = y * m * z;

    const vfloat_t fe = __builtin_convertvector(e, vfloat_t);
    y = y - fe * 2.12194440e-4f;
    y = y - 0.5f * z;
    return m + y + fe * 0.693359375f;
}

#endif // OPENFISH_SIMD_H
