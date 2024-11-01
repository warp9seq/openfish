#ifndef DECODE_HIP_H
#define DECODE_HIP_H

#include "decode.h"
#include <openfish/openfish.h>

#ifdef __cplusplus
extern "C" {
#endif

void decode_hip(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

#ifdef __cplusplus
}
#endif

#endif // DECODE_HIP_H