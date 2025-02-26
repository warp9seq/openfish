#ifndef DECODE_HIP_H
#define DECODE_HIP_H

#include "decode.h"
#include <openfish/openfish.h>

#ifdef __cplusplus
extern "C" {
#endif

openfish_gpubuf_t *gpubuf_init_hip(
    const int T,
    const int N,
    const int state_len
);

void gpubuf_free_hip(
    openfish_gpubuf_t *gpubuf
);

void decode_hip(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

void set_device_hip(
    int device
);

void *upload_scores_to_hip(
    const int T,
    const int N,
    const int C,
    const void *scores_TNC
);

void free_scores_hip(
    void *scores_TNC_gpu
);

void write_gpubuf_hip(
    const uint64_t T,
    const uint64_t N,
    const int state_len,
    const openfish_gpubuf_t *gpubuf
);

#ifdef __cplusplus
}
#endif

#endif // DECODE_HIP_H