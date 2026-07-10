#ifndef DECODE_CPU_H
#define DECODE_CPU_H

// Private (non-public) CPU decode entry point used by the in-memory test harness.
// It mirrors openfish_decode_cpu() but can also hand back the intermediate
// backward-guide / posterior / qual / total-probability tensors so the harness can
// compare them against the GPU path without any disk round-trip. Pass NULL for any
// *_out you don't need; the matching buffer is then freed internally. The caller owns
// (and must free) every non-NULL buffer that is returned.

#include <openfish/openfish.h>

#ifdef __cplusplus
extern "C" {
#endif

void openfish_decode_cpu_ex(
    int n_timesteps,
    int batch_size,
    int n_channels,
    int n_threads,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring,
    float **bwd_NTC_out,
    float **post_NTC_out,
    float **qual_data_out,
    float **total_probs_out
);

#ifdef __cplusplus
}
#endif

#endif // DECODE_CPU_H
