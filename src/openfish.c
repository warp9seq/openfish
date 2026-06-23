#include <openfish/openfish.h>
#include "decode.h"

openfish_opt_t openfish_decoder_default_opts(void) {
    openfish_opt_t opt = {100.0f, 2.0f, 0.0f, 1.0f};
    return opt;
}

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    const size_t num_states = (size_t)1 << (2 * state_len);
    return
        sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states +          // bwd_NTC
        sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states +          // post_NTC
        sizeof(uint8_t) * (size_t)batch_size * n_timesteps +                            // moves
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // sequence
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // qstring
        sizeof(beam_element_t) * (size_t)batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1) + // beam_vector
        sizeof(state_t) * (size_t)batch_size * n_timesteps +                            // states
        sizeof(float) * (size_t)batch_size * n_timesteps * NUM_BASES +                  // qual_data
        sizeof(float) * (size_t)batch_size * n_timesteps +                              // base_probs
        sizeof(float) * (size_t)batch_size * n_timesteps;                               // total_probs
}
