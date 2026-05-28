#include <openfish/openfish.h>
#include "decode.h"

openfish_opt_t openfish_decoder_default_opts(void) {
    openfish_opt_t opt = {32, 100.0f, 2.0f, 0.0f, 1.0f, 1.0f, false};
    return opt;
}

size_t openfish_gpubuf_size(
    const int T,
    const int N,
    const int state_len
) {
    const size_t num_states = (size_t)1 << (2 * state_len);
    return
        sizeof(float) * (size_t)N * (T + 1) * num_states +          // bwd_NTC
        sizeof(float) * (size_t)N * (T + 1) * num_states +          // post_NTC
        sizeof(uint8_t) * (size_t)N * T +                            // moves
        sizeof(char) * (size_t)N * T +                               // sequence
        sizeof(char) * (size_t)N * T +                               // qstring
        sizeof(beam_element_t) * (size_t)N * MAX_BEAM_WIDTH * (T + 1) + // beam_vector
        sizeof(state_t) * (size_t)N * T +                            // states
        sizeof(float) * (size_t)N * T * NUM_BASES +                  // qual_data
        sizeof(float) * (size_t)N * T +                              // base_probs
        sizeof(float) * (size_t)N * T;                               // total_probs
}
