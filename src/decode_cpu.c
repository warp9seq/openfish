#include "decode_cpu.h"
#include "beam_search.h"
#include "error.h"
#include "misc.h"

#include <openfish/openfish.h>

#include <math.h>
#include <pthread.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

static void backward_scan(const float *scores_in, float *out, const uint64_t chunk, const uint64_t T, const uint64_t N, const uint64_t num_states) {
    const float fixed_stay_score = 2.0f;

    const uint64_t ts_states = num_states * NUM_BASES;

    const float* const chunk_in = scores_in + chunk * ts_states; // should be half float (for GPU impl)
    float* const chunk_out = out + chunk * (T+1) * num_states;
    float* const alpha_init = chunk_out + num_states * T;
    for (uint64_t state = 0; state < num_states; ++state) { // (for GPU impl) its 1 thread per state, but below we iterate through all the states on 1 thread
        alpha_init[state] = 0.0f;
    }

    for (uint64_t ts = 0; ts < T; ++ts) {
        // threadgroup_barrier(mem_flags::medevice); // synchronize all threads before next time step (for GPU impl)
        const float* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        float* const ts_alpha_in = alpha_init - num_states * ts;
        float* const ts_alpha_out = ts_alpha_in - num_states;

        for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = (state * NUM_BASES) % num_states;
            const uint64_t step_trans_idx_a = step_state_idx_a * NUM_BASES +
                ((state * NUM_BASES) / num_states);

            float vals[NUM_TRANSITIONS];
            vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
            float max_val = vals[0];
            for (uint64_t base = 0; base < NUM_BASES; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    ts_in[step_trans_idx_a + base * NUM_BASES];
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            float sum = 0.0f;
            for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
                sum += exp(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + log(sum);
        }
    }
}

static void forward_scan(const float *scores_in, const float *bwd, float *out, const uint64_t chunk, const uint64_t _T, const uint64_t N, const uint64_t num_states) {
    const uint64_t T = _T+1; 
    const float kFixedStayScore = 2.0f;
    
    const uint64_t msb = num_states / NUM_BASES;
    const uint64_t ts_states = num_states * NUM_BASES;

    // This batch element's scores.
    const float *const chunk_scores = scores_in + chunk * ts_states;

    // Alternating forward guide buffers used for successive time steps.
    float ts_fwd[2][MAX_STATES]; // threadgroup

    // The forward guide input for the first step is 0.
    for (uint64_t state = 0; state < num_states; ++state) {
        ts_fwd[0][state] = 0.0f;
    }
    // threadgroup_barrier(mem_flags::mem_threadgroup); // ------------------------------------------------------------------

    for (uint64_t ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        // This time step's scores.
        const float *const ts_scores = chunk_scores + N * ts_states * ts;

        // Alternating TG buffer twiddling.
        const float *const ts_alpha_in = ts_fwd[ts & 1];
        float *const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = state / NUM_BASES;
            const uint64_t step_trans_idx_a = state * NUM_BASES;
            float vals[NUM_TRANSITIONS];
            float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            for (uint64_t base = 0; base < NUM_BASES; ++base) {
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                float ts_score = ts < _T ? ts_scores[step_trans_idx_a + base] : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * msb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            float fwd_sum = 0.0f;
            for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + log(fwd_sum);

            // Load the forward guide value calculated in the last time step for use
            // in this time step's posterior probability calculation.
            const float fwd_val = ts_alpha_in[state];

            // Calculate fwd/bwd guide product in log space.
            const float val = fwd_val + bwd[ts_idx + state];
            out[ts_idx + state] = val;
        }
    }
}

static void softmax(const float *fwd, float *out, const uint64_t chunk, const uint64_t _T, const uint64_t num_states) {
    const uint64_t T = _T+1; 
    for (uint64_t ts = 0; ts < T; ++ts) {
        const uint64_t ts_idx = (chunk * T + ts) * num_states;

        float max_val = fwd[ts_idx];
        for (uint64_t state = 0; state < num_states; ++state) {
            max_val = max_val > fwd[ts_idx + state] ? max_val : fwd[ts_idx + state];
        }

        float exp_sum = 0;
        float exp_vals[num_states];
        for (uint64_t state = 0; state < num_states; ++state) {
            const float val = fwd[ts_idx + state];
            const float exp_val = exp(val - max_val);
            exp_vals[state] = exp_val;
            exp_sum += exp_val;
        }

        for (uint64_t state = 0; state < num_states; ++state) {
            const float exp_val = exp_vals[state];

            // Write out the posterior probability 
            out[ts_idx + state] = (float)(exp_val / exp_sum);
        }
    }
}

typedef struct {
    const decoder_opts_t *options;
    const float *scores_TNC;
    float *bwd_NTC;
    float *fwd_NTC;
    float *post_NTC;
    int32_t start;
    int32_t end;
    int32_t state_len;
    int32_t T;
    int32_t N;
    int32_t C;
    state_t *states;
    uint8_t *moves;
    float *qual_data;
    float *base_probs;
    float *total_probs;
    char *sequence;
    char *qstring;
    beam_element_t *beam_vector;
} decode_thread_arg_t;

void* pthread_single_scan_score(void* voidargs) {
    decode_thread_arg_t* args = (decode_thread_arg_t*)voidargs;

    const int num_states = pow(NUM_BASES, args->state_len);

    const int T = args->T;
    const int N = args->N;

    for (int c = args->start; c < args->end; c++) {
        backward_scan(args->scores_TNC, args->bwd_NTC, c, T, N, num_states);
        forward_scan(args->scores_TNC, args->bwd_NTC, args->fwd_NTC, c, T, N, num_states);
        softmax(args->fwd_NTC, args->post_NTC, c, T, num_states);
    }

    pthread_exit(0);
}

void *pthread_single_beam_search(void *voidargs) {
    decode_thread_arg_t *args = (decode_thread_arg_t *)voidargs;
    const decoder_opts_t *options = args->options;
    
    const int num_states = pow(NUM_BASES, args->state_len);
    const int num_state_bits = (int)log2(num_states);
    const int T = args->T;
    const int N = args->N;
    const int C = args->C;

    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    for (int c = args->start; c < args->end; c++) {
        const float *scores = args->scores_TNC + c * (num_states * NUM_BASES);
        float *bwd = args->bwd_NTC + c * num_states * (T+1);
        float *post = args->post_NTC + c * num_states * (T+1);
        state_t *states = args->states + c * T;
        uint8_t *moves = args->moves + c * T;
        float *qual_data = args->qual_data + c * (T * NUM_BASES);
        float *base_probs = args->base_probs + c * T;
        float *total_probs = args->total_probs + c * T;
        char *sequence = args->sequence + c * T;
        char *qstring = args->qstring + c * T;
        beam_element_t *beam_vector = args->beam_vector + c * MAX_BEAM_WIDTH * (T+1);

        beam_search(scores, N * C, bwd, post, num_state_bits, T, beam_cut, fixed_stay_score, states, moves, qual_data, 1.0f, 1.0f, beam_vector);

        size_t seq_len = 0;
        for (int i = 0; i < T; ++i) {
            seq_len += moves[i];
            total_probs[i] = 0;
            base_probs[i] = 0;
        }

        generate_sequence(moves, states, qual_data, q_shift, q_scale, T, seq_len, base_probs, total_probs, sequence, qstring);
    }

    pthread_exit(0);
}

void decode_cpu(
    const int T,
    const int N,
    const int C,
    const int target_threads,
    float *scores_TNC,
    const int state_len,
    const decoder_opts_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
    // expect input already transformed
    // scores_TNC = scores_TNC.to(torch::kCPU).to(float).transpose(0, 1).contiguous();

    const int num_states = pow(NUM_BASES, state_len);

    OPENFISH_LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    float *bwd_NTC = (float *)calloc(N * (T + 1) * num_states, sizeof(float));
    float *fwd_NTC = (float *)calloc(N * (T + 1) * num_states, sizeof(float));
    float *post_NTC = (float *)calloc(N * (T + 1) * num_states, sizeof(float));

    // init results
    *moves = (uint8_t *)calloc(N * T, sizeof(uint8_t));
    MALLOC_CHK(*moves);

    *sequence = (char *)calloc(N * T, sizeof(char));
    MALLOC_CHK(*sequence);

    *qstring = (char *)calloc(N * T, sizeof(char));
    MALLOC_CHK(*qstring);

    // intermediate
    beam_element_t *beam_vector = (beam_element_t *)malloc(N * MAX_BEAM_WIDTH * (T + 1) * sizeof(beam_element_t));
    MALLOC_CHK(beam_vector);

    state_t *states = (state_t *)malloc(N * T * sizeof(state_t));
    MALLOC_CHK(states);

    float *qual_data = (float *)malloc(N * T * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);

    float *base_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(base_probs);

    float *total_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(total_probs);
    
    // create threads
    const int num_threads = N < target_threads ? N : target_threads;
    const int chunks_per_thread = N / num_threads;
    const int num_threads_with_one_more_chunk = N % num_threads;

    OPENFISH_LOG_DEBUG("dispatching %d threads for cpu decoding", num_threads);

    pthread_t tids[num_threads];
    decode_thread_arg_t pt_args[num_threads];
    int32_t t, ret;

    // set the data structures
    for (t = 0; t < num_threads; t++) {
        int extra = t < num_threads_with_one_more_chunk ? t : num_threads_with_one_more_chunk;
        pt_args[t].start = t * chunks_per_thread + extra;
        pt_args[t].end = pt_args[t].start + chunks_per_thread + (int)(t < num_threads_with_one_more_chunk);
        pt_args[t].scores_TNC = scores_TNC;
        pt_args[t].bwd_NTC = bwd_NTC;
        pt_args[t].fwd_NTC = fwd_NTC;
        pt_args[t].post_NTC = post_NTC;
        pt_args[t].options = options;
        pt_args[t].state_len = state_len;
        pt_args[t].T = T;
        pt_args[t].N = N;
        pt_args[t].C = C;
        pt_args[t].states = states;
        pt_args[t].moves = *moves;
        pt_args[t].qual_data = qual_data;
        pt_args[t].base_probs = base_probs;
        pt_args[t].total_probs = total_probs;
        pt_args[t].sequence = *sequence;
        pt_args[t].qstring = *qstring;
        pt_args[t].beam_vector = beam_vector;
    }

    // score tensors
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_scan_score, (void *)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < num_threads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

    // beam search
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_beam_search, (void *)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < num_threads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

#ifdef DEBUG
    // write tensors
    FILE *fp;
    fp = fopen("scores_TNC.blob", "w");
    fwrite(scores_TNC, sizeof(float), T * N * C, fp);
    fclose(fp);

    fp = fopen("bwd_NTC.blob", "w");
    fwrite(bwd_NTC, sizeof(float), N * (T + 1) * num_states, fp);
    fclose(fp);

    fp = fopen("fwd_NTC.blob", "w");
    fwrite(fwd_NTC, sizeof(float), N * (T + 1) * num_states, fp);
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    fwrite(post_NTC, sizeof(float), N * (T + 1) * num_states, fp);
    fclose(fp);

    // write beam results
    fp = fopen("qual_data.blob", "w");
    fwrite(qual_data, sizeof(float), N * T * NUM_BASES, fp);
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    fwrite(base_probs, sizeof(float), N * T, fp);
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    fwrite(total_probs, sizeof(float), N * T, fp);
    fclose(fp);
#endif

    // cleanup
    free(bwd_NTC);
    free(fwd_NTC);
    free(post_NTC);

    free(beam_vector);
    free(qual_data);
    free(states);
    free(base_probs);
    free(total_probs);
}
