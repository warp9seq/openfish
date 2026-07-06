#include "beam_search.h"
#include "error.h"
#include "misc.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <pthread.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

static void backward_scan(const float *scores_in, float *out, const uint64_t chunk, const uint64_t n_timesteps, const uint64_t num_states, const float score_scale) {
    const float fixed_stay_score = 2.0f;

    const uint64_t ts_states = num_states * NUM_BASES;

    // scores are NTC: each batch element's [T,C] block is contiguous, so its base
    // offset is chunk * T * C and successive timesteps are one C-stride apart.
    const float* const chunk_in = scores_in + chunk * n_timesteps * ts_states; // should be half float (for GPU impl)
    float* const chunk_out = out + chunk * (n_timesteps+1) * num_states;
    float* const alpha_init = chunk_out + num_states * n_timesteps;
    for (uint64_t state = 0; state < num_states; ++state) { // (for GPU impl) its 1 thread per state, but below we iterate through all the states on 1 thread
        alpha_init[state] = 0.0f;
    }

    for (uint64_t ts = 0; ts < n_timesteps; ++ts) {
        // threadgroup_barrier(mem_flags::medevice); // synchronize all threads before next time step (for GPU impl)
        const float* const ts_in = chunk_in + ts_states * (n_timesteps - ts - 1);
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
                    ts_in[step_trans_idx_a + base * NUM_BASES] * score_scale;
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            float sum = 0.0f;
            for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
                sum += expf(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + logf(sum);
        }
    }
}

static void forward_scan(const float *scores_in, const float *bwd, float *out, const uint64_t chunk, const uint64_t n_timesteps, const uint64_t num_states, const float score_scale) {
    const uint64_t n_ts = n_timesteps+1; // number of guide/posterior time steps
    const float kFixedStayScore = 2.0f;

    const uint64_t msb = num_states / NUM_BASES;
    const uint64_t ts_states = num_states * NUM_BASES;

    // This batch element's scores (NTC: [T,C] block contiguous per batch element).
    const float *const chunk_scores = scores_in + chunk * n_timesteps * ts_states;

    // Alternating forward guide buffers used for successive time steps.
    float ts_fwd[2][MAX_STATES]; // threadgroup

    // The forward guide input for the first step is 0.
    for (uint64_t state = 0; state < num_states; ++state) {
        ts_fwd[0][state] = 0.0f;
    }
    // threadgroup_barrier(mem_flags::mem_threadgroup); // ------------------------------------------------------------------

    for (uint64_t ts = 0; ts < n_ts; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.
        const uint64_t ts_idx = (chunk * n_ts + ts) * num_states;

        // Alternating TG buffer twiddling.
        const float *const ts_alpha_in = ts_fwd[ts & 1];
        float *const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the fwd/bwd guide product in log space for this time step's
        // posterior. This is required for all n_ts (= n_timesteps + 1) time steps.
        for (uint64_t state = 0; state < num_states; ++state) {
            // The forward guide value at this time step (alpha[ts]), calculated
            // in the previous iteration.
            const float fwd_val = ts_alpha_in[state];
            out[ts_idx + state] = fwd_val + bwd[ts_idx + state];
        }

        // Calculate the next time step's forward guide from this time step's
        // scores and forward guide. It's written to threadgroup memory for use
        // in the next iteration. The guide is only defined over the n_timesteps score
        // time steps, so we skip the final iteration (which would otherwise read
        // past the scores buffer to produce a result that is never read).
        if (ts < n_timesteps) {
            // This time step's scores.
            const float *const ts_scores = chunk_scores + ts_states * ts;

            for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
                const uint64_t stay_state_idx = state;
                const uint64_t step_state_idx_a = state / NUM_BASES;
                const uint64_t step_trans_idx_a = state * NUM_BASES;
                float vals[NUM_TRANSITIONS];
                float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
                for (uint64_t base = 0; base < NUM_BASES; ++base) {
                    vals[base + 1] = ts_alpha_in[step_state_idx_a + base * msb] +
                        ts_scores[step_trans_idx_a + base] * score_scale;
                    fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
                }
                float fwd_sum = 0.0f;
                for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
                    fwd_sum += expf(vals[i] - fwd_max_val);
                }
                ts_alpha_out[state] = fwd_max_val + logf(fwd_sum);
            }
        }
    }
}

static void softmax(const float *fwd, float *out, const uint64_t chunk, const uint64_t n_timesteps, const uint64_t num_states) {
    const uint64_t n_ts = n_timesteps+1; // number of guide/posterior time steps
    for (uint64_t ts = 0; ts < n_ts; ++ts) {
        const uint64_t ts_idx = (chunk * n_ts + ts) * num_states;

        float max_val = fwd[ts_idx];
        for (uint64_t state = 0; state < num_states; ++state) {
            max_val = max_val > fwd[ts_idx + state] ? max_val : fwd[ts_idx + state];
        }

        float exp_sum = 0;
        float exp_vals[num_states];
        for (uint64_t state = 0; state < num_states; ++state) {
            const float val = fwd[ts_idx + state];
            const float exp_val = expf(val - max_val);
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
    const openfish_opt_t *options;
    const float *scores_NTC;
    float score_scale;
    float *bwd_NTC;
    float *post_NTC;
    int32_t start;
    int32_t end;
    int32_t state_len;
    int32_t n_timesteps;
    int32_t batch_size;
    int32_t n_channels;
    state_t *states;
    uint8_t *moves;
    float *qual_data;
    float *base_probs;
    float *total_probs;
    char *sequence;
    char *qstring;
    beam_element_t *beam_vector;
} decode_thread_arg_t;

static void* pthread_single_scan_score(void* voidargs) {
    decode_thread_arg_t* args = (decode_thread_arg_t*)voidargs;

    const int num_states = pow(NUM_BASES, args->state_len);

    const int n_timesteps = args->n_timesteps;

    const float score_scale = args->score_scale;

    for (int c = args->start; c < args->end; c++) {
        backward_scan(args->scores_NTC, args->bwd_NTC, c, n_timesteps, num_states, score_scale);
        // forward_scan writes the fwd/bwd guide product into post_NTC, then softmax
        // normalises it in place. post_NTC doubles as the forward buffer, so no
        // separate fwd_NTC allocation is needed (mirrors the fused GPU fwd_post_scan).
        forward_scan(args->scores_NTC, args->bwd_NTC, args->post_NTC, c, n_timesteps, num_states, score_scale);
        softmax(args->post_NTC, args->post_NTC, c, n_timesteps, num_states);
    }

    pthread_exit(0);
}

static void *pthread_single_beam_search(void *voidargs) {
    decode_thread_arg_t *args = (decode_thread_arg_t *)voidargs;
    const openfish_opt_t *options = args->options;
    
    const int num_states = pow(NUM_BASES, args->state_len);
    const int num_state_bits = (int)log2(num_states);
    const int n_timesteps = args->n_timesteps;
    const int n_channels = args->n_channels;

    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;

    for (int c = args->start; c < args->end; c++) {
        const float *scores = args->scores_NTC + c * n_timesteps * (num_states * NUM_BASES);
        float *bwd = args->bwd_NTC + c * num_states * (n_timesteps+1);
        float *post = args->post_NTC + c * num_states * (n_timesteps+1);
        state_t *states = args->states + c * n_timesteps;
        uint8_t *moves = args->moves + c * n_timesteps;
        float *qual_data = args->qual_data + c * (n_timesteps * NUM_BASES);
        float *base_probs = args->base_probs + c * n_timesteps;
        float *total_probs = args->total_probs + c * n_timesteps;
        char *sequence = args->sequence + c * n_timesteps;
        char *qstring = args->qstring + c * n_timesteps;
        beam_element_t *beam_vector = args->beam_vector + c * MAX_BEAM_WIDTH * (n_timesteps+1);

        openfish_beam_search_cpu(scores, n_channels, bwd, post, num_state_bits, n_timesteps, beam_cut, fixed_stay_score, states, moves, qual_data, args->score_scale, 1.0f, beam_vector);

        size_t seq_len = 0;
        for (int i = 0; i < n_timesteps; ++i) {
            seq_len += moves[i];
            total_probs[i] = 0;
            base_probs[i] = 0;
        }

        openfish_generate_sequence_cpu(moves, states, qual_data, q_shift, q_scale, n_timesteps, seq_len, base_probs, total_probs, sequence, qstring);
    }

    pthread_exit(0);
}

void openfish_decode_cpu(
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
    char **qstring
) {
    const int num_states = pow(NUM_BASES, state_len);

    // The CPU pipeline operates on float32 scores. For int8 input we widen and
    // dequantize once up front, then run the standard float path with the scale already applied.
    // This keeps the float hot loops untouched (float decode stays byte-identical) at the cost of
    // one float scratch copy of the scores. slorado still calls the CPU path with float scores.
    const float *scores_f32;
    float *dequant_scores = NULL;
    float pipeline_scale;
    if (score_dtype == OPENFISH_SCORE_I8) {
        const size_t n = (size_t)batch_size * n_timesteps * n_channels;
        dequant_scores = (float *)malloc(n * sizeof(float));
        MALLOC_CHK(dequant_scores);
        const int8_t *q = (const int8_t *)scores_NTC;
        for (size_t i = 0; i < n; ++i) {
            dequant_scores[i] = (float)q[i] * score_scale;
        }
        scores_f32 = dequant_scores;
        pipeline_scale = 1.0f; // dequant already folded in above
    } else {
        scores_f32 = (const float *)scores_NTC;
        pipeline_scale = score_scale;
    }

    OPENFISH_LOG_TRACE("scores tensor dim (NTC): %d, %d, %d", batch_size, n_timesteps, n_channels);

    float *bwd_NTC = (float *)calloc(batch_size * (n_timesteps + 1) * num_states, sizeof(float));
    float *post_NTC = (float *)calloc(batch_size * (n_timesteps + 1) * num_states, sizeof(float));

    // init results
    *moves = (uint8_t *)calloc(batch_size * n_timesteps, sizeof(uint8_t));
    MALLOC_CHK(*moves);

    *sequence = (char *)calloc(batch_size * n_timesteps, sizeof(char));
    MALLOC_CHK(*sequence);

    *qstring = (char *)calloc(batch_size * n_timesteps, sizeof(char));
    MALLOC_CHK(*qstring);

    // intermediate
    beam_element_t *beam_vector = (beam_element_t *)malloc(batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1) * sizeof(beam_element_t));
    MALLOC_CHK(beam_vector);

    state_t *states = (state_t *)malloc(batch_size * n_timesteps * sizeof(state_t));
    MALLOC_CHK(states);

    float *qual_data = (float *)malloc(batch_size * n_timesteps * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);

    float *base_probs = (float *)malloc(batch_size * n_timesteps * sizeof(float));
    MALLOC_CHK(base_probs);

    float *total_probs = (float *)malloc(batch_size * n_timesteps * sizeof(float));
    MALLOC_CHK(total_probs);
    
    // create threads
    n_threads = batch_size < n_threads ? batch_size : n_threads;
    const int chunks_per_thread = batch_size / n_threads;
    const int num_threads_with_one_more_chunk = batch_size % n_threads;

    OPENFISH_LOG_TRACE("dispatching %d threads for cpu decoding", n_threads);

    pthread_t tids[n_threads];
    decode_thread_arg_t pt_args[n_threads];
    int32_t t, ret;

    // set the data structures
    for (t = 0; t < n_threads; t++) {
        int extra = t < num_threads_with_one_more_chunk ? t : num_threads_with_one_more_chunk;
        pt_args[t].start = t * chunks_per_thread + extra;
        pt_args[t].end = pt_args[t].start + chunks_per_thread + (int)(t < num_threads_with_one_more_chunk);
        pt_args[t].scores_NTC = scores_f32;
        pt_args[t].score_scale = pipeline_scale;
        pt_args[t].bwd_NTC = bwd_NTC;
        pt_args[t].post_NTC = post_NTC;
        pt_args[t].options = options;
        pt_args[t].state_len = state_len;
        pt_args[t].n_timesteps = n_timesteps;
        pt_args[t].batch_size = batch_size;
        pt_args[t].n_channels = n_channels;
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
    for (t = 0; t < n_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_scan_score, (void *)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < n_threads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

    // beam search
    for (t = 0; t < n_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_beam_search, (void *)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < n_threads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

#ifdef DEBUG
    // write tensors
    FILE *fp;
    
    fp = fopen("bwd_NTC.blob", "w");
    F_CHK(fp, "bwd_NTC.blob");
    if (fwrite(bwd_NTC, sizeof(float), batch_size * (n_timesteps + 1) * num_states, fp) != batch_size * (n_timesteps + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    F_CHK(fp, "post_NTC.blob");
    if (fwrite(post_NTC, sizeof(float), batch_size * (n_timesteps + 1) * num_states, fp) != batch_size * (n_timesteps + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    // write beam results
    fp = fopen("qual_data.blob", "w");
    F_CHK(fp, "qual_data.blob");
    if (fwrite(qual_data, sizeof(float), batch_size * n_timesteps * NUM_BASES, fp) != batch_size * n_timesteps * NUM_BASES) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    F_CHK(fp, "base_probs.blob");
    if (fwrite(base_probs, sizeof(float), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    F_CHK(fp, "total_probs.blob");
    if (fwrite(total_probs, sizeof(float), batch_size * n_timesteps, fp) != batch_size * n_timesteps) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);
#endif

    // cleanup
    free(dequant_scores); // NULL for the float path
    free(bwd_NTC);
    free(post_NTC);

    free(beam_vector);
    free(qual_data);
    free(states);
    free(base_probs);
    free(total_probs);
}
