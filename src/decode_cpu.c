#include "openfish_defs.h"
#include "error.h"
#include "misc.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <float.h>
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

// ==================== beam search (merged from beam_search.c) ====================

static void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

static int partitionf(float *nums, int left, int right) {
	float pivot = nums[left];
    int l = left+1;
    int r = right;
	while (l <= r) {
		if (nums[l] < pivot && nums[r] > pivot) {
            swapf(&nums[l], &nums[r]);
            l += 1;
            r -= 1;
        }
		if (nums[l] >= pivot) ++l;
		if (nums[r] <= pivot) --r;
	}
	swapf(&nums[left], &nums[r]);
	return r;
}

static float kth_largestf(float *nums, int k, int n) {
	int left = 0;
    int right = n-1;
    int idx = 0;
	while (true) {
		idx = partitionf(nums, left, right);
		if (idx == k) break;
		else if (idx < k) left = idx+1;
		else right = idx-1;
	}
	return nums[idx];
}

static float log_sum_exp(float x, float y) {
    float abs_diff = fabs(x - y);
    float m = x > y ? x : y;
    return m + ((abs_diff < 17.0f) ? (log( 1.0 + exp(-abs_diff))) : 0.0f);
}

static void openfish_generate_sequence_cpu(
    const uint8_t *moves,
    const state_t *states,
    const float *qual_data,
    float shift,
    float scale,
    size_t n_timesteps,
    size_t seq_len,
    float *base_probs,
    float *total_probs,
    char *sequence,
    char *qstring
) {
    size_t seq_pos = 0;

    const char alphabet[4] = {'A', 'C', 'G', 'T'};

    for (size_t blk = 0; blk < n_timesteps; ++blk) {
        int state = states[blk];
        int move = (int)moves[blk];
        int base = state & 3;
        int offset = (blk == 0) ? 0 : move - 1;
        int probPos = (int)(seq_pos + offset);

        // get the probability for the called base.
        base_probs[probPos] += qual_data[blk * NUM_BASES + base];

        // accumulate the total probability for all possible bases at this position, for normalization.
        for (size_t k = 0; k < NUM_BASES; ++k) {
            total_probs[probPos] += qual_data[blk * NUM_BASES + k];
        }

        if (blk == 0) {
            sequence[seq_pos++] = (char)base;
        } else {
            for (int j = 0; j < move; ++j) {
                sequence[seq_pos++] = (char)base;
            }
        }
    }

    for (size_t i = 0; i < seq_len; ++i) {
        sequence[i] = alphabet[(int)sequence[i]];
        base_probs[i] = 1.0f - (base_probs[i] / total_probs[i]);
        base_probs[i] = -10.0f * log10f(base_probs[i]);
        float qscore = base_probs[i] * scale + shift;
        if (qscore > 50.0f) qscore = 50.0f;
        if (qscore < 1.0f) qscore = 1.0f;
        qstring[i] = (char)(33.5f + qscore);
    }
}

// Incorporates NUM_NEW_BITS into a Castagnoli CRC32, aka CRC32C
// (not the same polynomial as CRC32 as used in zip/ethernet).
static uint32_t crc32c(uint32_t crc, uint32_t new_bits, int num_new_bits) {
    // Note that this is the reversed polynomial.
    const uint32_t POLYNOMIAL = 0x82f63b78u;
    for (int i = 0; i < num_new_bits; ++i) {
        uint32_t b = (new_bits ^ crc) & 1;
        crc >>= 1;
        if (b) {
            crc ^= POLYNOMIAL;
        }
        new_bits >>= 1;
    }
    return crc;
}

static void openfish_beam_search_cpu(
    const float *scores_NTC,
    size_t scores_block_stride,
    const float *bwd_NTC,
    const float *post_NTC,
    int num_state_bits,
    size_t n_timesteps,
    float beam_cut,
    float fixed_stay_score,
    state_t *states,
    uint8_t *moves,
    float *qual_data,
    float score_scale,
    float posts_scale,
    beam_element_t *beam_vector
) {
    const size_t num_states = 1ull << num_state_bits;
    const state_t states_mask = (state_t)(num_states - 1);
    const float log_beam_cut = (beam_cut > 0.0f) ? logf(beam_cut) : FLT_MAX;

    // create the previous and current beam fronts
    // each existing beam element can be extended by one of NUM_BASES, or be a stay (for a single beam).

    // current set of beam fronts (candidates)
    beam_front_element_t current_beam_front[MAX_BEAM_CANDIDATES];

    // the last set of beam fronts
    beam_front_element_t prev_beam_front[MAX_BEAM_CANDIDATES];

    // scores for each possible transition for each state (k-mer)
    float current_scores[MAX_BEAM_CANDIDATES];
    float prev_scores[MAX_BEAM_CANDIDATES];

    // Find the score an initial element needs in order to make it into the beam
    float beam_init_threshold = -FLT_MAX;
    if (MAX_BEAM_WIDTH < num_states) {
        if (MAX_BEAM_WIDTH < num_states) {
            // copy the first set of back guides and sort to extract max_beam_width highest elements
            float sorted_bwd_NTCs[MAX_STATES];
            for (size_t i = 0; i < num_states; ++i) {
                sorted_bwd_NTCs[i] = bwd_NTC[i];
            }

            // note that we don't need a full sort here to get the max_beam_width highest values
            beam_init_threshold = kth_largestf(sorted_bwd_NTCs, MAX_BEAM_WIDTH-1, num_states);
        }
    }

    // initialise all beams
    // go through all state scores in the first block of the back_guide
    for (size_t state = 0, beam_element = 0; state < num_states && beam_element < MAX_BEAM_WIDTH; state++) {
        if (bwd_NTC[state] >= beam_init_threshold) {
            // note that this first element has a prev_element_index of 0
            beam_front_element_t new_elem = {crc32c(CRC_SEED, (uint32_t)state, 32), (state_t)state, 0, false};
            prev_beam_front[beam_element] = new_elem;
            prev_scores[beam_element] = 0.0f;
            ++beam_element;
        }
    }

    // copy all beam fronts into the beam persistent state
    size_t current_beam_width = MAX_BEAM_WIDTH < num_states ? MAX_BEAM_WIDTH : num_states;
    for (size_t element_idx = 0; element_idx < current_beam_width; ++element_idx) {
        beam_vector[element_idx].state = prev_beam_front[element_idx].state;
        beam_vector[element_idx].prev_element_index = prev_beam_front[element_idx].prev_element_index;
        beam_vector[element_idx].stay = prev_beam_front[element_idx].stay;
    }

    // essentially a k=1 Bloom filter, indicating the presence of steps with particular
    // sequence hashes.  Avoids comparing stay hashes against all possible progenitor
    // states where none of them has the requisite sequence hash.
    bool step_hash_present[HASH_PRESENT_BITS];  // Default constructor zeros content.

    // iterate through blocks, extending beam
    for (size_t block_idx = 0; block_idx < n_timesteps; ++block_idx) {
        const float *const block_scores = scores_NTC + (block_idx * scores_block_stride);
        const float *const block_back_scores = bwd_NTC + ((block_idx + 1) << num_state_bits);

        float max_score = -FLT_MAX;

        // reset bloom filter
        for (uint32_t i = 0; i < HASH_PRESENT_BITS; ++i) {
            step_hash_present[i] = false;
        }

        // generate list of candidate elements for this timestep (block)
        // update the max score and scores for each possible transition for each beam
        size_t new_elem_count = 0;
        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const beam_front_element_t *previous_element = &prev_beam_front[prev_elem_idx];

            // expand all the possible steps
            for (int new_base = 0; new_base < NUM_BASES; new_base++) {

                /* kmer transitions order:
                *  N^K , N array
                *  elements stored as resulting kmer and modifying action (stays have a fixed score and are not computed)
                *  kmer index is lexicographic with most recent base in the fastest index
                *
                *  e.g.  AGT has index (4^2, 4, 1) . (0, 2, 3) == 11
                *  the modifying action is
                *    0: Remove A from beginning
                *    1: Remove C from beginning
                *    2: Remove G from beginning
                *    3: Remove T from beginning
                *
                *  transition (movement) ACGTT (111) -> CGTTG (446) has index 446 * 4 + 0 = 1784
                */

                // shift the state (k-mer) left and append the new base to the end of bitset
                // transition to a new k-mer (see explanation above)
                state_t new_state = ((state_t)((previous_element->state << NUM_BASE_BITS) & states_mask) | (state_t)(new_base));

                // get the score of this transition (see explanation above)
                const state_t move_idx = (state_t)((new_state << NUM_BASE_BITS) + (((previous_element->state << NUM_BASE_BITS) >> num_state_bits)));

                float block_score = (float)block_scores[move_idx] * score_scale;
                float new_score = prev_scores[prev_elem_idx] + block_score + (float)block_back_scores[new_state];

                // generate hash from previous element and new state
                uint32_t new_hash = crc32c(previous_element->hash, new_base, NUM_BASE_BITS);
                step_hash_present[new_hash & HASH_PRESENT_MASK] = true;

                uint32_t new_elem_idx = new_elem_count;

                // add new element to candidates
                beam_front_element_t new_beam_elem = {
                    new_hash,
                    new_state,
                    (uint8_t)prev_elem_idx,
                    false // this is never a stay, these are possible steps
                };
                current_beam_front[new_elem_idx] = new_beam_elem;

                // update scores
                current_scores[new_elem_idx] = new_score;
                max_score = max_score > new_score ? max_score : new_score;
                ++new_elem_count;
            }
        }

        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const beam_front_element_t *previous_element = &prev_beam_front[prev_elem_idx];
            uint32_t new_elem_idx = new_elem_count;

            // score for possible stay
            // if it's a stay, it's a kmer that repeats in the sequence
            const float stay_score = prev_scores[prev_elem_idx]                                         // score of prev elem
                                    + fixed_stay_score                                                  // + some static score for possible stays
                                    + (float)block_back_scores[previous_element->state];   // + score of previous state being in this timestep

            // add stay to candidates
            // since we will always have the step as a candidate (from our previous step) in our current_beam_front and current_scores, we only need to create candidate stay beam elements
            beam_front_element_t new_beam_elem = {
                previous_element->hash,
                previous_element->state,
                (uint8_t)prev_elem_idx,
                true // this is always stay, a stay represents a beam_element that has not changed since the last timestep
            };
            current_beam_front[new_elem_idx] = new_beam_elem;
            current_scores[new_elem_idx] = stay_score;

            max_score = max_score > stay_score ? max_score : stay_score;

            // determine whether the path including this stay duplicates another sequence ending in a step equal to the state of the stay
            // this "bloom filter" is supposed to help avoid comparing hash against every single step generated before
            if (step_hash_present[previous_element->hash & HASH_PRESENT_MASK]) {
                // left shift by 2 and then add the previous elem idx
                size_t stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_elem_idx;

                // latest base is in smallest bits
                int stay_latest_base = (int)(previous_element->state & 3);

                // go through all the possible step extensions that match this destination base with the stay and compare their hashes, merging if we find any
                for (size_t prev_elem_comp_idx = 0; prev_elem_comp_idx < current_beam_width; prev_elem_comp_idx++) {

                    // it's a step if it's a previous kmer with a repeated base
                    size_t step_elem_idx = (prev_elem_comp_idx << NUM_BASE_BITS) | stay_latest_base;

                    // compare hashes of step extension and possible stay, if they're equal, we merge
                    if (current_beam_front[stay_elem_idx].hash == current_beam_front[step_elem_idx].hash) {
                        const float folded_score = log_sum_exp(current_scores[stay_elem_idx], current_scores[step_elem_idx]);
                        // merge: 
                        //      modify score of whatever scored better with folded score
                        //      basically discard the worst scoring one
                        if (current_scores[stay_elem_idx] > current_scores[step_elem_idx]) {
                            // fold the step into the stay
                            current_scores[stay_elem_idx] = folded_score;
                            // the step element will end up last, sorted by score
                            current_scores[step_elem_idx] = -FLT_MAX;
                        } else {
                            // fold the stay into the step
                            current_scores[step_elem_idx] = folded_score;
                            // the stay element will end up last, sorted by score
                            current_scores[stay_elem_idx] = -FLT_MAX;
                        }
                        max_score = folded_score > max_score ? folded_score : max_score;
                    }
                }
            }
            ++new_elem_count;
        }

        // cut off to fit beam width
        // starting point for finding the cutoff score is the beam cut score
        float beam_cutoff_score = max_score - log_beam_cut;
        float *score_ptr;
        size_t elem_count;

        // count the elements which meet the min score
        elem_count = 0;
        score_ptr = current_scores;
        for (int i = (int)(new_elem_count); i; --i) {
            if (*score_ptr >= beam_cutoff_score) ++elem_count;
            ++score_ptr;
        }

        // binary search to find a score which doesn't return too many scores, but doesn't reduce beam width too much
        if (elem_count > MAX_BEAM_WIDTH) {
            size_t min_beam_width = (MAX_BEAM_WIDTH * 8) / 10;  // 80% of beam width is the minimum we accept
            float low_score = beam_cutoff_score;
            float hi_score = max_score;
            int num_guesses = 1;
            const int MAX_GUESSES = 10;

            while ((elem_count > MAX_BEAM_WIDTH || elem_count < min_beam_width) && num_guesses < MAX_GUESSES) {
                if (elem_count > MAX_BEAM_WIDTH) {
                    // make a higher guess
                    low_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0f;
                } else {
                    // make a lower guess
                    hi_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0f;
                }
                elem_count = 0;
                score_ptr = current_scores;
                for (int i = (int)new_elem_count; i; --i) {
                    if (*score_ptr >= beam_cutoff_score) ++elem_count;
                    ++score_ptr;
                }

                ++num_guesses;
            }

            // if we made 10 guesses and didn't find a suitable score, a couple of things may have happened:
            // 1: we just haven't completed the binary search yet (there is a good score in there somewhere but we didn't find it)
            //  - in this case we should just pick the higher of the two current search limits to get the top N elements)
            // 2: there is no good score, as max_score returns more than beam_width elements (i.e. more than the whole beam width has max_score)
            //  - in this case we should just take MAX_BEAM_WIDTH of the top-scoring elements
            // 3: there is no good score as all the elements from <80% of the beam to >100% have the same score
            //  - in this case we should just take the hi_score and accept it will return us less than 80% of the beam
            if (num_guesses == MAX_GUESSES) {
                beam_cutoff_score = hi_score;
                elem_count = 0;
                score_ptr = current_scores;
                for (int i = (int)new_elem_count; i; --i) {
                    if (*score_ptr >= beam_cutoff_score) ++elem_count;
                    ++score_ptr;
                }
            }

            // clamp the element count to the max beam width in case of failure 2 from above
            elem_count = elem_count < MAX_BEAM_WIDTH ? elem_count : MAX_BEAM_WIDTH;
        }

        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < new_elem_count; ++read_idx) {
            if (current_scores[read_idx] >= beam_cutoff_score) {
                if (write_idx < MAX_BEAM_WIDTH) {
                    prev_beam_front[write_idx] = current_beam_front[read_idx];
                    prev_scores[write_idx] = current_scores[read_idx];
                    ++write_idx;
                } else {
                    break;
                }
            }
        }

        // at the last timestep, we need to ensure the best path corresponds to element 0
        // the other elements don't matter
        if (block_idx == n_timesteps - 1) {
            float best_score = -FLT_MAX;
            size_t best_score_index = 0;
            for (size_t i = 0; i < elem_count; i++) {
                if (prev_scores[i] > best_score) {
                    best_score = prev_scores[i];
                    best_score_index = i;
                }
            }
            beam_front_element_t temp0 = prev_beam_front[0];
            prev_beam_front[0] = prev_beam_front[best_score_index];
            prev_beam_front[best_score_index] = temp0;

            float temp1 = prev_scores[0];
            prev_scores[0] = prev_scores[best_score_index];
            prev_scores[best_score_index] = temp1;
        }

        // copy this new beam front into the beam persistent state
        size_t beam_offset = (block_idx + 1) * MAX_BEAM_WIDTH;
        for (size_t i = 0; i < elem_count; ++i) {
            // remove backwards contribution from score
            prev_scores[i] -= (float)block_back_scores[prev_beam_front[i].state];

            beam_vector[beam_offset + i].state = prev_beam_front[i].state;
            beam_vector[beam_offset + i].prev_element_index = prev_beam_front[i].prev_element_index;
            beam_vector[beam_offset + i].stay = prev_beam_front[i].stay;
        }
        // adjust current beam width
        current_beam_width = elem_count;
    }

    // write out sequence bases and move table
    // note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
    uint8_t element_index = 0;
    for (size_t beam_idx = n_timesteps; beam_idx != 0; --beam_idx) {
        size_t beam_addr = beam_idx * MAX_BEAM_WIDTH + element_index;
        states[beam_idx - 1] = (int32_t)beam_vector[beam_addr].state;
        moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
        element_index = beam_vector[beam_addr].prev_element_index;
    }
    moves[0] = 1;  // always step in the first event

    int shifted_states[2 * NUM_BASES];

    // compute per-base qual data
    for (size_t block_idx = 0; block_idx < n_timesteps; ++block_idx) {
        int state = states[block_idx];
        states[block_idx] = states[block_idx] % NUM_BASES;
        int base_to_emit = states[block_idx];

        // compute a probability for this block, based on the path kmer
        const float *const timestep_posts = post_NTC + ((block_idx + 1) << num_state_bits);

        float block_prob = (float)(timestep_posts[state]) * posts_scale;

        // get indices of left- and right-shifted kmers
        int l_shift_idx = state >> NUM_BASE_BITS;
        int r_shift_idx = (state << NUM_BASE_BITS) % num_states;
        int msb = ((int)num_states) >> NUM_BASE_BITS;
        int l_shift_state, r_shift_state;
        for (int shift_base = 0; shift_base < NUM_BASES; ++shift_base) {
            l_shift_state = l_shift_idx + msb * shift_base;
            shifted_states[2 * shift_base] = l_shift_state;

            r_shift_state = r_shift_idx + shift_base;
            shifted_states[2 * shift_base + 1] = r_shift_state;
        }

        // add probabilities for unique states
        int candidate_state;
        for (size_t state_idx = 0; state_idx < 2 * NUM_BASES; ++state_idx) {
            candidate_state = shifted_states[state_idx];
            // don't double-count this shifted state if it matches the current state
            bool count_state = (candidate_state != state);
            // or any other shifted state that we've seen so far
            if (count_state) {
                for (size_t inner_state = 0; inner_state < state_idx; ++inner_state) {
                    if (shifted_states[inner_state] == candidate_state) {
                        count_state = false;
                        break;
                    }
                }
            }
            if (count_state) {
                block_prob += (float)(timestep_posts[candidate_state]) * posts_scale;
            }
        }
        if (block_prob > 1.0f) block_prob = 1.0f;
        else if (block_prob < 0.0f) block_prob = 0.0f;
        block_prob = powf(block_prob, 0.4f); // power fudge factor

        // calculate a placeholder qscore for the "wrong" bases
        const float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (size_t base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] = ((int)base == base_to_emit ? block_prob : wrong_base_prob);
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

// Extended CPU decode used by the in-memory test harness. Identical to openfish_decode_cpu
// but, when the corresponding *_out pointer is non-NULL, hands back the intermediate
// backward-guide / posterior / qual / total-probability tensors instead of freeing them
// (the caller then owns and frees them). Passing NULL for an out pointer keeps the old
// behaviour of freeing it internally. Declared in src/decode_cpu.h (private, not part of
// the public API).
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

    // cleanup
    free(dequant_scores); // NULL for the float path
    free(beam_vector);
    free(states);
    free(base_probs);

    // Hand back the requested intermediate tensors (test harness compares them against the
    // GPU path); otherwise free them. Ownership of any returned buffer passes to the caller.
    if (bwd_NTC_out)     *bwd_NTC_out = bwd_NTC;       else free(bwd_NTC);
    if (post_NTC_out)    *post_NTC_out = post_NTC;     else free(post_NTC);
    if (qual_data_out)   *qual_data_out = qual_data;   else free(qual_data);
    if (total_probs_out) *total_probs_out = total_probs; else free(total_probs);
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
    openfish_decode_cpu_ex(
        n_timesteps, batch_size, n_channels, n_threads, scores_NTC, score_dtype,
        score_scale, state_len, options, moves, sequence, qstring,
        NULL, NULL, NULL, NULL
    );
}
