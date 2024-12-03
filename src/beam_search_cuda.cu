#include "beam_search_cuda.cuh"
#include "decode.h"
#include "cuda_utils.cuh"

#include <math.h>
#include <float.h>
#include <cuda_fp16.h>

__device__ static __forceinline__ void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

__device__ static __forceinline__ int partitionf(float *nums, int left, int right) {
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

__device__ static __forceinline__ float kth_largestf(float *nums, int k, int n) {
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

__device__ static __forceinline__ float log_sum_exp(float x, float y) {
    float abs_diff = fabsf(x - y);
    float m = x > y ? x : y;
    return m + ((abs_diff < 17.0f) ? (__logf(1.0 + __expf(-abs_diff))) : 0.0f);
}

__global__ void generate_sequence(
    const beam_args_t args,
    const uint8_t *_moves,
    const state_t *_states,
    const float *_qual_data,
    float *_base_probs,
    float *_total_probs,
    char *_sequence,
    char *_qstring,
    const float shift,
    const float scale
) {
    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);

    if (chunk >= args.N) {
		return;
	}

    const uint64_t T = args.T;
    const uint8_t *moves = _moves + chunk * T;
    const state_t *states = _states + chunk * T;
    const float *qual_data = _qual_data + chunk * T * NUM_BASES;
    float *base_probs = _base_probs + chunk * T;
    float *total_probs = _total_probs + chunk * T;
    char *sequence = _sequence + chunk * T;
    char *qstring = _qstring + chunk * T;

    size_t seq_len = 0;
    for (size_t i = 0; i < T; ++i) {
        seq_len += moves[i];
        base_probs[i] = 0.0f;
        total_probs[i] = 0.0f;
    }

    size_t seq_pos = 0;

    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    
    for (size_t blk = 0; blk < T; ++blk) {
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
        base_probs[i] = -10.0f * __log10f(base_probs[i]);
        float qscore = base_probs[i] * scale + shift;
        if (qscore > 50.0f) qscore = 50.0f;
        if (qscore < 1.0f) qscore = 1.0f;
        qstring[i] = (char)(33.5f + qscore);
    }
    sequence[seq_len] = '\0';
    qstring[seq_len] = '\0';
}

// incorporates NUM_NEW_BITS into a Castagnoli CRC32, aka CRC32C
// (not the same polynomial as CRC32 as used in zip/ethernet).
__device__ static __forceinline__ uint32_t crc32c(uint32_t crc, uint32_t new_bits, int num_new_bits) {
    // note that this is the reversed polynomial
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

__global__ void beam_search(
    const beam_args_t args,
    state_t *_states,
    uint8_t *_moves,
    beam_element_t *_beam_vector,
    const float beam_cut,
    const float fixed_stay_score,
    const float score_scale
) {
    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
    const uint64_t tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const uint64_t nthreads = MAX_BEAM_WIDTH * NUM_BASES;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const unsigned mask = 0xFFFFFFFFU;
    (void)mask;
    
    if (chunk >= args.N || tid >= nthreads) {
		return;
	}

    const uint64_t T = args.T;
    const uint64_t N = args.N;
    const uint64_t C = args.C;

    const int num_state_bits = args.num_state_bits;
    const size_t num_states = 1ull << num_state_bits;
    const state_t states_mask = (state_t)(num_states - 1);
    const size_t scores_block_stride = N * C;
    const float log_beam_cut = (beam_cut > 0.0f) ? __logf(beam_cut) : FLT_MAX;

    const half *scores_TNC = (half *)args.scores_TNC + chunk * (num_states * NUM_BASES);
    const float *bwd_NTC = args.bwd_NTC + chunk * num_states * (T + 1);
    state_t *states = _states + chunk * T;
    uint8_t *moves = _moves + chunk * T;

    // this contains all the beams we're keeping track of 
    beam_element_t *beam_vector = _beam_vector + chunk * MAX_BEAM_WIDTH * (T + 1);

    // create the previous and current beam fronts
    // each existing beam element can be extended by one of NUM_BASES, or be a stay (for a single beam).

    // current set of beam fronts (candidates)
    __shared__ beam_front_element_t current_beam_front[MAX_BEAM_CANDIDATES];

    // the last set of beam fronts
    __shared__ beam_front_element_t prev_beam_front[MAX_BEAM_CANDIDATES];

    // scores for each possible transition for each state (k-mer)
    __shared__ float current_scores[MAX_BEAM_CANDIDATES];
    __shared__ float prev_scores[MAX_BEAM_CANDIDATES];

    // a k=1 Bloom filter, indicating the presence of steps with particular sequence hashes
    // avoids comparing stay hashes against all possible progenitor states where none of them has the requisite sequence hash
    __shared__ bool step_hash_present[HASH_PRESENT_BITS];

    __shared__ size_t current_beam_width;
    __shared__ float beam_init_threshold;
    if (tid == 0) {
        beam_init_threshold = -FLT_MAX;
        current_beam_width = MAX_BEAM_WIDTH < num_states ? MAX_BEAM_WIDTH : num_states;
    }
    __syncthreads();

    if (MAX_BEAM_WIDTH < num_states) {
        // copy the first set of back guides and sort to extract max_beam_width highest elements
        __shared__ float sorted_back_guides[MAX_STATES];
        for (size_t i = tid; i < num_states; i += nthreads) {
            sorted_back_guides[i] = bwd_NTC[i];
        }
        __syncthreads();

        // note that we don't need a full sort here to get the max_beam_width highest values
        if (tid == 0) beam_init_threshold = kth_largestf(sorted_back_guides, MAX_BEAM_WIDTH-1, num_states);
    }


    // find the score a state needs to make it into the first set of beam elements
    if (tid == 0) {
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
    }
    __syncthreads();

    // copy all beam fronts into the beam persistent state
    for (size_t element_idx = tid; element_idx < current_beam_width; element_idx += nthreads) {
        beam_vector[element_idx].state = prev_beam_front[element_idx].state;
        beam_vector[element_idx].prev_element_index = prev_beam_front[element_idx].prev_element_index;
        beam_vector[element_idx].stay = prev_beam_front[element_idx].stay;
    }
    __syncthreads();
    
    // iterate through blocks, extending each beam
    __shared__ int elem_count;
    __shared__ float block_buf[32];
    __shared__ float max_score;
    __shared__ uint32_t new_elem_count;
    __shared__ float beam_cutoff_score;
    for (size_t block_idx = 0; block_idx < T; ++block_idx) {
        const half *const block_scores = scores_TNC + (block_idx * scores_block_stride);
        const float *const block_back_scores = bwd_NTC + ((block_idx + 1) << num_state_bits);

        float warp_max = -FLT_MAX;
        if (tid == 0) {
            new_elem_count = 0;
        }
        
        // reset bloom filter
        for (uint32_t i = tid; i < HASH_PRESENT_BITS; i += nthreads) {
            step_hash_present[i] = false;
        }
        __syncthreads();
        
        // generate list of candidate elements for this timestep (block)
        // update the max score and scores for each possible transition for each beam
        for (size_t prev_elem_idx = (tid / NUM_BASES); prev_elem_idx < current_beam_width; prev_elem_idx += nthreads) {
            const beam_front_element_t *previous_element = &prev_beam_front[prev_elem_idx];
            const int new_base = tid % NUM_BASES;

            /*  kmer transitions order:
            *  N^K , N array
            *  Elements stored as resulting kmer and modifying action (stays have a fixed score and are not computed).
            *  Kmer index is lexicographic with most recent base in the fastest index
            *
            *  E.g.  AGT has index (4^2, 4, 1) . (0, 2, 3) == 11
            *  The modifying action is
            *    0: Remove A from beginning
            *    1: Remove C from beginning
            *    2: Remove G from beginning
            *    3: Remove T from beginning
            *
            *  Transition (movement) ACGTT (111) -> CGTTG (446) has index 446 * 4 + 0 = 1784
            */

            // shift the state (k-mer) left and append the new base to the end of bitset
            // transition to a new k-mer (see explanation above)
            state_t new_state = ((state_t)((previous_element->state << NUM_BASE_BITS) & states_mask) | (state_t)(new_base));

            // get the score of this transition (see explanation above)
            const state_t move_idx = (state_t)((new_state << NUM_BASE_BITS) + (((previous_element->state << NUM_BASE_BITS) >> num_state_bits)));

            float block_score = __half2float(block_scores[move_idx]) * score_scale;
            float new_score = prev_scores[prev_elem_idx] + block_score + (float)block_back_scores[new_state];

            // generate hash from previous element and new state
            uint32_t new_hash = crc32c(previous_element->hash, new_base, NUM_BASE_BITS);
            step_hash_present[new_hash & HASH_PRESENT_MASK] = true;

            uint32_t new_elem_idx = new_elem_count + (prev_elem_idx * NUM_BASES) + new_base;

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
            warp_max = max(warp_max, new_score);
        }
        __syncthreads();
        if (tid == 0) new_elem_count += current_beam_width * NUM_BASES;
        __syncthreads();

        // generate stays

        // stay: signal is on the same "timestep" with the same base
        // step: signal is on a different "timestep" with the same base
        
        // iter through each previous element to create candidate stays and compare it against the score of a corresponding candidate step
        // compare each candidate stay with steps of the same hash
        // whichever stay or step scores worse gets discarded
        // whichever scores better gets an updated score
        
        // e.g.
        //      if our stay s1 looks like this:
        //          s0{state: TCGG, stay: false} -> s1{state: TCGG, stay: true}
        //
        //      we compare its score to the step p1 that looks like this:
        //          p0{state: ATCG, stay: false} -> p1{state: TCGG, stay: false}
        //
        //      note: both must also have stemmed from the same sequence, so we keep a hash to keep track
        for (size_t prev_elem_idx = (tid / NUM_BASES); prev_elem_idx < current_beam_width; prev_elem_idx += nthreads) {
            const beam_front_element_t *previous_element = &prev_beam_front[prev_elem_idx];
            uint32_t new_elem_idx = new_elem_count + prev_elem_idx;

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

            warp_max = max(warp_max, stay_score);

            // determine whether the path including this stay duplicates another sequence ending in a step equal to the state of the stay
            // this "bloom filter" is supposed to help avoid comparing hash against every single step generated before
            if (step_hash_present[previous_element->hash & HASH_PRESENT_MASK]) {
                // left shift by 2 and then add the previous elem idx
                size_t stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_elem_idx;

                // latest base is in smallest bits
                int stay_latest_base = (int)(previous_element->state & 3);

                // go through all the possible step extensions that match this destination base with the stay and compare their hashes, merging if we find any
                for (size_t prev_elem_comp_idx = (tid % NUM_BASES); prev_elem_comp_idx < current_beam_width; prev_elem_comp_idx += NUM_BASES) {

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
                        warp_max = max(warp_max, folded_score);
                    }
                }
            }
        }
        __syncthreads();
        if (tid == 0) { new_elem_count += current_beam_width; }
        __syncthreads();

        // find max val in warp
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            warp_max = max(warp_max, __shfl_down_sync(mask, warp_max, offset));
        }
        if (lane_id == 0) block_buf[warp_id] = warp_max;
        __syncthreads();

        // set max val in all warps
        if (warp_id == 0) {
            warp_max = (tid < nthreads/warpSize) ? block_buf[lane_id] : 0;

            for (int offset = warpSize/2; offset > 0; offset >>= 1) {
                warp_max = max(warp_max, __shfl_down_sync(mask, warp_max, offset));
            }
            
            if (tid == 0) max_score = warp_max;
        }
        __syncthreads();

        // cut off to fit beam width
        // starting point for finding the cutoff score is the beam cut score
        if (tid == 0) {
            elem_count = 0;
            beam_cutoff_score = max_score - log_beam_cut;
        }
        __syncthreads();

        // count the elements which meet the min score
        float *score_ptr = current_scores + tid;
        for (int i = tid+1; i <= (int)(new_elem_count); i += nthreads) {
            if (*score_ptr >= beam_cutoff_score) atomicAdd(&elem_count, 1);
            score_ptr += nthreads;
        }
        __syncthreads();

        // binary search to find a score which doesn't return too many scores, but doesn't reduce beam width too much
        if (elem_count > MAX_BEAM_WIDTH) {
            if (tid == 0) {
                beam_cutoff_score = max_score;
                elem_count = 0;
            }
            __syncthreads();
            float *score_ptr = current_scores + tid;
            for (int i = tid+1; i <= (int)(new_elem_count); i += nthreads) {
                if (*score_ptr >= beam_cutoff_score) atomicAdd(&elem_count, 1);
                score_ptr += nthreads;
            }
            __syncthreads();
            if (tid == 0) elem_count = min(elem_count, MAX_BEAM_WIDTH);
            __syncthreads();
        }
        // write current scores and beam fronts to prev
        if (tid == 0) {
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
        }
        __syncthreads();

        // at the last timestep, we need to ensure the best path corresponds to element 0
        // the other elements don't matter
        if (tid == 0 && block_idx == T - 1) {
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
        __syncthreads();

        // copy this new beam front into the beam persistent state
        size_t beam_offset = (block_idx + 1) * MAX_BEAM_WIDTH;
        for (size_t i = tid; i < elem_count; i += nthreads) {
            // remove backwards contribution from score
            prev_scores[i] -= (float)block_back_scores[prev_beam_front[i].state];

            beam_vector[beam_offset + i].state = prev_beam_front[i].state;
            beam_vector[beam_offset + i].prev_element_index = prev_beam_front[i].prev_element_index;
            beam_vector[beam_offset + i].stay = prev_beam_front[i].stay;
        }
        // adjust current beam width
        current_beam_width = elem_count;
        __syncthreads();
    }

    // write out sequence bases and move table
    if (tid == 0) {
        // note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
        uint8_t element_index = 0;
        for (size_t beam_idx = T; beam_idx != 0; --beam_idx) {
            size_t beam_addr = beam_idx * MAX_BEAM_WIDTH + element_index;
            states[beam_idx - 1] = (int32_t)beam_vector[beam_addr].state;
            moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
            element_index = beam_vector[beam_addr].prev_element_index;
        }
        moves[0] = 1;  // always step in the first event
    }
}

__global__ void compute_qual_data(
    const beam_args_t args,
    state_t *_states,
    float *_qual_data,
    const float posts_scale
) {
    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
    if (chunk >= args.N) {
		return;
	}

    const uint64_t T = args.T;

    const size_t num_states = 1ull << args.num_state_bits;
    const size_t num_state_bits = args.num_state_bits;

    const float *post_NTC = args.post_NTC + chunk * num_states * (T + 1);
    state_t *states = _states + chunk * T;
    float *qual_data = _qual_data + chunk * (T * NUM_BASES);

    int shifted_states[2 * NUM_BASES];

    // compute per-base qual data
    for (size_t block_idx = 0; block_idx < T; ++block_idx) {
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
        block_prob = __powf(block_prob, 0.4f); // power fudge factor

        // calculate a placeholder qscore for the "wrong" bases
        const float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (size_t base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] = ((int)base == base_to_emit ? block_prob : wrong_base_prob);
        }
    }
}