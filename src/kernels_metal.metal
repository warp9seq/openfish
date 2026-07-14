// Metal Shading Language port of the CRF-CTC decoding kernels.
//
// This mirrors the CUDA kernels in kernels_cuda.h. Each threadgroup
// decodes one chunk (batch element), exactly like one CUDA block per chunk. CUDA warps map
// to Metal SIMD-groups (32-wide on Apple GPUs), __syncthreads() maps to a threadgroup barrier,
// __shared__ maps to threadgroup memory, and __shfl_down_sync maps to simd_shuffle_down.
//
// The kernels are dispatched with threadsPerThreadgroup == num_states (scan kernels),
// MAX_BEAM_WIDTH*NUM_BASES (beam search), or 1 (sequence/qual generation), so the
// chunk/thread bounds checks never actually fire and every thread reaches each barrier.

#include <metal_stdlib>
using namespace metal;

// The constants, state_t, beam_element_t / beam_front_element_t and the scan_params_t /
// beam_params_t argument blocks come from openfish_defs.h, which the Makefile concatenates ahead
// of this source at build time (newLibraryWithSource: can't resolve local #includes). Only the
// two shader-private constants below are defined here.

#define SIMD_WIDTH (32)
// -FLT_MAX sentinel (matches CUDA's -FLT_MAX)
#define OF_FLT_MAX (0x1.fffffep127f)

// Score element type. The kernel source is compiled twice (decode_metal.mm prepends the define):
// half for native fp16 scores, char for int8-quantized scores The raw value is
// multiplied by score_scale on read (1.0 for fp16, e.g. 5/127 for int8) -- mirrors the CUDA
// load_score<ScoreT>() * score_scale in kernels_cuda.h.
#ifndef OF_SCORE_T
#define OF_SCORE_T half
#endif

// ------------------------------------------------------------------ scan kernels

kernel void bwd_scan(
    constant scan_params_t &args      [[buffer(0)]],
    device const OF_SCORE_T *scores_in [[buffer(1)]],
    device float         *out       [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    const ulong chunk = tgid;
    const ulong state = tid;

    const ulong num_states  = args.num_states;
    const ulong n_timesteps = args.n_timesteps;
    const ulong batch_size  = args.batch_size;

    if (chunk >= batch_size || tid >= num_states) {
        return;
    }

    const float fixed_stay_score = args.fixed_stay_score;
    const float score_scale = args.score_scale;
    const ulong ts_states = num_states * NUM_BASES;

    // scores are NTC: each batch element's [T,C] block is contiguous
    device const OF_SCORE_T *const chunk_in   = scores_in + chunk * n_timesteps * ts_states;
    device float       *const chunk_out  = out + chunk * (n_timesteps + 1) * num_states;
    device float       *const alpha_init = chunk_out + num_states * n_timesteps;
    alpha_init[state] = 0.0f;

    for (ulong ts = 0; ts < n_timesteps; ++ts) {
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        device const OF_SCORE_T *const ts_in       = chunk_in + ts_states * (n_timesteps - ts - 1);
        device float      *const ts_alpha_in = alpha_init - num_states * ts;
        device float      *const ts_alpha_out = ts_alpha_in - num_states;

        const ulong stay_state_idx  = state;
        const ulong step_state_idx_a = (state * NUM_BASES) % num_states;
        const ulong step_trans_idx_a = step_state_idx_a * NUM_BASES + ((state * NUM_BASES) / num_states);

        float vals[NUM_TRANSITIONS];
        vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
        float max_val = vals[0];
        for (ulong base = 0; base < NUM_BASES; ++base) {
            vals[base + 1] = ts_alpha_in[step_state_idx_a + base] + float(ts_in[step_trans_idx_a + base * NUM_BASES]) * score_scale;
            max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
        }
        float sum = 0.0f;
        for (ulong i = 0; i < NUM_TRANSITIONS; ++i) {
            sum += exp(vals[i] - max_val);
        }
        ts_alpha_out[state] = max_val + log(sum);
    }
}

kernel void fwd_post_scan(
    constant scan_params_t &args      [[buffer(0)]],
    device const OF_SCORE_T *scores_in [[buffer(1)]],
    device const float   *bwd       [[buffer(2)]],
    device float         *out       [[buffer(3)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint nthreads[[threads_per_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]]
) {
    const ulong chunk = tgid;

    const ulong num_states  = args.num_states;
    const ulong n_timesteps = args.n_timesteps;
    const ulong n_ts        = n_timesteps + 1;
    const ulong batch_size  = args.batch_size;

    if (chunk >= batch_size || tid >= num_states) {
        return;
    }

    const ulong state = tid;
    const float fixed_stay_score = args.fixed_stay_score;
    const float score_scale = args.score_scale;

    const ulong msb = num_states / NUM_BASES;
    const ulong ts_states = num_states * NUM_BASES;

    threadgroup float fwd_vals[MAX_STATES];
    threadgroup float fwd_maxs[32];
    threadgroup float exp_vals[MAX_STATES];
    threadgroup float exp_sums[32];
    threadgroup float ts_fwd[2][MAX_STATES];
    float warp_max;

    // scores are NTC: [T,C] block contiguous per batch element
    device const OF_SCORE_T *const chunk_scores = scores_in + chunk * n_timesteps * ts_states;

    for (ulong s = tid; s < num_states; s += nthreads) {
        ts_fwd[0][s] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (ulong ts = 0; ts < n_ts; ++ts) {
        warp_max = -OF_FLT_MAX;
        const ulong ts_idx = (chunk * n_ts + ts) * num_states;

        threadgroup const float *const ts_alpha_in  = ts_fwd[ts & 1];
        threadgroup float       *const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        if (ts < n_timesteps) {
            device const OF_SCORE_T *const ts_scores = chunk_scores + ts_states * ts;

            const ulong stay_state_idx  = state;
            const ulong step_state_idx_a = state / NUM_BASES;
            const ulong step_trans_idx_a = state * NUM_BASES;
            float vals[NUM_TRANSITIONS];
            float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
            for (ulong base = 0; base < NUM_BASES; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * msb] +
                    float(ts_scores[step_trans_idx_a + base]) * score_scale;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            float fwd_sum = 0.0f;
            for (ulong i = 0; i < NUM_TRANSITIONS; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + log(fwd_sum);
        }

        const float fwd_val = ts_alpha_in[state];
        const float val = fwd_val + bwd[ts_idx + state];

        fwd_vals[state] = val;
        warp_max = max(warp_max, val);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // find max fwd val in simdgroup
        for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
            warp_max = max(warp_max, simd_shuffle_down(warp_max, offset));
        }
        if (lane_id == 0) fwd_maxs[warp_id] = warp_max;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // set max fwd vals across all simdgroups
        if (warp_id == 0) {
            warp_max = (tid < num_states / SIMD_WIDTH) ? fwd_maxs[lane_id] : 0;
            for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                warp_max = max(warp_max, simd_shuffle_down(warp_max, offset));
            }
            if (tid == 0) fwd_maxs[0] = warp_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // enter exp vals
        float warp_sum = 0.0f;
        exp_vals[state] = exp(fwd_vals[state] - fwd_maxs[0]);
        warp_sum += exp_vals[state];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // sum exp vals in simdgroup
        for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
            warp_sum += simd_shuffle_down(warp_sum, offset);
        }
        if (lane_id == 0) exp_sums[warp_id] = warp_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // sum exp vals across all simdgroups
        if (warp_id == 0) {
            warp_sum = (tid < num_states / SIMD_WIDTH) ? exp_sums[lane_id] : 0;
            for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                warp_sum += simd_shuffle_down(warp_sum, offset);
            }
            if (tid == 0) exp_sums[0] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        out[ts_idx + state] = exp_vals[state] / exp_sums[0];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

// ------------------------------------------------------------------ beam search helpers

static inline void swapf(thread float *a, thread float *b) {
    float temp = *a; *a = *b; *b = temp;
}

static inline int partitionf(threadgroup float *nums, int left, int right) {
    int pivot = left;
    { float t = nums[pivot]; nums[pivot] = nums[right]; nums[right] = t; }

    for (int i = left; i < right; ++i) {
        if (nums[i] > nums[right]) {
            float t = nums[pivot]; nums[pivot] = nums[i]; nums[i] = t;
            ++pivot;
        }
    }
    { float t = nums[pivot]; nums[pivot] = nums[right]; nums[right] = t; }
    return pivot;
}

static inline float kth_largestf(threadgroup float *nums, int k, int n) {
    int left = 0;
    int right = n - 1;
    while (left <= right) {
        int pivot = partitionf(nums, left, right);
        if (pivot < k) {
            left = pivot + 1;
        } else if (pivot > k) {
            right = pivot - 1;
        } else {
            return nums[pivot];
        }
    }
    return -OF_FLT_MAX;
}

static inline float log_sum_exp(float x, float y) {
    float abs_diff = fabs(x - y);
    float m = x > y ? x : y;
    return m + ((abs_diff < 17.0f) ? (log(1.0f + exp(-abs_diff))) : 0.0f);
}

// Castagnoli CRC32 (CRC32C), reversed polynomial.
static inline uint crc32c(uint crc, uint new_bits, int num_new_bits) {
    const uint POLYNOMIAL = 0x82f63b78u;
    for (int i = 0; i < num_new_bits; ++i) {
        uint b = (new_bits ^ crc) & 1;
        crc >>= 1;
        if (b) {
            crc ^= POLYNOMIAL;
        }
        new_bits >>= 1;
    }
    return crc;
}

// ------------------------------------------------------------------ beam search

kernel void beam_search(
    constant beam_params_t &beam_args        [[buffer(0)]],
    device const OF_SCORE_T *scores_NTC_in    [[buffer(1)]],
    device const float   *bwd_NTC_in       [[buffer(2)]],
    device state_t       *_states          [[buffer(3)]],
    device uchar         *_moves           [[buffer(4)]],
    device beam_element_t *_beam_vector     [[buffer(5)]],
    constant float       &beam_cut         [[buffer(6)]],
    constant float       &fixed_stay_score [[buffer(7)]],
    constant float       &score_scale      [[buffer(8)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]]
) {
    const ulong chunk = tgid;
    const ulong n_threads = MAX_BEAM_WIDTH * NUM_BASES;

    if (chunk >= beam_args.batch_size || tid >= n_threads) {
        return;
    }

    const ulong n_timesteps = beam_args.n_timesteps;
    const ulong n_channels  = beam_args.n_channels;

    const int num_state_bits = beam_args.num_state_bits;
    const ulong num_states = 1ull << num_state_bits;
    const state_t states_mask = (state_t)(num_states - 1);
    // scores are NTC: successive timesteps are one C-stride apart; batch block at chunk * T * C
    const ulong scores_block_stride = n_channels;
    const float log_beam_cut = (beam_cut > 0.0f) ? log(beam_cut) : OF_FLT_MAX;

    device const OF_SCORE_T *scores_NTC = scores_NTC_in + chunk * n_timesteps * (num_states * NUM_BASES);
    device const float *bwd_NTC = bwd_NTC_in + chunk * num_states * (n_timesteps + 1);
    device state_t *states = _states + chunk * n_timesteps;
    device uchar *moves = _moves + chunk * n_timesteps;
    device beam_element_t *beam_vector = _beam_vector + chunk * MAX_BEAM_WIDTH * (n_timesteps + 1);

    threadgroup beam_front_element_t current_beam_front[MAX_BEAM_CANDIDATES];
    threadgroup beam_front_element_t prev_beam_front[MAX_BEAM_CANDIDATES];
    threadgroup float current_scores[MAX_BEAM_CANDIDATES];
    threadgroup float prev_scores[MAX_BEAM_CANDIDATES];

    // candidate-phase scratch: k=1 Bloom filter during candidate/stay generation, then reused
    // as the stream-compaction prefix-sum storage (compact_offsets) in the pruning phase.
    threadgroup bool cand_scratch[HASH_PRESENT_BITS];
    threadgroup bool *const step_hash_present = cand_scratch;
    threadgroup int  *const compact_offsets   = (threadgroup int *)cand_scratch;

    threadgroup ulong current_beam_width;
    threadgroup float beam_init_threshold;

    // back-guide sort scratch (num_states floats; sized to the MAX_STATES upper bound here).
    threadgroup float sorted_back_guides[MAX_STATES];

    for (ulong beam_element = tid; beam_element < MAX_BEAM_CANDIDATES; beam_element += n_threads) {
        current_beam_front[beam_element] = beam_front_element_t{0, 0, 0, false};
        prev_beam_front[beam_element] = beam_front_element_t{0, 0, 0, false};
        current_scores[beam_element] = 0.0f;
        prev_scores[beam_element] = 0.0f;
    }

    if (tid == 0) {
        beam_init_threshold = -OF_FLT_MAX;
        current_beam_width = MAX_BEAM_WIDTH < num_states ? MAX_BEAM_WIDTH : num_states;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (MAX_BEAM_WIDTH < num_states) {
        for (ulong i = tid; i < num_states; i += n_threads) {
            sorted_back_guides[i] = bwd_NTC[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tid == 0) beam_init_threshold = kth_largestf(sorted_back_guides, MAX_BEAM_WIDTH - 1, (int)num_states);
    }

    if (tid == 0) {
        for (ulong state = 0, beam_element = 0; state < num_states && beam_element < MAX_BEAM_WIDTH; state++) {
            if (bwd_NTC[state] >= beam_init_threshold) {
                beam_front_element_t new_elem = {crc32c(CRC_SEED, (uint)state, 32), (state_t)state, 0, false};
                prev_beam_front[beam_element] = new_elem;
                prev_scores[beam_element] = 0.0f;
                ++beam_element;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (ulong element_idx = tid; element_idx < current_beam_width; element_idx += n_threads) {
        beam_vector[element_idx].state = prev_beam_front[element_idx].state;
        beam_vector[element_idx].prev_element_index = prev_beam_front[element_idx].prev_element_index;
        beam_vector[element_idx].stay = prev_beam_front[element_idx].stay;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    threadgroup int elem_count;
    threadgroup float warp_buf[64];
    threadgroup float *const max_buf = warp_buf;
    threadgroup int   *const count_buf = (threadgroup int *)warp_buf;
    threadgroup float max_score;
    threadgroup uint new_elem_count;
    threadgroup float beam_cutoff_score;
    threadgroup int entered_search;
    threadgroup int bs_active;

    for (ulong block_idx = 0; block_idx < n_timesteps; ++block_idx) {
        device const OF_SCORE_T *const block_scores = scores_NTC + (block_idx * scores_block_stride);
        device const float *const block_back_scores = bwd_NTC + ((block_idx + 1) << num_state_bits);

        float warp_max = -OF_FLT_MAX;
        if (tid == 0) {
            new_elem_count = 0;
        }

        for (uint i = tid; i < HASH_PRESENT_BITS; i += n_threads) {
            step_hash_present[i] = false;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // generate candidate step elements for this timestep
        for (ulong prev_elem_idx = (tid / NUM_BASES); prev_elem_idx < current_beam_width; prev_elem_idx += n_threads) {
            threadgroup const beam_front_element_t *previous_element = &prev_beam_front[prev_elem_idx];
            const int new_base = tid % NUM_BASES;

            state_t new_state = ((state_t)((previous_element->state << NUM_BASE_BITS) & states_mask) | (state_t)(new_base));
            const state_t move_idx = (state_t)((new_state << NUM_BASE_BITS) + (((previous_element->state << NUM_BASE_BITS) >> num_state_bits)));

            float block_score = float(block_scores[move_idx]) * score_scale;
            float new_score = prev_scores[prev_elem_idx] + block_score + (float)block_back_scores[new_state];

            uint new_hash = crc32c(previous_element->hash, new_base, NUM_BASE_BITS);
            step_hash_present[new_hash & HASH_PRESENT_MASK] = true;

            uint new_elem_idx = new_elem_count + (prev_elem_idx * NUM_BASES) + new_base;

            beam_front_element_t new_beam_elem = {
                new_hash,
                new_state,
                (uchar)prev_elem_idx,
                false
            };
            current_beam_front[new_elem_idx] = new_beam_elem;
            current_scores[new_elem_idx] = new_score;
            warp_max = max(warp_max, new_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tid == 0) new_elem_count += current_beam_width * NUM_BASES;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // generate stays and fold duplicate step/stay paths
        for (ulong prev_elem_idx = (tid / NUM_BASES); prev_elem_idx < current_beam_width; prev_elem_idx += n_threads) {
            threadgroup const beam_front_element_t *previous_element = &prev_beam_front[prev_elem_idx];
            uint new_elem_idx = new_elem_count + prev_elem_idx;

            const float stay_score = prev_scores[prev_elem_idx]
                                    + fixed_stay_score
                                    + (float)block_back_scores[previous_element->state];

            beam_front_element_t new_beam_elem = {
                previous_element->hash,
                previous_element->state,
                (uchar)prev_elem_idx,
                true
            };
            current_beam_front[new_elem_idx] = new_beam_elem;
            current_scores[new_elem_idx] = stay_score;

            warp_max = max(warp_max, stay_score);

            if (step_hash_present[previous_element->hash & HASH_PRESENT_MASK]) {
                ulong stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_elem_idx;
                int stay_latest_base = (int)(previous_element->state & 3);

                for (ulong prev_elem_comp_idx = (tid % NUM_BASES); prev_elem_comp_idx < current_beam_width; prev_elem_comp_idx += NUM_BASES) {
                    ulong step_elem_idx = (prev_elem_comp_idx << NUM_BASE_BITS) | stay_latest_base;

                    if (current_beam_front[stay_elem_idx].hash == current_beam_front[step_elem_idx].hash) {
                        const float folded_score = log_sum_exp(current_scores[stay_elem_idx], current_scores[step_elem_idx]);
                        if (current_scores[stay_elem_idx] > current_scores[step_elem_idx]) {
                            current_scores[stay_elem_idx] = folded_score;
                            current_scores[step_elem_idx] = -OF_FLT_MAX;
                        } else {
                            current_scores[step_elem_idx] = folded_score;
                            current_scores[stay_elem_idx] = -OF_FLT_MAX;
                        }
                        warp_max = max(warp_max, folded_score);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tid == 0) { new_elem_count += current_beam_width; }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // block-wide max reduction
        for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
            warp_max = max(warp_max, simd_shuffle_down(warp_max, offset));
        }
        if (lane_id == 0) max_buf[warp_id] = warp_max;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        if (warp_id == 0) {
            warp_max = (tid < n_threads / SIMD_WIDTH) ? max_buf[lane_id] : 0;
            for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                warp_max = max(warp_max, simd_shuffle_down(warp_max, offset));
            }
            if (tid == 0) max_score = warp_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        if (tid == 0) {
            beam_cutoff_score = max_score - log_beam_cut;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // count elements >= cutoff (block-wide reduction; result in count_buf[0])
        {
            int local = 0;
            for (int i = tid; i < (int)new_elem_count; i += n_threads) {
                if (current_scores[i] >= beam_cutoff_score) ++local;
            }
            for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) local += simd_shuffle_down(local, offset);
            if (lane_id == 0) count_buf[warp_id] = local;
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            if (warp_id == 0) {
                int v = (tid < (int)(n_threads / SIMD_WIDTH)) ? count_buf[lane_id] : 0;
                for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) v += simd_shuffle_down(v, offset);
                if (tid == 0) count_buf[0] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
        if (tid == 0) {
            elem_count = count_buf[0];
            entered_search = (elem_count > MAX_BEAM_WIDTH) ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // binary search for a cutoff score that keeps the beam within [80%, 100%] of MAX_BEAM_WIDTH.
        if (entered_search) {
            const ulong min_beam_width = (MAX_BEAM_WIDTH * 8) / 10;
            float low_score = beam_cutoff_score;
            float hi_score = max_score;
            int num_guesses = 1;
            const int MAX_GUESSES = 10;

            while (true) {
                if (tid == 0) {
                    if ((elem_count > MAX_BEAM_WIDTH || elem_count < (int)min_beam_width) && num_guesses < MAX_GUESSES) {
                        if (elem_count > MAX_BEAM_WIDTH) {
                            low_score = beam_cutoff_score;
                            beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0f;
                        } else {
                            hi_score = beam_cutoff_score;
                            beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0f;
                        }
                        ++num_guesses;
                        bs_active = 1;
                    } else {
                        bs_active = 0;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                if (!bs_active) break;

                {
                    int local = 0;
                    for (int i = tid; i < (int)new_elem_count; i += n_threads) {
                        if (current_scores[i] >= beam_cutoff_score) ++local;
                    }
                    for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) local += simd_shuffle_down(local, offset);
                    if (lane_id == 0) count_buf[warp_id] = local;
                    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                    if (warp_id == 0) {
                        int v = (tid < (int)(n_threads / SIMD_WIDTH)) ? count_buf[lane_id] : 0;
                        for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) v += simd_shuffle_down(v, offset);
                        if (tid == 0) count_buf[0] = v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                }
                if (tid == 0) elem_count = count_buf[0];
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            }

            if (tid == 0) {
                bs_active = (num_guesses == MAX_GUESSES) ? 1 : 0;
                if (bs_active) beam_cutoff_score = hi_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            if (bs_active) {
                int local = 0;
                for (int i = tid; i < (int)new_elem_count; i += n_threads) {
                    if (current_scores[i] >= beam_cutoff_score) ++local;
                }
                for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) local += simd_shuffle_down(local, offset);
                if (lane_id == 0) count_buf[warp_id] = local;
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                if (warp_id == 0) {
                    int v = (tid < (int)(n_threads / SIMD_WIDTH)) ? count_buf[lane_id] : 0;
                    for (int offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) v += simd_shuffle_down(v, offset);
                    if (tid == 0) count_buf[0] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                if (tid == 0) elem_count = count_buf[0];
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            }

            if (tid == 0) elem_count = elem_count < MAX_BEAM_WIDTH ? elem_count : MAX_BEAM_WIDTH;
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }

        // stream compaction: exclusive prefix sum of the >= cutoff predicate gives each passing
        // element its serial write index; gate on dst < MAX_BEAM_WIDTH to keep the first 32 in order.
        for (int i = tid; i < (int)new_elem_count; i += n_threads) {
            compact_offsets[i] = (current_scores[i] >= beam_cutoff_score) ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        for (int d = 1; d < (int)new_elem_count; d <<= 1) {
            int a0 = 0, a1 = 0;
            int i0 = tid, i1 = tid + (int)n_threads;
            if (i0 < (int)new_elem_count && i0 >= d) a0 = compact_offsets[i0 - d];
            if (i1 < (int)new_elem_count && i1 >= d) a1 = compact_offsets[i1 - d];
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            if (i0 < (int)new_elem_count && i0 >= d) compact_offsets[i0] += a0;
            if (i1 < (int)new_elem_count && i1 >= d) compact_offsets[i1] += a1;
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
        for (int i = tid; i < (int)new_elem_count; i += n_threads) {
            if (current_scores[i] >= beam_cutoff_score) {
                int dst = compact_offsets[i] - 1;
                if (dst < MAX_BEAM_WIDTH) {
                    prev_beam_front[dst] = current_beam_front[i];
                    prev_scores[dst] = current_scores[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tid == 0) {
            int total_pass = (new_elem_count > 0) ? compact_offsets[new_elem_count - 1] : 0;
            elem_count = total_pass < MAX_BEAM_WIDTH ? total_pass : MAX_BEAM_WIDTH;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // at the last timestep make the best path element 0
        if (tid == 0 && block_idx == n_timesteps - 1) {
            float best_score = -OF_FLT_MAX;
            ulong best_score_index = 0;
            for (ulong i = 0; i < (ulong)elem_count; i++) {
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
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // copy this new beam front into persistent beam state
        ulong beam_offset = (block_idx + 1) * MAX_BEAM_WIDTH;
        for (ulong i = tid; i < (ulong)elem_count; i += n_threads) {
            prev_scores[i] -= (float)block_back_scores[prev_beam_front[i].state];

            beam_vector[beam_offset + i].state = prev_beam_front[i].state;
            beam_vector[beam_offset + i].prev_element_index = prev_beam_front[i].prev_element_index;
            beam_vector[beam_offset + i].stay = prev_beam_front[i].stay;
        }
        if (tid == 0) current_beam_width = elem_count;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // trace back the best path into states/moves
    if (tid == 0) {
        uchar element_index = 0;
        for (ulong beam_idx = n_timesteps; beam_idx != 0; --beam_idx) {
            ulong beam_addr = beam_idx * MAX_BEAM_WIDTH + element_index;
            states[beam_idx - 1] = (state_t)beam_vector[beam_addr].state;
            moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
            element_index = beam_vector[beam_addr].prev_element_index;
        }
        moves[0] = 1;
    }
}

// ------------------------------------------------------------------ qual / sequence

kernel void compute_qual_data(
    constant beam_params_t &beam_args   [[buffer(0)]],
    device const float   *post_NTC_in [[buffer(1)]],
    device state_t       *_states     [[buffer(2)]],
    device float         *_qual_data  [[buffer(3)]],
    constant float       &posts_scale [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    const ulong chunk = tgid;
    if (chunk >= beam_args.batch_size) {
        return;
    }

    const ulong n_timesteps = beam_args.n_timesteps;
    const ulong num_states = 1ull << beam_args.num_state_bits;
    const ulong num_state_bits = beam_args.num_state_bits;

    device const float *post_NTC = post_NTC_in + chunk * num_states * (n_timesteps + 1);
    device state_t *states = _states + chunk * n_timesteps;
    device float *qual_data = _qual_data + chunk * (n_timesteps * NUM_BASES);

    int shifted_states[2 * NUM_BASES];

    for (ulong block_idx = 0; block_idx < n_timesteps; ++block_idx) {
        int state = states[block_idx];
        states[block_idx] = states[block_idx] % NUM_BASES;
        int base_to_emit = states[block_idx];

        device const float *const timestep_posts = post_NTC + ((block_idx + 1) << num_state_bits);

        float block_prob = (float)(timestep_posts[state]) * posts_scale;

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

        int candidate_state;
        for (ulong state_idx = 0; state_idx < 2 * NUM_BASES; ++state_idx) {
            candidate_state = shifted_states[state_idx];
            bool count_state = (candidate_state != state);
            if (count_state) {
                for (ulong inner_state = 0; inner_state < state_idx; ++inner_state) {
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
        block_prob = fmin(fmax(block_prob, 0.0f), 1.0f);
        block_prob = pow(block_prob, 0.4f);

        const float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (ulong base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] = ((int)base == base_to_emit ? block_prob : wrong_base_prob);
        }
    }
}

kernel void generate_sequence(
    constant beam_params_t &args         [[buffer(0)]],
    device const uchar   *_moves       [[buffer(1)]],
    device const state_t *_states      [[buffer(2)]],
    device const float   *_qual_data   [[buffer(3)]],
    device float         *_base_probs  [[buffer(4)]],
    device float         *_total_probs [[buffer(5)]],
    device char          *_sequence    [[buffer(6)]],
    device char          *_qstring     [[buffer(7)]],
    constant float       &shift        [[buffer(8)]],
    constant float       &scale        [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    const ulong chunk = tgid;
    if (chunk >= args.batch_size) {
        return;
    }

    const ulong n_timesteps = args.n_timesteps;
    device const uchar *moves = _moves + chunk * n_timesteps;
    device const state_t *states = _states + chunk * n_timesteps;
    device const float *qual_data = _qual_data + chunk * n_timesteps * NUM_BASES;
    device float *base_probs = _base_probs + chunk * n_timesteps;
    device float *total_probs = _total_probs + chunk * n_timesteps;
    device char *sequence = _sequence + chunk * n_timesteps;
    device char *qstring = _qstring + chunk * n_timesteps;

    ulong seq_len = 0;
    for (ulong i = 0; i < n_timesteps; ++i) {
        seq_len += moves[i];
        base_probs[i] = 0.0f;
        total_probs[i] = 0.0f;
    }

    ulong seq_pos = 0;
    const char alphabet[4] = {'A', 'C', 'G', 'T'};

    for (ulong blk = 0; blk < n_timesteps; ++blk) {
        int state = states[blk];
        int move = (int)moves[blk];
        int base = state & 3;
        int offset = (blk == 0) ? 0 : move - 1;
        int probPos = (int)(seq_pos + offset);

        base_probs[probPos] += qual_data[blk * NUM_BASES + base];
        for (ulong k = 0; k < NUM_BASES; ++k) {
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

    for (ulong i = 0; i < seq_len; ++i) {
        sequence[i] = alphabet[(int)sequence[i]];
        base_probs[i] = 1.0f - (base_probs[i] / total_probs[i]);
        base_probs[i] = -10.0f * log10(base_probs[i]);
        float qscore = fmin(fmax(base_probs[i] * scale + shift, 1.0f), 50.0f);
        qstring[i] = (char)((int)(33.5f + qscore));
    }
}
