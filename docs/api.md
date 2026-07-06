# Openfish API

**Note: As Openfish is still in early stages, the API may change in future versions. This message will be removed when it is stabilised.**

All public symbols are declared in `include/openfish/openfish.h` and prefixed with `openfish_`. Throughout the API the score/activation tensor dimensions are named:

| Name | Meaning |
|------|---------|
| `n_timesteps` | number of time steps (the `T` axis of the `scores_TNC` tensor) |
| `batch_size`  | number of chunks/reads in the batch (the `N` axis) |
| `n_channels`  | per-time-step score width = `4^state_len × 4` (the `C` axis) |
| `state_len`   | CTC k-mer state length (3 = fast, 4 = hac, 5 = sup) |

**Decoder options** — call `openfish_decoder_default_opts()` for ONT DNA v4.2.0 model defaults:

```c
typedef struct openfish_opt {
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
} openfish_opt_t;

// returns { beam_cut, blank_score, q_shift, q_scale }
openfish_opt_t opt = openfish_decoder_default_opts(); // { 100.0, 2.0, 0.0, 1.0 }
```

## CPU decoding

```c
void openfish_decode_cpu(
    int n_timesteps,             // number of time steps (T)
    int batch_size,              // number of chunks (N)
    int n_channels,              // per-time-step score width (C)
    int n_threads,                // number of CPU threads
    const void *scores_TNC,      // input score tensor [T × N × C], float32 (read-only)
    int state_len,               // CTC state length (3=fast, 4=hac, 5=sup)
    const openfish_opt_t *options,
    uint8_t **moves,             // output: move array
    char **sequence,             // output: base sequence
    char **qstring               // output: quality string
);
```

Example usage:

```c
torch::Tensor scores = module.forward(some_signal_data);
const openfish_opt_t opt = openfish_decoder_default_opts();

const int n_timesteps = scores.size(0);
const int batch_size  = scores.size(1);
const int n_channels  = scores.size(2);

const int state_len = 3; // depends on model
const int n_threads  = 32;

// pointers to results (allocated by openfish)
uint8_t *moves;
char *sequence;
char *qstring;

openfish_decode_cpu(n_timesteps, batch_size, n_channels, n_threads,
                    scores.data_ptr(), state_len, &opt,
                    &moves, &sequence, &qstring);

// iterate through each chunk
for (size_t chunk = 0; chunk < batch_size; ++chunk) {
    size_t idx = chunk * n_timesteps;

    // collect results based on move table
    std::vector<uint8_t> chunk_moves(moves + idx, moves + idx + n_timesteps);
    size_t num_bases = 0;
    for (uint8_t move : chunk_moves) {
        num_bases += move;
    }
    std::string chunk_seq  = std::string(sequence + idx, num_bases);
    std::string chunk_qstr = std::string(qstring + idx, num_bases);

    // do something with chunk_moves, chunk_seq, chunk_qstr here
}

// free memory allocated by openfish
free(moves);
free(sequence);
free(qstring);
```

## GPU decoding

Requires a `cuda=1`, `rocm=1` or `metal=1` build. Scores must be in device memory and are **float16** for the GPU path (CPU path is float32). For the Metal backend a device buffer is an `MTLBuffer` (the `scores_TNC` handle returned by the harness upload helper); on Apple Silicon's unified memory the `gpubuf` result pointers alias the shared buffers directly.

```c
// Allocate persistent GPU working buffers (once per n_timesteps/batch_size/state_len combination)
openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(n_timesteps, batch_size, state_len);

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_TNC,          // device pointer; float16 (read-only)
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,                 // outputs are allocated on the host and copied back
    char **sequence,
    char **qstring
);

openfish_gpubuf_free(gpubuf);
```

`openfish_gpubuf_size(n_timesteps, batch_size, state_len)` returns the total number of device bytes a `gpubuf` of those dimensions occupies (useful for VRAM accounting before `openfish_gpubuf_init`).
