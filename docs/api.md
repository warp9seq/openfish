# Openfish API

**Note: As Openfish is still in early stages, the API may change in future versions. This message will be removed when it is stabilised.**

All public symbols are declared in `include/openfish/openfish.h` and prefixed with `openfish_`. Throughout the API the score/activation tensor dimensions are named:

| Name | Meaning |
|------|---------|
| `n_timesteps` | number of time steps (the `T` axis of the `scores_NTC` tensor) |
| `batch_size`  | number of chunks in the batch (the `N` axis) |
| `n_channels`  | per-time-step score width = `4^state_len × 4` (the `C` axis) |
| `state_len`   | CTC k-mer state length (3 = fast, 4 = hac, 5 = sup) |

The score tensor is consumed in **`NTC` layout** (`[N × T × C]`, batch-major): each batch element's
`[T × C]` block is contiguous in memory. This matches the natural output layout of the model, so no
transpose is needed on the caller's side.

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

**Score element type** — every decode entry point takes a `score_dtype` and a `score_scale` that
describe the element type of `scores_NTC`:

```c
typedef enum {
    OPENFISH_SCORE_F16 = 0,  // native float scores (fp16 on the GPU path, fp32 on the CPU path)
    OPENFISH_SCORE_I8  = 1   // int8 quantized scores in [-127, 127]
} openfish_score_dtype_t;
```

Each raw score is dequantized on read as `(float)s * score_scale`:

- `OPENFISH_SCORE_F16` — the unquantized pipeline; pass `score_scale = 1.0f`.
- `OPENFISH_SCORE_I8` — int8 quantized scores; `score_scale` is the dequant factor (e.g. `5.0f/127.0f`).

## CPU decoding

```c
void openfish_decode_cpu(
    int n_timesteps,             // number of time steps (T)
    int batch_size,              // number of chunks (N)
    int n_channels,              // per-time-step score width (C)
    int n_threads,               // number of CPU threads
    const void *scores_NTC,      // input score tensor [N × T × C] (read-only)
    openfish_score_dtype_t score_dtype, // OPENFISH_SCORE_F16 (fp32 here) or OPENFISH_SCORE_I8
    float score_scale,           // dequant factor applied on read (1.0 for the fp32 path)
    int state_len,               // CTC state length (3=fast, 4=hac, 5=sup)
    const openfish_opt_t *options,
    uint8_t **moves,             // output: move array
    char **sequence,             // output: base sequence
    char **qstring               // output: quality string
);
```

Example usage:

```c
torch::Tensor scores = module.forward(some_signal_data); // NTC: [N × T × C]
const openfish_opt_t opt = openfish_decoder_default_opts();

const int batch_size  = scores.size(0);
const int n_timesteps = scores.size(1);
const int n_channels  = scores.size(2);

const int state_len = 3; // depends on model
const int n_threads  = 32;

// pointers to results (allocated by openfish)
uint8_t *moves;
char *sequence;
char *qstring;

openfish_decode_cpu(n_timesteps, batch_size, n_channels, n_threads,
                    scores.data_ptr(), OPENFISH_SCORE_F16, 1.0f, state_len, &opt,
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

### CPU beam search over pre-computed posteriors

`openfish_decode_cpu_beam` runs only the second half of the decoder — the beam search and
sequence generation — over posteriors that have **already** been computed by a forward/backward
scan. The scan output must be supplied in `gpubuf->bwd_NTC` and `gpubuf->post_NTC`. This is meant to
be paired with `openfish_decode_gpu_scan` on a **unified-memory GPU** (Apple Silicon, or an APU): the
GPU runs the heavy scan and the CPU beams over the shared posteriors with no device→host copy.

```c
void openfish_decode_cpu_beam(
    int n_timesteps,
    int batch_size,
    int n_channels,
    int n_threads,
    const void *scores_NTC,          // same raw scores the scan consumed (int8 or fp16); read sparsely
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf, // must have bwd_NTC / post_NTC already filled by a scan
    uint8_t **moves,                 // outputs allocated on the host
    char **sequence,
    char **qstring
);
```

```c
// unified-memory GPU: scan on the GPU, beam on the CPU over the shared posteriors
openfish_decode_gpu_scan(n_timesteps, batch_size, n_channels, scores_gpu,
                         OPENFISH_SCORE_I8, 5.0f/127.0f, state_len, &opt, gpubuf);
openfish_decode_cpu_beam(n_timesteps, batch_size, n_channels, n_threads, scores_host,
                         OPENFISH_SCORE_I8, 5.0f/127.0f, state_len, &opt, gpubuf,
                         &moves, &sequence, &qstring);
```

## GPU decoding

Requires a `cuda=1`, `rocm=1` or `metal=1` build. Scores must be in device memory. The GPU path accepts either **float16** (`OPENFISH_SCORE_F16`) or **int8** (`OPENFISH_SCORE_I8`) scores — selected via `score_dtype` — whereas the `OPENFISH_SCORE_F16` CPU path is float32. For the Metal backend a device buffer is an `MTLBuffer` (the `scores_NTC` handle returned by the harness upload helper); on Apple Silicon's unified memory the `gpubuf` result pointers alias the shared buffers directly.

```c
// Allocate persistent GPU working buffers (once per n_timesteps/batch_size/state_len combination)
openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(n_timesteps, batch_size, state_len);

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,          // device pointer; [N × T × C] (read-only)
    openfish_score_dtype_t score_dtype, // OPENFISH_SCORE_F16 (fp16 here) or OPENFISH_SCORE_I8
    float score_scale,               // dequant factor applied on read (1.0 for the fp16 path)
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

### GPU scan only

`openfish_decode_gpu_scan` runs only the forward/backward scan on the GPU — it fills
`gpubuf->bwd_NTC` and `gpubuf->post_NTC` and returns, with no beam search and no host result
buffers. Pair it with `openfish_decode_cpu_beam` (see above) to split the decode across the GPU
(scan) and CPU (beam); on a unified-memory GPU the CPU reads the posteriors with no copy.

```c
void openfish_decode_gpu_scan(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,          // device pointer; [N × T × C] (read-only)
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf  // bwd_NTC / post_NTC are filled on return
);
```
