# Openfish API

**Note: As Openfish is still in early stages, the API may change in the future versions. This message will be removed when it is stabilished.**

All public symbols are declared in `include/openfish/openfish.h`.

**Decoder options** — populate with the `DECODER_INIT` macro for ONT DNA v4.2.0 model defaults:

```c
openfish_opt_t opt = DECODER_INIT;
// DECODER_INIT expands to: {32, 100.0, 2.0, 0.0, 1.0, 1.0, false}
// fields: beam_width, beam_cut, blank_score, q_shift, q_scale, temperature, move_pad
```

**CPU decoding:**

```c
void openfish_decode_cpu(
    int T,                       // number of time steps
    int N,                       // batch size (number of chunks)
    int C,                       // score state size
    int nthreads,                // number of CPU threads
    void *scores_TNC,            // input score tensor [T × N × C], float32
    int state_len,               // CTC state length (3=fast, 4=hac, 5=sup)
    const openfish_opt_t *opt,
    uint8_t **moves,             // output: move array
    char **sequence,             // output: base sequence
    char **qstring               // output: quality string
);

// example usage
torch::Tensor scores = module.forward(some_signal_data);
const openfish_opt_t opt = DECODER_INIT; // default values for ONT DNA v4.2.0 models

const int T = scores.size(0);
const int N = scores.size(1);
const int C = scores.size(2);

const int state_len = 3; // depends on model
const int nthreads = 32;

// pointer to results
uint8_t *moves;
char *sequence;
char *qstring;

openfish_decode_cpu(T, N, C, nthreads, scores.data_ptr(), state_len, &opt, &moves, &sequence, &qstring);

// iterate through each chunk
for (size_t chunk = 0; chunk < N; ++chunk) {
    size_t idx = chunk * T;

    // collect results based on move table
    std::string chunk_moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
    size_t num_bases = 0;
    for (uint8_t move: chunk_moves) {
        num_bases += move;
    }
    std::string chunk_seq = std::string(sequence + idx, num_bases);
    std::string chunk_qstr = std::string(qstring + idx, num_bases);

    // do something with chunk_moves, chunk_seq, chunk_qstr here
}

// free memory allocated by openfish
free(moves);
free(sequence);
free(qstring);
```

**GPU decoding** (requires a `cuda=1` or `rocm=1` build):

```c
// Allocate persistent GPU working buffers (once per T/N/state_len combination)
openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(T, N, state_len);

void openfish_decode_gpu(
    int T, int N, int C,
    void *scores_TNC,                // input scores on device memory; float16 for GPU path
    int state_len,
    const openfish_opt_t *opt,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,                 // output is automatically copied to host memory
    char **sequence,
    char **qstring
);

openfish_gpubuf_free(gpubuf);
```

**Rotary embeddings** (used by transformer-based models):

```c
// CPU
void openfish_rotary_emb_cpu(void *x, void *sin_buf, void *cos_buf,
    int batch_size, int seqlen, int nheads, int head_dim, int rotary_half,
    int stride_batch, int stride_seq, int stride_head, int nthreads);

// GPU (x_gpu / sin_gpu / cos_gpu must be device pointers)
void openfish_rotary_emb_gpu(void *x_gpu, void *sin_gpu, void *cos_gpu,
    int batch_size, int seqlen, int nheads, int head_dim, int rotary_half,
    int stride_batch, int stride_seq, int stride_head);
```
