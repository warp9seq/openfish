# Openfish API

**Note: As Openfish is still in early stages, the API may change in the future versions. This message will be removed when it is stabilished.**

All public symbols are declared in `include/openfish/openfish.h`.

**Decoder options** — populate with the `DECODER_INIT` macro for sensible defaults:

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
    int C,                       // number of score classes (alphabet size)
    int nthreads,                // number of CPU threads
    void *scores_TNC,            // input score tensor [T × N × C], float32
    int state_len,               // CTC state length (3=fast, 4=hac, 5=sup)
    const openfish_opt_t *opt,
    uint8_t **moves,             // output: move array (caller-allocated, length N*T)
    char **sequence,             // output: base sequence (caller-allocated)
    char **qstring               // output: quality string (caller-allocated)
);
```

**GPU decoding** (requires a `cuda=1` or `rocm=1` build):

```c
// Allocate persistent GPU working buffers (once per T/N/state_len combination)
openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(T, N, state_len);

void openfish_decode_gpu(
    int T, int N, int C,
    void *scores_TNC,            // input scores on host memory; float16 for GPU path
    int state_len,
    const openfish_opt_t *opt,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
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