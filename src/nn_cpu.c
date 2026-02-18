#include "error.h"
#include "misc.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <pthread.h>

static void rotary_emb(
	float *x,
    float *_cos,
    float *_sin,
    const uint64_t seqlen,
    const uint64_t stride_batch,
    const uint64_t stride_seq,
    const uint64_t stride_head,
    const uint64_t rotary_half,
    const uint64_t batch,
    const uint64_t head,
    const uint64_t rot
) {
    float *_o0 = x + (batch * stride_batch) + (head * stride_head) + rot;
    float *_o1 = x + (batch * stride_batch) + (head * stride_head) + rotary_half + rot;

    for (int seq = 0; seq < seqlen; ++seq) {
        float cos_val = *(_cos + (seq * rotary_half) + rot);
        float sin_val = *(_sin + (seq * rotary_half) + rot);

        float *o0 = _o0 + (seq * stride_seq);
        float *o1 = _o1 + (seq * stride_seq);

        float x0 = *o0;
        float x1 = *o1;

        *o0 = x0 * cos_val - x1 * sin_val;
        *o1 = x0 * sin_val + x1 * cos_val;
    }
}

typedef struct {
    float *x;
    float *sin_buf;
    float *cos_buf;
    uint64_t start;
    uint64_t end;
    uint64_t seqlen;
    uint64_t nheads;
    uint64_t head_dim;
    uint64_t rotary_half;
    uint64_t stride_batch;
    uint64_t stride_seq;
    uint64_t stride_head;
} rotary_emb_thread_arg_t;

static void* pthread_single_rotary_emb(void* voidargs) {
    rotary_emb_thread_arg_t* args = (rotary_emb_thread_arg_t*)voidargs;

    for (uint64_t batch = args->start; batch < args->end; ++batch) {
        for (uint64_t head = 0; head < args->nheads; ++head) {
            for (uint64_t rot = 0; rot < args->rotary_half; ++rot) {
                rotary_emb(
                    args->x,
                    args->cos_buf,
                    args->sin_buf,
                    args->seqlen,
                    args->stride_batch,
                    args->stride_seq,
                    args->stride_head,
                    args->rotary_half,
                    batch, head, rot
                );
            }
        }
    }

    pthread_exit(0);
}

void openfish_rotary_emb_cpu(
    void *x,
    void *sin_buf,
    void *cos_buf,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head,
    int nthreads
) {
    // create threads
    nthreads = batch_size < nthreads ? batch_size : nthreads;
    const int chunks_per_thread = batch_size / nthreads;
    const int num_threads_with_one_more_chunk = batch_size % nthreads;

    OPENFISH_LOG_TRACE("dispatching %d threads for cpu decoding", nthreads);

    pthread_t tids[nthreads];
    rotary_emb_thread_arg_t pt_args[nthreads];
    int32_t t, ret;

    // set the data structures
    for (t = 0; t < nthreads; t++) {
        int extra = t < num_threads_with_one_more_chunk ? t : num_threads_with_one_more_chunk;
        pt_args[t].start = t * chunks_per_thread + extra;
        pt_args[t].end = pt_args[t].start + chunks_per_thread + (int)(t < num_threads_with_one_more_chunk);
        pt_args[t].x = (float *)x;
        pt_args[t].sin_buf = (float *)sin_buf;
        pt_args[t].cos_buf = (float *)cos_buf;
        pt_args[t].seqlen = seqlen;
        pt_args[t].nheads = nheads;
        pt_args[t].head_dim = head_dim;
        pt_args[t].rotary_half = rotary_half;
        pt_args[t].stride_batch = stride_batch;
        pt_args[t].stride_seq = stride_seq;
        pt_args[t].stride_head = stride_head;
    }

    // score tensors
    for (t = 0; t < nthreads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_rotary_emb, (void *)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < nthreads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }
}
