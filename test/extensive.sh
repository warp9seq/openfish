#!/bin/bash

DATA_DIR="/data/bonwon/slorado_test_data/blobs"

# terminate script
die() {
	echo "$1" >&2
	echo
	exit 1
}

gpu=0

if [ "$1" = 'mem' ]; then
    mem=1
else
    mem=0
fi

if [ "$1" = 'mem_gpu' ]; then
    mem=1
    gpu=1
else
    mem=0
fi

if [ "$1" = 'cuda' ]; then
    cuda=1
    gpu=1
else
    cuda=0
fi

ex() {
    if [ $mem -eq 1 ]; then
        /install/valgrind-3.21/bin/valgrind --error-exitcode=1  --leak-check=full --show-leak-kinds=all --suppressions=test/valgrind.supp --gen-suppressions=yes "$@"
    elif [ $cuda -eq 1 ]; then
        cuda-memcheck --leak-check full "$@"
    else
        "$@"
    fi
}

quick_test() {
    MODEL=$1

    STATE_LEN=3
    BATCH_SIZE=1000
    if [ "$MODEL" = "fast" ]; then
        BATCH_SIZE=1000
        STATE_LEN=3
    fi
    if [ "$MODEL" = "hac" ]; then
        BATCH_SIZE=400
        STATE_LEN=4
    fi
    if [ "$MODEL" = "sup" ]; then
        BATCH_SIZE=200
        STATE_LEN=5
    fi

    # warning: if the gpu decoder takes in full float values its not going to panic
    SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC.blob
    if [ $gpu -eq 1 ]; then
        SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC_half.blob
    fi

    ex ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BATCH_SIZE} ${STATE_LEN} || die "tool failed"
}

quick_test fast || die "running the tool failed"
quick_test hac || die "running the tool failed"
quick_test sup || die "running the tool failed"

echo "tests passed"