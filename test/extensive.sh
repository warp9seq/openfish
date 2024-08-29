#!/bin/bash

DATA_DIR=/data/bonwon/slorado_test_data/tensors

# terminate script
die() {
	echo "$1" >&2
	echo
	exit 1
}

if [ "$1" = 'mem' ]; then
    mem=1
else
    mem=0
fi

ex() {
    if [ $mem -eq 1 ]; then
        valgrind --leak-check=full --error-exitcode=1 "$@"
    else
        "$@"
    fi
}

quick_test() {
    MODEL=$1
    BATCH_SIZE=1000
    if [ "$MODEL" = "fast" ]; then
        BATCH_SIZE=1000
    fi
    if [ "$MODEL" = "hac" ]; then
        BATCH_SIZE=400
    fi
    if [ "$MODEL" = "sup" ]; then
        BATCH_SIZE=200
    fi
    SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC.pt

    ex ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BATCH_SIZE} || die "tool failed"

    diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC.pt scores_TNC.pt || die "failed diff"
    diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_bwd_NTC.pt bwd_NTC.pt || die "failed diff"
    diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_fwd_NTC.pt fwd_NTC.pt || die "failed diff"
    diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_post_NTC.pt post_NTC.pt || die "failed diff"

    echo "tests passed for ${MODEL}"
}

quick_test fast || die "running the tool failed"
quick_test hac || die "running the tool failed"
quick_test sup || die "running the tool failed"

echo "tests passed"