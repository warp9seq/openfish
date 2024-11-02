#!/bin/bash

# make sure to build with debug=1

die() {
	echo "$1" >&2
	echo
	exit 1
}

if [ "$#" -ne 1 ]; then
    die "usage: ./quick_run.sh <model>"
fi

if [ ! -f "compare_blob" ]; then
    g++ -o compare_blob test/compare_blob.cpp
fi

MODEL=$1

STATE_LEN=3
BATCH_SIZE=1000
TIMESTEPS=1666
TENS_LEN=0
INTENS_LEN=0
if [ "$MODEL" = "fast" ]; then
    BATCH_SIZE=1000
    STATE_LEN=3
    TENS_LEN=$(( BATCH_SIZE*(TIMESTEPS+1)*64 ))
    INTENS_LEN=$(( BATCH_SIZE*(TIMESTEPS) ))
fi
if [ "$MODEL" = "hac" ]; then
    BATCH_SIZE=400
    STATE_LEN=4
    TENS_LEN=$(( BATCH_SIZE*(TIMESTEPS+1)*256 ))
    INTENS_LEN=$(( BATCH_SIZE*(TIMESTEPS) ))
fi
if [ "$MODEL" = "sup" ]; then
    BATCH_SIZE=200
    STATE_LEN=5
    TENS_LEN=$(( BATCH_SIZE*(TIMESTEPS+1)*1024 ))
    INTENS_LEN=$(( BATCH_SIZE*(TIMESTEPS) ))
fi

DATA_DIR=/data/bonwon/slorado_test_data/blobs
SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC.blob

/usr/bin/time  --verbose ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BATCH_SIZE} ${STATE_LEN} || die "tool failed"

./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_bwd_NTC.blob bwd_NTC.blob $TENS_LEN
./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_fwd_NTC.blob fwd_NTC.blob $TENS_LEN
./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_post_NTC.blob post_NTC.blob $TENS_LEN
./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_qual_data.blob qual_data.blob $(( 4*(INTENS_LEN) ))
./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_total_probs.blob total_probs.blob $INTENS_LEN
./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_base_probs.blob base_probs.blob $INTENS_LEN
