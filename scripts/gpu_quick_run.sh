#!/bin/bash

die() {
	echo "$1" >&2
	echo
	exit 1
}

if [ "$#" -ne 1 ]; then
    die "usage: ./quick_run.sh <model>"
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
SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC_half.blob

/usr/bin/time  --verbose ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BATCH_SIZE} ${STATE_LEN} || die "tool failed"

# todo: check if files exist
# todo: compare qual base data and stuff
# echo "comparing bwd tensors..."
# ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_bwd_NTC.blob bwd_NTC.blob $TENS_LEN
# echo "comparing post tensors..."
# ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_post_NTC.blob post_NTC.blob $TENS_LEN

echo "diff moves..."
diff moves.gpu_ref moves.blob && echo "passed!" || die "diff failed"
echo "diff sequence..."
diff sequence.gpu_ref sequence.blob && echo "passed!" || die "diff failed"
# echo "diff qstring..."
# diff qstring.gpu_ref qstring.blob && echo "passed!" || die "diff failed"

echo "tests passed for ${MODEL}"

# .gpu refs are for checking if that 
# num batches for each model: 20k reads
# fast - 140
# hac - 345
# sup - 685