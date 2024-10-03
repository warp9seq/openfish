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

DATA_DIR=/data/bonwon/slorado_test_data/blobs
SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC.blob

/usr/bin/time  --verbose ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BATCH_SIZE} ${STATE_LEN} || die "tool failed"

echo "diff bwd_NTC..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_bwd_NTC.blob bwd_NTC.blob || die "failed diff"
echo "passed!"
echo "diff fwd_NTC..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_fwd_NTC.blob fwd_NTC.blob || die "failed diff"
echo "passed!"
echo "diff post_NTC..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_post_NTC.blob post_NTC.blob || die "failed diff"
echo "passed!"
echo "diff moves..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_moves.blob moves.blob || die "failed diff"
echo "passed!"
echo "diff sequence..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_sequence.blob sequence.blob || die "failed diff"
echo "passed!"
echo "diff qual_data..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_qual_data.blob qual_data.blob || die "failed diff"
echo "passed!"
echo "diff base_probs..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_base_probs.blob base_probs.blob || die "failed diff"
echo "passed!"
echo "diff total_probs..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_total_probs.blob total_probs.blob || die "failed diff"
echo "passed!"
echo "diff qstring..."
diff ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_qstring.blob qstring.blob || die "failed diff"
echo "passed!"

echo "tests passed for ${MODEL}"