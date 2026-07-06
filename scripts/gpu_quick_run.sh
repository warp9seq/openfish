#!/bin/bash

# make sure to build with debug=1

die() {
	echo "$1" >&2
	echo
	exit 1
}

DATA_URL="https://unsw-my.sharepoint.com/:u:/g/personal/z5136909_ad_unsw_edu_au/EewLv0Ei2U9NmR33xj8Q-mEBODG5Q1-900FUM7KIBH1HmQ?download=1"
DATA_DIR=test/openfish-blobs
# download test set given url
#
DOWNLOAD_TEST_DATA() {
	# data set exists
	if [ -d ${DATA_DIR} ]; then
		return
	fi

	mkdir -p test
	tar_path=test/data.tgz
	if command -v wget >/dev/null 2>&1; then
		wget -O $tar_path ${DATA_URL} || rm -rf $tar_path ${testdir}
	else
		curl -L -o $tar_path ${DATA_URL} || rm -rf $tar_path ${testdir}
	fi
	echo "Extracting. Please wait."
	tar -xf $tar_path -C test || rm -rf $tar_path ${testdir}
	rm -f $tar_path
}

# portable timing wrapper: GNU time uses --verbose, BSD/macOS time uses -l; fall back to none.
if /usr/bin/time --verbose true >/dev/null 2>&1; then
    TIME="/usr/bin/time --verbose"
elif /usr/bin/time -l true >/dev/null 2>&1; then
    TIME="/usr/bin/time -l"
else
    TIME=""
fi


if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    die "usage: ./gpu_quick_run.sh <model> [f16|i8]
  f16 (default): decode fp16 scores and compare against the reference blobs
  i8           : decode fp16 and int8 (quantised, score_scale = 5/127) and report their divergence"
fi

if [ ! -f "compare_blob" ]; then
    g++ -o compare_blob test/compare_blob.cpp
fi

MODEL=$1
MODE=${2:-f16}
if [ "$MODE" != "f16" ] && [ "$MODE" != "i8" ]; then
    die "unknown mode '$MODE' (expected f16 or i8)"
fi

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

SCORES=${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_scores_TNC_half.blob

DOWNLOAD_TEST_DATA

if [ "$MODE" = "i8" ]; then
    # int8 quantised path. The same fp16 score blob is decoded twice: once as fp16 (the
    # reference), and once as int8 (main.c quantises it on the host to round(tanh(x)*127) and
    # decodes with score_scale = 5/127). We then report how far the int8 result diverges from
    # fp16 -- int8 is inherently lossy, so this is a divergence report, not a pass/fail check.
    echo "=== decoding fp16 (reference) ==="
    ./openfish ${SCORES} ${BATCH_SIZE} ${STATE_LEN} 0 || die "fp16 decode failed"
    for f in bwd_NTC post_NTC qual_data total_probs moves sequence qstring; do
        mv ${f}.blob ${f}.f16.blob
    done

    echo "=== decoding int8 (quantised, score_scale = 5/127) ==="
    $TIME ./openfish ${SCORES} ${BATCH_SIZE} ${STATE_LEN} 1 || die "int8 decode failed"

    echo "=== int8 vs fp16: guide / posterior tensor divergence (max & avg elem diff) ==="
    ./compare_blob bwd_NTC.f16.blob     bwd_NTC.blob
    ./compare_blob post_NTC.f16.blob    post_NTC.blob
    ./compare_blob qual_data.f16.blob   qual_data.blob
    ./compare_blob total_probs.f16.blob total_probs.blob

    echo "=== int8 vs fp16: output divergence (moves is the primary signal; sequence/qstring"
    echo "    byte diffs are inflated by move-shift realignment) ==="
    for f in moves sequence qstring; do
        total=$(wc -c < ${f}.blob)
        ndiff=$(cmp -l ${f}.blob ${f}.f16.blob 2>/dev/null | wc -l)
        pct=$(awk "BEGIN{printf \"%.3f\", ($total>0)?100*$ndiff/$total:0}")
        echo "${f}: ${ndiff} / ${total} bytes differ (${pct}%)"
    done
else
    $TIME ./openfish ${SCORES} ${BATCH_SIZE} ${STATE_LEN} || die "tool failed"

    ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_bwd_NTC.blob bwd_NTC.blob
    ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_post_NTC.blob post_NTC.blob
    ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_qual_data.blob qual_data.blob
    ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_total_probs.blob total_probs.blob
    # ./compare_blob ${DATA_DIR}/${MODEL}_${BATCH_SIZE}c_base_probs.blob base_probs.blob
fi