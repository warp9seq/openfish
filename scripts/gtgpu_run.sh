#!/bin/bash

# make clean && make -j cuda=1

MODEL=sup
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

SCORES=test/data/${MODEL}_${BATCH_SIZE}c_scores_TNC.pt

/usr/bin/time  --verbose ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BATCH_SIZE}

diff test/data/${MODEL}_${BATCH_SIZE}c_scores_TNC.pt scores_TNC.pt && echo "no diff between scores_TNC"
diff test/data/${MODEL}_${BATCH_SIZE}c_bwd_NTC.pt bwd_NTC.pt && echo "no diff between bwd_NTC"
diff test/data/${MODEL}_${BATCH_SIZE}c_fwd_NTC.pt fwd_NTC.pt && echo "no diff between fwd_NTC"
diff test/data/${MODEL}_${BATCH_SIZE}c_post_NTC.pt post_NTC.pt && echo "no diff between post_NTC"