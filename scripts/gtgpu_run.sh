#!/bin/bash

make clean && make -j cuda=1

MODEL=fast
SCORES=test/data/scores_TNC.pt

/usr/bin/time  --verbose ./openfish ${SCORES} models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0

diff test/data/scores_TNC.pt scores_TNC.pt && echo "no diff between scores_TNC"
diff test/data/bwd_NTC.pt bwd_NTC.pt && echo "no diff between bwd_NTC"
diff test/data/fwd_NTC.pt fwd_NTC.pt && echo "no diff between fwd_NTC"
diff test/data/post_NTC.pt post_NTC.pt && echo "no diff between post_NTC"