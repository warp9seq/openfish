#!/bin/bash

# make sure to build with debug=1

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
        valgrind --error-exitcode=1 --leak-check=full --show-leak-kinds=all --suppressions=test/valgrind.supp "$@"
    else
        "$@"
    fi
}

MODEL=fast

STATE_LEN=3
BATCH_SIZE=1
TIMESTEPS=1666
TENS_LEN=$(( BATCH_SIZE*(TIMESTEPS+1)*64 ))
INTENS_LEN=$(( BATCH_SIZE*(TIMESTEPS) ))

SCORES=test/data/scores.blob

ex ./openfish ${SCORES} ${BATCH_SIZE} ${STATE_LEN} || die "tool failed"

diff moves.blob test/data/moves.blob || die "tool failed"
diff sequence.blob test/data/sequence.blob || die "tool failed"
diff qstring.blob test/data/qstring.blob || die "tool failed"
