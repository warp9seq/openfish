#!/bin/bash

# In-memory CPU-vs-GPU decode test (the CPU decoder is treated as ground truth).
#
# Fully hermetic: the harness synthesises its own scores in memory, so there is nothing to
# download and nothing is read from or written to disk.
#
# Build first, then run:
#   make          && make test     # CPU-only: determinism + sanity check
#   make cuda=1   && make test      # compares GPU against CPU
#   make rocm=1   && make test
#
# `./test/test.sh mem` runs the build under valgrind.

die() {
	echo "$1" >&2
	echo
	exit 1
}

mem=0
[ "$1" = "mem" ] && mem=1

ex() {
    if [ $mem -eq 1 ]; then
        valgrind --error-exitcode=1 --leak-check=full --show-leak-kinds=all --suppressions=test/valgrind.supp "$@"
    else
        "$@"
    fi
}

[ -x ./test_openfish ] || die "test_openfish not built -- run: make [cuda=1|rocm=1|metal=1] test"

# One case per model (state_len). Batch sizes kept modest so the run is quick while still
# generating plenty of positions to surface any CPU/GPU divergence. Under valgrind, shrink
# the batches so the run finishes in reasonable time.
if [ $mem -eq 1 ]; then
    ex ./test_openfish 2 3 || die "fast test failed"
    ex ./test_openfish 2 4 || die "hac test failed"
    ex ./test_openfish 2 5 || die "sup test failed"
else
    ex ./test_openfish 100 3 || die "fast test failed"
    ex ./test_openfish 40  4 || die "hac test failed"
    ex ./test_openfish 20  5 || die "sup test failed"
fi

echo "all tests passed"
