#!/bin/bash

if [ ! -f "compare_blob" ]; then
    g++ -o compare_blob test/compare_blob.cpp || die "Could not create compile compare_blob"
fi

DATA_DIR=test/data/openfish-blobs/
LINK="https://unsw-my.sharepoint.com/:u:/g/personal/z5136909_ad_unsw_edu_au/EewLv0Ei2U9NmR33xj8Q-mEBODG5Q1-900FUM7KIBH1HmQ?download=1"
TAR="openfish-blobs.tar.gz"
if [ ! -d ${DATA_DIR} ]; then
    test -e ${TAR} && rm ${TAR}
    wget ${LINK} -O ${TAR}|| die "Could not download openfish-blobs"
    tar -xvf ${TAR} --one-top-level=test/data/ || die "Could not extract openfish-blobs"
    rm ${TAR}
fi