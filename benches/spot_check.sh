#!/bin/bash

# this script is going to be psuedo-recursive. 
#It lives in the fekan repo, but it also clones the fekan repo
# to make sure it has the most updated context whereever it runs.
# Yay, portability!

set -e # exit on error
set -x # print commands


DATA_FILE=$(mktemp)".json"
trap 'rm -f $DATA_FILE' EXIT

git rev-parse HEAD

python3 generate_feynman_I-9-18_data.py 6000 > "$DATA_FILE"


cargo run --features serialization --profile release -- build regressor --data "$DATA_FILE" \
   --learning-rate 0.001 \
   --no-save \
   --coefs 15 \
   --hidden-layer-sizes "6,4,1" \
   -e 300 \



