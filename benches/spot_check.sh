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

python3 generate_exp_data.py 1000 > "$DATA_FILE"


cargo run --features serialization -- build regressor --data "$DATA_FILE" \
   --hidden-layer-sizes "5" \
   --learning-rate 0.01 \
   --validate-each-epoch \
   --num-threads 4 \
   --no-save \
   --degree 3 \
   --knot-extension-targets 10,20,50 \
   --knot-extension-times 10,20,30 \
   --epochs 100


