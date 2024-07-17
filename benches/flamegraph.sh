#!/bin/sh
mv flamegraph.svg flamegraph_old.svg
DATA_FILE=$(mktemp)".json"
trap "rm -f $temp_file" EXIT

python3 generate_ellipj_data.py 100000 > $DATA_FILE

sudo cargo flamegraph --features="serialization" -- \
build regressor \
--data $DATA_FILE \
--no-save \
--hidden-layer-sizes "2,2" \
-e 5 \
--max-knot-length 1000 \
--learning-rate 0.01 \
--validate-each-epoch 
sudo cargo clean