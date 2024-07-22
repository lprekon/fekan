#!/bin/sh
mv flamegraph.svg flamegraph_old.svg
DATA_FILE=$(mktemp)".json"
trap 'rm -f $DATA_FILE' EXIT

python3 generate_ellipj_data.py 100000 > "$DATA_FILE"

sudo cargo flamegraph --features="serialization" -- \
build regressor \
--data "$DATA_FILE" \
--no-save \
--hidden-layer-sizes "2" \
-e 10 \
--coefs 5 \
--knot-extension-targets 20 \
--knot-extension-times 4 \
--learning-rate 0.01 \
--validate-each-epoch  \
--num-threads 1 
sudo cargo clean