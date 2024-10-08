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
--knot-extension-times "10" \
--sym-times "10" \
--sym-threshold 0.98 \
--prune-times "10" \
--prune-threshold 0.01 \
--learning-rate 0.01 \
--validate-each-epoch  \
--num-threads 8 
sudo cargo clean