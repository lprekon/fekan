#!/bin/bash

LOG_FILE = "feynman_regression_accuracy.log"
DATA_FILE = $(mktemp)+".json"
trap "rm -f $DATA_FILE" EXIT

echo $(git rev-parse HEAD) > $LOG_FILE

python3 generate_feynman_I-9-18_data.py 1000000 > $DATA_FILE

fekan build regressor --data $DATA_FILE \
    --hidden-layer-sizes "6,4,2,1" \
    --learning-rate 0.001 \
    --validate-each-epoch \
    --log-output \
    --no-save \
    > $LOG_FILE