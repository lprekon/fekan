#!/bin/bash

LOG_FILE="ellipj_regression_accuracy.log"
DATA_FILE=$(mktemp)+".json"
trap "rm -f $DATA_FILE" EXIT

which python3
python3 generate_ellipj_data.py 1000000 > $DATA_FILE

fekan build regressor --data $DATA_FILE \
    --hidden-layer-sizes "2,2" \
    --learning-rate 0.001 \
    --validate-each-epoch \
    --log-output \
    --no-save \
    > $LOG_FILE