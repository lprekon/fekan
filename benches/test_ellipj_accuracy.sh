#!/bin/bash

# this script is going to be psuedo-recursive. 
#It lives in the fekan repo, but it also clones the fekan repo
# to make sure it has the most updated context whereever it runs.
# Yay, portability!

set -e # exit on error

git clone https://github.com/lprekon/fekan.git
cd fekan
cargo install fekan --path . --features "serialization"

cd benches

LOG_FILE="ellipj_regression_accuracy.log"
touch $LOG_FILE
DATA_FILE=$(mktemp)".json"
trap "rm -f $DATA_FILE" EXIT

if [ -n $S3_BUCKET ]; then
    echo "S3 target detected. Checking connection..."
    aws s3 ls s3://$S3_BUCKET
    if [$? -ne 0]; then
        echo "Error: could not connect to s3 bucket $S3_BUCKET"
        exit 1
    else 
        echo "Connection successful"
    fi
fi

echo $(git rev-parse HEAD) > $LOG_FILE

echo "ensuring python packages are installed"
pip3 install -r requirements.txt

echo "generating data"
python3 generate_ellipj_data.py 1000000 > $DATA_FILE

echo "running regression"
fekan build regressor --data $DATA_FILE \
   --hidden-layer-sizes "2,2" \
   --learning-rate 0.001 \
   --validate-each-epoch \
   --log-output \
   --no-save \
   > $LOG_FILE
echo "regression complete"

if [ -n $S3_BUCKET ]; then
    echo "uploading log file to s3"
    aws s3 cp $LOG_FILE s3://$S3_BUCKET
fi