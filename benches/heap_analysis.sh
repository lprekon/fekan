#!/bin/sh
set -e # exit on error
set -x # print commands

if [ -n "$S3_BUCKET" ]; then
    echo "S3 target detected. Checking connection..."
    aws s3 ls s3://$S3_BUCKET
    if [$? -ne 0]; then
        echo "Error: could not connect to s3 bucket $S3_BUCKET"
        exit 1
    else 
        echo "Connection successful"
    fi
fi

# grab the latest code
git clone https://github.com/lprekon/fekan.git
cd fekan
git checkout "$BRANCH"
git rev-parse HEAD
cargo build --release --features serialization

DATA_FILE=$(mktemp)".json"
trap 'rm -f $DATA_FILE' EXIT
pip3 install -r benches/requirements.txt
python3 benches/generate_ellipj_data.py 10000 > $DATA_FILE

valgrind --tool=dhat target/release/fekan build regressor \
    --data $DATA_FILE \
    --hidden-layer-sizes 2 \
    --num-epochs 500 \
    --coefs 16 \
    --no-save

if [ -n "$S3_BUCKET" ]; then
    aws s3 cp dhat.out.* s3://$S3_BUCKET
fi


