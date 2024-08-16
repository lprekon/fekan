#!/bin/sh
set -e # exit on error
set -x # print commands

# make sure we have all the variables and accesses we need

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
# set our defaults if we don't have them
if [ -z "$BRANCH" ]; then
    BRANCH="main"
fi
if [ -n "$KNOT_EXTENSION_TARGETS" ]; then
    KNOT_EXTENSION_FLAG="--knot-extension-targets $KNOT_EXTENSION_TARGETS --knot-extension-times $KNOT_EXTENSION_TIMES"
fi
if [ -n "$SYMBOLIFICATION_TIMES" ]; then
    SYMBOLIFICATION_FLAG="--sym-times $SYMBOLIFICATION_TIMES --sym-threshold $SYMBOLIFICATION_THRESHOLD"
fi
if [ -n "$PRUNING_TIMES" ]; then
    PRUNING_FLAG="--prune-times $PRUNING_TIMES --prune-threshold $PRUNING_THRESHOLD"
fi
if [ -n "$L1_LAMBDA" ] then
    L1_LAMBDA_FLAG="--l1 $L1_LAMBDA"
fi
if [ -n "$ENTROPY_LAMBDA" ] then
    ENTROPY_FLAG="--entropy $ENTROPY_LAMBDA"
fi
if [ -n "$LEARNING_RATE" ]; then
    LEARNING_RATE_FLAG="--learning-rate $LEARNING_RATE"
fi
if [ -n "$HIDDEN_LAYER_SIZES" ]; then
    HIDDEN_LAYER_SIZES_FLAG="--hidden-layer-sizes $HIDDEN_LAYER_SIZES"
fi
if [ -n "$NUM_EPOCHS" ]; then
    NUM_EPOCHS_FLAG="--num-epochs $NUM_EPOCHS"
fi
if [ -n "$DEGREE" ]; then
    DEGREE_FLAG="--degree $DEGREE"
fi
if [ -n "$COEFS" ]; then
    COEFS_FLAG="--coefs $COEFS"
fi
if [ -n "$BATCH_SIZE" ]; then
    BATCH_SIZE_FLAG="--batch-size $BATCH_SIZE"
fi
if [ -n "$VERBOSITY" ]; then
    VERBOSITY_FLAG="-$VERBOSITY"
fi
# grab the latest code
git clone https://github.com/lprekon/fekan.git
cd fekan
git checkout "$BRANCH"
git rev-parse HEAD
cargo install fekan --path . --features "serialization"

cd benches

if [ -z "$BENCHMARK" ]; then
    # error out - we need to know where out data is coming from
    echo "Error: BENCHMARK environment variable not set"
    exit 1
else
    if [ ! -f "generate_${BENCHMARK}_data.py" ]; then
        echo "Error: generate_${BENCHMARK}_data.py not found"
        exit 1
    fi
fi

LOG_FILE="${BENCHMARK}_accuracy_$BRANCH_$NAMETAG.log"
touch $LOG_FILE
# tag the log file with the git hash
git rev-parse HEAD > $LOG_FILE
# generate test data
DATA_FILE=$(mktemp)".json"
trap 'rm -f $DATA_FILE' EXIT
pip3 install -r requirements.txt
python3 generate_${BENCHMARK}_data.py 100000 > $DATA_FILE

fekan build regressor \
    --data $DATA_FILE \
    $HIDDEN_LAYER_SIZES_FLAG \
    $NUM_EPOCHS_FLAG \
    $DEGREE_FLAG \
    $COEFS_FLAG \
    $KNOT_EXTENSION_FLAG \
    $KNOT_UPDATE_INTERVAL_FLAG \
    $SYMBOLIFICATION_FLAG \
    $PRUNING_FLAG \
    $L1_LAMBDA_FLAG \
    $ENTROPY_FLAG \
    $LEARNING_RATE_FLAG \
    --validate-each-epoch \
    --log-output \
    --no-save \
    >> $LOG_FILE

if [ -n "$S3_BUCKET" ]; then
    aws s3 cp $LOG_FILE s3://$S3_BUCKET
fi
