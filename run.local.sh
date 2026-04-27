#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOT'
Usage: ./run.local.sh <modelname> <gpu_id> [extra train args...]

Example:
  ./run.local.sh dcnv2 6 --num_epochs 3 --batch_size 128

Behavior:
  1. Create <modelname>/<modelname>_local from sample/ if missing
  2. Merge data/toss/tiny_train.parquet, valid.parquet, test.parquet
  3. Re-split merged toss data into train/valid/test = 8:1:1
  4. Train using <modelname>/<modelname>_local/model_training
  5. Write a run log under top-level log/
  6. Write checkpoints under top-level checkpoint/
  7. Run inference on held-out test using <modelname>/<modelname>_local/model_evaluation
  8. Print test AUC and LogLoss

Requirements:
  - data/toss/tiny_train.parquet
  - data/toss/valid.parquet
  - data/toss/test.parquet
  - data/toss/schema.json or data/schema.json
  - current Python environment must have pyarrow, torch, sklearn
EOT
}

MODEL_NAME="${1:-}"
GPU_ID="${2:-}"
if [[ -z "$MODEL_NAME" || -z "$GPU_ID" ]]; then
    usage
    exit 1
fi
shift 2 || true

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_DIR="$ROOT_DIR/sample"
MODEL_DIR="$ROOT_DIR/$MODEL_NAME"
LOCAL_DIR="$MODEL_DIR/${MODEL_NAME}_local"
LOCAL_TRAIN_DIR="$LOCAL_DIR/model_training"
LOCAL_EVAL_DIR="$LOCAL_DIR/model_evaluation"
DATA_DIR="$ROOT_DIR/data"
TOSS_DATA_DIR="$DATA_DIR/toss"
SOURCE_TINY_TRAIN_PARQUET="$TOSS_DATA_DIR/tiny_train.parquet"
SOURCE_VALID_PARQUET="$TOSS_DATA_DIR/valid.parquet"
SOURCE_TEST_PARQUET="$TOSS_DATA_DIR/test.parquet"
SOURCE_SCHEMA="$TOSS_DATA_DIR/schema.json"
if [[ ! -f "$SOURCE_SCHEMA" ]]; then
    SOURCE_SCHEMA="$DATA_DIR/schema.json"
fi

USER_NAME="$(whoami)"
RUN_STAMP="$(date '+%y%m%d_%H%M%S')"
RUN_TAG="${MODEL_NAME}_${USER_NAME}_gpu${GPU_ID}_${RUN_STAMP}"

TOP_LOG_DIR="$ROOT_DIR/log"
TOP_CKPT_DIR="$ROOT_DIR/checkpoint"
TOP_RUN_DIR="$ROOT_DIR/local_runs"
LEGACY_CKPT_DIR="$ROOT_DIR/chekpoint"
if [[ -d "$LEGACY_CKPT_DIR" && ! -d "$TOP_CKPT_DIR" ]]; then
    TOP_CKPT_DIR="$LEGACY_CKPT_DIR"
fi

RUN_LOG_FILE="$TOP_LOG_DIR/${RUN_TAG}.log"
RUN_CKPT_DIR="$TOP_CKPT_DIR/${RUN_TAG}"
RUN_DATA_DIR="$TOP_RUN_DIR/${RUN_TAG}"

for required_file in "$SOURCE_TINY_TRAIN_PARQUET" "$SOURCE_VALID_PARQUET" "$SOURCE_TEST_PARQUET"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing parquet file: $required_file" >&2
        exit 1
    fi
done

if [[ ! -f "$SOURCE_SCHEMA" ]]; then
    echo "Missing schema file: $SOURCE_SCHEMA" >&2
    echo "Place schema.json under KDD_submission/data/toss/ or KDD_submission/data/ before running local validation." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR" "$TOP_LOG_DIR" "$TOP_CKPT_DIR" "$TOP_RUN_DIR"

if [[ ! -d "$LOCAL_TRAIN_DIR" ]]; then
    mkdir -p "$LOCAL_DIR"
    cp -R "$SAMPLE_DIR/model_training" "$LOCAL_TRAIN_DIR"
fi

if [[ ! -d "$LOCAL_EVAL_DIR" ]]; then
    mkdir -p "$LOCAL_DIR"
    cp -R "$SAMPLE_DIR/model_evaluation" "$LOCAL_EVAL_DIR"
fi

TRAIN_DIR="$RUN_DATA_DIR/data_train"
VALID_DIR="$RUN_DATA_DIR/data_valid"
TEST_DIR="$RUN_DATA_DIR/data_test"
TF_EVENTS_DIR="$RUN_DATA_DIR/tf_events"
RESULT_DIR="$RUN_DATA_DIR/eval_results"

rm -rf "$RUN_DATA_DIR"
mkdir -p "$TRAIN_DIR" "$VALID_DIR" "$TEST_DIR" "$RUN_CKPT_DIR" "$TF_EVENTS_DIR" "$RESULT_DIR"

cp "$SOURCE_SCHEMA" "$TRAIN_DIR/schema.json"
cp "$SOURCE_SCHEMA" "$VALID_DIR/schema.json"
cp "$SOURCE_SCHEMA" "$TEST_DIR/schema.json"

echo "Preparing merged toss local split from:"
echo "  $SOURCE_TINY_TRAIN_PARQUET"
echo "  $SOURCE_VALID_PARQUET"
echo "  $SOURCE_TEST_PARQUET"
SPLIT_SCRIPT="$RUN_DATA_DIR/build_split.py"
cat > "$SPLIT_SCRIPT" <<PY
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

paths = [
    Path(r"$SOURCE_TINY_TRAIN_PARQUET"),
    Path(r"$SOURCE_VALID_PARQUET"),
    Path(r"$SOURCE_TEST_PARQUET"),
]
train_dir = Path(r"$TRAIN_DIR")
valid_dir = Path(r"$VALID_DIR")
test_dir = Path(r"$TEST_DIR")

tables = [pq.read_table(p) for p in paths]
table = pa.concat_tables(tables, promote=True)
num_rows = table.num_rows
train_rows = int(num_rows * 0.8)
valid_rows = int(num_rows * 0.1)
test_rows = num_rows - train_rows - valid_rows

train = table.slice(0, train_rows)
valid = table.slice(train_rows, valid_rows)
test = table.slice(train_rows + valid_rows, test_rows)

pq.write_table(train, train_dir / 'train_0000.parquet', row_group_size=1024)
pq.write_table(valid, valid_dir / 'valid_0000.parquet', row_group_size=1024)
pq.write_table(test, test_dir / 'test_0000.parquet', row_group_size=1024)

print(f'Prepared merged toss split rows={num_rows}')
print(f'  train={train.num_rows}, valid={valid.num_rows}, test={test.num_rows}')
PY
conda run -n fuxictr python "$SPLIT_SCRIPT"

echo "Running local training for model=$MODEL_NAME on gpu=$GPU_ID"
(
    export TRAIN_DATA_PATH="$TRAIN_DIR"
    export TRAIN_CKPT_PATH="$RUN_CKPT_DIR"
    export TRAIN_LOG_PATH="$TOP_LOG_DIR"
    export TRAIN_TF_EVENTS_PATH="$TF_EVENTS_DIR"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    cd "$LOCAL_TRAIN_DIR"
    bash ./run.sh --device cuda --valid_ratio 0.0 "$@"
) 2>&1 | tee "$RUN_LOG_FILE"

BEST_CKPT_DIR="$(find "$RUN_CKPT_DIR" -maxdepth 1 -type d -name '*.best_model' | sort | tail -n 1)"
if [[ -z "$BEST_CKPT_DIR" ]]; then
    echo "Failed to locate best checkpoint under $RUN_CKPT_DIR" >&2
    exit 1
fi

echo "Using checkpoint: $BEST_CKPT_DIR" | tee -a "$RUN_LOG_FILE"
(
    export MODEL_OUTPUT_PATH="$BEST_CKPT_DIR"
    export EVAL_DATA_PATH="$TEST_DIR"
    export EVAL_RESULT_PATH="$RESULT_DIR"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    cd "$LOCAL_EVAL_DIR"
    python infer.py
) 2>&1 | tee -a "$RUN_LOG_FILE"

echo "Computing held-out test metrics" | tee -a "$RUN_LOG_FILE"
python <<PY | tee -a "$RUN_LOG_FILE"
import json
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

result_path = Path(r"$RESULT_DIR") / 'predictions.json'
test_parquet = Path(r"$TEST_DIR") / 'test_0000.parquet'

with result_path.open('r', encoding='utf-8') as f:
    predictions = json.load(f)['predictions']

table = pq.read_table(test_parquet, columns=['user_id', 'label_type'])
user_ids = table.column('user_id').to_pylist()
raw_labels = table.column('label_type').to_pylist()
labels = [1 if value == 2 else 0 for value in raw_labels]

y_true = []
y_prob = []
for user_id, label in zip(user_ids, labels):
    pred = predictions.get(str(user_id))
    if pred is None:
        pred = predictions.get(user_id)
    if pred is None:
        raise KeyError(f'Missing prediction for user_id={user_id}')
    y_true.append(label)
    y_prob.append(float(pred))

valid_pairs = [(yt, yp) for yt, yp in zip(y_true, y_prob) if yp == yp]
n_nan = len(y_prob) - len(valid_pairs)
if n_nan > 0:
    print(f'Filtered {n_nan}/{len(y_prob)} NaN predictions before metric computation')

if not valid_pairs:
    auc = 0.0
    ll = float('inf')
else:
    y_true = [yt for yt, _ in valid_pairs]
    y_prob = [yp for _, yp in valid_pairs]
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    ll = log_loss(y_true, y_prob, labels=[0, 1])

print(f'Local test AUC: {auc:.6f}')
print(f'Local test LogLoss: {ll:.6f}')
PY

echo "Run log: $RUN_LOG_FILE"
echo "Checkpoint root: $RUN_CKPT_DIR"
