#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOT'
Usage: ./run.submit.sh <modelname> [num_epochs]

Creates:
  <modelname>/<modelname>_submission/<modelname>_training
  <modelname>/<modelname>_submission/<modelname>_evaluation

The contents are copied from sample/ or <modelname>/<modelname>_local and follow
the competition file contract:
  - training: 7 files
  - evaluation: 3 files

If num_epochs is provided, the copied training train.py will be rewritten so
its --num_epochs default uses that value.
EOT
}

MODEL_NAME="${1:-}"
NUM_EPOCHS="${2:-}"
if [[ -z "$MODEL_NAME" ]]; then
    usage
    exit 1
fi
if [[ -n "$NUM_EPOCHS" && ! "$NUM_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "num_epochs must be an integer: $NUM_EPOCHS" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_DIR="$ROOT_DIR/sample"
MODEL_DIR="$ROOT_DIR/$MODEL_NAME"
LOCAL_DIR="$MODEL_DIR/${MODEL_NAME}_local"
LOCAL_TRAIN_DIR="$LOCAL_DIR/model_training"
LOCAL_EVAL_DIR="$LOCAL_DIR/model_evaluation"
SUBMISSION_DIR="$MODEL_DIR/${MODEL_NAME}_submission"
TRAINING_DIR="$SUBMISSION_DIR/${MODEL_NAME}_training"
EVALUATION_DIR="$SUBMISSION_DIR/${MODEL_NAME}_evaluation"

TRAIN_SRC_DIR="$SAMPLE_DIR/model_training"
EVAL_SRC_DIR="$SAMPLE_DIR/model_evaluation"
if [[ -d "$LOCAL_TRAIN_DIR" ]]; then
    TRAIN_SRC_DIR="$LOCAL_TRAIN_DIR"
fi
if [[ -d "$LOCAL_EVAL_DIR" ]]; then
    EVAL_SRC_DIR="$LOCAL_EVAL_DIR"
fi

mkdir -p "$TRAINING_DIR" "$EVALUATION_DIR"

cp -f "$TRAIN_SRC_DIR/dataset.py" "$TRAINING_DIR/"
cp -f "$TRAIN_SRC_DIR/model.py" "$TRAINING_DIR/"
cp -f "$TRAIN_SRC_DIR/ns_groups.json" "$TRAINING_DIR/"
cp -f "$TRAIN_SRC_DIR/run.sh" "$TRAINING_DIR/"
cp -f "$TRAIN_SRC_DIR/train.py" "$TRAINING_DIR/"
cp -f "$TRAIN_SRC_DIR/trainer.py" "$TRAINING_DIR/"
cp -f "$TRAIN_SRC_DIR/utils.py" "$TRAINING_DIR/"

cp -f "$EVAL_SRC_DIR/dataset.py" "$EVALUATION_DIR/"
cp -f "$EVAL_SRC_DIR/infer.py" "$EVALUATION_DIR/"
cp -f "$EVAL_SRC_DIR/model.py" "$EVALUATION_DIR/"

if [[ -n "$NUM_EPOCHS" ]]; then
    python - <<PY
from pathlib import Path
import re

path = Path(r"$TRAINING_DIR/train.py")
text = path.read_text()
pattern = r"(parser\.add_argument\('--num_epochs',\s*type=int,\s*default=)(\d+)"
new_text, count = re.subn(pattern, rf"\\g<1>{int('$NUM_EPOCHS')}", text, count=1)
if count != 1:
    raise SystemExit(f"Failed to update --num_epochs default in {path}")
path.write_text(new_text)
PY
fi

echo "Created submission package for model=$MODEL_NAME"
echo "Training dir: $TRAINING_DIR"
echo "Evaluation dir: $EVALUATION_DIR"
if [[ -n "$NUM_EPOCHS" ]]; then
    echo "Applied train.py default num_epochs=$NUM_EPOCHS"
fi
