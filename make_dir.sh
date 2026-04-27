#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOT'
Usage: ./make_dir.sh <model>

Creates:
  <model>/<model>_local/model_training
  <model>/<model>_local/model_evaluation
  <model>/<model>_local/model_submission/model_training
  <model>/<model>_local/model_submission/model_evaluation
EOT
}

MODEL_NAME="${1:-}"
if [[ -z "$MODEL_NAME" ]]; then
    usage
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$ROOT_DIR/$MODEL_NAME"

mkdir -p \
    "$BASE_DIR/${MODEL_NAME}_local/model_training" \
    "$BASE_DIR/${MODEL_NAME}_local/model_evaluation" \
    "$BASE_DIR/${MODEL_NAME}_submission/${MODEL_NAME}_training" \
    "$BASE_DIR/${MODEL_NAME}_submission/${MODEL_NAME}_evaluation"

echo "Created directories for model=$MODEL_NAME"
echo "  $BASE_DIR/model_training"
echo "  $BASE_DIR/model_evaluation"
echo "  $BASE_DIR/model_submission/model_training"
echo "  $BASE_DIR/model_submission/model_evaluation"
