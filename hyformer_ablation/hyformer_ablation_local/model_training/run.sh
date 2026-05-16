#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

BASE_TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?TRAIN_DATA_PATH is required}"
BASE_CKPT_PATH="${TRAIN_CKPT_PATH:?TRAIN_CKPT_PATH is required}"
BASE_LOG_PATH="${TRAIN_LOG_PATH:?TRAIN_LOG_PATH is required}"
BASE_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:?TRAIN_TF_EVENTS_PATH is required}"

COMMON_ARGS=(
  --ns_tokenizer_type rankmixer
  --user_ns_tokens 5
  --item_ns_tokens 2
  --num_queries 2
  --ns_groups_json ""
  --emb_skip_threshold 1000000
  --num_workers 8
)

USER_ARGS=("$@")

# Keep the default suite broad but still tractable for a 10-epoch local run.
# Add more variants by extending the ABLATION_VARIANTS env var with names below.
DEFAULT_VARIANTS=(
  baseline
  no_time
  rank_ffn_only
  rank_none
  query_ns_only
  query_seq_only
  query_zero_seq
  seq_swiglu
  seq_longer
  num_queries_1
  rope
  no_sparse_reinit
)

if [[ -n "${ABLATION_VARIANTS:-}" ]]; then
  # shellcheck disable=SC2206
  VARIANTS=(${ABLATION_VARIANTS})
else
  VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

variant_args() {
  case "$1" in
    baseline)
      ;;
    no_time)
      echo --no_time_buckets
      ;;
    rank_ffn_only)
      echo --rank_mixer_mode ffn_only
      ;;
    rank_none)
      echo --rank_mixer_mode none
      ;;
    query_ns_only)
      echo --query_context_mode ns_only
      ;;
    query_seq_only)
      echo --query_context_mode seq_only
      ;;
    query_zero_seq)
      echo --query_context_mode ns_zero_seq
      ;;
    seq_swiglu)
      echo --seq_encoder_type swiglu
      ;;
    seq_longer)
      echo --seq_encoder_type longer --seq_top_k 50
      ;;
    num_queries_1)
      echo --num_queries 1 --rank_mixer_mode ffn_only
      ;;
    rope)
      echo --use_rope
      ;;
    no_sparse_reinit)
      echo --reinit_sparse_after_epoch 999999
      ;;
    *)
      echo "Unknown ablation variant: $1" >&2
      return 1
      ;;
  esac
}

rm -rf "${BASE_CKPT_PATH:?}/"*
mkdir -p "$BASE_CKPT_PATH" "$BASE_LOG_PATH" "$BASE_TF_EVENTS_PATH"

echo "HyFormer ablation suite:"
printf '  %s\n' "${VARIANTS[@]}"

for variant in "${VARIANTS[@]}"; do
  echo "========== Ablation variant: ${variant} =========="
  VARIANT_ROOT="$BASE_CKPT_PATH/$variant"
  VARIANT_CKPT="$VARIANT_ROOT/ckpt"
  VARIANT_LOG="$VARIANT_ROOT/logs"
  VARIANT_TF="$VARIANT_ROOT/tf_events"
  mkdir -p "$VARIANT_CKPT" "$VARIANT_LOG" "$VARIANT_TF"

  export TRAIN_DATA_PATH="$BASE_TRAIN_DATA_PATH"
  export TRAIN_CKPT_PATH="$VARIANT_CKPT"
  export TRAIN_LOG_PATH="$VARIANT_LOG"
  export TRAIN_TF_EVENTS_PATH="$VARIANT_TF"

  # shellcheck disable=SC2207
  EXTRA_ARGS=($(variant_args "$variant"))
  python3 -u "$SCRIPT_DIR/train.py" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${USER_ARGS[@]}"
done

RESULT_DIR="$BASE_CKPT_PATH/ablation_results"
python3 "$SCRIPT_DIR/analyze_ablation.py" --root "$BASE_CKPT_PATH" --out-dir "$RESULT_DIR"

BEST_VARIANT="$(python3 - "$RESULT_DIR/ablation_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best_variant"])
PY
)"
BEST_DIR="$(find "$BASE_CKPT_PATH/$BEST_VARIANT/ckpt" -maxdepth 1 -type d -name '*.best_model' | sort | tail -n 1)"
if [[ -z "$BEST_DIR" ]]; then
  echo "Could not find best checkpoint for variant=$BEST_VARIANT" >&2
  exit 1
fi

TOP_BEST_DIR="$BASE_CKPT_PATH/ablation_best.${BEST_VARIANT}.best_model"
rm -rf "$TOP_BEST_DIR"
cp -R "$BEST_DIR" "$TOP_BEST_DIR"

echo "Ablation result CSV: $RESULT_DIR/ablation_results.csv"
echo "Ablation result SVG: $RESULT_DIR/ablation_auc.svg"
if [[ -f "$RESULT_DIR/ablation_auc.png" ]]; then
  echo "Ablation result PNG: $RESULT_DIR/ablation_auc.png"
fi
echo "Copied best variant checkpoint for run.local.sh evaluation:"
echo "  $TOP_BEST_DIR"
