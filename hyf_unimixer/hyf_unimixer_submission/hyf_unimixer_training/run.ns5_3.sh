#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# HyFormer UniMixer tuning: slightly more item-side NS capacity.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type unimixer \
    --rank_mixer_mode unimixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 3 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
