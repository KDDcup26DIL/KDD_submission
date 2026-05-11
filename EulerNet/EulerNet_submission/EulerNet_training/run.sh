#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --embedding_dim 16 \
    --shape 52,52,52,52,52 \
    --net_ex_dropout 0.1 \
    --net_im_dropout 0.1 \
    --num_workers 8 \
    "$@"