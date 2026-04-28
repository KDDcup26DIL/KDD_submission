#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_OR_DIR [--schema PATH] [--output-dir DIR] [--prefix NAME] [--no-png]" >&2
  echo "Examples:" >&2
  echo "  $0 Wukong" >&2
  echo "  $0 dcnv2" >&2
  echo "  $0 checkpoint/Wukong_hun_gpu0_260427_171827/global_step860.layer=3.head=4.hidden=64.best_model" >&2
  exit 2
fi

MODEL_OR_DIR="$1"
shift

python3 -u "${SCRIPT_DIR}/tools/visualize_architecture.py" "${MODEL_OR_DIR}" "$@"
