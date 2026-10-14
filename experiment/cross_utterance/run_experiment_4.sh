#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${RUN_DIR:-}}"
SEED="${SEED:-2024}"

if [[ -z "$TRAIN_RUN_DIR" || -z "$EVAL_RUN_DIR" ]]; then
  echo "ERROR: Set TRAIN_RUN_DIR and EVAL_RUN_DIR (or RUN_DIR) first." >&2
  exit 2
fi

exec "$PYTHON" "$SCRIPT_DIR/train_harm_predictor.py" \
  --train-run-dir "$TRAIN_RUN_DIR" \
  --eval-run-dir "$EVAL_RUN_DIR" \
  --seed "$SEED" \
  "$@"
