#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

exec "${PYTHON:-python}" -u "$SCRIPT_DIR/train.py" \
  --selection-protocol validation \
  --Dataset MELD --fusion-variant sdt \
  --epochs 50 --batch-size 8 --hidden-dim 1024 --n-head 8 \
  --lr 0.000005 --l2 0.00001 --dropout 0.5 --temp 8 \
  --gamma-1 1 --gamma-2 1 --gamma-3 1 \
  --seed 2024 "$@"
