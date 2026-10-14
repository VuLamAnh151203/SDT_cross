#!/usr/bin/env bash
# Shared configuration for the four independent entry points.
# Edit the defaults here once; environment variables can still override them.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SDT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-C:/path/to/fully_trained/best_checkpoint.pt}"
FEATURE_PATH="${FEATURE_PATH:-$SDT_ROOT/../SDT/data/iemocap_multimodal_features.pkl}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/outputs/full_run}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
NUM_THREADS="${NUM_THREADS:-0}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-0}"
ALLOW_SMOKE_CHECKPOINT="${ALLOW_SMOKE_CHECKPOINT:-0}"
PROB_ATOL="${PROB_ATOL:-2e-4}"
PROB_RTOL="${PROB_RTOL:-2e-4}"

if [[ "$CHECKPOINT" == "C:/path/to/fully_trained/best_checkpoint.pt" ]]; then
  echo "ERROR: Edit CHECKPOINT in $SCRIPT_DIR/experiment_config.sh to the real fully trained checkpoint." >&2
  exit 2
fi

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --feature-path "$FEATURE_PATH"
  --run-dir "$RUN_DIR"
  --split "$SPLIT"
  --device "$DEVICE"
  --gpu-id "$GPU_ID"
  --seed "$SEED"
  --num-threads "$NUM_THREADS"
  --max-conversations "$MAX_CONVERSATIONS"
  --prob-atol "$PROB_ATOL"
  --prob-rtol "$PROB_RTOL"
)

if [[ "$ALLOW_SMOKE_CHECKPOINT" == "1" ]]; then
  COMMON_ARGS+=(--allow-smoke-checkpoint)
fi

