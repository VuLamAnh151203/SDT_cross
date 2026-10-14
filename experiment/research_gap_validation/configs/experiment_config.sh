#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$CONFIG_DIR/.." && pwd)"
SDT_ROOT="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-C:/path/to/fully_trained/best_checkpoint.pt}"
FEATURE_PATH="${FEATURE_PATH:-$SDT_ROOT/../SDT/data/iemocap_multimodal_features.pkl}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$SDT_ROOT/experiment/cross_utterance/outputs/full_run}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/outputs/full_run}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
SEEDS="${SEEDS:-2024 2025 2026}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-2000}"
IMPACT_EPS="${IMPACT_EPS:-0.0001}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-64}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RUN_DONOR="${RUN_DONOR:-1}"
RUN_FULL_E3="${RUN_FULL_E3:-0}"
ALLOW_SMOKE_CHECKPOINT="${ALLOW_SMOKE_CHECKPOINT:-0}"
RENORMALIZE_ATTENTION="${RENORMALIZE_ATTENTION:-0}"

if [[ "$CHECKPOINT" == "C:/path/to/fully_trained/best_checkpoint.pt" ]]; then
  echo "ERROR: set CHECKPOINT in configs/experiment_config.sh or environment." >&2
  exit 2
fi

MODEL_ARGS=(
  --checkpoint "$CHECKPOINT" --feature-path "$FEATURE_PATH"
  --device "$DEVICE" --gpu-id "$GPU_ID"
)
if [[ "$ALLOW_SMOKE_CHECKPOINT" == "1" ]]; then
  MODEL_ARGS+=(--allow-smoke-checkpoint)
fi

ATTENTION_ARGS=()
if [[ "$RENORMALIZE_ATTENTION" == "1" ]]; then
  ATTENTION_ARGS+=(--renormalize-attention)
fi

