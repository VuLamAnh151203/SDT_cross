#!/usr/bin/env bash
# ETCR-specific analysis settings. Model/data settings come from the parent config.

set -euo pipefail

ETCR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ETCR_SCRIPT_DIR/../experiment_config.sh"

IMPACT_EPS="${IMPACT_EPS:-0.0001}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-2000}"
EMBEDDING_COMPONENTS="${EMBEDDING_COMPONENTS:-32}"
ALLOW_MISSING_BASELINE="${ALLOW_MISSING_BASELINE:-0}"

ETCR_GENERATOR_ARGS=(
  --checkpoint "$CHECKPOINT"
  --feature-path "$FEATURE_PATH"
  --run-dir "$RUN_DIR"
  --device "$DEVICE"
  --gpu-id "$GPU_ID"
  --seed "$SEED"
  --num-threads "$NUM_THREADS"
  --max-conversations "$MAX_CONVERSATIONS"
  --prob-atol "$PROB_ATOL"
  --prob-rtol "$PROB_RTOL"
)

if [[ "$ALLOW_SMOKE_CHECKPOINT" == "1" ]]; then
  ETCR_GENERATOR_ARGS+=(--allow-smoke-checkpoint)
fi
if [[ "$ALLOW_MISSING_BASELINE" == "1" ]]; then
  ETCR_GENERATOR_ARGS+=(--allow-missing-baseline)
fi

