#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

TWO_STAGE_D1_DIR="${TWO_STAGE_D1_DIR:-$OUTPUT_DIR/D1}"
TWO_STAGE_OUTPUT_DIR="${TWO_STAGE_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/two_stage_target_gate}"
TWO_STAGE_SEEDS="${TWO_STAGE_SEEDS:-2024 2025 2026 2027 2028}"
TWO_STAGE_COVERAGES="${TWO_STAGE_COVERAGES:-0.01 0.02 0.05 0.10 0.20}"
TWO_STAGE_ALPHAS="${TWO_STAGE_ALPHAS:-0.25 0.5 0.75 1.0}"
TWO_STAGE_TARGET_EPS="${TWO_STAGE_TARGET_EPS:-0.05}"
read -r -a SEEDS <<< "$TWO_STAGE_SEEDS"
read -r -a COVERAGES <<< "$TWO_STAGE_COVERAGES"
read -r -a ALPHAS <<< "$TWO_STAGE_ALPHAS"

"$PYTHON" "$SCRIPT_DIR/e3_target_coverage_abstention.py" \
  "${MODEL_ARGS[@]}" "${ATTENTION_ARGS[@]}" --target-model \
  --d1-dir "$TWO_STAGE_D1_DIR" --output-dir "$TWO_STAGE_OUTPUT_DIR" \
  --seeds "${SEEDS[@]}" --coverages "${COVERAGES[@]}" --alphas "${ALPHAS[@]}" \
  --target-improvement-eps "$TWO_STAGE_TARGET_EPS" \
  --selection-metric weighted_f1 --batch-size "$BATCH_SIZE" \
  --predictor-batch-size 8192 --num-workers 0 --max-batches "$MAX_BATCHES"

echo "Selected: $TWO_STAGE_OUTPUT_DIR/selected_config.csv"
echo "Test: $TWO_STAGE_OUTPUT_DIR/test_summary.csv"
