#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Confirm the best no-top-K configurations from the single-seed diagnostic:
#   B3: text-only baseline, margin=0, K=0
#   B4: text-only, impact_min_margin=0.005, K=0
#
# The underlying strong-impact runner trains B4, locates or trains the matching
# B3 baseline, selects on validation, evaluates the selected configurations on
# test, and writes paired B4-B3 deltas.
CONFIRM_OUTPUT_DIR="${CONFIRM_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/no_topk_confirmation}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-2024 2025 2026 2027 2028}"
CONFIRM_EPOCHS="${CONFIRM_EPOCHS:-30}"
CONFIRM_PATIENCE="${CONFIRM_PATIENCE:-7}"
CONFIRM_GATE_LR="${CONFIRM_GATE_LR:-0.0003}"
CONFIRM_REUSE_COMPLETED_RUNS="${CONFIRM_REUSE_COMPLETED_RUNS:-1}"

echo "Five-seed no-top-K confirmation in SDT_cross"
echo "Seeds: $CONFIRM_SEEDS"
echo "B3: text-only baseline, margin=0, K=0"
echo "B4: text-only, margin=0.005, K=0"
echo "Selection: validation/weighted_f1"

STRONG_TOPK_OUTPUT_DIR="$CONFIRM_OUTPUT_DIR" \
STRONG_SEEDS="$CONFIRM_SEEDS" \
STRONG_MIN_MARGINS="0.005" \
STRONG_INITIAL_GATE="0.90" \
STRONG_LAMBDA_IMPACT="0.10" \
STRONG_EPOCHS="$CONFIRM_EPOCHS" \
STRONG_PATIENCE="$CONFIRM_PATIENCE" \
STRONG_GATE_LR="$CONFIRM_GATE_LR" \
STRONG_SELECTION_METRIC="weighted_f1" \
STRONG_TOP_K_VALUES="0" \
STRONG_REUSE_COMPLETED_RUNS="$CONFIRM_REUSE_COMPLETED_RUNS" \
bash "$SCRIPT_DIR/run_e3_strong_topk_diagnostic.sh"

echo "No-top-K confirmation complete: $CONFIRM_OUTPUT_DIR/selection"
echo "Validation summary: $CONFIRM_OUTPUT_DIR/selection/topk_validation_summary.csv"
echo "Test summary: $CONFIRM_OUTPUT_DIR/selection/topk_selected_test_summary.csv"
echo "Paired deltas: $CONFIRM_OUTPUT_DIR/selection/topk_selected_paired_deltas.csv"
