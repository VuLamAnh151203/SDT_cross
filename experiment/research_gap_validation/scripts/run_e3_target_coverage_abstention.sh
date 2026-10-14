#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

COVERAGE_D1_DIR="${COVERAGE_D1_DIR:-$OUTPUT_DIR/D1}"
COVERAGE_OUTPUT_DIR="${COVERAGE_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/target_coverage_abstention}"
COVERAGE_SEEDS="${COVERAGE_SEEDS:-2024 2025 2026 2027 2028}"
COVERAGE_RATES="${COVERAGE_RATES:-0.01 0.02 0.05 0.10 0.20}"
COVERAGE_ALPHAS="${COVERAGE_ALPHAS:-0.25 0.5 0.75 1.0}"
COVERAGE_SELECTION_METRIC="${COVERAGE_SELECTION_METRIC:-weighted_f1}"
COVERAGE_REUSE_COMPLETED_RUN="${COVERAGE_REUSE_COMPLETED_RUN:-1}"

read -r -a SEEDS <<< "$COVERAGE_SEEDS"
read -r -a RATES <<< "$COVERAGE_RATES"
read -r -a ALPHAS <<< "$COVERAGE_ALPHAS"

[[ -s "$COVERAGE_D1_DIR/edge_impact_pairs.csv" ]] || { echo "Missing D1 dataset; run scripts/run_d1.sh first" >&2; exit 2; }
for seed in "${SEEDS[@]}"; do
  [[ -s "$COVERAGE_D1_DIR/checkpoints/three_class_seed${seed}.pt" ]] || { echo "Missing three_class_seed${seed}.pt" >&2; exit 2; }
done
if [[ "$COVERAGE_REUSE_COMPLETED_RUN" == "1" && -s "$COVERAGE_OUTPUT_DIR/manifest.json" ]]; then
  echo "Reusing completed run: $COVERAGE_OUTPUT_DIR"
  exit 0
fi

echo "Target coverage sweep: ${RATES[*]} (selection: validation/$COVERAGE_SELECTION_METRIC)"
"$PYTHON" "$SCRIPT_DIR/e3_target_coverage_abstention.py" \
  "${MODEL_ARGS[@]}" "${ATTENTION_ARGS[@]}" \
  --d1-dir "$COVERAGE_D1_DIR" --output-dir "$COVERAGE_OUTPUT_DIR" \
  --seeds "${SEEDS[@]}" --coverages "${RATES[@]}" --alphas "${ALPHAS[@]}" \
  --selection-metric "$COVERAGE_SELECTION_METRIC" --batch-size "$BATCH_SIZE" \
  --predictor-batch-size 8192 --num-workers 0 --max-batches "$MAX_BATCHES"

echo "Selected config: $COVERAGE_OUTPUT_DIR/selected_config.csv"
echo "Test summary: $COVERAGE_OUTPUT_DIR/test_summary.csv"
