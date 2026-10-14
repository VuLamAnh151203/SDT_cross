#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

ABSTENTION_D1_DIR="${ABSTENTION_D1_DIR:-$OUTPUT_DIR/D1}"
ABSTENTION_OUTPUT_DIR="${ABSTENTION_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/three_class_abstention}"
ABSTENTION_SEEDS="${ABSTENTION_SEEDS:-2024 2025 2026 2027 2028}"
ABSTENTION_PROBABILITY_THRESHOLDS="${ABSTENTION_PROBABILITY_THRESHOLDS:-0.5 0.6 0.7 0.8 0.9}"
ABSTENTION_MARGIN_THRESHOLDS="${ABSTENTION_MARGIN_THRESHOLDS:-0 0.1 0.2 0.3}"
ABSTENTION_ALPHAS="${ABSTENTION_ALPHAS:-0.25 0.5 0.75 1.0}"
ABSTENTION_SELECTION_METRIC="${ABSTENTION_SELECTION_METRIC:-weighted_f1}"
ABSTENTION_MODALITY="${ABSTENTION_MODALITY:-text}"
ABSTENTION_REUSE_COMPLETED_RUN="${ABSTENTION_REUSE_COMPLETED_RUN:-1}"

if [[ ! -s "$ABSTENTION_D1_DIR/edge_impact_pairs.csv" ]]; then
  echo "ERROR: missing D1 dataset: $ABSTENTION_D1_DIR/edge_impact_pairs.csv" >&2
  echo "Run scripts/run_d1.sh first." >&2
  exit 2
fi

read -r -a SEED_ARRAY <<< "$ABSTENTION_SEEDS"
read -r -a PROBABILITY_ARRAY <<< "$ABSTENTION_PROBABILITY_THRESHOLDS"
read -r -a MARGIN_ARRAY <<< "$ABSTENTION_MARGIN_THRESHOLDS"
read -r -a ALPHA_ARRAY <<< "$ABSTENTION_ALPHAS"

for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="$ABSTENTION_D1_DIR/checkpoints/three_class_seed${seed}.pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: missing D1 three-class checkpoint: $checkpoint" >&2
    echo "Run D1 with the same seeds before this diagnostic." >&2
    exit 2
  fi
done

echo "Three-class selective gating with abstention"
echo "D1 directory: $ABSTENTION_D1_DIR"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Probability thresholds: ${PROBABILITY_ARRAY[*]}"
echo "Harm margins: ${MARGIN_ARRAY[*]}"
echo "Suppression alphas: ${ALPHA_ARRAY[*]}"
echo "Selection: validation/$ABSTENTION_SELECTION_METRIC"
echo "Test is evaluated only after validation selection."

if [[ "$ABSTENTION_REUSE_COMPLETED_RUN" == "1" \
      && -s "$ABSTENTION_OUTPUT_DIR/manifest.json" ]]; then
  echo "Reusing completed abstention diagnostic: $ABSTENTION_OUTPUT_DIR"
  exit 0
fi

"$PYTHON" "$SCRIPT_DIR/e3_three_class_abstention.py" \
  "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --d1-dir "$ABSTENTION_D1_DIR" \
  --output-dir "$ABSTENTION_OUTPUT_DIR" \
  --seeds "${SEED_ARRAY[@]}" \
  --modality "$ABSTENTION_MODALITY" \
  --probability-thresholds "${PROBABILITY_ARRAY[@]}" \
  --margin-thresholds "${MARGIN_ARRAY[@]}" \
  --alphas "${ALPHA_ARRAY[@]}" \
  --selection-metric "$ABSTENTION_SELECTION_METRIC" \
  --batch-size "$BATCH_SIZE" \
  --predictor-batch-size 8192 \
  --num-workers 0 \
  --max-batches "$MAX_BATCHES"

echo "Three-class abstention complete: $ABSTENTION_OUTPUT_DIR"
echo "Selected configuration: $ABSTENTION_OUTPUT_DIR/selected_config.csv"
echo "Test summary: $ABSTENTION_OUTPUT_DIR/test_summary.csv"
echo "Paired deltas: $ABSTENTION_OUTPUT_DIR/paired_deltas.csv"
