#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

STRONG_TOPK_OUTPUT_DIR="${STRONG_TOPK_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/strong_impact_topk}"
STRONG_SEEDS="${STRONG_SEEDS:-2024 2025 2026 2027 2028}"
STRONG_MIN_MARGINS="${STRONG_MIN_MARGINS:-0 0.001 0.005 0.01}"
STRONG_INITIAL_GATE="${STRONG_INITIAL_GATE:-0.90}"
STRONG_LAMBDA_IMPACT="${STRONG_LAMBDA_IMPACT:-0.10}"
STRONG_EPOCHS="${STRONG_EPOCHS:-30}"
STRONG_PATIENCE="${STRONG_PATIENCE:-7}"
STRONG_GATE_LR="${STRONG_GATE_LR:-0.0003}"
STRONG_SELECTION_METRIC="${STRONG_SELECTION_METRIC:-weighted_f1}"
STRONG_TOP_K_VALUES="${STRONG_TOP_K_VALUES:-0 1 3 5}"
STRONG_REUSE_COMPLETED_RUNS="${STRONG_REUSE_COMPLETED_RUNS:-1}"
STRONG_B3_CHECKPOINT_DIR="${STRONG_B3_CHECKPOINT_DIR:-$OUTPUT_DIR/diagnostics/five_diagnostics/05_text_only/E3/checkpoints}"

mkdir -p "$STRONG_TOPK_OUTPUT_DIR"
read -r -a SEED_ARRAY <<< "$STRONG_SEEDS"
read -r -a MARGIN_ARRAY <<< "$STRONG_MIN_MARGINS"
read -r -a TOP_K_ARRAY <<< "$STRONG_TOP_K_VALUES"

echo "[strong 1/3] Training text-only B4 with strong-impact thresholds"
B4_CHECKPOINTS=()
for min_margin in "${MARGIN_ARRAY[@]}"; do
  run_dir="$STRONG_TOPK_OUTPUT_DIR/margin_${min_margin}"
  if [[ "$STRONG_REUSE_COMPLETED_RUNS" == "1" \
        && -s "$run_dir/E3/manifest.json" ]]; then
    echo "Reusing completed strong-impact run: margin=$min_margin"
  else
    DIAGNOSTIC_OUTPUT_DIR="$run_dir" \
    DIAGNOSTIC_MODELS="B4" \
    DIAGNOSTIC_SEEDS="$STRONG_SEEDS" \
    DIAGNOSTIC_EPOCHS="$STRONG_EPOCHS" \
    DIAGNOSTIC_PATIENCE="$STRONG_PATIENCE" \
    DIAGNOSTIC_GATE_LR="$STRONG_GATE_LR" \
    DIAGNOSTIC_SELECTION_SPLIT="valid" \
    DIAGNOSTIC_SELECTION_METRIC="$STRONG_SELECTION_METRIC" \
    DIAGNOSTIC_INITIAL_GATE="$STRONG_INITIAL_GATE" \
    DIAGNOSTIC_LAMBDA_IMPACT="$STRONG_LAMBDA_IMPACT" \
    DIAGNOSTIC_IMPACT_MIN_MARGIN="$min_margin" \
    DIAGNOSTIC_GATE_MODALITIES="text" \
    DIAGNOSTIC_DEFER_TEST_EVALUATION="1" \
    bash "$SCRIPT_DIR/run_e3_gate_only_diagnostic.sh"
  fi
  for seed in "${SEED_ARRAY[@]}"; do
    checkpoint="$run_dir/E3/checkpoints/gate_only_diagnostic_B4_seed${seed}.pt"
    if [[ ! -s "$checkpoint" ]]; then
      echo "ERROR: missing strong-impact checkpoint: $checkpoint" >&2
      exit 2
    fi
    B4_CHECKPOINTS+=("$checkpoint")
  done
done

echo "[strong 2/3] Locating or training the text-only B3 baseline"
B3_CHECKPOINTS=()
missing_baseline=0
for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="$STRONG_B3_CHECKPOINT_DIR/gate_only_diagnostic_B3_seed${seed}.pt"
  if [[ -s "$checkpoint" ]]; then
    B3_CHECKPOINTS+=("$checkpoint")
  else
    missing_baseline=1
  fi
done
if [[ "$missing_baseline" == "1" ]]; then
  baseline_dir="$STRONG_TOPK_OUTPUT_DIR/b3_text_baseline"
  echo "Text-only B3 checkpoints were not complete; training baseline once."
  DIAGNOSTIC_OUTPUT_DIR="$baseline_dir" \
  DIAGNOSTIC_MODELS="B3" \
  DIAGNOSTIC_SEEDS="$STRONG_SEEDS" \
  DIAGNOSTIC_EPOCHS="$STRONG_EPOCHS" \
  DIAGNOSTIC_PATIENCE="$STRONG_PATIENCE" \
  DIAGNOSTIC_GATE_LR="$STRONG_GATE_LR" \
  DIAGNOSTIC_SELECTION_SPLIT="valid" \
  DIAGNOSTIC_SELECTION_METRIC="$STRONG_SELECTION_METRIC" \
  DIAGNOSTIC_INITIAL_GATE="0.99" \
  DIAGNOSTIC_GATE_MODALITIES="text" \
  DIAGNOSTIC_DEFER_TEST_EVALUATION="1" \
  bash "$SCRIPT_DIR/run_e3_gate_only_diagnostic.sh"
  B3_CHECKPOINTS=()
  for seed in "${SEED_ARRAY[@]}"; do
    checkpoint="$baseline_dir/E3/checkpoints/gate_only_diagnostic_B3_seed${seed}.pt"
    if [[ ! -s "$checkpoint" ]]; then
      echo "ERROR: missing B3 baseline checkpoint: $checkpoint" >&2
      exit 2
    fi
    B3_CHECKPOINTS+=("$checkpoint")
  done
fi

echo "[strong 3/3] Selecting margin and conservative top-K on validation"
selection_dir="$STRONG_TOPK_OUTPUT_DIR/selection"
if [[ "$STRONG_REUSE_COMPLETED_RUNS" == "1" \
      && -s "$selection_dir/topk_manifest.json" ]]; then
  echo "Reusing completed strong-impact/top-K selection"
else
  "$PYTHON" "$SCRIPT_DIR/e3_select_conservative_topk.py" \
    "${MODEL_ARGS[@]}" \
    --trained-checkpoints "${B3_CHECKPOINTS[@]}" "${B4_CHECKPOINTS[@]}" \
    --output-dir "$selection_dir" \
    --top-k "${TOP_K_ARRAY[@]}" \
    --selection-metric "$STRONG_SELECTION_METRIC" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --max-batches "$MAX_BATCHES"
fi

echo "Strong-impact + conservative top-K diagnostic complete: $selection_dir"
echo "Primary result: $selection_dir/topk_selected_test_summary.csv"
