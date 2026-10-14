#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

DIAGNOSTIC_OUTPUT_DIR="${DIAGNOSTIC_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/gate_only}"
AUDIT_FOLD_NAME="${AUDIT_FOLD_NAME:-gate_only_diagnostic}"
AUDIT_MODELS="${AUDIT_MODELS:-B0 B1 B2}"
AUDIT_SEEDS="${AUDIT_SEEDS:-$SEEDS}"
AUDIT_STRATEGIES="${AUDIT_STRATEGIES:-train_mean}"

read -r -a MODEL_ARRAY <<< "$AUDIT_MODELS"
read -r -a SEED_ARRAY <<< "$AUDIT_SEEDS"
read -r -a STRATEGY_ARRAY <<< "$AUDIT_STRATEGIES"

TRAINED_CHECKPOINTS=()
for seed in "${SEED_ARRAY[@]}"; do
  for model in "${MODEL_ARRAY[@]}"; do
    checkpoint="$DIAGNOSTIC_OUTPUT_DIR/E3/checkpoints/${AUDIT_FOLD_NAME}_${model}_seed${seed}.pt"
    if [[ ! -s "$checkpoint" ]]; then
      echo "ERROR: missing diagnostic checkpoint: $checkpoint" >&2
      exit 2
    fi
    TRAINED_CHECKPOINTS+=("$checkpoint")
  done
done

echo "Auditing models: ${MODEL_ARRAY[*]}"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Checkpoints: ${#TRAINED_CHECKPOINTS[@]}"
echo "Output: $DIAGNOSTIC_OUTPUT_DIR/E3"

"$PYTHON" "$SCRIPT_DIR/e3_posttraining_audit.py" \
  "${MODEL_ARGS[@]}" \
  --trained-checkpoints "${TRAINED_CHECKPOINTS[@]}" \
  --output-dir "$DIAGNOSTIC_OUTPUT_DIR" \
  --split test \
  --modalities text \
  --strategies "${STRATEGY_ARRAY[@]}" \
  --impact-eps "$IMPACT_EPS" \
  --max-conversations "$MAX_CONVERSATIONS"

echo "Targeted audit complete: $DIAGNOSTIC_OUTPUT_DIR/E3"
