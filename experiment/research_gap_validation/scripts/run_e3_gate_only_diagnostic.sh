#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

DIAGNOSTIC_OUTPUT_DIR="${DIAGNOSTIC_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/gate_only}"
DIAGNOSTIC_MODELS="${DIAGNOSTIC_MODELS:-B0 B1 B2 B3 B4 B6}"
DIAGNOSTIC_SEEDS="${DIAGNOSTIC_SEEDS:-2024}"
DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-30}"
DIAGNOSTIC_PATIENCE="${DIAGNOSTIC_PATIENCE:-7}"
DIAGNOSTIC_GATE_LR="${DIAGNOSTIC_GATE_LR:-0.0003}"
DIAGNOSTIC_SELECTION_SPLIT="${DIAGNOSTIC_SELECTION_SPLIT:-valid}"
DIAGNOSTIC_SELECTION_METRIC="${DIAGNOSTIC_SELECTION_METRIC:-weighted_f1}"
DIAGNOSTIC_INITIAL_GATE="${DIAGNOSTIC_INITIAL_GATE:-0.99}"
DIAGNOSTIC_GATE_MODALITIES="${DIAGNOSTIC_GATE_MODALITIES:-text audio visual}"
DIAGNOSTIC_LAMBDA_IDENTITY="${DIAGNOSTIC_LAMBDA_IDENTITY:-0.001}"
DIAGNOSTIC_LAMBDA_IMPACT="${DIAGNOSTIC_LAMBDA_IMPACT:-0.01}"
DIAGNOSTIC_IMPACT_LOSS="${DIAGNOSTIC_IMPACT_LOSS:-soft_mse}"
DIAGNOSTIC_TEACHER_GATES="${DIAGNOSTIC_TEACHER_GATES:-0 0.25 0.5 0.75 1}"
DIAGNOSTIC_TEACHER_WEIGHT_SCALE="${DIAGNOSTIC_TEACHER_WEIGHT_SCALE:-0.01}"
DIAGNOSTIC_TEACHER_MARGIN_EPS="${DIAGNOSTIC_TEACHER_MARGIN_EPS:-0.0001}"

TEACHER_FILE="$OUTPUT_DIR/manifests/train_edge_impact_teacher.csv"
mkdir -p "$DIAGNOSTIC_OUTPUT_DIR" "$OUTPUT_DIR/manifests"
read -r -a TEACHER_GATE_ARRAY <<< "$DIAGNOSTIC_TEACHER_GATES"

if [[ ! -s "$TEACHER_FILE" ]]; then
  echo "[diagnostic 1/2] Generating train-only propagation-edge teacher..."
  "$PYTHON" "$SCRIPT_DIR/generate_train_edge_teacher.py" \
    "${MODEL_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --modalities text \
    --gates "${TEACHER_GATE_ARRAY[@]}" \
    --intervention-batch-size "$INTERVENTION_BATCH_SIZE" \
    --teacher-weight-scale "$DIAGNOSTIC_TEACHER_WEIGHT_SCALE" \
    --teacher-margin-eps "$DIAGNOSTIC_TEACHER_MARGIN_EPS" \
    --seed "$SEED" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "[diagnostic 1/2] Reusing impact teacher: $TEACHER_FILE"
fi

read -r -a MODEL_ARRAY <<< "$DIAGNOSTIC_MODELS"
read -r -a SEED_ARRAY <<< "$DIAGNOSTIC_SEEDS"
read -r -a GATE_MODALITY_ARRAY <<< "$DIAGNOSTIC_GATE_MODALITIES"

echo "[diagnostic 2/2] Frozen SDT/classifier; training gates only"
echo "Models: ${MODEL_ARRAY[*]}"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Gate LR: $DIAGNOSTIC_GATE_LR"
echo "Gate modalities: ${GATE_MODALITY_ARRAY[*]}"
echo "Initial gate: $DIAGNOSTIC_INITIAL_GATE"

"$PYTHON" "$SCRIPT_DIR/e3_train_compare.py" \
  "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --output-dir "$DIAGNOSTIC_OUTPUT_DIR" \
  --impact-teacher "$TEACHER_FILE" \
  --models "${MODEL_ARRAY[@]}" \
  --seeds "${SEED_ARRAY[@]}" \
  --epochs "$DIAGNOSTIC_EPOCHS" \
  --patience "$DIAGNOSTIC_PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --lr "$DIAGNOSTIC_GATE_LR" \
  --gate-lr "$DIAGNOSTIC_GATE_LR" \
  --initial-gate "$DIAGNOSTIC_INITIAL_GATE" \
  --gate-modalities "${GATE_MODALITY_ARRAY[@]}" \
  --lambda-identity "$DIAGNOSTIC_LAMBDA_IDENTITY" \
  --lambda-impact "$DIAGNOSTIC_LAMBDA_IMPACT" \
  --impact-loss "$DIAGNOSTIC_IMPACT_LOSS" \
  --impact-scale "$DIAGNOSTIC_TEACHER_WEIGHT_SCALE" \
  --impact-eps "$IMPACT_EPS" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --max-batches "$MAX_BATCHES" \
  --fold-name gate_only_diagnostic \
  --selection-split "$DIAGNOSTIC_SELECTION_SPLIT" \
  --selection-metric "$DIAGNOSTIC_SELECTION_METRIC" \
  --freeze-classifier

echo "Gate-only diagnostic complete: $DIAGNOSTIC_OUTPUT_DIR/E3"
