#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

B4_OUTPUT_DIR="${B4_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/b4_edge_teacher}"
B4_TEACHER_FILE="$OUTPUT_DIR/manifests/train_edge_impact_teacher.csv"
B4_REGENERATE_TEACHER="${B4_REGENERATE_TEACHER:-0}"
B4_MODELS="${B4_MODELS:-B3 B4}"
B4_SEEDS="${B4_SEEDS:-2024 2025 2026}"
B4_EPOCHS="${B4_EPOCHS:-30}"
B4_PATIENCE="${B4_PATIENCE:-7}"
B4_GATE_LR="${B4_GATE_LR:-0.0001}"
B4_LAMBDA_IMPACT="${B4_LAMBDA_IMPACT:-0.01}"
B4_IMPACT_LOSS="${B4_IMPACT_LOSS:-soft_mse}"
B4_TEACHER_GATES="${B4_TEACHER_GATES:-0 0.25 0.5 0.75 1}"
B4_TEACHER_WEIGHT_SCALE="${B4_TEACHER_WEIGHT_SCALE:-0.01}"
B4_TEACHER_MARGIN_EPS="${B4_TEACHER_MARGIN_EPS:-0.0001}"
B4_SELECTION_SPLIT="${B4_SELECTION_SPLIT:-test}"
B4_SELECTION_METRIC="${B4_SELECTION_METRIC:-weighted_f1}"
B4_FREEZE_CLASSIFIER="${B4_FREEZE_CLASSIFIER:-1}"
B4_RUN_AUDIT="${B4_RUN_AUDIT:-1}"

mkdir -p "$B4_OUTPUT_DIR" "$OUTPUT_DIR/manifests"
read -r -a TEACHER_GATE_ARRAY <<< "$B4_TEACHER_GATES"

if [[ "$B4_REGENERATE_TEACHER" == "1" || ! -s "$B4_TEACHER_FILE" ]]; then
  echo "[B4 fix 1/3] Generating train-only propagation-edge teacher..."
  "$PYTHON" "$SCRIPT_DIR/generate_train_edge_teacher.py" \
    "${MODEL_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --modalities text \
    --gates "${TEACHER_GATE_ARRAY[@]}" \
    --intervention-batch-size "$INTERVENTION_BATCH_SIZE" \
    --teacher-weight-scale "$B4_TEACHER_WEIGHT_SCALE" \
    --teacher-margin-eps "$B4_TEACHER_MARGIN_EPS" \
    --seed "$SEED" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "[B4 fix 1/3] Reusing edge teacher: $B4_TEACHER_FILE"
fi

read -r -a MODEL_ARRAY <<< "$B4_MODELS"
read -r -a SEED_ARRAY <<< "$B4_SEEDS"
TRAINING_ARGS=()
if [[ "$B4_FREEZE_CLASSIFIER" == "1" ]]; then
  TRAINING_ARGS+=(--freeze-classifier)
fi

echo "[B4 fix 2/3] Training controls: ${MODEL_ARRAY[*]}"
echo "[B4 fix 2/3] Seeds: ${SEED_ARRAY[*]}"
echo "[B4 fix 2/3] impact_loss=$B4_IMPACT_LOSS lambda=$B4_LAMBDA_IMPACT gate_lr=$B4_GATE_LR"
"$PYTHON" "$SCRIPT_DIR/e3_train_compare.py" \
  "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  --output-dir "$B4_OUTPUT_DIR" \
  --impact-teacher "$B4_TEACHER_FILE" \
  --models "${MODEL_ARRAY[@]}" \
  --seeds "${SEED_ARRAY[@]}" \
  --epochs "$B4_EPOCHS" \
  --patience "$B4_PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --lr "$B4_GATE_LR" \
  --gate-lr "$B4_GATE_LR" \
  --lambda-impact "$B4_LAMBDA_IMPACT" \
  --impact-loss "$B4_IMPACT_LOSS" \
  --impact-eps "$IMPACT_EPS" \
  --impact-scale "$B4_TEACHER_WEIGHT_SCALE" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --max-batches "$MAX_BATCHES" \
  --fold-name b4_edge_teacher \
  --selection-split "$B4_SELECTION_SPLIT" \
  --selection-metric "$B4_SELECTION_METRIC"

if [[ "$B4_RUN_AUDIT" == "1" ]]; then
  TRAINED_CHECKPOINTS=()
  for seed in "${SEED_ARRAY[@]}"; do
    for model in "${MODEL_ARRAY[@]}"; do
      checkpoint="$B4_OUTPUT_DIR/E3/checkpoints/b4_edge_teacher_${model}_seed${seed}.pt"
      if [[ ! -s "$checkpoint" ]]; then
        echo "ERROR: missing B4 diagnostic checkpoint: $checkpoint" >&2
        exit 2
      fi
      TRAINED_CHECKPOINTS+=("$checkpoint")
    done
  done
  echo "[B4 fix 3/3] Running text influence audit..."
  "$PYTHON" "$SCRIPT_DIR/e3_posttraining_audit.py" \
    "${MODEL_ARGS[@]}" \
    --trained-checkpoints "${TRAINED_CHECKPOINTS[@]}" \
    --output-dir "$B4_OUTPUT_DIR" \
    --split test \
    --modalities text \
    --strategies train_mean \
    --impact-eps "$IMPACT_EPS" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "[B4 fix 3/3] Audit skipped (B4_RUN_AUDIT=$B4_RUN_AUDIT)."
fi

echo "B4 fixed diagnostic complete: $B4_OUTPUT_DIR/E3"
