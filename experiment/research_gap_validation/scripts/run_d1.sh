#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

D1_OUTPUT_DIR="${D1_OUTPUT_DIR:-$OUTPUT_DIR}"
D1_DATASET="$D1_OUTPUT_DIR/D1/edge_impact_pairs.csv"
D1_REGENERATE_DATA="${D1_REGENERATE_DATA:-0}"
D1_MODALITIES="${D1_MODALITIES:-text}"
D1_VALIDATION_SESSIONS="${D1_VALIDATION_SESSIONS:-}"
D1_SUPERVISIONS="${D1_SUPERVISIONS:-binary three_class magnitude}"
D1_STRONG_EPS="${D1_STRONG_EPS:-0.0001 0.001 0.01}"
D1_SEEDS="${D1_SEEDS:-$SEEDS}"
D1_EPOCHS="${D1_EPOCHS:-40}"
D1_PATIENCE="${D1_PATIENCE:-6}"
D1_BATCH_SIZE="${D1_BATCH_SIZE:-8192}"
D1_LEARNING_RATE="${D1_LEARNING_RATE:-0.0003}"
D1_HIDDEN_DIM="${D1_HIDDEN_DIM:-64}"
D1_DROPOUT="${D1_DROPOUT:-0.1}"
D1_MAX_CONVERSATIONS_PER_SPLIT="${D1_MAX_CONVERSATIONS_PER_SPLIT:-0}"
D1_MAX_TRAIN_ROWS="${D1_MAX_TRAIN_ROWS:-0}"

read -r -a MODALITY_ARRAY <<< "$D1_MODALITIES"
read -r -a VALIDATION_SESSION_ARRAY <<< "$D1_VALIDATION_SESSIONS"
read -r -a SUPERVISION_ARRAY <<< "$D1_SUPERVISIONS"
read -r -a EPS_ARRAY <<< "$D1_STRONG_EPS"
read -r -a SEED_ARRAY <<< "$D1_SEEDS"

VALIDATION_ARGS=()
if [[ ${#VALIDATION_SESSION_ARRAY[@]} -gt 0 ]]; then
  VALIDATION_ARGS+=(--validation-sessions "${VALIDATION_SESSION_ARRAY[@]}")
fi

if [[ "$D1_REGENERATE_DATA" == "1" || ! -s "$D1_DATASET" ]]; then
  echo "[D1 1/2] Generating frozen direct-edge impacts and observable features..."
  PREPARE_ARGS=()
  if [[ "$D1_REGENERATE_DATA" == "1" ]]; then
    PREPARE_ARGS+=(--overwrite)
  fi
  "$PYTHON" "$SCRIPT_DIR/d1_prepare_edge_dataset.py" \
    "${MODEL_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${PREPARE_ARGS[@]}" \
    --output-dir "$D1_OUTPUT_DIR" \
    --modalities "${MODALITY_ARRAY[@]}" \
    --intervention-batch-size "$INTERVENTION_BATCH_SIZE" \
    --seed "$SEED" \
    --max-conversations-per-split "$D1_MAX_CONVERSATIONS_PER_SPLIT"
else
  echo "[D1 1/2] Reusing dataset: $D1_DATASET"
fi

echo "[D1 2/2] Comparing binary, three-class, and magnitude supervision..."
"$PYTHON" "$SCRIPT_DIR/d1_harm_learnability.py" \
  --dataset "$D1_DATASET" \
  --output-dir "$D1_OUTPUT_DIR" \
  --supervisions "${SUPERVISION_ARRAY[@]}" \
  --strong-eps-candidates "${EPS_ARRAY[@]}" \
  --base-impact-eps "$IMPACT_EPS" \
  --selection-metric auroc \
  --seeds "${SEED_ARRAY[@]}" \
  --epochs "$D1_EPOCHS" \
  --patience "$D1_PATIENCE" \
  --batch-size "$D1_BATCH_SIZE" \
  --learning-rate "$D1_LEARNING_RATE" \
  --hidden-dim "$D1_HIDDEN_DIM" \
  --dropout "$D1_DROPOUT" \
  --device "$DEVICE" \
  --gpu-id "$GPU_ID" \
  --max-train-rows "$D1_MAX_TRAIN_ROWS"

echo "D1 complete: $D1_OUTPUT_DIR/D1"
