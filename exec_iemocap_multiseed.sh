#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
RUN_NAME="${RUN_NAME:-iemocap_multiseed_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RESULTS_DIR/$RUN_NAME}"
DEVICE="${DEVICE:-auto}"
GPU_ID="${GPU_ID:-0}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-validation}"
VALID_RATIO="${VALID_RATIO:-0.1}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
N_HEAD="${N_HEAD:-8}"
MAX_BATCHES="${MAX_BATCHES:-0}"
FEATURE_PATH="${FEATURE_PATH:-}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if (($#)); then
  SEED_ARRAY=("$@")
else
  read -r -a SEED_ARRAY <<< "${SEEDS:-2024 2025 2026 2027 2028}"
fi

if ((${#SEED_ARRAY[@]} == 0)); then
  echo "ERROR: provide at least one seed." >&2
  exit 2
fi

declare -A SEEN_SEEDS=()
for seed in "${SEED_ARRAY[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid integer seed: $seed" >&2
    exit 2
  fi
  if [[ -n "${SEEN_SEEDS[$seed]:-}" ]]; then
    echo "ERROR: duplicate seed: $seed" >&2
    exit 2
  fi
  SEEN_SEEDS[$seed]=1
done

if [[ -e "$RUN_DIR" ]]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  echo "Set RUN_NAME or RUN_DIR to a new path." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

printf 'Multi-seed output: %s\n' "$RUN_DIR"
printf 'Seeds: %s\n' "${SEED_ARRAY[*]}"
printf 'Checkpoint selection: %s\n' "$SELECTION_PROTOCOL"

OPTIONAL_ARGS=()
if [[ -n "$FEATURE_PATH" ]]; then
  OPTIONAL_ARGS+=(--feature-path "$FEATURE_PATH")
fi
if [[ "$MAX_BATCHES" != "0" ]]; then
  OPTIONAL_ARGS+=(--max-batches "$MAX_BATCHES")
fi

for seed in "${SEED_ARRAY[@]}"; do
  echo "================ seed=$seed ================"
  "$PYTHON" -u "$SCRIPT_DIR/train.py" \
    --selection-protocol "$SELECTION_PROTOCOL" \
    --valid-ratio "$VALID_RATIO" \
    --Dataset IEMOCAP --fusion-variant sdt \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" --n-head "$N_HEAD" \
    --lr 0.0001 --l2 0.00001 --dropout 0.5 --temp 1 \
    --gamma-1 1 --gamma-2 1 --gamma-3 1 \
    --device "$DEVICE" --gpu-id "$GPU_ID" \
    --output-dir "$RUN_DIR" --seed "$seed" \
    "${OPTIONAL_ARGS[@]}"
done

"$PYTHON" -u "$SCRIPT_DIR/summarize_multiseed.py" --run-dir "$RUN_DIR"
echo "All results and the mean +/- std summary are in: $RUN_DIR"
