#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# The train and validation propagation-edge teachers live in OUTPUT_DIR/manifests
# and are intentionally shared across all runs.
FIVE_OUTPUT_DIR="${FIVE_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/five_diagnostics}"
FIVE_SEEDS="${FIVE_SEEDS:-2024 2025 2026 2027 2028}"
FIVE_MODELS="${FIVE_MODELS:-B0 B3 B4}"
FIVE_EPOCHS="${FIVE_EPOCHS:-30}"
FIVE_PATIENCE="${FIVE_PATIENCE:-7}"
FIVE_GATE_LR="${FIVE_GATE_LR:-0.0003}"
FIVE_SELECTION_METRIC="${FIVE_SELECTION_METRIC:-weighted_f1}"
FIVE_LAMBDA_IMPACT="${FIVE_LAMBDA_IMPACT:-0.01}"
FIVE_INITIAL_GATE="${FIVE_INITIAL_GATE:-0.99}"
FIVE_REUSE_COMPLETED_RUNS="${FIVE_REUSE_COMPLETED_RUNS:-1}"

SWEEP_SEEDS="${SWEEP_SEEDS:-2024}"
SWEEP_MODELS="${SWEEP_MODELS:-B3 B4}"
SWEEP_INITIAL_GATES="${SWEEP_INITIAL_GATES:-0.90 0.95 0.99}"
SWEEP_LAMBDA_IMPACTS="${SWEEP_LAMBDA_IMPACTS:-0.01 0.05 0.10}"

mkdir -p "$FIVE_OUTPUT_DIR"
read -r -a FIVE_SEED_ARRAY <<< "$FIVE_SEEDS"

echo "[1/5] Multi-seed B0/B3/B4 gate-only diagnostic"
if [[ "$FIVE_REUSE_COMPLETED_RUNS" == "1" \
      && -s "$FIVE_OUTPUT_DIR/01_multiseed/E3/manifest.json" ]]; then
  echo "Reusing completed step 1"
else
  DIAGNOSTIC_OUTPUT_DIR="$FIVE_OUTPUT_DIR/01_multiseed" \
  DIAGNOSTIC_MODELS="$FIVE_MODELS" \
  DIAGNOSTIC_SEEDS="$FIVE_SEEDS" \
  DIAGNOSTIC_EPOCHS="$FIVE_EPOCHS" \
  DIAGNOSTIC_PATIENCE="$FIVE_PATIENCE" \
  DIAGNOSTIC_GATE_LR="$FIVE_GATE_LR" \
  DIAGNOSTIC_SELECTION_SPLIT="valid" \
  DIAGNOSTIC_SELECTION_METRIC="$FIVE_SELECTION_METRIC" \
  DIAGNOSTIC_INITIAL_GATE="$FIVE_INITIAL_GATE" \
  DIAGNOSTIC_LAMBDA_IMPACT="$FIVE_LAMBDA_IMPACT" \
  DIAGNOSTIC_GATE_MODALITIES="text audio visual" \
  bash "$SCRIPT_DIR/run_e3_gate_only_diagnostic.sh"
fi

echo "[2/5] Cross-seed paired deltas and per-modality gate statistics"
"$PYTHON" "$SCRIPT_DIR/summarize_e3_diagnostic.py" \
  --e3-dir "$FIVE_OUTPUT_DIR/01_multiseed/E3"

echo "[3/5] Validation teacher-fit: weighted MSE, Spearman and Precision@K"
VALID_TEACHER="$OUTPUT_DIR/manifests/valid_edge_impact_teacher.csv"
if [[ ! -s "$VALID_TEACHER" ]]; then
  read -r -a TEACHER_GATE_ARRAY <<< "${DIAGNOSTIC_TEACHER_GATES:-0 0.25 0.5 0.75 1}"
  "$PYTHON" "$SCRIPT_DIR/generate_train_edge_teacher.py" \
    "${MODEL_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --split valid \
    --modalities text \
    --gates "${TEACHER_GATE_ARRAY[@]}" \
    --intervention-batch-size "$INTERVENTION_BATCH_SIZE" \
    --teacher-weight-scale "${DIAGNOSTIC_TEACHER_WEIGHT_SCALE:-0.01}" \
    --teacher-margin-eps "${DIAGNOSTIC_TEACHER_MARGIN_EPS:-0.0001}" \
    --seed "$SEED" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "Reusing validation teacher: $VALID_TEACHER"
fi

FIT_CHECKPOINTS=()
for seed in "${FIVE_SEED_ARRAY[@]}"; do
  for model in B3 B4; do
    checkpoint="$FIVE_OUTPUT_DIR/01_multiseed/E3/checkpoints/gate_only_diagnostic_${model}_seed${seed}.pt"
    if [[ ! -s "$checkpoint" ]]; then
      echo "ERROR: step 3 requires checkpoint $checkpoint" >&2
      echo "Keep B3 and B4 in FIVE_MODELS." >&2
      exit 2
    fi
    FIT_CHECKPOINTS+=("$checkpoint")
  done
done
mkdir -p "$FIVE_OUTPUT_DIR/03_teacher_fit"
if [[ "$FIVE_REUSE_COMPLETED_RUNS" == "1" \
      && -s "$FIVE_OUTPUT_DIR/03_teacher_fit/teacher_fit_manifest.json" ]]; then
  echo "Reusing completed step 3"
else
  "$PYTHON" "$SCRIPT_DIR/evaluate_gate_teacher_fit.py" \
    "${MODEL_ARGS[@]}" \
    --trained-checkpoints "${FIT_CHECKPOINTS[@]}" \
    --teacher-file "$VALID_TEACHER" \
    --output-dir "$FIVE_OUTPUT_DIR/03_teacher_fit" \
    --split valid \
    --precision-k 1 3 5 \
    --max-conversations "$MAX_CONVERSATIONS"
fi

echo "[4/5] Initial-gate x impact-loss-weight sweep (validation selection only)"
read -r -a SWEEP_GATE_ARRAY <<< "$SWEEP_INITIAL_GATES"
read -r -a SWEEP_LAMBDA_ARRAY <<< "$SWEEP_LAMBDA_IMPACTS"
read -r -a SWEEP_MODEL_ARRAY <<< "$SWEEP_MODELS"
for initial_gate in "${SWEEP_GATE_ARRAY[@]}"; do
  for lambda_impact in "${SWEEP_LAMBDA_ARRAY[@]}"; do
    run_dir="$FIVE_OUTPUT_DIR/04_sweep/gate_${initial_gate}_lambda_${lambda_impact}"
    if [[ "$FIVE_REUSE_COMPLETED_RUNS" == "1" \
          && -s "$run_dir/E3/manifest.json" ]]; then
      echo "Reusing completed sweep: $run_dir"
      continue
    fi
    # lambda_impact has no effect on B3, so run B3 only once per initial gate.
    RUN_SWEEP_MODELS=()
    for model in "${SWEEP_MODEL_ARRAY[@]}"; do
      if [[ "$model" != "B3" || "$lambda_impact" == "${SWEEP_LAMBDA_ARRAY[0]}" ]]; then
        RUN_SWEEP_MODELS+=("$model")
      fi
    done
    if [[ "${#RUN_SWEEP_MODELS[@]}" -eq 0 ]]; then
      continue
    fi
    echo "Sweep models=${RUN_SWEEP_MODELS[*]} initial_gate=$initial_gate lambda_impact=$lambda_impact"
    DIAGNOSTIC_OUTPUT_DIR="$run_dir" \
    DIAGNOSTIC_MODELS="${RUN_SWEEP_MODELS[*]}" \
    DIAGNOSTIC_SEEDS="$SWEEP_SEEDS" \
    DIAGNOSTIC_EPOCHS="$FIVE_EPOCHS" \
    DIAGNOSTIC_PATIENCE="$FIVE_PATIENCE" \
    DIAGNOSTIC_GATE_LR="$FIVE_GATE_LR" \
    DIAGNOSTIC_SELECTION_SPLIT="valid" \
    DIAGNOSTIC_SELECTION_METRIC="$FIVE_SELECTION_METRIC" \
    DIAGNOSTIC_INITIAL_GATE="$initial_gate" \
    DIAGNOSTIC_LAMBDA_IMPACT="$lambda_impact" \
    DIAGNOSTIC_GATE_MODALITIES="text audio visual" \
    bash "$SCRIPT_DIR/run_e3_gate_only_diagnostic.sh"
  done
done
"$PYTHON" "$SCRIPT_DIR/summarize_e3_sweep.py" \
  --sweep-dir "$FIVE_OUTPUT_DIR/04_sweep"

echo "[5/5] Text-only gates; audio and visual gates fixed exactly at 1"
if [[ "$FIVE_REUSE_COMPLETED_RUNS" == "1" \
      && -s "$FIVE_OUTPUT_DIR/05_text_only/E3/manifest.json" ]]; then
  echo "Reusing completed step 5"
else
  DIAGNOSTIC_OUTPUT_DIR="$FIVE_OUTPUT_DIR/05_text_only" \
  DIAGNOSTIC_MODELS="B3 B4" \
  DIAGNOSTIC_SEEDS="$FIVE_SEEDS" \
  DIAGNOSTIC_EPOCHS="$FIVE_EPOCHS" \
  DIAGNOSTIC_PATIENCE="$FIVE_PATIENCE" \
  DIAGNOSTIC_GATE_LR="$FIVE_GATE_LR" \
  DIAGNOSTIC_SELECTION_SPLIT="valid" \
  DIAGNOSTIC_SELECTION_METRIC="$FIVE_SELECTION_METRIC" \
  DIAGNOSTIC_INITIAL_GATE="$FIVE_INITIAL_GATE" \
  DIAGNOSTIC_LAMBDA_IMPACT="$FIVE_LAMBDA_IMPACT" \
  DIAGNOSTIC_GATE_MODALITIES="text" \
  bash "$SCRIPT_DIR/run_e3_gate_only_diagnostic.sh"
fi
"$PYTHON" "$SCRIPT_DIR/summarize_e3_diagnostic.py" \
  --e3-dir "$FIVE_OUTPUT_DIR/05_text_only/E3"

echo "All five E3 diagnostics completed: $FIVE_OUTPUT_DIR"
