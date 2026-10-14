#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

mkdir -p "$OUTPUT_DIR/manifests"

E3_IMPACT_LOSS="${E3_IMPACT_LOSS:-soft_mse}"
E3_LAMBDA_IMPACT="${E3_LAMBDA_IMPACT:-0.01}"
E3_TEACHER_GATES="${E3_TEACHER_GATES:-0 0.25 0.5 0.75 1}"
E3_TEACHER_WEIGHT_SCALE="${E3_TEACHER_WEIGHT_SCALE:-0.01}"
E3_TEACHER_MARGIN_EPS="${E3_TEACHER_MARGIN_EPS:-0.0001}"
read -r -a TEACHER_GATE_ARRAY <<< "$E3_TEACHER_GATES"

"$PYTHON" "$SCRIPT_DIR/audit_sdt_architecture.py" "${MODEL_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR"

IMPACT_FILES=(
  "$SOURCE_RUN_DIR/results/impact_pairs_train_mean.csv"
  "$SOURCE_RUN_DIR/results/impact_pairs_zero.csv"
)

if [[ "$RUN_DONOR" == "1" ]]; then
  "$PYTHON" "$SCRIPT_DIR/generate_donor_impacts.py" "${MODEL_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" --split test --modalities text \
    --seed "$SEED" --max-conversations "$MAX_CONVERSATIONS"
  IMPACT_FILES+=("$OUTPUT_DIR/E1/impact_pairs_train_donor.csv")
fi

"$PYTHON" "$SCRIPT_DIR/e1_mixed_contribution.py" \
  --impact-files "${IMPACT_FILES[@]}" \
  --baseline-file "$SOURCE_RUN_DIR/baseline/baseline_predictions.csv" \
  --output-dir "$OUTPUT_DIR" --eps "$IMPACT_EPS" \
  --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED"

"$PYTHON" "$SCRIPT_DIR/e2_stage_interventions.py" "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --input-impact-file "$SOURCE_RUN_DIR/results/impact_pairs_train_mean.csv" \
  --output-dir "$OUTPUT_DIR" --split test --modalities text \
  --intervention-batch-size "$INTERVENTION_BATCH_SIZE" \
  --bootstrap-reps "$BOOTSTRAP_REPS" --impact-eps "$IMPACT_EPS" --seed "$SEED" \
  --max-conversations "$MAX_CONVERSATIONS"

"$PYTHON" "$SCRIPT_DIR/e1_mixed_contribution.py" \
  --impact-files "${IMPACT_FILES[@]}" \
  --edge-file "$OUTPUT_DIR/E2/stage_intervention_pairs.csv" \
  --baseline-file "$SOURCE_RUN_DIR/baseline/baseline_predictions.csv" \
  --output-dir "$OUTPUT_DIR" --eps "$IMPACT_EPS" \
  --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED"

"$PYTHON" "$SCRIPT_DIR/build_report.py" --output-dir "$OUTPUT_DIR"
"$PYTHON" "$SCRIPT_DIR/make_figures.py" --output-dir "$OUTPUT_DIR"

if [[ "$RUN_FULL_E3" != "1" ]]; then
  echo "Stopped after E2 by pre-registered gate. Review $OUTPUT_DIR/report.md"
  echo "Set RUN_FULL_E3=1 only after accepting the E1/E2 premise."
  exit 0
fi

"$PYTHON" "$SCRIPT_DIR/generate_train_edge_teacher.py" "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" --modalities text \
  --gates "${TEACHER_GATE_ARRAY[@]}" \
  --intervention-batch-size "$INTERVENTION_BATCH_SIZE" \
  --teacher-weight-scale "$E3_TEACHER_WEIGHT_SCALE" \
  --teacher-margin-eps "$E3_TEACHER_MARGIN_EPS" --seed "$SEED" \
  --max-conversations "$MAX_CONVERSATIONS"

read -r -a SEED_ARRAY <<< "$SEEDS"
"$PYTHON" "$SCRIPT_DIR/e3_train_compare.py" "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --impact-teacher "$OUTPUT_DIR/manifests/train_edge_impact_teacher.csv" \
  --seeds "${SEED_ARRAY[@]}" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --lambda-impact "$E3_LAMBDA_IMPACT" --impact-loss "$E3_IMPACT_LOSS" \
  --impact-scale "$E3_TEACHER_WEIGHT_SCALE" \
  --impact-eps "$IMPACT_EPS" --bootstrap-reps "$BOOTSTRAP_REPS" \
  --max-batches "$MAX_BATCHES"

mapfile -t TRAINED_CHECKPOINTS < <(find "$OUTPUT_DIR/E3/checkpoints" -name '*.pt' -type f | sort)
"$PYTHON" "$SCRIPT_DIR/e3_posttraining_audit.py" "${MODEL_ARGS[@]}" \
  --trained-checkpoints "${TRAINED_CHECKPOINTS[@]}" \
  --output-dir "$OUTPUT_DIR" --split test --modalities text \
  --strategies train_mean --impact-eps "$IMPACT_EPS" \
  --max-conversations "$MAX_CONVERSATIONS"

mapfile -t B4_CHECKPOINTS < <(find "$OUTPUT_DIR/E3/checkpoints" -name '*_B4_seed*.pt' -type f | sort)
"$PYTHON" "$SCRIPT_DIR/e4_fixed_controls.py" "${MODEL_ARGS[@]}" \
  --trained-checkpoints "${B4_CHECKPOINTS[@]}" --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE"

"$PYTHON" "$SCRIPT_DIR/e4_robustness.py" --output-dir "$OUTPUT_DIR" --seed "$SEED"
"$PYTHON" "$SCRIPT_DIR/make_figures.py" --output-dir "$OUTPUT_DIR"
"$PYTHON" "$SCRIPT_DIR/build_report.py" --output-dir "$OUTPUT_DIR"

