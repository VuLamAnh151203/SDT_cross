#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment3_config.sh"

"$PYTHON" "$SCRIPT_DIR/generate_etcr_interventions.py" "${ETCR_GENERATOR_ARGS[@]}" \
  --splits train valid --strategies train_mean
"$PYTHON" "$SCRIPT_DIR/generate_etcr_interventions.py" "${ETCR_GENERATOR_ARGS[@]}" \
  --splits test --strategies train_mean zero
"$PYTHON" "$SCRIPT_DIR/phase_a_analyze_bias.py" --run-dir "$RUN_DIR" \
  --impact-eps "$IMPACT_EPS" --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED"
"$PYTHON" "$SCRIPT_DIR/phase_b_analyze_errors.py" --run-dir "$RUN_DIR" \
  --impact-eps "$IMPACT_EPS" --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED"
"$PYTHON" "$SCRIPT_DIR/phase_c_train_policy.py" --run-dir "$RUN_DIR" \
  --impact-eps "$IMPACT_EPS" --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED" \
  --embedding-components "$EMBEDDING_COMPONENTS"
"$PYTHON" "$SCRIPT_DIR/build_report.py" --run-dir "$RUN_DIR"

