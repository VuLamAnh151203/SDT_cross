#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment3_config.sh"

if [[ ! -f "$RUN_DIR/etcr/raw/test_train_mean.csv" || ! -f "$RUN_DIR/etcr/raw/test_zero.csv" ]]; then
  "$PYTHON" "$SCRIPT_DIR/generate_etcr_interventions.py" "${ETCR_GENERATOR_ARGS[@]}" \
    --splits test --strategies train_mean zero
fi
"$PYTHON" "$SCRIPT_DIR/phase_b_analyze_errors.py" --run-dir "$RUN_DIR" \
  --impact-eps "$IMPACT_EPS" --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED"

