#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment3_config.sh"

if [[ ! -f "$RUN_DIR/etcr/raw/train_train_mean.csv" || \
      ! -f "$RUN_DIR/etcr/raw/valid_train_mean.csv" || \
      ! -f "$RUN_DIR/etcr/raw/test_train_mean.csv" ]]; then
  "$PYTHON" "$SCRIPT_DIR/generate_etcr_interventions.py" "${ETCR_GENERATOR_ARGS[@]}" \
    --splits train valid test --strategies train_mean
fi
"$PYTHON" "$SCRIPT_DIR/phase_c_train_policy.py" --run-dir "$RUN_DIR" \
  --impact-eps "$IMPACT_EPS" --bootstrap-reps "$BOOTSTRAP_REPS" --seed "$SEED" \
  --embedding-components "$EMBEDDING_COMPONENTS"

