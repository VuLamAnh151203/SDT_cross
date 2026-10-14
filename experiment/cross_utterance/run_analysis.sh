#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

"$PYTHON" "$SCRIPT_DIR/analyze_impacts.py" \
  --run-dir "$RUN_DIR" \
  --seed "$SEED" \
  "$@"

"$PYTHON" "$SCRIPT_DIR/analyze_prediction_changes.py" \
  --run-dir "$RUN_DIR"

exec "$PYTHON" "$SCRIPT_DIR/compute_ablation_classification_metrics.py" \
  --run-dir "$RUN_DIR"

