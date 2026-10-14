#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

"$PYTHON" "$SCRIPT_DIR/analyze_contextual_relationships.py" \
  --run-dir "$RUN_DIR" \
  --seed "$SEED"

exec "$PYTHON" "$SCRIPT_DIR/fit_structural_harm_mixed_effects.py" \
  --run-dir "$RUN_DIR" \
  --seed "$SEED" \
  "$@"
