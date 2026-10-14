#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

exec "$PYTHON" "$SCRIPT_DIR/run_impact_analysis.py" baseline \
  "${COMMON_ARGS[@]}" "$@"

