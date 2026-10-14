"""Summarize and select E3 gate hyperparameters using validation metrics only."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import read_csv, write_csv  # noqa: E402


def summarize(sweep_dir: Path) -> None:
    rows = []
    for path in sorted(sweep_dir.glob("gate_*_lambda_*/E3/model_comparison.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["run_dir"] = str(path.parent.parent)
            rows.append(row)
    if not rows:
        raise ValueError(f"No sweep results found under {sweep_dir}")

    grouped = defaultdict(list)
    for row in rows:
        key = (row["model"], float(row["initial_gate"]), float(row["lambda_impact"]))
        grouped[key].append(row)
    summaries = []
    for (model, initial_gate, lambda_impact), selected in sorted(grouped.items()):
        summaries.append({
            "model": model, "initial_gate": initial_gate,
            "lambda_impact": lambda_impact, "seeds": len(selected),
            "valid_weighted_f1_mean": float(np.mean([
                float(row["best_selection_weighted_f1"]) for row in selected
            ])),
            "valid_accuracy_mean": float(np.mean([
                float(row["best_selection_accuracy"]) for row in selected
            ])),
            "valid_nll_mean": float(np.mean([
                float(row["best_selection_nll"]) for row in selected
            ])),
            "test_weighted_f1_mean_reporting_only": float(np.mean([
                float(row["weighted_f1"]) for row in selected
            ])),
            "run_dirs": ";".join(sorted({row["run_dir"] for row in selected})),
        })

    best = []
    for model in sorted({row["model"] for row in summaries}):
        candidates = [row for row in summaries if row["model"] == model]
        winner = max(candidates, key=lambda row: (
            row["valid_weighted_f1_mean"], row["valid_accuracy_mean"],
            -row["valid_nll_mean"],
        ))
        best.append({**winner, "selected_on": "validation_weighted_f1"})
    write_csv(sweep_dir / "sweep_summary.csv", summaries)
    write_csv(sweep_dir / "best_validation_config.csv", best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.sweep_dir.resolve())


if __name__ == "__main__":
    main()
