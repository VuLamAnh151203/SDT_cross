"""Create compact cross-seed and per-modality summaries for an E3 run."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import read_csv, write_csv  # noqa: E402


TEST_METRICS = ("accuracy", "weighted_f1", "macro_f1", "nll")
SELECTION_METRICS = ("best_selection_accuracy", "best_selection_weighted_f1",
                     "best_selection_nll")
MODALITIES = ("text", "audio", "visual")


def _stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def summarize(e3_dir: Path) -> None:
    rows = read_csv(e3_dir / "model_comparison.csv")
    if not rows:
        raise ValueError(f"No model rows in {e3_dir}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    seed_summary = []
    modality_summary = []
    for model, selected in sorted(grouped.items()):
        summary = {"model": model, "seeds": len(selected)}
        for metric in TEST_METRICS + SELECTION_METRICS:
            mean, std = _stats([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        seed_summary.append(summary)
        for modality in MODALITIES:
            gate_mean, gate_std = _stats([
                float(row[f"mean_gate_{modality}"]) for row in selected
            ])
            rate_mean, rate_std = _stats([
                float(row[f"strong_suppression_target_rate_{modality}"])
                for row in selected
            ])
            modality_summary.append({
                "model": model, "modality": modality, "seeds": len(selected),
                "mean_gate_mean": gate_mean, "mean_gate_std": gate_std,
                "strong_suppression_target_rate_mean": rate_mean,
                "strong_suppression_target_rate_std": rate_std,
            })

    by_model_seed = {(row["model"], int(row["seed"])): row for row in rows}
    delta_rows = []
    for left, right in (("B3", "B0"), ("B4", "B0"), ("B4", "B3")):
        seeds = sorted(
            {seed for model, seed in by_model_seed if model == left}
            & {seed for model, seed in by_model_seed if model == right}
        )
        if not seeds:
            continue
        for metric in TEST_METRICS:
            deltas = [
                float(by_model_seed[(left, seed)][metric])
                - float(by_model_seed[(right, seed)][metric])
                for seed in seeds
            ]
            mean, std = _stats(deltas)
            sem = std / math.sqrt(len(deltas)) if deltas else float("nan")
            delta_rows.append({
                "comparison": f"{left}-{right}", "metric": metric,
                "seeds": len(seeds), "delta_mean": mean, "delta_std": std,
                "normal_approx_ci95_low": mean - 1.96 * sem,
                "normal_approx_ci95_high": mean + 1.96 * sem,
                "note": "NLL delta below zero is better; score deltas above zero are better.",
            })

    write_csv(e3_dir / "cross_seed_summary.csv", seed_summary)
    write_csv(e3_dir / "cross_seed_paired_deltas.csv", delta_rows)
    write_csv(e3_dir / "modality_gate_summary.csv", modality_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.e3_dir.resolve())


if __name__ == "__main__":
    main()
