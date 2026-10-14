"""Phase A: quantify previous-emotion bias on previous-same-speaker pairs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from etcr_common import (
    EPSILON, as_bool, as_float, cluster_bootstrap, read_rows,
    validate_target_rows, write_rows,
)


METRICS = {
    "B": "old_emotion_probability_bias_B",
    "R": "old_vs_current_logit_shift_R",
    "G": "current_emotion_recovery_G",
    "signed_impact": "signed_impact",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--impact-eps", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def _summary(group_type: str, group_value: str, rows: Sequence[Mapping[str, str]],
             bootstrap_reps: int, seed: int) -> list[dict[str, object]]:
    output = []
    for short_name, column in METRICS.items():
        values = np.asarray([as_float(row, column) for row in rows], dtype=np.float64)
        low, high = cluster_bootstrap(
            rows, lambda sample, name=column: float(np.mean(
                [as_float(row, name) for row in sample]
            )), bootstrap_reps, seed,
        )
        output.append({
            "group_type": group_type, "group_value": group_value,
            "count": len(rows), "metric": short_name,
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "ci95_low": low, "ci95_high": high,
        })
    for name, predicate in (
        ("fraction_R_positive", lambda row: as_float(row, METRICS["R"]) > 0),
        ("fraction_B_positive", lambda row: as_float(row, METRICS["B"]) > 0),
    ):
        estimate = float(np.mean([predicate(row) for row in rows]))
        low, high = cluster_bootstrap(
            rows, lambda sample, pred=predicate: float(np.mean([pred(row) for row in sample])),
            bootstrap_reps, seed + 17,
        )
        output.append({
            "group_type": group_type, "group_value": group_value,
            "count": len(rows), "metric": name, "mean": estimate,
            "median": "", "ci95_low": low, "ci95_high": high,
        })
    return output


def _make_figure(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Phase A figure requires matplotlib") from error
    transition = [row for row in rows if as_bool(row, "transition")]
    r_values = np.asarray([as_float(row, METRICS["R"]) for row in transition])
    impacts = np.asarray([as_float(row, "signed_impact") for row in transition])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(r_values, bins=60, color="#4472C4", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(title="Transition targets: old-emotion preference shift",
                xlabel="R (positive = full context favors old emotion)", ylabel="Targets")
    axes[1].scatter(r_values, impacts, s=10, alpha=0.35, color="#C55A11")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Bias versus target loss impact", xlabel="R",
                ylabel="NLL(mask) - NLL(full)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    root = args.run_dir.expanduser().resolve() / "etcr"
    primary = read_rows(root / "raw" / "test_train_mean.csv")
    validate_target_rows(primary, "train_mean")
    eligible = [row for row in primary if as_bool(row, "eligible")]
    if not eligible:
        raise ValueError("No eligible test targets")
    pair_dir = root / "phase_a"
    write_rows(pair_dir / "emotion_persistence_pairs.csv", eligible)
    zero_path = root / "raw" / "test_zero.csv"
    if zero_path.is_file():
        zero = read_rows(zero_path)
        validate_target_rows(zero, "zero")
        write_rows(pair_dir / "emotion_persistence_pairs_zero.csv",
                   [row for row in zero if as_bool(row, "eligible")])

    groups: list[tuple[str, str, list[dict[str, str]]]] = []
    groups.append(("relation", "continuity",
                   [row for row in eligible if as_bool(row, "continuity")]))
    transitions = [row for row in eligible if as_bool(row, "transition")]
    groups.append(("relation", "transition", transitions))
    groups.extend([
        ("transition_impact", "harmful",
         [row for row in transitions if as_float(row, "signed_impact") < -args.impact_eps]),
        ("transition_impact", "beneficial",
         [row for row in transitions if as_float(row, "signed_impact") > args.impact_eps]),
        ("transition_impact", "near_zero",
         [row for row in transitions if abs(as_float(row, "signed_impact")) <= args.impact_eps]),
    ])
    for correctness in (0, 1):
        groups.append(("transition_baseline_correct", str(correctness), [
            row for row in transitions if int(row["correct_full"]) == correctness
        ]))
    pairs = sorted({(int(row["source_label"]), int(row["target_label"]))
                    for row in transitions})
    for source, target in pairs:
        groups.append(("transition_emotion_pair", f"{source}->{target}", [
            row for row in transitions
            if int(row["source_label"]) == source and int(row["target_label"]) == target
        ]))
    summaries = []
    for index, (group_type, group_value, rows) in enumerate(groups):
        if rows:
            summaries.extend(_summary(group_type, group_value, rows,
                                      args.bootstrap_reps, args.seed + index * 101))
    write_rows(pair_dir / "emotion_persistence_summary.csv", summaries)
    _make_figure(primary, root / "figures" / "phase_a_bias.png")
    print(f"Phase A: {len(eligible)} eligible targets; {len(transitions)} transitions")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.impact_eps < 0 or args.bootstrap_reps < 0:
        raise ValueError("impact-eps and bootstrap-reps must be nonnegative")
    analyze(args)


if __name__ == "__main__":
    main()

