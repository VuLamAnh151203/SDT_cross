"""Summarize completed zero and train-mean cross-utterance impacts."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sdt_cross_utterance_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from impact_core import (
    IMPACT_FIELDS,
    MODALITIES,
    merge_csv_files,
    read_json,
    validate_impact_csv,
    write_csv_atomic,
    write_json_atomic,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    parser.add_argument(
        "--eps-sensitivity", type=float, nargs="+", default=(1e-6, 1e-4, 1e-3)
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-scatter-points", type=int, default=50000)
    return parser.parse_args(argv)


def _float(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}")
    return value


def _impact_class(values: np.ndarray, epsilon: float) -> Tuple[float, float, float]:
    if not len(values):
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(values > epsilon)),
        float(np.mean(values < -epsilon)),
        float(np.mean(np.abs(values) <= epsilon)),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _bootstrap_conversation_mean(
    conversation_means: np.ndarray, repetitions: int, rng: np.random.Generator
) -> Tuple[float | None, float | None]:
    if not len(conversation_means) or repetitions < 1:
        return None, None
    sample_indices = rng.integers(
        0, len(conversation_means), size=(repetitions, len(conversation_means))
    )
    estimates = conversation_means[sample_indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _write_rows(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    write_csv_atomic(path, rows, list(rows[0].keys()))


def _collect(paths: Mapping[str, Path]):
    impacts = defaultdict(list)
    distance_impacts = defaultdict(list)
    speaker_impacts = defaultdict(list)
    conversation_totals = defaultdict(lambda: [0.0, 0])
    transitions = Counter()
    pair_maps: Dict[str, Dict[tuple, float]] = {"zero": {}, "train_mean": {}}

    for strategy, path in paths.items():
        with path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                modality = row["modality"]
                value = _float(row, "signed_impact")
                conversation_id = row["conversation_id"]
                source_idx = int(row["source_idx"])
                target_idx = int(row["target_idx"])
                key = (conversation_id, source_idx, target_idx, modality)
                if key in pair_maps[strategy]:
                    raise ValueError(f"Duplicate exact pair key in {strategy}: {key}")
                pair_maps[strategy][key] = value
                impacts[(strategy, modality)].append(value)
                distance_impacts[
                    (strategy, modality, int(row["absolute_distance"]),
                     int(row["source_before_target"]))
                ].append(value)
                relation = "same_speaker" if int(row["same_speaker"]) else "cross_speaker"
                speaker_impacts[(strategy, modality, relation)].append(value)
                total = conversation_totals[(strategy, modality, conversation_id)]
                total[0] += value
                total[1] += 1
                transition = (
                    "correct_to_correct" if int(row["correct_full"]) and int(row["correct_mask"])
                    else "correct_to_wrong" if int(row["correct_full"])
                    else "wrong_to_correct" if int(row["correct_mask"])
                    else "wrong_to_wrong"
                )
                transitions[(strategy, modality, transition)] += 1
    if pair_maps["zero"].keys() != pair_maps["train_mean"].keys():
        missing_mean = list(pair_maps["zero"].keys() - pair_maps["train_mean"].keys())[:5]
        missing_zero = list(pair_maps["train_mean"].keys() - pair_maps["zero"].keys())[:5]
        raise ValueError(
            f"Zero/train-mean exact keys differ; missing_mean={missing_mean}, "
            f"missing_zero={missing_zero}"
        )
    return (
        impacts, distance_impacts, speaker_impacts, conversation_totals,
        transitions, pair_maps,
    )


def _summary_by_modality(
    impacts, conversation_totals, epsilon, bootstrap_reps, seed
) -> List[Mapping[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for strategy in ("zero", "train_mean"):
        for modality in MODALITIES:
            values = np.asarray(impacts[(strategy, modality)], dtype=np.float64)
            positive, negative, near_zero = _impact_class(values, epsilon)
            conversation_means = np.asarray([
                total / count
                for (item_strategy, item_modality, _), (total, count)
                in conversation_totals.items()
                if item_strategy == strategy and item_modality == modality
            ])
            ci_low, ci_high = _bootstrap_conversation_mean(
                conversation_means, bootstrap_reps, rng
            )
            rows.append({
                "replacement_strategy": strategy,
                "modality": modality,
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "impact_eps": epsilon,
                "positive_fraction": positive,
                "negative_fraction": negative,
                "near_zero_fraction": near_zero,
                "conversation_count": len(conversation_means),
                "mean_of_conversation_means": float(conversation_means.mean()),
                "conversation_bootstrap_ci95_low": ci_low,
                "conversation_bootstrap_ci95_high": ci_high,
            })
    return rows


def _distance_summary(distance_impacts) -> List[Mapping[str, object]]:
    rows = []
    for (strategy, modality, distance, source_before), values in sorted(
        distance_impacts.items()
    ):
        array = np.asarray(values, dtype=np.float64)
        rows.append({
            "replacement_strategy": strategy,
            "modality": modality,
            "absolute_distance": distance,
            "source_before_target": source_before,
            "count": len(array),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std()),
        })
    return rows


def _speaker_summary(speaker_impacts, epsilon) -> List[Mapping[str, object]]:
    rows = []
    for (strategy, modality, relation), values in sorted(speaker_impacts.items()):
        array = np.asarray(values, dtype=np.float64)
        positive, negative, near_zero = _impact_class(array, epsilon)
        rows.append({
            "replacement_strategy": strategy,
            "modality": modality,
            "speaker_relation": relation,
            "count": len(array),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std()),
            "positive_fraction": positive,
            "negative_fraction": negative,
            "near_zero_fraction": near_zero,
        })
    return rows


def _robustness(pair_maps, epsilon) -> Tuple[List[Mapping[str, object]], Dict[str, tuple]]:
    rows = []
    paired = {}
    all_keys = pair_maps["zero"].keys()
    for modality in MODALITIES:
        keys = [key for key in all_keys if key[-1] == modality]
        zero = np.asarray([pair_maps["zero"][key] for key in keys], dtype=np.float64)
        mean = np.asarray([pair_maps["train_mean"][key] for key in keys], dtype=np.float64)
        paired[modality] = (zero, mean)
        substantial = (np.abs(zero) > epsilon) & (np.abs(mean) > epsilon)
        sign_agreement = (
            float(np.mean(np.sign(zero[substantial]) == np.sign(mean[substantial])))
            if substantial.any() else None
        )
        row = {
            "modality": modality,
            "pair_count": len(zero),
            "pearson": _correlation(zero, mean),
            "spearman": _correlation(_rankdata(zero), _rankdata(mean)),
            "impact_eps": epsilon,
            "both_substantial_count": int(substantial.sum()),
            "sign_agreement_outside_epsilon": sign_agreement,
            "either_near_zero_fraction": float(
                np.mean((np.abs(zero) <= epsilon) | (np.abs(mean) <= epsilon))
            ),
        }
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, int(round(len(zero) * fraction)))
            zero_helpful = set(np.argpartition(zero, -count)[-count:])
            mean_helpful = set(np.argpartition(mean, -count)[-count:])
            zero_harmful = set(np.argpartition(zero, count - 1)[:count])
            mean_harmful = set(np.argpartition(mean, count - 1)[:count])
            tag = f"top_{int(fraction * 100):02d}pct"
            row[f"{tag}_helpful_overlap"] = len(zero_helpful & mean_helpful) / count
            row[f"{tag}_harmful_overlap"] = len(zero_harmful & mean_harmful) / count
        rows.append(row)
    return rows, paired


def _sensitivity_rows(impacts, epsilons) -> List[Mapping[str, object]]:
    rows = []
    for epsilon in sorted(set(epsilons)):
        for strategy in ("zero", "train_mean"):
            for modality in MODALITIES:
                values = np.asarray(impacts[(strategy, modality)], dtype=np.float64)
                positive, negative, near_zero = _impact_class(values, epsilon)
                rows.append({
                    "impact_eps": epsilon,
                    "replacement_strategy": strategy,
                    "modality": modality,
                    "count": len(values),
                    "positive_fraction": positive,
                    "negative_fraction": negative,
                    "near_zero_fraction": near_zero,
                })
    return rows


def _plot_distributions(impacts, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    for row_index, strategy in enumerate(("zero", "train_mean")):
        for column_index, modality in enumerate(MODALITIES):
            values = np.asarray(impacts[(strategy, modality)], dtype=np.float64)
            low, high = np.quantile(values, (0.005, 0.995))
            if low == high:
                low, high = low - 1e-6, high + 1e-6
            axis = axes[row_index, column_index]
            axis.hist(values, bins=80, range=(low, high), color="#3973ac", alpha=0.85)
            axis.axvline(0, color="black", linewidth=1)
            axis.set_title(f"{strategy} · {modality} (n={len(values):,})")
            axis.set_xlabel("Signed impact (CE_mask − CE_full)")
            axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_distance(distance_rows, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, strategy in zip(axes, ("zero", "train_mean")):
        for modality in MODALITIES:
            for source_before, linestyle, suffix in ((1, "-", "before"), (0, "--", "after")):
                selected = [
                    row for row in distance_rows
                    if row["replacement_strategy"] == strategy
                    and row["modality"] == modality
                    and row["source_before_target"] == source_before
                ]
                selected.sort(key=lambda row: row["absolute_distance"])
                axis.plot(
                    [row["absolute_distance"] for row in selected],
                    [row["mean"] for row in selected],
                    linestyle=linestyle, label=f"{modality} · source {suffix}",
                )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(strategy)
        axis.set_xlabel("Absolute utterance distance")
        axis.set_ylabel("Mean signed impact")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_speaker(speaker_rows, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    x = np.arange(len(MODALITIES))
    width = 0.35
    for axis, strategy in zip(axes, ("zero", "train_mean")):
        for offset, relation in ((-width / 2, "same_speaker"), (width / 2, "cross_speaker")):
            means = []
            for modality in MODALITIES:
                match = next((
                    row for row in speaker_rows
                    if row["replacement_strategy"] == strategy
                    and row["modality"] == modality
                    and row["speaker_relation"] == relation
                ), None)
                means.append(float("nan") if match is None else match["mean"])
            axis.bar(x + offset, means, width, label=relation.replace("_", " "))
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, MODALITIES)
        axis.set_title(strategy)
        axis.set_ylabel("Mean signed impact")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_robustness(paired, path: Path, maximum: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, modality in zip(axes, MODALITIES):
        zero, mean = paired[modality]
        if len(zero) > maximum:
            indices = rng.choice(len(zero), size=maximum, replace=False)
            plot_zero, plot_mean = zero[indices], mean[indices]
        else:
            plot_zero, plot_mean = zero, mean
        axis.scatter(plot_zero, plot_mean, s=4, alpha=0.2, rasterized=True)
        low = min(float(plot_zero.min()), float(plot_mean.min()))
        high = max(float(plot_zero.max()), float(plot_mean.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.axhline(0, color="grey", linewidth=0.7)
        axis.axvline(0, color="grey", linewidth=0.7)
        axis.set_title(f"{modality} (shown n={len(plot_zero):,})")
        axis.set_xlabel("Zero impact")
        axis.set_ylabel("Train-mean impact")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(run_dir, manifest, summary_rows, robustness_rows, epsilon) -> None:
    lines = [
        "# Frozen SDT cross-utterance impact report",
        "",
        "## Provenance",
        "",
        f"- Dataset/split: `{manifest['dataset']}` / `{manifest['split']}`",
        f"- Checkpoint ID: `{manifest['checkpoint_id']}`",
        f"- Conversations: {manifest['conversation_count']}",
        f"- Valid utterances: {manifest['utterance_count']}",
        f"- Rows per strategy: {manifest['expected_rows_per_strategy']:,}",
        f"- Primary near-zero tolerance: `{epsilon:g}`",
        "",
        "## Aggregate signed impacts",
        "",
        "Positive values mean that replacing the source-modality feature increased "
        "the target true-class loss. Negative values mean the replacement decreased it.",
        "",
        "| Strategy | Modality | N | Mean | Median | Positive | Negative | Near zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['replacement_strategy']} | {row['modality']} | {row['count']:,} | "
            f"{row['mean']:.6g} | {row['median']:.6g} | "
            f"{row['positive_fraction']:.3f} | {row['negative_fraction']:.3f} | "
            f"{row['near_zero_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "## Zero versus training-mean robustness",
        "",
        "| Modality | Pearson | Spearman | Sign agreement (both substantial) |",
        "|---|---:|---:|---:|",
    ])
    for row in robustness_rows:
        pearson = "NA" if row["pearson"] is None else f"{row['pearson']:.4f}"
        spearman = "NA" if row["spearman"] is None else f"{row['spearman']:.4f}"
        agreement = (
            "NA" if row["sign_agreement_outside_epsilon"] is None
            else f"{row['sign_agreement_outside_epsilon']:.4f}"
        )
        lines.append(f"| {row['modality']} | {pearson} | {spearman} | {agreement} |")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "These values are model-specific, intervention-dependent predictive impacts. "
        "They are not real-world causal effects. The SDT attention path is bidirectional, "
        "so the results describe full-conversation contextual impact. Zero replacement "
        "also preserves the utterance position and speaker embedding. Correlated rows from "
        "the same conversation must not be treated as independent observations.",
        "",
        "Detailed tables are in `results/`; paper-ready figures are in `figures/`.",
        "",
    ])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.impact_eps < 0 or any(value < 0 for value in args.eps_sensitivity):
        raise ValueError("Impact tolerances must be nonnegative")
    if args.bootstrap_reps < 1 or args.max_scatter_points < 1:
        raise ValueError("Bootstrap repetitions and scatter limit must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = read_json(run_dir / "experiment_manifest.json")
    expected = int(manifest["expected_rows_per_strategy"])
    result_dir = run_dir / "results"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "zero": result_dir / "impact_pairs_zero.csv",
        "train_mean": result_dir / "impact_pairs_train_mean.csv",
    }
    for strategy, path in paths.items():
        validate_impact_csv(path, strategy, expected)

    combined_path = result_dir / "impact_pairs.csv"
    combined_count = merge_csv_files(
        [paths["zero"], paths["train_mean"]], combined_path
    )
    if combined_count != 2 * expected:
        raise AssertionError("Combined result row count is incorrect")

    (
        impacts, distance_impacts, speaker_impacts, conversation_totals,
        transitions, pair_maps,
    ) = _collect(paths)
    summary_rows = _summary_by_modality(
        impacts, conversation_totals, args.impact_eps, args.bootstrap_reps, args.seed
    )
    distance_rows = _distance_summary(distance_impacts)
    speaker_rows = _speaker_summary(speaker_impacts, args.impact_eps)
    robustness_rows, paired = _robustness(pair_maps, args.impact_eps)
    sensitivity_rows = _sensitivity_rows(
        impacts, list(args.eps_sensitivity) + [args.impact_eps]
    )
    transition_rows = [
        {
            "replacement_strategy": strategy,
            "modality": modality,
            "transition": transition,
            "count": count,
        }
        for (strategy, modality, transition), count in sorted(transitions.items())
    ]

    _write_rows(result_dir / "summary_by_modality.csv", summary_rows)
    _write_rows(result_dir / "summary_by_distance.csv", distance_rows)
    _write_rows(result_dir / "summary_by_speaker_relation.csv", speaker_rows)
    _write_rows(result_dir / "masking_robustness.csv", robustness_rows)
    _write_rows(result_dir / "impact_eps_sensitivity.csv", sensitivity_rows)
    _write_rows(result_dir / "prediction_transitions.csv", transition_rows)

    _plot_distributions(
        impacts, figure_dir / "impact_distribution_by_modality.png"
    )
    _plot_distance(distance_rows, figure_dir / "impact_vs_distance.png")
    _plot_speaker(
        speaker_rows, figure_dir / "impact_by_speaker_relation.png"
    )
    _plot_robustness(
        paired, figure_dir / "zero_vs_mean_scatter.png",
        args.max_scatter_points, args.seed,
    )
    _write_report(run_dir, manifest, summary_rows, robustness_rows, args.impact_eps)
    write_json_atomic(result_dir / "analysis_validation.json", {
        "combined_rows": combined_count,
        "exact_pair_keys_per_strategy": len(pair_maps["zero"]),
        "impact_eps": args.impact_eps,
        "eps_sensitivity": sorted(set(args.eps_sensitivity)),
        "bootstrap_reps": args.bootstrap_reps,
        "zero_mean_exact_keys_match": True,
    })
    print(f"Analysis complete: {run_dir / 'report.md'}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
