"""Structural-harm experiment: speaker, distance, and emotion transitions.

All uncertainty intervals in this script use cluster bootstrap resampling at
the conversation level.  Emotion labels are used only for offline analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from experiment_analysis_common import (
    MODALITIES,
    STRATEGIES,
    binary_int,
    finite_float,
    require_columns,
    resolve_impact_paths,
    write_rows,
)


REQUIRED = {
    "conversation_id", "source_idx", "target_idx", "source_speaker",
    "target_speaker", "same_speaker", "absolute_distance",
    "source_before_target", "modality", "replacement_strategy",
    "source_emotion", "target_emotion", "signed_impact",
}
METRIC_NAMES = (
    "mean_signed_impact",
    "mean_absolute_impact",
    "harmful_impact_rate",
    "conditional_harmful_fraction",
    "influential_rate",
    "mean_harmful_magnitude",
    "beneficial_impact_rate",
    "conditional_beneficial_fraction",
)
DISTANCE_LABELS = ("01-05", "06-10", "11-20", "21+")


@dataclass
class Totals:
    count: int = 0
    signed_sum: float = 0.0
    absolute_sum: float = 0.0
    harmful_count: int = 0
    harmful_magnitude_sum: float = 0.0
    beneficial_count: int = 0
    beneficial_magnitude_sum: float = 0.0

    @property
    def influential_count(self) -> int:
        return self.harmful_count + self.beneficial_count

    def add(self, impact: float, epsilon: float) -> None:
        self.count += 1
        self.signed_sum += impact
        self.absolute_sum += abs(impact)
        if impact < -epsilon:
            self.harmful_count += 1
            self.harmful_magnitude_sum += abs(impact)
        elif impact > epsilon:
            self.beneficial_count += 1
            self.beneficial_magnitude_sum += impact

    def merge(self, other: "Totals") -> None:
        self.count += other.count
        self.signed_sum += other.signed_sum
        self.absolute_sum += other.absolute_sum
        self.harmful_count += other.harmful_count
        self.harmful_magnitude_sum += other.harmful_magnitude_sum
        self.beneficial_count += other.beneficial_count
        self.beneficial_magnitude_sum += other.beneficial_magnitude_sum

    def metrics(self) -> Dict[str, float | None]:
        if self.count == 0:
            return {name: None for name in METRIC_NAMES}
        influential = self.influential_count
        return {
            "mean_signed_impact": self.signed_sum / self.count,
            "mean_absolute_impact": self.absolute_sum / self.count,
            "harmful_impact_rate": self.harmful_count / self.count,
            "conditional_harmful_fraction": (
                self.harmful_count / influential if influential else None
            ),
            "influential_rate": influential / self.count,
            "mean_harmful_magnitude": (
                self.harmful_magnitude_sum / self.harmful_count
                if self.harmful_count else None
            ),
            "beneficial_impact_rate": self.beneficial_count / self.count,
            "conditional_beneficial_fraction": (
                self.beneficial_count / influential if influential else None
            ),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    parser.add_argument(
        "--eps-values", type=float, nargs="+", default=(1e-6, 1e-4, 1e-3),
        help="Thresholds used for robustness summaries",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def distance_bin(distance: int) -> str:
    if distance <= 5:
        return "01-05"
    if distance <= 10:
        return "06-10"
    if distance <= 20:
        return "11-20"
    return "21+"


def _aggregate(items: Iterable[Totals]) -> Totals:
    total = Totals()
    for item in items:
        total.merge(item)
    return total


def _bootstrap_intervals(
    per_conversation: Mapping[str, Totals],
    conversations: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, tuple[float | None, float | None]]:
    values = {name: [] for name in METRIC_NAMES}
    for _ in range(repetitions):
        sampled = _aggregate(
            per_conversation.get(
                conversations[index], Totals()
            )
            for index in rng.integers(0, len(conversations), size=len(conversations))
        )
        for name, value in sampled.metrics().items():
            if value is not None and math.isfinite(value):
                values[name].append(value)
    intervals = {}
    for name, estimates in values.items():
        if estimates:
            low, high = np.quantile(np.asarray(estimates), (0.025, 0.975))
            intervals[name] = (float(low), float(high))
        else:
            intervals[name] = (None, None)
    return intervals


def _summary_rows(
    groups: Mapping[tuple, Mapping[str, Totals]],
    key_names: Sequence[str],
    conversations: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
):
    rows = []
    for key in sorted(groups):
        by_conversation = groups[key]
        total = _aggregate(by_conversation.values())
        intervals = _bootstrap_intervals(
            by_conversation, conversations, repetitions, rng
        )
        row = dict(zip(key_names, key))
        row.update({
            "count": total.count,
            "conversation_count_present": len(by_conversation),
            "harmful_count": total.harmful_count,
            "beneficial_count": total.beneficial_count,
            "influential_count": total.influential_count,
            **total.metrics(),
        })
        for name in METRIC_NAMES:
            row[f"{name}_ci95_low"] = intervals[name][0]
            row[f"{name}_ci95_high"] = intervals[name][1]
        rows.append(row)
    return rows


def _speaker_sequences(paths: Sequence[Path]):
    speakers: Dict[tuple[str, int], str] = {}
    with paths[0].open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require_columns(paths[0], reader.fieldnames or [], REQUIRED)
        for line_number, row in enumerate(reader, start=2):
            conversation_id = row["conversation_id"]
            for index_name, speaker_name in (
                ("source_idx", "source_speaker"),
                ("target_idx", "target_speaker"),
            ):
                key = (conversation_id, int(row[index_name]))
                speaker = row[speaker_name]
                previous = speakers.setdefault(key, speaker)
                if previous != speaker:
                    raise ValueError(
                        f"{paths[0]}:{line_number}: inconsistent speaker for {key}"
                    )
    by_conversation = defaultdict(list)
    for (conversation_id, index), speaker in speakers.items():
        by_conversation[conversation_id].append((index, speaker))
    previous_same_speaker = {}
    for conversation_id, sequence in by_conversation.items():
        last_seen = {}
        for index, speaker in sorted(sequence):
            if speaker in last_seen:
                previous_same_speaker[(conversation_id, index)] = last_seen[speaker]
            last_seen[speaker] = index
    return previous_same_speaker


def _difference_bootstrap(
    first: Mapping[str, Totals],
    second: Mapping[str, Totals],
    conversations: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
):
    estimates = {name: [] for name in METRIC_NAMES}
    for _ in range(repetitions):
        sampled_ids = [
            conversations[index]
            for index in rng.integers(0, len(conversations), size=len(conversations))
        ]
        first_total = _aggregate(first.get(item, Totals()) for item in sampled_ids)
        second_total = _aggregate(second.get(item, Totals()) for item in sampled_ids)
        first_metrics = first_total.metrics()
        second_metrics = second_total.metrics()
        for name in METRIC_NAMES:
            if first_metrics[name] is not None and second_metrics[name] is not None:
                estimates[name].append(first_metrics[name] - second_metrics[name])
    result = {}
    for name, values in estimates.items():
        if values:
            low, high = np.quantile(np.asarray(values), (0.025, 0.975))
            result[name] = (float(low), float(high))
        else:
            result[name] = (None, None)
    return result


def _emotion_contrasts(
    groups: Mapping[tuple, Mapping[str, Totals]],
    scope: str,
    epsilons: Sequence[float],
    conversations: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
):
    rows = []
    for epsilon in epsilons:
        for strategy in STRATEGIES:
            for modality in MODALITIES:
                same = groups.get(
                    (epsilon, strategy, modality, "same_emotion"), {}
                )
                shift = groups.get(
                    (epsilon, strategy, modality, "different_emotion"), {}
                )
                if not same or not shift:
                    continue
                same_total = _aggregate(same.values())
                shift_total = _aggregate(shift.values())
                same_metrics = same_total.metrics()
                shift_metrics = shift_total.metrics()
                intervals = _difference_bootstrap(
                    shift, same, conversations, repetitions, rng
                )
                row = {
                    "scope": scope,
                    "impact_eps": epsilon,
                    "replacement_strategy": strategy,
                    "modality": modality,
                    "same_emotion_count": same_total.count,
                    "emotion_shift_count": shift_total.count,
                }
                for name in METRIC_NAMES:
                    row[f"same_emotion_{name}"] = same_metrics[name]
                    row[f"emotion_shift_{name}"] = shift_metrics[name]
                    row[f"difference_shift_minus_same_{name}"] = (
                        shift_metrics[name] - same_metrics[name]
                        if shift_metrics[name] is not None
                        and same_metrics[name] is not None else None
                    )
                    row[f"difference_{name}_ci95_low"] = intervals[name][0]
                    row[f"difference_{name}_ci95_high"] = intervals[name][1]
                rows.append(row)
    return rows


def _strategy_contrasts(
    groups: Mapping[tuple, Mapping[str, Totals]],
    epsilons: Sequence[float],
    conversations: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
):
    rows = []
    for epsilon in epsilons:
        for modality in MODALITIES:
            for relation in ("same_speaker", "cross_speaker"):
                for band in DISTANCE_LABELS:
                    mean = groups.get(
                        (epsilon, "train_mean", modality, relation, band), {}
                    )
                    zero = groups.get(
                        (epsilon, "zero", modality, relation, band), {}
                    )
                    if not mean or not zero:
                        continue
                    mean_metrics = _aggregate(mean.values()).metrics()
                    zero_metrics = _aggregate(zero.values()).metrics()
                    intervals = _difference_bootstrap(
                        mean, zero, conversations, repetitions, rng
                    )
                    row = {
                        "impact_eps": epsilon,
                        "modality": modality,
                        "speaker_relation": relation,
                        "distance_bin": band,
                    }
                    for name in METRIC_NAMES:
                        row[f"zero_{name}"] = zero_metrics[name]
                        row[f"train_mean_{name}"] = mean_metrics[name]
                        row[f"difference_mean_minus_zero_{name}"] = (
                            mean_metrics[name] - zero_metrics[name]
                            if mean_metrics[name] is not None
                            and zero_metrics[name] is not None else None
                        )
                        row[f"difference_{name}_ci95_low"] = intervals[name][0]
                        row[f"difference_{name}_ci95_high"] = intervals[name][1]
                    rows.append(row)
    return rows


def _within_target_outputs(
    groups: Mapping[tuple, Totals],
    primary_epsilon: float,
    conversations: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
):
    details = []
    bases = sorted({key[:-1] for key in groups if key[0] == primary_epsilon})
    for base in bases:
        same = groups.get(base + ("same_speaker",))
        cross = groups.get(base + ("cross_speaker",))
        if same is None or cross is None:
            continue
        same_metrics = same.metrics()
        cross_metrics = cross.metrics()
        epsilon, strategy, modality, band, conversation_id, target_idx = base
        row = {
            "impact_eps": epsilon,
            "replacement_strategy": strategy,
            "modality": modality,
            "distance_bin": band,
            "conversation_id": conversation_id,
            "target_idx": target_idx,
            "same_speaker_count": same.count,
            "cross_speaker_count": cross.count,
        }
        for name in METRIC_NAMES:
            row[f"same_speaker_{name}"] = same_metrics[name]
            row[f"cross_speaker_{name}"] = cross_metrics[name]
            row[f"difference_same_minus_cross_{name}"] = (
                same_metrics[name] - cross_metrics[name]
                if same_metrics[name] is not None
                and cross_metrics[name] is not None else None
            )
        details.append(row)

    grouped = defaultdict(list)
    for row in details:
        grouped[(
            row["replacement_strategy"], row["modality"], row["distance_bin"]
        )].append(row)
    summaries = []
    for key in sorted(grouped):
        items = grouped[key]
        row = {
            "impact_eps": primary_epsilon,
            "replacement_strategy": key[0],
            "modality": key[1],
            "distance_bin": key[2],
            "matched_target_count": len(items),
            "conversation_count": len({item["conversation_id"] for item in items}),
        }
        for name in METRIC_NAMES:
            column = f"difference_same_minus_cross_{name}"
            valid = [item for item in items if item[column] is not None]
            row[f"mean_{column}"] = (
                float(np.mean([item[column] for item in valid])) if valid else None
            )
            estimates = []
            by_conversation = defaultdict(list)
            for item in valid:
                by_conversation[item["conversation_id"]].append(item[column])
            for _ in range(repetitions):
                sampled = [
                    conversations[index]
                    for index in rng.integers(
                        0, len(conversations), size=len(conversations)
                    )
                ]
                values = [
                    value for conversation_id in sampled
                    for value in by_conversation.get(conversation_id, [])
                ]
                if values:
                    estimates.append(float(np.mean(values)))
            if estimates:
                low, high = np.quantile(estimates, (0.025, 0.975))
                row[f"{column}_ci95_low"] = float(low)
                row[f"{column}_ci95_high"] = float(high)
            else:
                row[f"{column}_ci95_low"] = None
                row[f"{column}_ci95_high"] = None
        summaries.append(row)
    return details, summaries


def _plot_heatmaps(rows, figure_dir: Path, primary_epsilon: float) -> None:
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "sdt_structural_harm_matplotlib"),
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Heatmap generation requires matplotlib; install requirements.txt"
        ) from error

    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("harmful_impact_rate", "Harmful rate (all pairs)"),
        ("conditional_harmful_fraction", "Harmful fraction (influential only)"),
        ("influential_rate", "Influential rate"),
        ("mean_harmful_magnitude", "Mean harmful magnitude"),
    )
    lookup = {
        (row["replacement_strategy"], row["modality"],
         row["speaker_relation"], row["distance_bin"]): row
        for row in rows if float(row["impact_eps"]) == primary_epsilon
    }
    color_limits = {}
    for metric, _ in metrics:
        if metric != "mean_harmful_magnitude":
            color_limits[metric] = (0.0, 1.0)
        else:
            observed = [
                float(row[metric]) for row in rows if row[metric] is not None
            ]
            color_limits[metric] = (
                0.0, max(max(observed), 1e-12) if observed else 1.0
            )
    relations = ("same_speaker", "cross_speaker")
    for strategy in STRATEGIES:
        for modality in MODALITIES:
            figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            for axis, (metric, title) in zip(axes.flat, metrics):
                matrix = np.full((2, len(DISTANCE_LABELS)), np.nan)
                for row_index, relation in enumerate(relations):
                    for column_index, band in enumerate(DISTANCE_LABELS):
                        row = lookup.get((strategy, modality, relation, band))
                        if row is not None and row[metric] is not None:
                            matrix[row_index, column_index] = float(row[metric])
                image = axis.imshow(
                    matrix,
                    aspect="auto",
                    cmap="magma",
                    vmin=color_limits[metric][0],
                    vmax=color_limits[metric][1],
                )
                axis.set_title(title)
                axis.set_xticks(range(len(DISTANCE_LABELS)), DISTANCE_LABELS)
                axis.set_yticks(range(2), ("Same", "Cross"))
                axis.set_xlabel("Absolute temporal distance")
                axis.set_ylabel("Speaker relation")
                for row_index in range(2):
                    for column_index in range(len(DISTANCE_LABELS)):
                        value = matrix[row_index, column_index]
                        if np.isfinite(value):
                            label = (
                                f"{100 * value:.1f}%"
                                if metric != "mean_harmful_magnitude" else f"{value:.3g}"
                            )
                            axis.text(
                                column_index, row_index, label,
                                ha="center", va="center", color="white", fontsize=9,
                            )
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            figure.suptitle(
                f"Structural harm — {strategy}, {modality}, epsilon={primary_epsilon:g}"
            )
            figure.savefig(
                figure_dir / f"structural_harm_heatmap_{strategy}_{modality}.png",
                dpi=180,
            )
            plt.close(figure)


def analyze(
    run_dir: Path,
    impact_eps: float = 1e-4,
    eps_values: Sequence[float] = (1e-6, 1e-4, 1e-3),
    bootstrap_reps: int = 2000,
    seed: int = 2024,
    generate_figures: bool = True,
) -> Path:
    epsilons = tuple(sorted(set(float(value) for value in (*eps_values, impact_eps))))
    if any(value < 0 for value in epsilons):
        raise ValueError("Impact thresholds must be nonnegative")
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")

    run_dir = Path(run_dir).expanduser().resolve()
    result_dir = run_dir / "results"
    paths = resolve_impact_paths(run_dir)
    previous_same_speaker = _speaker_sequences(paths)
    relationship = defaultdict(lambda: defaultdict(Totals))
    speaker_distance = defaultdict(lambda: defaultdict(Totals))
    same_speaker_all = defaultdict(lambda: defaultdict(Totals))
    same_speaker_prior = defaultdict(lambda: defaultdict(Totals))
    previous_same = defaultdict(lambda: defaultdict(Totals))
    previous_transitions = defaultdict(lambda: defaultdict(Totals))
    within_target = defaultdict(Totals)
    conversations = set()

    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require_columns(path, reader.fieldnames or [], REQUIRED)
            for line_number, row in enumerate(reader, start=2):
                strategy = row["replacement_strategy"]
                modality = row["modality"]
                if strategy not in STRATEGIES or modality not in MODALITIES:
                    raise ValueError(f"{path}:{line_number}: invalid strategy/modality")
                conversation_id = row["conversation_id"]
                conversations.add(conversation_id)
                source_idx = int(row["source_idx"])
                target_idx = int(row["target_idx"])
                distance = int(row["absolute_distance"])
                if distance < 1:
                    raise ValueError(f"{path}:{line_number}: invalid cross distance")
                same_speaker = binary_int(row, "same_speaker")
                source_before = binary_int(row, "source_before_target")
                source_emotion = int(row["source_emotion"])
                target_emotion = int(row["target_emotion"])
                impact = finite_float(row, "signed_impact")
                relation = "same_speaker" if same_speaker else "cross_speaker"
                emotion_relation = (
                    "same_emotion" if source_emotion == target_emotion
                    else "different_emotion"
                )
                band = distance_bin(distance)
                is_previous_same = (
                    previous_same_speaker.get((conversation_id, target_idx)) == source_idx
                )
                for epsilon in epsilons:
                    relationship[
                        (epsilon, strategy, relation, band, emotion_relation, modality)
                    ][conversation_id].add(impact, epsilon)
                    speaker_distance[
                        (epsilon, strategy, modality, relation, band)
                    ][conversation_id].add(impact, epsilon)
                    within_target[
                        (epsilon, strategy, modality, band, conversation_id,
                         target_idx, relation)
                    ].add(impact, epsilon)
                    if same_speaker:
                        key = (epsilon, strategy, modality, emotion_relation)
                        same_speaker_all[key][conversation_id].add(impact, epsilon)
                        if source_before:
                            same_speaker_prior[key][conversation_id].add(impact, epsilon)
                        if is_previous_same:
                            previous_same[key][conversation_id].add(impact, epsilon)
                            previous_transitions[
                                (epsilon, strategy, modality,
                                 str(source_emotion), str(target_emotion))
                            ][conversation_id].add(impact, epsilon)

    ordered_conversations = sorted(conversations)
    rng = np.random.default_rng(seed)
    relationship_rows = _summary_rows(
        relationship,
        ("impact_eps", "replacement_strategy", "speaker_relation",
         "distance_bin", "emotion_relation", "modality"),
        ordered_conversations, bootstrap_reps, rng,
    )
    speaker_distance_rows = _summary_rows(
        speaker_distance,
        ("impact_eps", "replacement_strategy", "modality",
         "speaker_relation", "distance_bin"),
        ordered_conversations, bootstrap_reps, rng,
    )
    same_all_rows = _summary_rows(
        same_speaker_all,
        ("impact_eps", "replacement_strategy", "modality", "emotion_relation"),
        ordered_conversations, bootstrap_reps, rng,
    )
    prior_rows = _summary_rows(
        same_speaker_prior,
        ("impact_eps", "replacement_strategy", "modality", "emotion_relation"),
        ordered_conversations, bootstrap_reps, rng,
    )
    previous_rows = _summary_rows(
        previous_same,
        ("impact_eps", "replacement_strategy", "modality", "emotion_relation"),
        ordered_conversations, bootstrap_reps, rng,
    )
    transition_rows = _summary_rows(
        previous_transitions,
        ("impact_eps", "replacement_strategy", "modality",
         "source_emotion", "target_emotion"),
        ordered_conversations, bootstrap_reps, rng,
    )
    contrast_rows = [
        *_emotion_contrasts(
            same_speaker_all, "all_same_speaker_pairs", epsilons,
            ordered_conversations, bootstrap_reps, rng,
        ),
        *_emotion_contrasts(
            same_speaker_prior, "all_previous_same_speaker_sources", epsilons,
            ordered_conversations, bootstrap_reps, rng,
        ),
        *_emotion_contrasts(
            previous_same, "previous_utterance_in_same_speaker_sequence", epsilons,
            ordered_conversations, bootstrap_reps, rng,
        ),
    ]
    strategy_rows = _strategy_contrasts(
        speaker_distance, epsilons, ordered_conversations,
        bootstrap_reps, rng,
    )
    within_details, within_summary = _within_target_outputs(
        within_target, impact_eps, ordered_conversations, bootstrap_reps, rng
    )

    primary_relationship = [
        row for row in relationship_rows if float(row["impact_eps"]) == impact_eps
    ]
    primary_speaker_distance = [
        row for row in speaker_distance_rows
        if float(row["impact_eps"]) == impact_eps
    ]
    write_rows(result_dir / "experiment2_relationship_impact.csv", primary_relationship)
    write_rows(result_dir / "experiment2_harm_sensitivity.csv", relationship_rows)
    write_rows(
        result_dir / "experiment2_speaker_distance_harm.csv",
        speaker_distance_rows,
    )
    write_rows(
        result_dir / "experiment2_same_speaker_emotion_relation.csv", same_all_rows
    )
    write_rows(
        result_dir / "experiment2_same_speaker_prior_emotion_relation.csv", prior_rows
    )
    write_rows(
        result_dir / "experiment2_previous_same_speaker_transition.csv", previous_rows
    )
    write_rows(
        result_dir / "experiment2_previous_same_speaker_emotion_pairs.csv",
        transition_rows,
    )
    if contrast_rows:
        write_rows(result_dir / "experiment2_emotion_shift_contrasts.csv", contrast_rows)
    if strategy_rows:
        write_rows(result_dir / "experiment2_strategy_robustness.csv", strategy_rows)
    if within_details:
        write_rows(
            result_dir / "experiment2_within_target_matched_details.csv",
            within_details,
        )
        write_rows(
            result_dir / "experiment2_within_target_matched_summary.csv",
            within_summary,
        )
    if generate_figures:
        _plot_heatmaps(
            primary_speaker_distance,
            run_dir / "figures" / "structural_harm",
            impact_eps,
        )
    print(
        f"Structural-harm analysis complete: {len(primary_relationship)} primary "
        f"groups, {len(epsilons)} epsilon values; outputs={result_dir}",
        flush=True,
    )
    return result_dir


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return analyze(
        args.run_dir,
        args.impact_eps,
        args.eps_values,
        args.bootstrap_reps,
        args.seed,
    )


if __name__ == "__main__":
    main()
