"""Compute pair-weighted classification metrics from impact-pair CSV files.

Each target utterance occurs once for every ablated source utterance.  The
metrics produced here therefore describe intervention pairs; they are not the
ordinary utterance-level test metrics written by Task A.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REQUIRED_COLUMNS = {
    "replacement_strategy",
    "modality",
    "target_emotion",
    "pred_full",
    "pred_mask",
}

TEST_METRIC_COLUMNS = {
    "conversation_id",
    "conversation_length",
    "target_idx",
    "target_emotion",
    "pred_full",
}

OUTPUT_FIELDS = [
    "replacement_strategy",
    "modality",
    "count",
    "baseline_correct",
    "ablation_correct",
    "baseline_pair_weighted_accuracy",
    "ablation_pair_weighted_accuracy",
    "delta_accuracy",
    "baseline_pair_weighted_wf1",
    "ablation_pair_weighted_wf1",
    "delta_wf1",
    "baseline_pair_weighted_macro_f1",
    "ablation_pair_weighted_macro_f1",
    "delta_macro_f1",
    "changed_prediction_count",
    "changed_prediction_fraction",
]


@dataclass
class ClassificationAccumulator:
    count: int = 0
    baseline_correct: int = 0
    ablation_correct: int = 0
    changed_prediction_count: int = 0
    baseline_confusion: Counter = field(default_factory=Counter)
    ablation_confusion: Counter = field(default_factory=Counter)

    def add(self, target: int, baseline_prediction: int, ablation_prediction: int) -> None:
        self.count += 1
        self.baseline_correct += int(target == baseline_prediction)
        self.ablation_correct += int(target == ablation_prediction)
        self.changed_prediction_count += int(baseline_prediction != ablation_prediction)
        self.baseline_confusion[(target, baseline_prediction)] += 1
        self.ablation_confusion[(target, ablation_prediction)] += 1


def _classification_scores(
    confusion: Mapping[Tuple[int, int], int], count: int
) -> Tuple[float, float, float]:
    if count <= 0:
        raise ValueError("Cannot calculate metrics for an empty group")

    labels = sorted(
        {target for target, _ in confusion} | {prediction for _, prediction in confusion}
    )
    supports = {
        label: sum(value for (target, _), value in confusion.items() if target == label)
        for label in labels
    }
    correct = sum(confusion.get((label, label), 0) for label in labels)
    f1_scores = []
    for label in labels:
        true_positive = confusion.get((label, label), 0)
        false_positive = sum(
            value
            for (target, prediction), value in confusion.items()
            if prediction == label and target != label
        )
        false_negative = sum(
            value
            for (target, prediction), value in confusion.items()
            if target == label and prediction != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    accuracy = 100.0 * correct / count
    weighted_f1 = 100.0 * sum(
        score * supports[label] for label, score in zip(labels, f1_scores)
    ) / count
    macro_f1 = 100.0 * sum(f1_scores) / len(f1_scores)
    return accuracy, weighted_f1, macro_f1


def calculate_metrics(paths: Iterable[Path]) -> List[Dict[str, object]]:
    accumulators: Dict[Tuple[str, str], ClassificationAccumulator] = defaultdict(
        ClassificationAccumulator
    )
    found_path = False

    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        found_path = True
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")

            for line_number, row in enumerate(reader, start=2):
                strategy = row["replacement_strategy"].strip()
                modality = row["modality"].strip()
                if not strategy or not modality:
                    raise ValueError(
                        f"{path}:{line_number}: empty strategy or modality"
                    )
                try:
                    target = int(row["target_emotion"])
                    baseline_prediction = int(row["pred_full"])
                    ablation_prediction = int(row["pred_mask"])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid class identifier"
                    ) from error
                accumulators[(strategy, modality)].add(
                    target, baseline_prediction, ablation_prediction
                )

    if not found_path:
        raise ValueError("No impact-pair CSV was provided")
    if not accumulators:
        raise ValueError("The input CSV files contain no data rows")

    strategy_order = {"zero": 0, "train_mean": 1}
    modality_order = {"text": 0, "audio": 1, "visual": 2}
    ordered_keys = sorted(
        accumulators,
        key=lambda item: (
            strategy_order.get(item[0], len(strategy_order)),
            item[0],
            modality_order.get(item[1], len(modality_order)),
            item[1],
        ),
    )

    rows: List[Dict[str, object]] = []
    for strategy, modality in ordered_keys:
        accumulator = accumulators[(strategy, modality)]
        baseline_accuracy, baseline_wf1, baseline_macro_f1 = _classification_scores(
            accumulator.baseline_confusion, accumulator.count
        )
        ablation_accuracy, ablation_wf1, ablation_macro_f1 = _classification_scores(
            accumulator.ablation_confusion, accumulator.count
        )
        rows.append({
            "replacement_strategy": strategy,
            "modality": modality,
            "count": accumulator.count,
            "baseline_correct": accumulator.baseline_correct,
            "ablation_correct": accumulator.ablation_correct,
            "baseline_pair_weighted_accuracy": baseline_accuracy,
            "ablation_pair_weighted_accuracy": ablation_accuracy,
            "delta_accuracy": ablation_accuracy - baseline_accuracy,
            "baseline_pair_weighted_wf1": baseline_wf1,
            "ablation_pair_weighted_wf1": ablation_wf1,
            "delta_wf1": ablation_wf1 - baseline_wf1,
            "baseline_pair_weighted_macro_f1": baseline_macro_f1,
            "ablation_pair_weighted_macro_f1": ablation_macro_f1,
            "delta_macro_f1": ablation_macro_f1 - baseline_macro_f1,
            "changed_prediction_count": accumulator.changed_prediction_count,
            "changed_prediction_fraction": (
                accumulator.changed_prediction_count / accumulator.count
            ),
        })
    return rows


def calculate_test_metrics(paths: Iterable[Path]) -> Dict[str, object]:
    """Recover ordinary test metrics by keeping one baseline row per target."""
    targets: Dict[Tuple[str, int], Tuple[int, int]] = {}
    conversation_lengths: Dict[str, int] = {}

    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = sorted(TEST_METRIC_COLUMNS - columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")

            for line_number, row in enumerate(reader, start=2):
                conversation_id = row["conversation_id"].strip()
                try:
                    conversation_length = int(row["conversation_length"])
                    target_idx = int(row["target_idx"])
                    target = int(row["target_emotion"])
                    prediction = int(row["pred_full"])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid baseline target data"
                    ) from error

                previous_length = conversation_lengths.setdefault(
                    conversation_id, conversation_length
                )
                if previous_length != conversation_length:
                    raise ValueError(
                        f"Inconsistent length for conversation {conversation_id}"
                    )
                key = (conversation_id, target_idx)
                value = (target, prediction)
                previous = targets.setdefault(key, value)
                if previous != value:
                    raise ValueError(
                        f"Inconsistent label/pred_full for target {key}: "
                        f"{previous} versus {value}"
                    )

    if not targets:
        raise ValueError("The input CSV files contain no target predictions")

    for conversation_id, length in conversation_lengths.items():
        observed = {
            target_idx
            for item_conversation, target_idx in targets
            if item_conversation == conversation_id
        }
        expected = set(range(length))
        if observed != expected:
            missing = sorted(expected - observed)[:10]
            extra = sorted(observed - expected)[:10]
            raise ValueError(
                f"Incomplete targets for {conversation_id}: "
                f"missing={missing}, extra={extra}"
            )

    confusion = Counter(
        (target, prediction) for target, prediction in targets.values()
    )
    accuracy, weighted_f1, macro_f1 = _classification_scores(
        confusion, len(targets)
    )
    return {
        "source": "deduplicated_pred_full",
        "conversation_count": len(conversation_lengths),
        "utterance_count": len(targets),
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }


def write_metrics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_test_metrics(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing results/impact_pairs_{zero,train_mean}.csv",
    )
    source.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="One or more completed impact-pair CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: next to the input files in results/)",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        help="Deduplicated baseline test-metric CSV output",
    )
    return parser.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> Tuple[List[Path], Path, Path]:
    if args.run_dir is not None:
        result_dir = args.run_dir / "results"
        candidates = [
            result_dir / "impact_pairs_zero.csv",
            result_dir / "impact_pairs_train_mean.csv",
        ]
        paths = [path for path in candidates if path.is_file()]
        if not paths:
            raise FileNotFoundError(
                f"No impact_pairs_zero.csv or impact_pairs_train_mean.csv in {result_dir}"
            )
        output = args.output or result_dir / "ablation_classification_metrics.csv"
        test_output = args.test_output or result_dir / "test_classification_metrics.csv"
        return paths, output, test_output

    paths = list(args.input)
    output = args.output or paths[0].parent / "ablation_classification_metrics.csv"
    test_output = args.test_output or paths[0].parent / "test_classification_metrics.csv"
    return paths, output, test_output


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    paths, output, test_output = _resolve_paths(args)
    rows = calculate_metrics(paths)
    test_metrics = calculate_test_metrics(paths)
    write_metrics(output, rows)
    write_test_metrics(test_output, test_metrics)
    print(f"Wrote {len(rows)} metric rows to {output}", flush=True)
    print(
        f"Wrote standard metrics for {test_metrics['utterance_count']} utterances "
        f"to {test_output}",
        flush=True,
    )
    return output


if __name__ == "__main__":
    main()
