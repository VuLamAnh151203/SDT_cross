"""Shared helpers for Experiments 2--4.

This module intentionally depends only on the Python standard library and
NumPy.  Ground-truth labels are exposed to offline diagnostic analyses, but
Experiment 4 explicitly excludes them from predictor inputs.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


STRATEGIES = ("zero", "train_mean")
MODALITIES = ("text", "audio", "visual")


def finite_float(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {row[name]!r}")
    return value


def binary_int(row: Mapping[str, str], name: str) -> int:
    value = int(row[name])
    if value not in (0, 1):
        raise ValueError(f"Expected binary {name}, got {row[name]!r}")
    return value


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def resolve_impact_paths(run_dir: Path) -> List[Path]:
    result_dir = Path(run_dir).expanduser().resolve() / "results"
    paths = [
        result_dir / f"impact_pairs_{strategy}.csv" for strategy in STRATEGIES
    ]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No impact_pairs_zero.csv or impact_pairs_train_mean.csv in {result_dir}"
        )
    return existing


def require_columns(path: Path, fieldnames: Iterable[str], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def classification_scores(
    labels_and_predictions: Iterable[Tuple[int, int]],
) -> Tuple[float, float, float]:
    confusion = Counter(labels_and_predictions)
    count = sum(confusion.values())
    if count == 0:
        raise ValueError("Cannot calculate classification metrics for no targets")
    labels = sorted(
        {target for target, _ in confusion} | {prediction for _, prediction in confusion}
    )
    correct = sum(confusion.get((label, label), 0) for label in labels)
    supports: Dict[int, int] = {
        label: sum(
            value for (target, _), value in confusion.items() if target == label
        )
        for label in labels
    }
    f1_values = []
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
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    accuracy = 100.0 * correct / count
    weighted_f1 = 100.0 * sum(
        score * supports[label] for label, score in zip(labels, f1_values)
    ) / count
    macro_f1 = 100.0 * sum(f1_values) / len(f1_values)
    return accuracy, weighted_f1, macro_f1
