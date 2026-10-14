"""Shared, model-free utilities for Experiment 3 ETCR."""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


EPSILON = 1e-4
PRIMARY_STRATEGY = "train_mean"


def previous_same_speaker(speakers: Sequence[int]) -> list[int]:
    """Return the nearest strictly earlier index with the same speaker, or -1."""
    previous: dict[int, int] = {}
    result = []
    for index, speaker in enumerate(speakers):
        result.append(previous.get(int(speaker), -1))
        previous[int(speaker)] = index
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def require_columns(rows: Sequence[Mapping[str, str]], names: Iterable[str]) -> None:
    missing = sorted(set(names).difference(rows[0]))
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def as_bool(row: Mapping[str, str], name: str) -> bool:
    value = int(row[name])
    if value not in (0, 1):
        raise ValueError(f"{name} is not binary: {row[name]!r}")
    return bool(value)


def as_float(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} is non-finite: {row[name]!r}")
    return value


def probability_columns(row: Mapping[str, str], prefix: str) -> list[str]:
    columns = [name for name in row if name.startswith(prefix)]
    return sorted(columns, key=lambda name: int(name.rsplit("_", 1)[1]))


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Labels and predictions must have the same non-zero length")
    confusion = Counter(zip(labels, predictions))
    classes = sorted(set(labels) | set(predictions))
    correct = sum(confusion[(label, label)] for label in classes)
    f1s, supports = [], []
    for label in classes:
        tp = confusion[(label, label)]
        fp = sum(value for (truth, pred), value in confusion.items()
                 if pred == label and truth != label)
        fn = sum(value for (truth, pred), value in confusion.items()
                 if truth == label and pred != label)
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else 2 * tp / denominator)
        supports.append(sum(value for (truth, _), value in confusion.items() if truth == label))
    count = len(labels)
    return {
        "accuracy": 100.0 * correct / count,
        "weighted_f1": 100.0 * sum(f * n for f, n in zip(f1s, supports)) / count,
        "macro_f1": 100.0 * sum(f1s) / len(f1s),
    }


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    statistic: Callable[[Sequence[Mapping[str, object]]], float],
    repetitions: int,
    seed: int,
    cluster_name: str = "conversation_id",
) -> tuple[float, float]:
    """Percentile CI after resampling complete conversations with replacement."""
    if repetitions <= 0:
        return math.nan, math.nan
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_name])].append(row)
    clusters = sorted(grouped)
    if not clusters:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in chosen for row in grouped[str(cluster)]]
        try:
            estimates.append(float(statistic(sample)))
        except (ValueError, ZeroDivisionError):
            continue
    return percentile_interval(estimates)


def validate_target_rows(rows: Sequence[Mapping[str, str]], expected_strategy: str | None = None) -> None:
    require_columns(rows, {
        "split", "replacement_strategy", "conversation_id", "target_idx", "eligible",
        "target_label", "pred_full", "pred_mask", "nll_full", "nll_mask",
        "signed_impact", "baseline_margin",
    })
    keys = set()
    for row in rows:
        if expected_strategy and row["replacement_strategy"] != expected_strategy:
            raise ValueError(f"Unexpected strategy {row['replacement_strategy']!r}")
        key = (row["conversation_id"], int(row["target_idx"]))
        if key in keys:
            raise ValueError(f"Duplicate target row: {key}")
        keys.add(key)
        eligible = as_bool(row, "eligible")
        nll_full = as_float(row, "nll_full")
        nll_mask = as_float(row, "nll_mask")
        impact = as_float(row, "signed_impact")
        if not math.isclose(impact, nll_mask - nll_full, abs_tol=2e-6, rel_tol=2e-6):
            raise ValueError(f"Impact equation mismatch at {key}")
        if eligible:
            source_idx = int(row["source_idx"])
            target_idx = int(row["target_idx"])
            if not 0 <= source_idx < target_idx:
                raise ValueError(f"Invalid previous-source relation at {key}")
        elif int(row["pred_full"]) != int(row["pred_mask"]):
            raise ValueError(f"Ineligible target changed prediction at {key}")


def subset_name(row: Mapping[str, str]) -> str:
    if not as_bool(row, "eligible"):
        return "ineligible"
    return "continuity" if as_bool(row, "continuity") else "transition"

