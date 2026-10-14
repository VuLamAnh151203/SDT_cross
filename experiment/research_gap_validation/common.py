"""Shared utilities for research-gap validation experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "visual")
SHORT_MODALITIES = ("t", "a", "v")
EPS_VALUES = (1e-6, 1e-4, 1e-3)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    extras = sorted(set().union(*(set(row) for row in rows)).difference(fieldnames))
    fieldnames.extend(extras)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(rows: Sequence[Mapping[str, str]], required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def finite(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {row[name]!r}")
    return value


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    if not labels or len(labels) != len(predictions):
        raise ValueError("Expected equally sized non-empty labels and predictions")
    confusion = Counter(zip(labels, predictions))
    classes = sorted(set(labels) | set(predictions))
    f1s, supports = [], []
    for label in classes:
        tp = confusion[(label, label)]
        fp = sum(n for (truth, pred), n in confusion.items()
                 if pred == label and truth != label)
        fn = sum(n for (truth, pred), n in confusion.items()
                 if truth == label and pred != label)
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else 2 * tp / denominator)
        supports.append(sum(n for (truth, _), n in confusion.items() if truth == label))
    count = len(labels)
    return {
        "accuracy": 100.0 * sum(t == p for t, p in zip(labels, predictions)) / count,
        "weighted_f1": 100.0 * sum(f * n for f, n in zip(f1s, supports)) / count,
        "macro_f1": 100.0 * sum(f1s) / len(f1s),
    }


def percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if not len(array):
        return math.nan, math.nan
    low, high = np.quantile(array, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap(
    rows: Sequence[Mapping[str, object]], statistic: Callable[[Sequence[Mapping[str, object]]], float],
    repetitions: int, seed: int, cluster: str = "conversation_id",
) -> tuple[float, float]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster])].append(row)
    names = sorted(grouped)
    if not names or repetitions <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(names, len(names), replace=True)
        sample = [row for name in sampled for row in grouped[str(name)]]
        try:
            estimates.append(float(statistic(sample)))
        except (ValueError, ZeroDivisionError):
            continue
    return percentile_ci(estimates)


def session_id(conversation_id: str) -> str:
    if conversation_id.startswith("Ses") and len(conversation_id) >= 5:
        return conversation_id[:5]
    return conversation_id.split("_")[0]


def distance_bin(distance: int) -> str:
    if distance <= 5:
        return "01-05"
    if distance <= 10:
        return "06-10"
    if distance <= 20:
        return "11-20"
    return "21+"


def margin_bin(margin: float) -> str:
    if margin < 0.05:
        return "low_<0.05"
    if margin < 0.20:
        return "middle_0.05-0.20"
    return "high_>=0.20"

