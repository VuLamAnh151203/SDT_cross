"""Analyze which source-target interventions change the predicted class."""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


STRATEGIES = ("zero", "train_mean")
MODALITIES = ("text", "audio", "visual")
IMPACT_REQUIRED = {
    "conversation_id", "source_idx", "target_idx", "source_speaker",
    "target_speaker", "same_speaker", "absolute_distance",
    "source_before_target", "modality", "replacement_strategy",
    "source_emotion", "target_emotion", "signed_impact",
    "prob_full_true", "prob_mask_true", "delta_true_class_prob",
    "pred_full", "pred_mask", "correct_full", "correct_mask",
    "changed_prediction", "js_divergence", "source_feature_norm",
    "replacement_delta_l2",
}


@dataclass(frozen=True)
class BaselineTarget:
    label: int
    prediction: int
    correct: int
    top1_probability: float
    margin: float
    entropy: float


@dataclass
class GroupStats:
    count: int = 0
    changed: int = 0
    correct_to_wrong: int = 0
    wrong_to_correct: int = 0
    wrong_to_wrong_changed: int = 0
    sum_abs_impact: float = 0.0
    sum_js: float = 0.0
    sum_margin: float = 0.0
    sum_entropy: float = 0.0
    changed_abs_impacts: List[float] = field(default_factory=list)
    changed_js: List[float] = field(default_factory=list)
    changed_margins: List[float] = field(default_factory=list)
    changed_entropies: List[float] = field(default_factory=list)

    def add(
        self,
        changed: bool,
        correct_full: bool,
        correct_mask: bool,
        absolute_impact: float,
        js_divergence: float,
        margin: float,
        entropy: float,
    ) -> None:
        self.count += 1
        self.sum_abs_impact += absolute_impact
        self.sum_js += js_divergence
        self.sum_margin += margin
        self.sum_entropy += entropy
        if not changed:
            return
        self.changed += 1
        self.changed_abs_impacts.append(absolute_impact)
        self.changed_js.append(js_divergence)
        self.changed_margins.append(margin)
        self.changed_entropies.append(entropy)
        if correct_full and not correct_mask:
            self.correct_to_wrong += 1
        elif not correct_full and correct_mask:
            self.wrong_to_correct += 1
        elif not correct_full and not correct_mask:
            self.wrong_to_wrong_changed += 1


@dataclass
class TargetStats:
    source_count: int = 0
    flip_count: int = 0
    same_speaker_source_count: int = 0
    same_speaker_flip_count: int = 0
    source_before_count: int = 0
    source_before_flip_count: int = 0
    minimum_flip_distance: int | None = None
    maximum_abs_flip_impact: float = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum changed cases retained per strategy and modality",
    )
    return parser.parse_args(argv)


def _float(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {row[name]!r}")
    return value


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _distance_bin(distance: int) -> str:
    if distance <= 5:
        return "01-05"
    if distance <= 10:
        return "06-10"
    if distance <= 20:
        return "11-20"
    if distance <= 40:
        return "21-40"
    if distance <= 60:
        return "41-60"
    return "61+"


def _margin_bin(margin: float) -> str:
    if margin < 0.01:
        return "[0.00,0.01)"
    if margin < 0.02:
        return "[0.01,0.02)"
    if margin < 0.05:
        return "[0.02,0.05)"
    if margin < 0.10:
        return "[0.05,0.10)"
    if margin < 0.20:
        return "[0.10,0.20)"
    if margin < 0.40:
        return "[0.20,0.40)"
    return "[0.40,1.00]"


def _load_baseline(path: Path) -> Dict[Tuple[str, int], BaselineTarget]:
    if not path.is_file():
        raise FileNotFoundError(path)
    targets: Dict[Tuple[str, int], BaselineTarget] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        probability_columns = sorted(
            (name for name in fieldnames if name.startswith("prob_") and name != "prob_true"),
            key=lambda name: int(name.split("_")[1]),
        )
        required = {"conversation_id", "utterance_index", "label", "prediction", "correct"}
        missing = sorted(required - set(fieldnames))
        if missing or len(probability_columns) < 2:
            raise ValueError(
                f"Invalid baseline file; missing={missing}, "
                f"probability_columns={probability_columns}"
            )
        for line_number, row in enumerate(reader, start=2):
            key = (row["conversation_id"], int(row["utterance_index"]))
            probabilities = [_float(row, name) for name in probability_columns]
            ranked = sorted(probabilities, reverse=True)
            top1_probability = ranked[0]
            margin = ranked[0] - ranked[1]
            entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
            target = BaselineTarget(
                label=int(row["label"]),
                prediction=int(row["prediction"]),
                correct=int(row["correct"]),
                top1_probability=top1_probability,
                margin=margin,
                entropy=entropy,
            )
            if key in targets:
                raise ValueError(f"Duplicate baseline target at line {line_number}: {key}")
            targets[key] = target
    if not targets:
        raise ValueError("Baseline predictions are empty")
    return targets


def _rate_rows(
    counts: Mapping[Tuple[str, ...], List[int]], names: Sequence[str]
) -> List[Mapping[str, object]]:
    rows = []
    for key in sorted(counts):
        total, changed = counts[key]
        row = dict(zip(names, key))
        row.update({
            "count": total,
            "changed_prediction_count": changed,
            "changed_prediction_fraction": changed / total,
        })
        rows.append(row)
    return rows


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def analyze(run_dir: Path, top_k: int = 100) -> Path:
    if top_k < 1:
        raise ValueError("--top-k must be positive")
    run_dir = run_dir.expanduser().resolve()
    result_dir = run_dir / "results"
    baseline = _load_baseline(run_dir / "baseline" / "baseline_predictions.csv")

    groups: Dict[Tuple[str, str], GroupStats] = defaultdict(GroupStats)
    distance_counts = defaultdict(lambda: [0, 0])
    speaker_counts = defaultdict(lambda: [0, 0])
    direction_counts = defaultdict(lambda: [0, 0])
    target_emotion_counts = defaultdict(lambda: [0, 0])
    source_emotion_counts = defaultdict(lambda: [0, 0])
    margin_counts = defaultdict(lambda: [0, 0])
    transition_counts = Counter()
    target_stats: Dict[Tuple[str, str, str, int], TargetStats] = defaultdict(TargetStats)
    changed_pairs: Dict[str, Dict[Tuple[str, int, int, str], int]] = {
        strategy: {} for strategy in STRATEGIES
    }
    pair_counts = Counter()
    top_heaps: Dict[Tuple[str, str], list] = defaultdict(list)
    heap_serial = 0

    for strategy in STRATEGIES:
        path = result_dir / f"impact_pairs_{strategy}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(IMPACT_REQUIRED - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                row_strategy = row["replacement_strategy"]
                if row_strategy != strategy:
                    raise ValueError(
                        f"{path}:{line_number}: expected {strategy}, got {row_strategy}"
                    )
                modality = row["modality"]
                if modality not in MODALITIES:
                    raise ValueError(f"{path}:{line_number}: invalid modality {modality}")
                conversation_id = row["conversation_id"]
                source_idx = int(row["source_idx"])
                target_idx = int(row["target_idx"])
                baseline_key = (conversation_id, target_idx)
                if baseline_key not in baseline:
                    raise ValueError(f"Missing baseline target {baseline_key}")
                baseline_target = baseline[baseline_key]
                target_emotion = int(row["target_emotion"])
                pred_full = int(row["pred_full"])
                pred_mask = int(row["pred_mask"])
                if (
                    target_emotion != baseline_target.label
                    or pred_full != baseline_target.prediction
                ):
                    raise ValueError(
                        f"{path}:{line_number}: baseline mismatch for {baseline_key}"
                    )

                changed = bool(int(row["changed_prediction"]))
                if changed != (pred_full != pred_mask):
                    raise ValueError(f"{path}:{line_number}: inconsistent changed_prediction")
                correct_full = bool(int(row["correct_full"]))
                correct_mask = bool(int(row["correct_mask"]))
                same_speaker = bool(int(row["same_speaker"]))
                source_before = bool(int(row["source_before_target"]))
                distance = int(row["absolute_distance"])
                signed_impact = _float(row, "signed_impact")
                js_divergence = _float(row, "js_divergence")
                absolute_impact = abs(signed_impact)

                group_key = (strategy, modality)
                groups[group_key].add(
                    changed, correct_full, correct_mask, absolute_impact,
                    js_divergence, baseline_target.margin, baseline_target.entropy,
                )
                rate_keys = (
                    (distance_counts, group_key + (_distance_bin(distance),)),
                    (speaker_counts, group_key + (
                        "same_speaker" if same_speaker else "cross_speaker",
                    )),
                    (direction_counts, group_key + (
                        "source_before_target" if source_before else "source_after_target",
                    )),
                    (target_emotion_counts, group_key + (str(target_emotion),)),
                    (source_emotion_counts, group_key + (row["source_emotion"],)),
                    (margin_counts, group_key + (_margin_bin(baseline_target.margin),)),
                )
                for counter, key in rate_keys:
                    counter[key][0] += 1
                    counter[key][1] += int(changed)

                target_key = (strategy, modality, conversation_id, target_idx)
                target = target_stats[target_key]
                target.source_count += 1
                target.same_speaker_source_count += int(same_speaker)
                target.source_before_count += int(source_before)
                if changed:
                    target.flip_count += 1
                    target.same_speaker_flip_count += int(same_speaker)
                    target.source_before_flip_count += int(source_before)
                    target.minimum_flip_distance = (
                        distance if target.minimum_flip_distance is None
                        else min(target.minimum_flip_distance, distance)
                    )
                    target.maximum_abs_flip_impact = max(
                        target.maximum_abs_flip_impact, absolute_impact
                    )
                    transition = (
                        "correct_to_wrong" if correct_full and not correct_mask
                        else "wrong_to_correct" if not correct_full and correct_mask
                        else "wrong_to_wrong"
                    )
                    transition_counts[
                        (strategy, modality, str(target_emotion), str(pred_full),
                         str(pred_mask), transition)
                    ] += 1
                    exact_key = (conversation_id, source_idx, target_idx, modality)
                    if exact_key in changed_pairs[strategy]:
                        raise ValueError(f"Duplicate changed pair: {strategy} {exact_key}")
                    changed_pairs[strategy][exact_key] = pred_mask

                    case = {
                        "replacement_strategy": strategy,
                        "modality": modality,
                        "conversation_id": conversation_id,
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "source_speaker": row["source_speaker"],
                        "target_speaker": row["target_speaker"],
                        "same_speaker": int(same_speaker),
                        "source_before_target": int(source_before),
                        "absolute_distance": distance,
                        "source_emotion": row["source_emotion"],
                        "target_emotion": target_emotion,
                        "pred_full": pred_full,
                        "pred_mask": pred_mask,
                        "correct_full": int(correct_full),
                        "correct_mask": int(correct_mask),
                        "baseline_top1_probability": baseline_target.top1_probability,
                        "baseline_margin": baseline_target.margin,
                        "baseline_entropy": baseline_target.entropy,
                        "signed_impact": signed_impact,
                        "delta_true_class_prob": _float(row, "delta_true_class_prob"),
                        "js_divergence": js_divergence,
                        "source_feature_norm": _float(row, "source_feature_norm"),
                        "replacement_delta_l2": _float(row, "replacement_delta_l2"),
                    }
                    heap_serial += 1
                    heap = top_heaps[group_key]
                    item = (js_divergence, absolute_impact, heap_serial, case)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item[:3] > heap[0][:3]:
                        heapq.heapreplace(heap, item)
                pair_counts[group_key] += 1

    for modality in MODALITIES:
        zero_count = pair_counts[("zero", modality)]
        mean_count = pair_counts[("train_mean", modality)]
        if zero_count != mean_count:
            raise ValueError(
                f"Pair counts differ for {modality}: zero={zero_count}, mean={mean_count}"
            )

    summary_rows = []
    for strategy in STRATEGIES:
        for modality in MODALITIES:
            item = groups[(strategy, modality)]
            changed = max(item.changed, 1)
            unchanged = item.count - item.changed
            changed_abs_sum = sum(item.changed_abs_impacts)
            changed_js_sum = sum(item.changed_js)
            changed_margin_sum = sum(item.changed_margins)
            changed_entropy_sum = sum(item.changed_entropies)
            summary_rows.append({
                "replacement_strategy": strategy,
                "modality": modality,
                "count": item.count,
                "changed_prediction_count": item.changed,
                "changed_prediction_fraction": item.changed / item.count,
                "correct_to_wrong": item.correct_to_wrong,
                "wrong_to_correct": item.wrong_to_correct,
                "wrong_to_wrong_changed": item.wrong_to_wrong_changed,
                "mean_abs_signed_impact_all": item.sum_abs_impact / item.count,
                "mean_abs_signed_impact_changed": (
                    changed_abs_sum / changed if item.changed else None
                ),
                "mean_abs_signed_impact_unchanged": (
                    (item.sum_abs_impact - changed_abs_sum) / unchanged
                    if unchanged else None
                ),
                "median_abs_signed_impact_changed": _median(item.changed_abs_impacts),
                "mean_js_divergence_all": item.sum_js / item.count,
                "mean_js_divergence_changed": (
                    changed_js_sum / changed if item.changed else None
                ),
                "mean_js_divergence_unchanged": (
                    (item.sum_js - changed_js_sum) / unchanged if unchanged else None
                ),
                "median_js_divergence_changed": _median(item.changed_js),
                "mean_baseline_margin_all": item.sum_margin / item.count,
                "mean_baseline_margin_changed": (
                    changed_margin_sum / changed if item.changed else None
                ),
                "mean_baseline_margin_unchanged": (
                    (item.sum_margin - changed_margin_sum) / unchanged
                    if unchanged else None
                ),
                "median_baseline_margin_changed": _median(item.changed_margins),
                "mean_baseline_entropy_all": item.sum_entropy / item.count,
                "mean_baseline_entropy_changed": (
                    changed_entropy_sum / changed if item.changed else None
                ),
                "mean_baseline_entropy_unchanged": (
                    (item.sum_entropy - changed_entropy_sum) / unchanged
                    if unchanged else None
                ),
                "median_baseline_entropy_changed": _median(item.changed_entropies),
            })

    transition_rows = []
    for key in sorted(transition_counts):
        strategy, modality, target_emotion, pred_full, pred_mask, transition = key
        transition_rows.append({
            "replacement_strategy": strategy,
            "modality": modality,
            "target_emotion": target_emotion,
            "pred_full": pred_full,
            "pred_mask": pred_mask,
            "correctness_transition": transition,
            "count": transition_counts[key],
        })

    target_rows = []
    for key in sorted(target_stats):
        strategy, modality, conversation_id, target_idx = key
        item = target_stats[key]
        base = baseline[(conversation_id, target_idx)]
        target_rows.append({
            "replacement_strategy": strategy,
            "modality": modality,
            "conversation_id": conversation_id,
            "target_idx": target_idx,
            "target_emotion": base.label,
            "pred_full": base.prediction,
            "correct_full": base.correct,
            "baseline_top1_probability": base.top1_probability,
            "baseline_margin": base.margin,
            "baseline_entropy": base.entropy,
            "source_count": item.source_count,
            "flip_count": item.flip_count,
            "flip_fraction": item.flip_count / item.source_count,
            "same_speaker_source_count": item.same_speaker_source_count,
            "same_speaker_flip_count": item.same_speaker_flip_count,
            "source_before_count": item.source_before_count,
            "source_before_flip_count": item.source_before_flip_count,
            "minimum_flip_distance": item.minimum_flip_distance,
            "maximum_abs_flip_impact": item.maximum_abs_flip_impact,
        })

    robustness_rows = []
    zero_changed = changed_pairs["zero"]
    mean_changed = changed_pairs["train_mean"]
    for modality in MODALITIES:
        zero_keys = {key for key in zero_changed if key[3] == modality}
        mean_keys = {key for key in mean_changed if key[3] == modality}
        shared = zero_keys & mean_keys
        categories = Counter({
            "zero_only": len(zero_keys - mean_keys),
            "train_mean_only": len(mean_keys - zero_keys),
            "both_same_new_prediction": sum(
                zero_changed[key] == mean_changed[key] for key in shared
            ),
            "both_different_new_prediction": sum(
                zero_changed[key] != mean_changed[key] for key in shared
            ),
        })
        categories["neither"] = pair_counts[("zero", modality)] - len(
            zero_keys | mean_keys
        )
        for category in (
            "neither", "zero_only", "train_mean_only",
            "both_same_new_prediction", "both_different_new_prediction",
        ):
            robustness_rows.append({
                "modality": modality,
                "category": category,
                "count": categories[category],
                "fraction": categories[category] / pair_counts[("zero", modality)],
            })

    top_rows = []
    for strategy in STRATEGIES:
        for modality in MODALITIES:
            ordered = sorted(top_heaps[(strategy, modality)], reverse=True)
            for rank, (_, _, _, case) in enumerate(ordered, start=1):
                top_rows.append({"rank_within_strategy_modality": rank, **case})

    _write_rows(result_dir / "changed_prediction_summary.csv", summary_rows)
    _write_rows(
        result_dir / "flip_rate_by_distance.csv",
        _rate_rows(distance_counts, ("replacement_strategy", "modality", "distance_bin")),
    )
    _write_rows(
        result_dir / "flip_rate_by_speaker_relation.csv",
        _rate_rows(speaker_counts, ("replacement_strategy", "modality", "speaker_relation")),
    )
    _write_rows(
        result_dir / "flip_rate_by_source_direction.csv",
        _rate_rows(direction_counts, ("replacement_strategy", "modality", "source_direction")),
    )
    _write_rows(
        result_dir / "flip_rate_by_target_emotion.csv",
        _rate_rows(target_emotion_counts, ("replacement_strategy", "modality", "target_emotion")),
    )
    _write_rows(
        result_dir / "flip_rate_by_source_emotion.csv",
        _rate_rows(source_emotion_counts, ("replacement_strategy", "modality", "source_emotion")),
    )
    _write_rows(
        result_dir / "flip_rate_by_baseline_margin.csv",
        _rate_rows(margin_counts, ("replacement_strategy", "modality", "baseline_margin_bin")),
    )
    _write_rows(result_dir / "flip_class_transitions.csv", transition_rows)
    _write_rows(result_dir / "flip_target_susceptibility.csv", target_rows)
    _write_rows(result_dir / "flip_robustness_zero_vs_mean.csv", robustness_rows)
    _write_rows(result_dir / "top_changed_cases.csv", top_rows)

    print(
        f"Prediction-change analysis complete: {result_dir} "
        f"({sum(item.changed for item in groups.values())} changed rows)",
        flush=True,
    )
    return result_dir


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return analyze(args.run_dir, args.top_k)


if __name__ == "__main__":
    main()
