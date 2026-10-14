"""Evaluate learned E3 gates against an independently generated validation teacher."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
SDT_ROOT = ROOT.parents[1]
for path in (ROOT, CROSS, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import read_csv, write_csv, write_json  # noqa: E402
from impact_core import (dataset_lookup, load_conversation, load_experiment,
                         seed_everything)  # noqa: E402
from e3_posttraining_audit import _load_wrapper  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--trained-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--precision-k", nargs="+", type=int, default=(1, 3, 5))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    ranked_left, ranked_right = _rank_average(left), _rank_average(right)
    if ranked_left.std() == 0.0 or ranked_right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return float("nan"), float("nan")
    return (float(finite.mean()),
            float(finite.std(ddof=1)) if len(finite) > 1 else 0.0)


def _teacher_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty validation teacher: {path}")
    bad = sorted({row.get("split", "") for row in rows if row.get("split") != split})
    if bad:
        raise ValueError(f"Teacher contains non-{split} rows: {bad}")
    required = {"conversation_id", "source_idx", "target_idx", "modality",
                "teacher_gate", "teacher_weight", "teacher_decision"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Teacher is missing columns: {sorted(missing)}")
    return rows


def evaluate(args: argparse.Namespace) -> None:
    if not args.precision_k or any(value < 1 for value in args.precision_k):
        raise ValueError("--precision-k values must be positive")
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    teacher_rows = _teacher_rows(args.teacher_file, args.split)
    split_ids = set(experiment.split_ids[args.split])
    teacher_ids = {str(row["conversation_id"]) for row in teacher_rows}
    if not teacher_ids.issubset(split_ids):
        raise ValueError("Validation teacher contains conversations outside validation split")
    ids = sorted(teacher_ids)
    if args.max_conversations:
        ids = ids[:args.max_conversations]
    selected_ids = set(ids)
    teacher_by_conversation = defaultdict(list)
    for row in teacher_rows:
        if row["conversation_id"] in selected_ids:
            teacher_by_conversation[row["conversation_id"]].append(row)

    lookup = dataset_lookup(experiment.dataset)
    summaries, precision_rows, prediction_rows = [], [], []
    for checkpoint_path in sorted(args.trained_checkpoints, key=lambda path: path.name):
        wrapper, artifact = _load_wrapper(experiment, checkpoint_path)
        seed_everything(int(artifact["seed"]))
        model_rows = []
        for number, conversation in enumerate(ids, start=1):
            batch = load_conversation(
                experiment.dataset, lookup, conversation, experiment.device
            )
            with torch.inference_mode():
                output = wrapper(
                    batch.text, batch.visual, batch.audio, batch.mask,
                    batch.speakers, batch.lengths,
                )
            pair = output["reliability_pair_gate"]
            modality = output["reliability_modality_gate"]
            for teacher in teacher_by_conversation[conversation]:
                source = int(teacher["source_idx"])
                target = int(teacher["target_idx"])
                modality_index = MODALITY_INDEX[str(teacher["modality"])]
                if pair is not None:
                    predicted_gate = float(pair[0, target, source, modality_index].cpu())
                elif modality is not None:
                    predicted_gate = float(modality[0, target, modality_index].cpu())
                else:
                    predicted_gate = 1.0
                row = {
                    "model": artifact["model_id"], "seed": artifact["seed"],
                    "split": args.split, "conversation_id": conversation,
                    "source_idx": source, "target_idx": target,
                    "modality": teacher["modality"],
                    "predicted_gate": predicted_gate,
                    "teacher_gate": float(teacher["teacher_gate"]),
                    "teacher_weight": float(teacher["teacher_weight"]),
                    "teacher_decision": teacher["teacher_decision"],
                }
                model_rows.append(row)
                prediction_rows.append(row)
            print(
                f"teacher fit {artifact['model_id']} seed={artifact['seed']} "
                f"{number}/{len(ids)}",
                flush=True,
            )

        effective = [row for row in model_rows if float(row["teacher_weight"]) > 0.0]
        if not effective:
            raise ValueError("Validation teacher has no positive-weight rows")
        predicted = np.asarray([float(row["predicted_gate"]) for row in effective])
        expected = np.asarray([float(row["teacher_gate"]) for row in effective])
        weights = np.asarray([float(row["teacher_weight"]) for row in effective])
        errors = (predicted - expected) ** 2
        summaries.append({
            "model": artifact["model_id"], "seed": artifact["seed"],
            "split": args.split, "pairs": len(effective),
            "weighted_mse": float(np.sum(errors * weights) / np.sum(weights)),
            "mse": float(np.mean(errors)),
            "spearman_gate": _spearman(predicted, expected),
            "mean_predicted_gate": float(np.mean(predicted)),
            "mean_teacher_gate": float(np.mean(expected)),
            "teacher_suppress_rate": float(np.mean([
                row["teacher_decision"] == "suppress" for row in effective
            ])),
        })

        groups = defaultdict(list)
        for row in effective:
            groups[(row["conversation_id"], int(row["target_idx"]),
                    row["modality"])].append(row)
        for k in sorted(set(args.precision_k)):
            values, positives = [], []
            for group in groups.values():
                ranked = sorted(
                    group,
                    key=lambda row: (float(row["predicted_gate"]), int(row["source_idx"])),
                )
                count = min(k, len(ranked))
                if not count:
                    continue
                values.append(float(np.mean([
                    row["teacher_decision"] == "suppress" for row in ranked[:count]
                ])))
                positives.append(float(np.mean([
                    row["teacher_decision"] == "suppress" for row in group
                ])))
            precision_rows.append({
                "model": artifact["model_id"], "seed": artifact["seed"],
                "split": args.split, "k": k, "target_groups": len(values),
                "precision_at_k": float(np.mean(values)),
                "random_prevalence": float(np.mean(positives)),
                "lift_over_random": (
                    float(np.mean(values)) / float(np.mean(positives))
                    if values and np.mean(positives) > 0 else ""
                ),
            })

    output = args.output_dir.resolve()
    write_csv(output / "teacher_fit_metrics.csv", summaries)
    write_csv(output / "teacher_precision_at_k.csv", precision_rows)
    write_csv(output / "teacher_fit_predictions.csv", prediction_rows)
    cross_seed = []
    for model in sorted({row["model"] for row in summaries}):
        selected = [row for row in summaries if row["model"] == model]
        summary = {"model": model, "seeds": len(selected), "split": args.split}
        for metric in ("weighted_mse", "mse", "spearman_gate",
                       "mean_predicted_gate", "mean_teacher_gate",
                       "teacher_suppress_rate"):
            mean, std = _mean_std([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        cross_seed.append(summary)
    precision_cross_seed = []
    for model in sorted({row["model"] for row in precision_rows}):
        for k in sorted(set(args.precision_k)):
            selected = [row for row in precision_rows
                        if row["model"] == model and int(row["k"]) == k]
            precision_mean, precision_std = _mean_std([
                float(row["precision_at_k"]) for row in selected
            ])
            prevalence_mean, prevalence_std = _mean_std([
                float(row["random_prevalence"]) for row in selected
            ])
            precision_cross_seed.append({
                "model": model, "k": k, "seeds": len(selected),
                "precision_at_k_mean": precision_mean,
                "precision_at_k_std": precision_std,
                "random_prevalence_mean": prevalence_mean,
                "random_prevalence_std": prevalence_std,
            })
    write_csv(output / "teacher_fit_cross_seed_summary.csv", cross_seed)
    write_csv(output / "teacher_precision_cross_seed_summary.csv", precision_cross_seed)
    write_json(output / "teacher_fit_manifest.json", {
        "base_checkpoint_id": experiment.checkpoint_id,
        "split": args.split,
        "teacher_file": str(args.teacher_file.resolve()),
        "trained_checkpoints": [str(path.resolve()) for path in args.trained_checkpoints],
        "conversation_ids": ids,
        "precision_k": sorted(set(args.precision_k)),
        "note": "Validation teacher is used only for diagnosis, never as an E3 training target.",
    })


def main(argv: Sequence[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
