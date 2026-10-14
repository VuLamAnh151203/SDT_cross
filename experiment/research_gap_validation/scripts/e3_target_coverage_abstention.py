"""Validation-calibrated target-coverage abstention for D1 three-class gates."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
SDT_ROOT = ROOT.parents[1]
for path in (ROOT, HERE, CROSS, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import sha256, write_csv, write_json
from dataloader import make_loaders
from e3_three_class_abstention import (
    METRICS, GateConfiguration, _evaluate, _load_json, _load_predictor,
    _mean_std, _predict_edge_probabilities, _read_observable_features,
    _summary_selection_key,
)
from impact_core import load_experiment, seed_everything
from reliability_model import ReliabilitySDT
from d1_prepare_edge_dataset import FEATURE_NAMES


@dataclass(frozen=True)
class CoverageConfiguration:
    coverage: float
    alpha: float

    @property
    def identity(self) -> bool:
        return self.coverage == 0.0 or self.alpha == 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--d1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=(2024, 2025, 2026, 2027, 2028))
    parser.add_argument("--coverages", nargs="+", type=float,
                        default=(0.01, 0.02, 0.05, 0.10, 0.20))
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--selection-metric", choices=("weighted_f1", "accuracy", "nll"),
                        default="weighted_f1")
    parser.add_argument("--modality", choices=("text",), default="text")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--predictor-batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--renormalize-attention", action="store_true")
    parser.add_argument("--target-model", action="store_true",
                        help="Rank targets with a train-only target classifier")
    parser.add_argument("--target-improvement-eps", type=float, default=0.05,
                        help="Train target label: best edge delta-NLL < -epsilon")
    return parser.parse_args(argv)


def target_scores(probabilities: Mapping[str, np.ndarray]) -> list[tuple[float, str, int]]:
    """Best P(harmful)-P(beneficial) for every target, deterministically sorted."""
    rows: list[tuple[float, str, int]] = []
    for conversation, matrix in probabilities.items():
        for target in range(matrix.shape[0]):
            sources = [source for source in range(matrix.shape[1]) if source != target]
            if not sources:
                continue
            values = matrix[target, sources]
            if not np.isfinite(values).all():
                raise ValueError(f"Missing probabilities for {conversation} target {target}")
            score = float(np.max(values[:, 0] - values[:, 2]))
            rows.append((score, str(conversation), target))
    return sorted(rows, key=lambda row: (-row[0], row[1], row[2]))


def validation_coverage_selection(
    rows: Sequence[tuple[float, str, int]], coverage: float,
) -> tuple[set[tuple[str, int]], float]:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("Coverage must be in (0, 1]")
    count = min(len(rows), max(1, math.ceil(coverage * len(rows))))
    chosen = {(conversation, target) for _, conversation, target in rows[:count]}
    return chosen, float(rows[count - 1][0])


def threshold_selection(
    rows: Sequence[tuple[float, str, int]], cutoff: float,
) -> set[tuple[str, int]]:
    return {(conversation, target) for score, conversation, target in rows if score >= cutoff}


def target_level_dataset(path: Path, modality: str, epsilon: float):
    """Aggregate observable edge features; labels never enter valid/test features."""
    groups: dict[tuple[str, str, int], list[tuple[np.ndarray, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["modality"] != modality:
                continue
            key = (row["predictor_split"], row["conversation_id"], int(row["target_idx"]))
            vector = np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64)
            groups.setdefault(key, []).append((vector, float(row["signed_impact"])))
    result = {split: {"keys": [], "x": [], "y": []} for split in ("train", "valid", "test")}
    for (split, conversation, target), values in groups.items():
        vectors = np.stack([value[0] for value in values])
        # Distribution summaries plus candidate count capture max-score/length bias.
        features = np.concatenate((vectors.mean(0), vectors.std(0), vectors.min(0),
                                   vectors.max(0), [math.log1p(len(vectors))]))
        result[split]["keys"].append((conversation, target))
        result[split]["x"].append(features)
        result[split]["y"].append(int(min(value[1] for value in values) < -epsilon))
    for split in result:
        result[split]["x"] = np.asarray(result[split]["x"], dtype=np.float64)
        result[split]["y"] = np.asarray(result[split]["y"], dtype=np.int64)
    return result


def fit_target_scores(dataset, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(dataset["train"]["y"])) != 2:
        raise ValueError("Target labels contain only one class; adjust --target-improvement-eps")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed,
    ))
    model.fit(dataset["train"]["x"], dataset["train"]["y"])
    output = {}
    for split in ("valid", "test"):
        scores = model.predict_proba(dataset[split]["x"])[:, 1]
        output[split] = sorted(
            [(float(score), str(key[0]), int(key[1]))
             for score, key in zip(scores, dataset[split]["keys"])],
            key=lambda row: (-row[0], row[1], row[2]),
        )
    return output


def run(args: argparse.Namespace) -> Path:
    coverages = sorted(set(float(value) for value in args.coverages))
    alphas = sorted(set(float(value) for value in args.alphas))
    if not coverages or any(not 0.0 < value <= 1.0 for value in coverages):
        raise ValueError("Coverages must be in (0, 1]")
    if not alphas or any(not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("Alphas must be in (0, 1]")
    configurations = [CoverageConfiguration(0.0, 0.0)] + [
        CoverageConfiguration(coverage, alpha)
        for coverage in coverages for alpha in alphas
    ]
    seed_everything(args.seeds[0])
    experiment = load_experiment(args.checkpoint, args.feature_path, args.device,
                                 args.gpu_id, args.allow_smoke_checkpoint)
    d1_dir = args.d1_dir.resolve()
    dataset_path = d1_dir / "edge_impact_pairs.csv"
    dataset_manifest = _load_json(d1_dir / "dataset_manifest.json")
    experiment_manifest = _load_json(d1_dir / "experiment_manifest.json")
    if dataset_manifest.get("checkpoint_id") != experiment.checkpoint_id:
        raise ValueError("D1 dataset and SDT checkpoint do not match")
    if experiment_manifest.get("dataset_sha256") != sha256(dataset_path):
        raise ValueError("D1 predictor manifest and edge dataset do not match")
    if bool(dataset_manifest.get("renormalize_attention", False)) != bool(args.renormalize_attention):
        raise ValueError("D1 dataset and run disagree on attention renormalization")
    split_ids = {name: list(values) for name, values in dataset_manifest["split_ids"].items()}
    if not split_ids.get("valid") or not split_ids.get("test"):
        raise ValueError("Non-empty D1 validation and test splits are required")
    if set(split_ids["valid"]) & set(split_ids["test"]):
        raise ValueError("D1 validation/test conversation leakage")
    feature_rows = _read_observable_features(dataset_path, args.modality)
    wrapper = ReliabilitySDT(experiment.model, "B0",
                             renormalize_attention=args.renormalize_attention).to(experiment.device).eval()
    loaders = make_loaders(experiment.dataset, {key: split_ids[key] for key in ("valid", "test")},
                           args.batch_size, args.seeds[0], args.num_workers,
                           experiment.device.type == "cuda")
    identity_gate = GateConfiguration(1.0, 1.0, 0.0)
    active_gate = lambda alpha: GateConfiguration(0.0, -1.0, alpha)
    baseline_valid, _ = _evaluate(wrapper, loaders["valid"], None, identity_gate,
                                  args.modality, experiment.device, args.max_batches)
    baseline_test, baseline_predictions = _evaluate(
        wrapper, loaders["test"], None, identity_gate, args.modality,
        experiment.device, args.max_batches)
    baseline_by_target = {(str(row["conversation_id"]), int(row["target_idx"])): row
                          for row in baseline_predictions}

    validation_rows: list[dict[str, object]] = []
    probabilities_by_seed = {}
    cutoffs: dict[tuple[int, float], float] = {}
    artifacts = {}
    target_rankings = {}
    checkpoints = []
    for seed in args.seeds:
        checkpoint = d1_dir / "checkpoints" / f"three_class_seed{seed}.pt"
        model, mean, std, artifact = _load_predictor(checkpoint, experiment.device)
        probabilities = {split: _predict_edge_probabilities(
            model, mean, std, feature_rows[split], experiment.device,
            args.predictor_batch_size) for split in ("valid", "test")}
        probabilities_by_seed[seed] = probabilities
        artifacts[seed] = artifact
        if args.target_model:
            target_data = target_level_dataset(
                dataset_path, args.modality, args.target_improvement_eps
            )
            target_rankings[seed] = fit_target_scores(target_data, seed)
        checkpoints.append(str(checkpoint))
        ranked = (target_rankings[seed]["valid"] if args.target_model
                  else target_scores(probabilities["valid"]))
        for configuration in configurations:
            if configuration.identity:
                metrics, cutoff = dict(baseline_valid), math.inf
            else:
                allowed, cutoff = validation_coverage_selection(ranked, configuration.coverage)
                metrics, _ = _evaluate(wrapper, loaders["valid"], probabilities["valid"],
                                       active_gate(configuration.alpha), args.modality,
                                       experiment.device, args.max_batches, allowed)
                cutoffs[(seed, configuration.coverage)] = cutoff
            validation_rows.append({
                "seed": seed, "coverage": configuration.coverage,
                "alpha": configuration.alpha, "identity_candidate": int(configuration.identity),
                "validation_score_cutoff": cutoff, "split": "valid", **metrics,
            })
        print(f"coverage validation seed={seed}: {len(configurations)} configurations", flush=True)

    summary_rows = []
    for configuration in configurations:
        rows = [row for row in validation_rows
                if float(row["coverage"]) == configuration.coverage
                and float(row["alpha"]) == configuration.alpha]
        summary: dict[str, object] = {
            "coverage": configuration.coverage, "alpha": configuration.alpha,
            "identity_candidate": int(configuration.identity), "seeds": len(rows),
        }
        for metric in (*METRICS, "intervention_target_rate", "mean_selected_gate"):
            mean, std = _mean_std([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = mean, std
        summary_rows.append(summary)
    winner = max(summary_rows, key=lambda row: _summary_selection_key(row, args.selection_metric))
    selected = CoverageConfiguration(float(winner["coverage"]), float(winner["alpha"]))
    selected_row = {**winner, "selection_split": "valid",
                    "selection_metric": args.selection_metric,
                    "selection_aggregation": "mean_across_seeds"}
    print(f"selected coverage={selected.coverage:g} alpha={selected.alpha:g} "
          f"valid_WF1={float(winner['weighted_f1_mean']):.3f}", flush=True)

    test_rows, prediction_rows = [], []
    for seed in args.seeds:
        if selected.identity:
            metrics, predictions, cutoff = dict(baseline_test), list(baseline_predictions), math.inf
        else:
            cutoff = cutoffs[(seed, selected.coverage)]
            ranked_test = (target_rankings[seed]["test"] if args.target_model
                           else target_scores(probabilities_by_seed[seed]["test"]))
            allowed = threshold_selection(ranked_test, cutoff)
            metrics, predictions = _evaluate(
                wrapper, loaders["test"], probabilities_by_seed[seed]["test"],
                active_gate(selected.alpha), args.modality, experiment.device,
                args.max_batches, allowed)
        row: dict[str, object] = {
            "seed": seed, "coverage": selected.coverage, "alpha": selected.alpha,
            "validation_score_cutoff": cutoff,
            "selected_strong_eps": float(artifacts[seed]["selected_strong_eps"]),
            "selection_split": "valid", "selection_metric": args.selection_metric,
        }
        for metric in METRICS:
            row[f"baseline_{metric}"] = baseline_test[metric]
            row[f"selective_{metric}"] = metrics[metric]
            row[f"delta_{metric}"] = float(metrics[metric]) - float(baseline_test[metric])
        for metric in ("utterances", "intervened_targets", "intervention_target_rate",
                       "mean_selected_gate", "seconds"):
            row[metric] = metrics[metric]
        test_rows.append(row)
        for prediction in predictions:
            key = (str(prediction["conversation_id"]), int(prediction["target_idx"]))
            baseline = baseline_by_target[key]
            prediction_rows.append({
                "seed": seed, "coverage": selected.coverage, "alpha": selected.alpha,
                "validation_score_cutoff": cutoff, **prediction,
                "baseline_prediction": baseline["prediction"], "baseline_nll": baseline["nll"],
                "delta_nll": float(prediction["nll"]) - float(baseline["nll"]),
            })
        print(f"coverage test seed={seed}: rate={float(metrics['intervention_target_rate']):.3f} "
              f"WF1={float(metrics['weighted_f1']):.3f} delta={float(row['delta_weighted_f1']):+.3f}",
              flush=True)

    test_summary: dict[str, object] = {"seeds": len(test_rows)}
    paired_rows = []
    for metric in METRICS:
        for prefix in ("baseline", "selective", "delta"):
            mean, std = _mean_std([float(row[f"{prefix}_{metric}"]) for row in test_rows])
            test_summary[f"{prefix}_{metric}_mean"] = mean
            test_summary[f"{prefix}_{metric}_std"] = std
        deltas = [float(row[f"delta_{metric}"]) for row in test_rows]
        mean, std = _mean_std(deltas)
        half = 1.96 * std / math.sqrt(len(deltas))
        paired_rows.append({"comparison": "target_coverage_abstention-B0", "metric": metric,
                            "seeds": len(deltas), "delta_mean": mean, "delta_std": std,
                            "normal_approx_ci95_low": mean - half,
                            "normal_approx_ci95_high": mean + half,
                            "note": "NLL delta below zero is better; score deltas above zero are better."})
    for metric in ("intervention_target_rate", "mean_selected_gate"):
        mean, std = _mean_std([float(row[metric]) for row in test_rows])
        test_summary[f"{metric}_mean"], test_summary[f"{metric}_std"] = mean, std

    output = args.output_dir.resolve()
    write_csv(output / "validation_grid.csv", validation_rows)
    write_csv(output / "validation_summary.csv", summary_rows)
    write_csv(output / "selected_config.csv", [selected_row])
    write_csv(output / "test_runs.csv", test_rows)
    write_csv(output / "test_summary.csv", [test_summary])
    write_csv(output / "paired_deltas.csv", paired_rows)
    write_csv(output / "test_predictions.csv", prediction_rows)
    write_json(output / "manifest.json", {
        "base_checkpoint_id": experiment.checkpoint_id,
        "d1_dataset_sha256": sha256(dataset_path), "d1_checkpoints": checkpoints,
        "seeds": list(args.seeds), "coverages": coverages, "alphas": alphas,
        "selection_split": "valid", "selection_metric": args.selection_metric,
        "target_score": ("train-only logistic target classifier" if args.target_model
                         else "max_source(P(harmful)-P(beneficial))"),
        "target_improvement_eps": args.target_improvement_eps if args.target_model else None,
        "calibration": "per-seed validation quantile; fixed numeric cutoff applied to test",
        "test_used_for_training_or_selection": False,
    })
    print(f"Target-coverage abstention results: {output}", flush=True)
    return output


if __name__ == "__main__":
    run(parse_args())
