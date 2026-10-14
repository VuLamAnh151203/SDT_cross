"""Evaluate D1 three-class selective propagation gating with abstention.

The D1 predictor sees only pre-intervention observable features.  For every
target, this diagnostic ranks text propagation edges by
``P(harmful) - P(beneficial)`` and changes at most one edge.  It abstains when
the best edge does not pass both validation-selected confidence thresholds.
The SDT checkpoint stays frozen throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
SDT_ROOT = ROOT.parents[1]
for path in (ROOT, HERE, CROSS, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import classification_metrics, sha256, write_csv, write_json  # noqa: E402
from dataloader import make_loaders  # noqa: E402
from d1_harm_learnability import ImpactMLP  # noqa: E402
from d1_prepare_edge_dataset import FEATURE_NAMES  # noqa: E402
from impact_core import load_experiment, seed_everything  # noqa: E402
from reliability_model import ReliabilitySDT, source_branches  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}
METRICS = ("accuracy", "weighted_f1", "macro_f1", "nll")


@dataclass(frozen=True)
class GateConfiguration:
    probability_threshold: float
    margin_threshold: float
    alpha: float

    @property
    def identity(self) -> bool:
        return self.alpha == 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--d1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=(2024, 2025, 2026, 2027, 2028))
    parser.add_argument("--modality", choices=tuple(MODALITY_INDEX), default="text")
    parser.add_argument("--probability-thresholds", nargs="+", type=float,
                        default=(0.5, 0.6, 0.7, 0.8, 0.9))
    parser.add_argument("--margin-thresholds", nargs="+", type=float,
                        default=(0.0, 0.1, 0.2, 0.3))
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--selection-metric",
                        choices=("weighted_f1", "accuracy", "nll"),
                        default="weighted_f1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--predictor-batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--renormalize-attention", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_torch(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility.
        return torch.load(path, map_location="cpu")


def _validate_grid(args: argparse.Namespace) -> list[GateConfiguration]:
    probabilities = sorted(set(float(value) for value in args.probability_thresholds))
    margins = sorted(set(float(value) for value in args.margin_thresholds))
    alphas = sorted(set(float(value) for value in args.alphas))
    if not probabilities or any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("Probability thresholds must be in [0, 1]")
    if not margins or any(not 0.0 <= value <= 1.0 for value in margins):
        raise ValueError("Margin thresholds must be in [0, 1]")
    if not alphas or any(not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("Suppression alphas must be in (0, 1]")
    # Identity is an explicit candidate.  This lets validation conclude that
    # the predictor should abstain everywhere rather than forcing a bad gate.
    configurations = [GateConfiguration(1.0, 1.0, 0.0)]
    configurations.extend(
        GateConfiguration(probability, margin, alpha)
        for probability, margin, alpha in product(probabilities, margins, alphas)
    )
    return configurations


def _read_observable_features(
    dataset_path: Path,
    modality: str,
) -> dict[str, dict[str, list[tuple[int, int, np.ndarray]]]]:
    required = {
        "predictor_split", "conversation_id", "source_idx", "target_idx",
        "modality", *FEATURE_NAMES,
    }
    grouped: dict[str, dict[str, list[tuple[int, int, np.ndarray]]]] = {
        "valid": {}, "test": {},
    }
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    with dataset_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"D1 dataset missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            split = row["predictor_split"]
            if split not in grouped or row["modality"] != modality:
                continue
            vector = np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"{dataset_path}:{line_number}: non-finite feature")
            conversation = row["conversation_id"]
            grouped[split].setdefault(conversation, []).append((
                int(row["target_idx"]), int(row["source_idx"]), vector,
            ))
    if not grouped["valid"] or not grouped["test"]:
        raise ValueError("D1 observable dataset must contain valid and test rows")
    return grouped


def _load_predictor(path: Path, device: torch.device):
    artifact = _load_torch(path)
    if artifact.get("supervision") != "three_class":
        raise ValueError(f"Expected a three_class D1 checkpoint: {path}")
    if tuple(artifact.get("feature_names", ())) != tuple(FEATURE_NAMES):
        raise ValueError(f"D1 feature schema mismatch: {path}")
    config = artifact["model_config"]
    if int(config["output_dim"]) != 3:
        raise ValueError(f"D1 checkpoint is not three-class: {path}")
    model = ImpactMLP(
        int(config["input_dim"]), int(config["hidden_dim"]),
        int(config["output_dim"]), float(config["dropout"]),
    ).to(device)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.eval()
    feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(artifact["feature_std"], dtype=np.float32)
    if feature_mean.shape != (len(FEATURE_NAMES),) or feature_std.shape != feature_mean.shape:
        raise ValueError(f"Invalid D1 normalization arrays: {path}")
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all():
        raise ValueError(f"Non-finite D1 normalization arrays: {path}")
    if np.any(feature_std <= 0.0):
        raise ValueError(f"Non-positive D1 feature std: {path}")
    return model, feature_mean, feature_std, artifact


def _predict_edge_probabilities(
    model: torch.nn.Module,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    rows_by_conversation: Mapping[str, Sequence[tuple[int, int, np.ndarray]]],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for conversation, rows in rows_by_conversation.items():
        length = max(max(target, source) for target, source, _ in rows) + 1
        expected = length * (length - 1)
        if len(rows) != expected:
            raise ValueError(
                f"Incomplete D1 edge features for {conversation}: {len(rows)} != {expected}"
            )
        features = np.stack([row[2] for row in rows])
        normalized = (features - feature_mean) / feature_std
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(normalized), batch_size):
                tensor = torch.from_numpy(normalized[start:start + batch_size]).to(device)
                chunks.append(torch.softmax(model(tensor), dim=-1).cpu().numpy())
        probabilities = np.concatenate(chunks, axis=0)
        matrix = np.full((length, length, 3), np.nan, dtype=np.float32)
        for (target, source, _), probability in zip(rows, probabilities):
            if np.isfinite(matrix[target, source]).any():
                raise ValueError(f"Duplicate D1 edge feature: {conversation} {target} {source}")
            matrix[target, source] = probability
        output[conversation] = matrix
    return output


def selective_edge_gate(
    probability_matrix: np.ndarray,
    configuration: GateConfiguration,
    modality_index: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Return one conversation's gate and per-target abstention decisions."""
    if probability_matrix.ndim != 3 or probability_matrix.shape[2] != 3:
        raise ValueError("Expected [target, source, 3] class probabilities")
    length = probability_matrix.shape[0]
    if probability_matrix.shape[1] != length:
        raise ValueError("Expected a square source/target probability matrix")
    gate = np.ones((length, length, 3), dtype=np.float32)
    decisions: list[dict[str, float | int]] = []
    for target in range(length):
        sources = [source for source in range(length) if source != target]
        if not sources or configuration.identity:
            decisions.append({
                "selected_source_idx": -1, "selected_gate": 1.0,
                "selected_harm_probability": 0.0,
                "selected_beneficial_probability": 0.0,
                "selected_harm_margin": 0.0, "intervened": 0,
            })
            continue
        values = probability_matrix[target, sources]
        if not np.isfinite(values).all():
            raise ValueError(f"Missing class probability for target {target}")
        harm_margin = values[:, 0] - values[:, 2]
        local = int(np.argmax(harm_margin))
        source = sources[local]
        harmful = float(values[local, 0])
        beneficial = float(values[local, 2])
        margin = float(harm_margin[local])
        intervene = (
            harmful >= configuration.probability_threshold
            and margin >= configuration.margin_threshold
        )
        applied = max(0.0, min(1.0, 1.0 - configuration.alpha * harmful)) if intervene else 1.0
        if intervene:
            gate[target, source, modality_index] = applied
        decisions.append({
            "selected_source_idx": source if intervene else -1,
            "selected_gate": applied,
            "selected_harm_probability": harmful,
            "selected_beneficial_probability": beneficial,
            "selected_harm_margin": margin,
            "intervened": int(intervene),
        })
    return gate, decisions


def _evaluate(
    wrapper: ReliabilitySDT,
    loader,
    edge_probabilities: Mapping[str, np.ndarray] | None,
    configuration: GateConfiguration,
    modality: str,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    labels_all: list[int] = []
    predictions_all: list[int] = []
    prediction_rows: list[dict[str, object]] = []
    nll_sum = 0.0
    utterances = interventions = 0
    selected_gate_sum = 0.0
    modality_index = MODALITY_INDEX[modality]
    started = time.perf_counter()
    wrapper.eval()
    with torch.inference_mode():
        for batch_number, data in enumerate(loader):
            if max_batches and batch_number >= max_batches:
                break
            text, visual, audio, speaker_labels, mask, labels = [
                item.to(device) for item in data[:6]
            ]
            conversation_ids = data[-1]
            speakers = speaker_labels.transpose(0, 1)
            valid = mask.bool()
            lengths = [int(value) for value in valid.sum(dim=1).tolist()]
            edge_gate = torch.ones(
                len(conversation_ids), mask.size(1), mask.size(1), 3,
                device=device,
            )
            batch_decisions: list[list[dict[str, float | int]]] = []
            for batch_index, (conversation, length) in enumerate(zip(conversation_ids, lengths)):
                if configuration.identity:
                    matrix = np.zeros((length, length, 3), dtype=np.float32)
                else:
                    if edge_probabilities is None or conversation not in edge_probabilities:
                        raise KeyError(f"Missing D1 probabilities for {conversation}")
                    matrix = edge_probabilities[conversation]
                    if matrix.shape[:2] != (length, length):
                        raise ValueError(
                            f"D1/model length mismatch for {conversation}: "
                            f"{matrix.shape[:2]} != {(length, length)}"
                        )
                gate, decisions = selective_edge_gate(
                    matrix, configuration, modality_index,
                )
                edge_gate[batch_index, :length, :length] = torch.from_numpy(gate).to(device)
                batch_decisions.append(decisions)
            output = wrapper(
                text, visual, audio, mask, speakers, lengths,
                fixed_edge_gate=edge_gate,
                active_branches=source_branches(modality_index),
            )
            prediction = output["logits"].argmax(dim=-1)
            log_prob = output["log_prob"]
            for batch_index, (conversation, length) in enumerate(zip(conversation_ids, lengths)):
                for target in range(length):
                    label = int(labels[batch_index, target].item())
                    predicted = int(prediction[batch_index, target].item())
                    target_nll = float(-log_prob[batch_index, target, label].cpu())
                    decision = batch_decisions[batch_index][target]
                    labels_all.append(label)
                    predictions_all.append(predicted)
                    nll_sum += target_nll
                    utterances += 1
                    interventions += int(decision["intervened"])
                    if int(decision["intervened"]):
                        selected_gate_sum += float(decision["selected_gate"])
                    prediction_rows.append({
                        "conversation_id": conversation,
                        "target_idx": target,
                        "target_label": label,
                        "prediction": predicted,
                        "nll": target_nll,
                        **decision,
                    })
    if not utterances:
        raise ValueError("Empty validation/test evaluation")
    metrics = classification_metrics(labels_all, predictions_all)
    metrics.update({
        "nll": nll_sum / utterances,
        "utterances": utterances,
        "intervened_targets": interventions,
        "intervention_target_rate": interventions / utterances,
        "mean_selected_gate": (
            selected_gate_sum / interventions if interventions else 1.0
        ),
        "seconds": time.perf_counter() - started,
    })
    return metrics, prediction_rows


def _summary_selection_key(row: Mapping[str, object], metric: str) -> tuple[float, ...]:
    accuracy = float(row["accuracy_mean"])
    weighted_f1 = float(row["weighted_f1_mean"])
    negative_nll = -float(row["nll_mean"])
    if metric == "accuracy":
        scores = (accuracy, weighted_f1, negative_nll)
    elif metric == "nll":
        scores = (negative_nll, weighted_f1, accuracy)
    else:
        scores = (weighted_f1, accuracy, negative_nll)
    return (*scores, -float(row["intervention_target_rate_mean"]))


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1 or args.predictor_batch_size < 1 or args.max_batches < 0:
        raise ValueError("Invalid batch size or max-batches")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("D1 predictor seeds must be unique")
    configurations = _validate_grid(args)
    seed_everything(args.seeds[0])
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    d1_dir = args.d1_dir.resolve()
    dataset_path = d1_dir / "edge_impact_pairs.csv"
    dataset_manifest = _load_json(d1_dir / "dataset_manifest.json")
    experiment_manifest = _load_json(d1_dir / "experiment_manifest.json")
    if dataset_manifest.get("checkpoint_id") != experiment.checkpoint_id:
        raise ValueError("D1 dataset and SDT checkpoint do not match")
    if experiment_manifest.get("dataset_sha256") != sha256(dataset_path):
        raise ValueError("D1 predictor manifest and edge dataset do not match")
    if args.modality not in set(dataset_manifest.get("modalities", ())):
        raise ValueError(f"D1 dataset does not contain modality {args.modality!r}")
    if bool(dataset_manifest.get("renormalize_attention", False)) != bool(
        args.renormalize_attention
    ):
        raise ValueError("D1 dataset and abstention run disagree on attention renormalization")
    split_ids = {
        name: list(values)
        for name, values in dataset_manifest.get("split_ids", {}).items()
    }
    if not split_ids.get("valid") or not split_ids.get("test"):
        raise ValueError("Three-class abstention requires non-empty D1 valid/test splits")
    if set(split_ids["valid"]) & set(split_ids["test"]):
        raise ValueError("D1 validation/test conversation leakage")

    feature_rows = _read_observable_features(dataset_path, args.modality)
    for split in ("valid", "test"):
        if set(feature_rows[split]) != set(split_ids[split]):
            raise ValueError(f"D1 {split} feature rows do not match dataset manifest")

    wrapper = ReliabilitySDT(
        experiment.model, "B0",
        renormalize_attention=args.renormalize_attention,
    ).to(experiment.device).eval()
    loaders = make_loaders(
        experiment.dataset, {"valid": split_ids["valid"], "test": split_ids["test"]},
        args.batch_size, args.seeds[0], args.num_workers,
        experiment.device.type == "cuda",
    )
    identity = GateConfiguration(1.0, 1.0, 0.0)
    baseline_valid, _ = _evaluate(
        wrapper, loaders["valid"], None, identity, args.modality,
        experiment.device, args.max_batches,
    )
    baseline_test, baseline_predictions = _evaluate(
        wrapper, loaders["test"], None, identity, args.modality,
        experiment.device, args.max_batches,
    )
    baseline_by_target = {
        (row["conversation_id"], int(row["target_idx"])): row
        for row in baseline_predictions
    }

    output = args.output_dir.resolve()
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    checkpoint_paths: list[str] = []
    probabilities_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    artifacts_by_seed: dict[int, Mapping[str, object]] = {}

    for seed in args.seeds:
        seed_everything(seed)
        checkpoint_path = d1_dir / "checkpoints" / f"three_class_seed{seed}.pt"
        model, feature_mean, feature_std, artifact = _load_predictor(
            checkpoint_path, experiment.device,
        )
        if int(artifact.get("seed", -1)) != seed:
            raise ValueError(f"D1 seed mismatch: {checkpoint_path}")
        checkpoint_paths.append(str(checkpoint_path))
        probabilities = {
            split: _predict_edge_probabilities(
                model, feature_mean, feature_std, feature_rows[split],
                experiment.device, args.predictor_batch_size,
            )
            for split in ("valid", "test")
        }
        probabilities_by_seed[seed] = probabilities
        artifacts_by_seed[seed] = artifact
        for number, configuration in enumerate(configurations, start=1):
            if configuration.identity:
                metrics = dict(baseline_valid)
            else:
                metrics, _ = _evaluate(
                    wrapper, loaders["valid"], probabilities["valid"], configuration,
                    args.modality, experiment.device, args.max_batches,
                )
            row = {
                "seed": seed,
                "probability_threshold": configuration.probability_threshold,
                "margin_threshold": configuration.margin_threshold,
                "alpha": configuration.alpha,
                "identity_candidate": int(configuration.identity),
                "split": "valid",
                **metrics,
            }
            validation_rows.append(row)
            if number % 20 == 0 or number == len(configurations):
                print(
                    f"abstention validation seed={seed}: "
                    f"{number}/{len(configurations)} configurations",
                    flush=True,
                )

    validation_summary: list[dict[str, object]] = []
    for configuration in configurations:
        rows = [
            row for row in validation_rows
            if float(row["probability_threshold"]) == configuration.probability_threshold
            and float(row["margin_threshold"]) == configuration.margin_threshold
            and float(row["alpha"]) == configuration.alpha
        ]
        summary_row: dict[str, object] = {
            "probability_threshold": configuration.probability_threshold,
            "margin_threshold": configuration.margin_threshold,
            "alpha": configuration.alpha,
            "identity_candidate": int(configuration.identity),
            "seeds": len(rows),
        }
        for metric in (*METRICS, "intervention_target_rate", "mean_selected_gate"):
            mean, std = _mean_std([float(row[metric]) for row in rows])
            summary_row[f"{metric}_mean"] = mean
            summary_row[f"{metric}_std"] = std
        validation_summary.append(summary_row)
    winner = max(
        validation_summary,
        key=lambda row: _summary_selection_key(row, args.selection_metric),
    )
    selected = GateConfiguration(
        float(winner["probability_threshold"]),
        float(winner["margin_threshold"]),
        float(winner["alpha"]),
    )
    selected_row = {
        **winner,
        "selection_split": "valid",
        "selection_metric": args.selection_metric,
        "selection_aggregation": "mean_across_seeds",
    }
    print(
        "selected global abstention configuration: "
        f"tau_p={selected.probability_threshold:g} "
        f"tau_m={selected.margin_threshold:g} alpha={selected.alpha:g} "
        f"mean_valid_WF1={float(winner['weighted_f1_mean']):.3f}",
        flush=True,
    )

    for seed in args.seeds:
        artifact = artifacts_by_seed[seed]
        probabilities = probabilities_by_seed[seed]
        if selected.identity:
            test_metrics = dict(baseline_test)
            selected_predictions = [dict(row) for row in baseline_predictions]
        else:
            test_metrics, selected_predictions = _evaluate(
                wrapper, loaders["test"], probabilities["test"], selected,
                args.modality, experiment.device, args.max_batches,
            )
        test_row: dict[str, object] = {
            "seed": seed,
            "probability_threshold": selected.probability_threshold,
            "margin_threshold": selected.margin_threshold,
            "alpha": selected.alpha,
            "selected_strong_eps": float(artifact["selected_strong_eps"]),
            "selection_split": "valid",
            "selection_metric": args.selection_metric,
        }
        for metric in METRICS:
            test_row[f"baseline_{metric}"] = baseline_test[metric]
            test_row[f"selective_{metric}"] = test_metrics[metric]
            test_row[f"delta_{metric}"] = float(test_metrics[metric]) - float(
                baseline_test[metric]
            )
        for metric in ("utterances", "intervened_targets", "intervention_target_rate",
                       "mean_selected_gate", "seconds"):
            test_row[metric] = test_metrics[metric]
        test_rows.append(test_row)

        for row in selected_predictions:
            key = (row["conversation_id"], int(row["target_idx"]))
            baseline = baseline_by_target[key]
            prediction_rows.append({
                "seed": seed,
                "probability_threshold": selected.probability_threshold,
                "margin_threshold": selected.margin_threshold,
                "alpha": selected.alpha,
                **row,
                "baseline_prediction": baseline["prediction"],
                "baseline_nll": baseline["nll"],
                "delta_nll": float(row["nll"]) - float(baseline["nll"]),
            })
        print(
            f"three-class abstention seed={seed}: "
            f"tau_p={selected.probability_threshold:g} "
            f"tau_m={selected.margin_threshold:g} alpha={selected.alpha:g} "
            f"test_WF1={float(test_metrics['weighted_f1']):.3f} "
            f"delta={float(test_row['delta_weighted_f1']):+.3f}",
            flush=True,
        )

    summary: dict[str, object] = {"seeds": len(test_rows)}
    for metric in METRICS:
        for prefix in ("baseline", "selective", "delta"):
            mean, std = _mean_std([float(row[f"{prefix}_{metric}"]) for row in test_rows])
            summary[f"{prefix}_{metric}_mean"] = mean
            summary[f"{prefix}_{metric}_std"] = std
    for metric in ("intervention_target_rate", "mean_selected_gate"):
        mean, std = _mean_std([float(row[metric]) for row in test_rows])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std

    paired_rows = []
    for metric in METRICS:
        values = [float(row[f"delta_{metric}"]) for row in test_rows]
        mean, std = _mean_std(values)
        sem = std / math.sqrt(len(values)) if values else math.nan
        paired_rows.append({
            "comparison": "three_class_abstention-B0",
            "metric": metric,
            "seeds": len(values),
            "delta_mean": mean,
            "delta_std": std,
            "normal_approx_ci95_low": mean - 1.96 * sem,
            "normal_approx_ci95_high": mean + 1.96 * sem,
            "note": "NLL delta below zero is better; score deltas above zero are better.",
        })

    write_csv(output / "validation_grid.csv", validation_rows)
    write_csv(output / "validation_summary.csv", validation_summary)
    write_csv(output / "selected_config.csv", [selected_row])
    write_csv(output / "test_runs.csv", test_rows)
    write_csv(output / "test_summary.csv", [summary])
    write_csv(output / "paired_deltas.csv", paired_rows)
    write_csv(output / "test_predictions.csv", prediction_rows)
    write_json(output / "manifest.json", {
        "base_checkpoint_id": experiment.checkpoint_id,
        "base_checkpoint_sha256": experiment.checkpoint_hash,
        "d1_dataset": str(dataset_path),
        "d1_dataset_sha256": sha256(dataset_path),
        "d1_checkpoints": checkpoint_paths,
        "seeds": list(args.seeds),
        "modality": args.modality,
        "selection_split": "valid",
        "selection_metric": args.selection_metric,
        "selection_aggregation": "mean_across_seeds",
        "ranking_score": "P(harmful)-P(beneficial)",
        "maximum_edges_per_target": 1,
        "abstention_rule": "P(harmful)>=tau_probability and harm_margin>=tau_margin",
        "gate_rule": "1-alpha*P(harmful)",
        "identity_candidate_in_validation_grid": True,
        "probability_thresholds": sorted(set(args.probability_thresholds)),
        "margin_thresholds": sorted(set(args.margin_thresholds)),
        "alphas": sorted(set(args.alphas)),
        "test_used_for_training_or_selection": False,
        "smoke_limited": bool(args.max_batches),
    })
    print(f"Three-class abstention results written to: {output}", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
