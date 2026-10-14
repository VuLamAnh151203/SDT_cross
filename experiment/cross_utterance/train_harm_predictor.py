"""Experiment 4: predict harmful context without ground-truth inference inputs.

Training labels are derived from signed impacts on a TRAIN intervention run.
Evaluation labels are used only for scoring on disjoint held-out conversations.
Neither emotion labels nor signed impacts enter the predictor feature matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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


IMPACT_REQUIRED = {
    "split", "conversation_id", "conversation_length", "source_idx", "target_idx",
    "same_speaker", "signed_offset", "absolute_distance", "source_before_target",
    "modality", "replacement_strategy", "signed_impact", "source_feature_norm",
    "replacement_delta_l2",
}
BASELINE_REQUIRED = {"conversation_id", "utterance_index"}
FEATURE_NAMES = (
    "baseline_margin",
    "baseline_top1_probability",
    "baseline_entropy",
    "log1p_absolute_distance",
    "normalized_signed_offset",
    "same_speaker",
    "source_before_target",
    "log1p_conversation_length",
    "log1p_source_feature_norm",
    "log1p_replacement_delta_l2",
    "modality_audio",
    "modality_visual",
    "strategy_train_mean",
)
MODEL_FEATURES = {
    "margin_only": ("baseline_margin",),
    "distance_only": ("log1p_absolute_distance",),
    "speaker_only": ("same_speaker",),
    "relationship_baseline": (
        "baseline_margin", "log1p_absolute_distance", "same_speaker",
        "source_before_target",
    ),
    "full_lightweight": FEATURE_NAMES,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-dir", required=True, type=Path)
    parser.add_argument("--eval-run-dir", required=True, type=Path)
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: <eval-run-dir>/results/experiment4_harm_predictor",
    )
    return parser.parse_args(argv)


def _load_baseline_features(run_dir: Path) -> Dict[Tuple[str, int], Tuple[float, float, float]]:
    path = Path(run_dir) / "baseline" / "baseline_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        require_columns(path, fieldnames, BASELINE_REQUIRED)
        probability_columns = sorted(
            (name for name in fieldnames if name.startswith("prob_") and name != "prob_true"),
            key=lambda name: int(name.split("_")[1]),
        )
        if len(probability_columns) < 2:
            raise ValueError(f"{path} has fewer than two probability columns")
        for line_number, row in enumerate(reader, start=2):
            probabilities = np.asarray(
                [finite_float(row, name) for name in probability_columns],
                dtype=np.float64,
            )
            if np.any(probabilities < 0):
                raise ValueError(f"{path}:{line_number}: negative probability")
            ranked = np.sort(probabilities)
            top1 = float(ranked[-1])
            margin = float(ranked[-1] - ranked[-2])
            entropy = float(-sum(
                value * math.log(value) for value in probabilities if value > 0
            ))
            key = (row["conversation_id"], int(row["utterance_index"]))
            if key in result:
                raise ValueError(f"{path}:{line_number}: duplicate baseline target {key}")
            result[key] = (margin, top1, entropy)
    if not result:
        raise ValueError(f"No baseline predictions in {path}")
    return result


def _count_rows(paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require_columns(path, reader.fieldnames or [], IMPACT_REQUIRED)
            count += sum(1 for _ in reader)
    return count


def _load_matrix(
    run_dir: Path,
    impact_eps: float,
    require_train_split: bool,
) -> Tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    paths = resolve_impact_paths(run_dir)
    baseline = _load_baseline_features(run_dir)
    count = _count_rows(paths)
    features = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(count, dtype=np.int8)
    conversations = set()
    splits = set()
    cursor = 0
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require_columns(path, reader.fieldnames or [], IMPACT_REQUIRED)
            for line_number, row in enumerate(reader, start=2):
                split = row["split"].strip().lower()
                splits.add(split)
                if require_train_split and split != "train":
                    raise ValueError(
                        f"{path}:{line_number}: predictor training requires split=train, got {split!r}"
                    )
                if not require_train_split and split == "train":
                    raise ValueError(
                        f"{path}:{line_number}: held-out evaluation cannot use split=train"
                    )
                conversation_id = row["conversation_id"]
                conversations.add(conversation_id)
                target_key = (conversation_id, int(row["target_idx"]))
                if target_key not in baseline:
                    raise ValueError(f"{path}:{line_number}: missing baseline {target_key}")
                margin, top1, entropy = baseline[target_key]
                length = int(row["conversation_length"])
                distance = int(row["absolute_distance"])
                signed_offset = int(row["signed_offset"])
                modality = row["modality"]
                strategy = row["replacement_strategy"]
                if modality not in MODALITIES or strategy not in STRATEGIES:
                    raise ValueError(f"{path}:{line_number}: invalid strategy/modality")
                vector = (
                    margin,
                    top1,
                    entropy,
                    math.log1p(distance),
                    signed_offset / max(length - 1, 1),
                    binary_int(row, "same_speaker"),
                    binary_int(row, "source_before_target"),
                    math.log1p(length),
                    math.log1p(max(finite_float(row, "source_feature_norm"), 0.0)),
                    math.log1p(max(finite_float(row, "replacement_delta_l2"), 0.0)),
                    int(modality == "audio"),
                    int(modality == "visual"),
                    int(strategy == "train_mean"),
                )
                features[cursor] = vector
                labels[cursor] = int(finite_float(row, "signed_impact") < -impact_eps)
                cursor += 1
    if cursor != count:
        raise AssertionError(f"Loaded {cursor} rows, expected {count}")
    if len(np.unique(labels)) != 2:
        raise ValueError("Harm labels must contain both classes")
    return features, labels, conversations, splits


def _training_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15
    )
    return float(thresholds[int(np.argmax(f1))])


def _metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Mapping[str, float | int]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = probabilities >= threshold
    return {
        "count": len(y_true),
        "harmful_count": int(y_true.sum()),
        "harmful_prevalence": float(y_true.mean()),
        "threshold_selected_on_train": threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def train_and_evaluate(
    train_run_dir: Path,
    eval_run_dir: Path,
    impact_eps: float = 1e-4,
    seed: int = 2024,
    max_iter: int = 500,
    output_dir: Path | None = None,
) -> Path:
    if impact_eps < 0:
        raise ValueError("impact_eps must be nonnegative")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("Experiment 4 requires scikit-learn") from error

    x_train, y_train, train_conversations, train_splits = _load_matrix(
        train_run_dir, impact_eps, require_train_split=True
    )
    x_eval, y_eval, eval_conversations, eval_splits = _load_matrix(
        eval_run_dir, impact_eps, require_train_split=False
    )
    overlap = sorted(train_conversations & eval_conversations)
    if overlap:
        raise ValueError(f"Train/eval conversation leakage: {overlap[:10]}")

    output_dir = output_dir or (
        Path(eval_run_dir).expanduser().resolve() / "results" / "experiment4_harm_predictor"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    metric_rows: List[Mapping[str, object]] = []
    coefficient_rows: List[Mapping[str, object]] = []

    # Constant prevalence is a non-learned reference for AP and Brier score.
    prevalence = float(y_train.mean())
    constant_train = np.full(len(y_train), prevalence)
    constant_eval = np.full(len(y_eval), prevalence)
    constant_threshold = _training_threshold(y_train, constant_train)
    metric_rows.append({
        "model": "constant_prevalence",
        "feature_count": 0,
        **_metrics(y_eval, constant_eval, constant_threshold),
    })

    for model_name, names in MODEL_FEATURES.items():
        indices = [feature_index[name] for name in names]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed),
        )
        model.fit(x_train[:, indices], y_train)
        train_probabilities = model.predict_proba(x_train[:, indices])[:, 1]
        threshold = _training_threshold(y_train, train_probabilities)
        eval_probabilities = model.predict_proba(x_eval[:, indices])[:, 1]
        metric_rows.append({
            "model": model_name,
            "feature_count": len(indices),
            **_metrics(y_eval, eval_probabilities, threshold),
        })
        logistic = model.named_steps["logisticregression"]
        for name, coefficient in zip(names, logistic.coef_[0]):
            coefficient_rows.append({
                "model": model_name,
                "feature": name,
                "standardized_log_odds_coefficient": float(coefficient),
            })
        coefficient_rows.append({
            "model": model_name,
            "feature": "intercept",
            "standardized_log_odds_coefficient": float(logistic.intercept_[0]),
        })

    simple_ap = max(
        float(row["average_precision"])
        for row in metric_rows
        if row["model"] in {"margin_only", "distance_only", "speaker_only"}
    )
    metric_rows = [
        {
            **row,
            "beats_all_single_feature_baselines_by_ap": (
                int(float(row["average_precision"]) > simple_ap)
                if row["model"] == "full_lightweight" else ""
            ),
        }
        for row in metric_rows
    ]
    write_rows(output_dir / "harm_predictor_metrics.csv", metric_rows)
    write_rows(output_dir / "harm_predictor_coefficients.csv", coefficient_rows)
    metadata = {
        "impact_eps": impact_eps,
        "seed": seed,
        "max_iter": max_iter,
        "train_run_dir": str(Path(train_run_dir).expanduser().resolve()),
        "eval_run_dir": str(Path(eval_run_dir).expanduser().resolve()),
        "train_splits": sorted(train_splits),
        "eval_splits": sorted(eval_splits),
        "train_conversation_count": len(train_conversations),
        "eval_conversation_count": len(eval_conversations),
        "train_row_count": len(y_train),
        "eval_row_count": len(y_eval),
        "feature_names": list(FEATURE_NAMES),
        "forbidden_predictor_inputs": [
            "source_emotion", "target_emotion", "signed_impact", "correct_mask",
            "pred_mask", "loss_mask", "prob_mask_true",
        ],
        "note": (
            "Signed impact is used only to create train labels and score held-out labels; "
            "it is never part of the predictor input."
        ),
    }
    metadata_path = output_dir / "harm_predictor_metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(
        f"Experiment 4 complete: train={len(y_train)} rows, eval={len(y_eval)} rows; "
        f"outputs={output_dir}",
        flush=True,
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return train_and_evaluate(
        args.train_run_dir,
        args.eval_run_dir,
        args.impact_eps,
        args.seed,
        args.max_iter,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
