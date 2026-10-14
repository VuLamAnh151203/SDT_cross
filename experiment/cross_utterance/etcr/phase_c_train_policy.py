"""Phase C: predict when masking previous same-speaker text will reduce NLL."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from etcr_common import (
    EPSILON, as_bool, as_float, classification_metrics, cluster_bootstrap,
    probability_columns, read_rows, validate_target_rows, write_rows,
)


FORBIDDEN_FEATURE_TOKENS = (
    "label", "emotion", "transition", "continuity", "impact", "correct",
    "nll_mask", "pred_mask", "prob_mask", "logit_mask",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--impact-eps", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--embedding-components", type=int, default=32)
    return parser.parse_args(argv)


def _load_split(root: Path, split: str):
    rows = read_rows(root / "raw" / f"{split}_train_mean.csv")
    validate_target_rows(rows, "train_mean")
    archive = np.load(root / "raw" / f"{split}_text_embeddings.npz")
    source = archive["source"].astype(np.float32)
    target = archive["target"].astype(np.float32)
    eligible = archive["eligible"].astype(bool)
    keys = archive["row_key"].astype(str)
    expected = np.asarray([f"{row['conversation_id']}\t{row['target_idx']}" for row in rows])
    if len(rows) != source.shape[0] or not np.array_equal(keys, expected):
        raise ValueError(f"Embedding/CSV row alignment failed for {split}")
    if not np.array_equal(eligible, np.asarray([as_bool(row, "eligible") for row in rows])):
        raise ValueError(f"Embedding eligibility mismatch for {split}")
    return rows, source, target


def _observable_features(rows, source, target, pca=None, fit_pca=False,
                         components=32, seed=2024):
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("Phase C requires scikit-learn") from error
    indices = np.asarray([index for index, row in enumerate(rows) if as_bool(row, "eligible")])
    selected = [rows[index] for index in indices]
    prob_names = probability_columns(rows[0], "prob_full_")
    uncertainty_names = [*prob_names, "baseline_margin", "full_entropy"]
    uncertainty = np.asarray([
        [*[as_float(row, name) for name in prob_names],
         as_float(row, "baseline_margin"), as_float(row, "full_entropy")]
        for row in selected
    ], dtype=np.float64)
    structure_names = [
        "log1p_source_distance", "source_position", "target_position",
        "text_cosine_similarity", "log1p_text_l2_distance",
        "log1p_source_text_norm", "log1p_target_text_norm",
    ]
    structure = np.asarray([
        [math.log1p(int(row["source_distance"])), as_float(row, "source_position"),
         as_float(row, "target_position"), as_float(row, "text_cosine_similarity"),
         math.log1p(as_float(row, "text_l2_distance")),
         math.log1p(as_float(row, "source_text_norm")),
         math.log1p(as_float(row, "target_text_norm"))]
        for row in selected
    ], dtype=np.float64)
    raw_embedding = np.concatenate(
        [source[indices], target[indices], target[indices] - source[indices]], axis=1
    )
    if fit_pca:
        count = min(components, raw_embedding.shape[0] - 1, raw_embedding.shape[1])
        if count < 1:
            raise ValueError("Not enough training targets for embedding PCA")
        pca = PCA(n_components=count, svd_solver="randomized", random_state=seed)
        embedding = pca.fit_transform(raw_embedding)
    else:
        if pca is None:
            raise ValueError("A train-fitted PCA is required")
        embedding = pca.transform(raw_embedding)
    embedding_names = [f"train_pca_embedding_{index}" for index in range(embedding.shape[1])]
    feature_names = {
        "uncertainty_only": uncertainty_names,
        "structure_only": structure_names,
        "learned_full": uncertainty_names + structure_names + embedding_names,
    }
    matrices = {
        "uncertainty_only": uncertainty,
        "structure_only": structure,
        "learned_full": np.concatenate([uncertainty, structure, embedding], axis=1),
    }
    for names in feature_names.values():
        lowered = [name.lower() for name in names]
        leakage = [name for name in lowered
                   if any(token in name for token in FORBIDDEN_FEATURE_TOKENS)]
        if leakage:
            raise AssertionError(f"Forbidden Phase C predictor features: {leakage}")
    return indices, matrices, feature_names, pca


def _fit_models(matrices, labels, max_iter: int, seed: int):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("Phase C requires scikit-learn") from error
    if np.unique(labels).size != 2:
        raise ValueError("Training help_to_mask labels contain only one class")
    models = {}
    for name, matrix in matrices.items():
        models[name] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs"),
        ).fit(matrix, labels)
    return models


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(labels[selected])) - float(np.mean(scores[selected]))
            )
    return result


def _predictor_metric(model_name: str, split: str, labels: np.ndarray,
                      scores: np.ndarray, threshold: float) -> dict[str, object]:
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except ImportError as error:
        raise RuntimeError("Phase C requires scikit-learn") from error
    action = scores >= threshold
    return {
        "model": model_name, "split": split, "count": len(labels),
        "positive_count": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else "",
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_10_bins": _ece(labels, scores),
        "threshold_selected_on_validation": threshold,
        "action_count": int(action.sum()), "action_rate": float(np.mean(action)),
        "action_precision": float(np.mean(labels[action])) if np.any(action) else "",
    }


def _select_threshold(rows, eligible_indices: np.ndarray, scores: np.ndarray) -> float:
    """Select the validation threshold giving the lowest observed policy NLL."""
    delta = np.asarray([
        as_float(rows[index], "nll_mask") - as_float(rows[index], "nll_full")
        for index in eligible_indices
    ])
    candidates = np.concatenate(([math.inf], np.unique(scores)))
    objectives = np.asarray([float(np.sum(delta[scores >= threshold]))
                             for threshold in candidates])
    best = int(np.argmin(objectives))
    return float(candidates[best])


def _policy_masks(rows, eligible_indices, learned_scores, uncertainty_scores,
                  structure_scores, threshold, seed):
    eligible = np.asarray([as_bool(row, "eligible") for row in rows])
    learned_local = learned_scores >= threshold
    learned = np.zeros(len(rows), dtype=bool)
    learned[eligible_indices] = learned_local
    budget = int(learned.sum())

    def top_budget(scores):
        result = np.zeros(len(rows), dtype=bool)
        if budget:
            order = np.argsort(-scores, kind="stable")[:budget]
            result[eligible_indices[order]] = True
        return result

    rng = np.random.default_rng(seed)
    random_mask = np.zeros(len(rows), dtype=bool)
    eligible_global = np.flatnonzero(eligible)
    if budget:
        random_mask[rng.choice(eligible_global, budget, replace=False)] = True
    low_margin = np.zeros(len(rows), dtype=bool)
    if budget:
        margins = np.asarray([as_float(rows[index], "baseline_margin")
                              for index in eligible_indices])
        low_margin[eligible_indices[np.argsort(margins, kind="stable")[:budget]]] = True
    oracle = np.asarray([
        as_bool(row, "eligible") and as_float(row, "nll_mask") < as_float(row, "nll_full")
        for row in rows
    ])
    return {
        "original_sdt": np.zeros(len(rows), dtype=bool),
        "always_mask": eligible,
        "random_same_budget": random_mask,
        "low_margin_same_budget": low_margin,
        "uncertainty_same_budget": top_budget(uncertainty_scores),
        "structure_same_budget": top_budget(structure_scores),
        "learned_policy": learned,
        "oracle_ground_truth": oracle,
    }


def _policy_score(rows, mask: np.ndarray):
    labels = [int(row["target_label"]) for row in rows]
    predictions = [int(row["pred_mask"] if action else row["pred_full"])
                   for row, action in zip(rows, mask)]
    nll = [as_float(row, "nll_mask" if action else "nll_full")
           for row, action in zip(rows, mask)]
    metrics = classification_metrics(labels, predictions)
    metrics["nll"] = float(np.mean(nll))
    return metrics, predictions


def _mitigation_rows(rows, policies, repetitions: int, seed: int):
    subsets = {
        "all_test": np.ones(len(rows), dtype=bool),
        "all_eligible": np.asarray([as_bool(row, "eligible") for row in rows]),
        "continuity": np.asarray([as_bool(row, "continuity") for row in rows]),
        "transition": np.asarray([as_bool(row, "transition") for row in rows]),
    }
    baseline_mask = policies["original_sdt"]
    output = []
    for policy_offset, (policy, mask) in enumerate(policies.items()):
        for subset_offset, (subset, selection) in enumerate(subsets.items()):
            indices = np.flatnonzero(selection)
            selected_rows = [rows[index] for index in indices]
            selected_mask = mask[indices]
            scores, predictions = _policy_score(selected_rows, selected_mask)
            baseline, baseline_predictions = _policy_score(
                selected_rows, np.zeros(len(selected_rows), dtype=bool)
            )
            row: dict[str, object] = {
                "policy": policy, "subset": subset, "count": len(indices),
                "mask_count": int(selected_mask.sum()),
                "action_rate": float(np.mean(selected_mask)),
                "corrected_count": sum(
                    int(base != int(item["target_label"]) and final == int(item["target_label"]))
                    for item, base, final in zip(selected_rows, baseline_predictions, predictions)
                ),
                "damaged_count": sum(
                    int(base == int(item["target_label"]) and final != int(item["target_label"]))
                    for item, base, final in zip(selected_rows, baseline_predictions, predictions)
                ),
                "uses_ground_truth_for_decision": int(policy == "oracle_ground_truth"),
            }
            for metric in ("accuracy", "weighted_f1", "macro_f1", "nll"):
                delta = scores[metric] - baseline[metric]
                augmented = []
                for local_index, item in enumerate(selected_rows):
                    copy = dict(item)
                    copy["_action"] = int(selected_mask[local_index])
                    augmented.append(copy)
                low, high = cluster_bootstrap(
                    augmented,
                    lambda sample, name=metric: (
                        _policy_score(sample, np.asarray([bool(int(x["_action"])) for x in sample]))[0][name]
                        - _policy_score(sample, np.zeros(len(sample), dtype=bool))[0][name]
                    ), repetitions, seed + policy_offset * 101 + subset_offset,
                )
                row.update({f"baseline_{metric}": baseline[metric], f"policy_{metric}": scores[metric],
                            f"delta_{metric}": delta, f"delta_{metric}_ci95_low": low,
                            f"delta_{metric}_ci95_high": high})
            output.append(row)
    return output


def train_and_evaluate(args: argparse.Namespace) -> None:
    root = args.run_dir.expanduser().resolve() / "etcr"
    loaded = {split: _load_split(root, split) for split in ("train", "valid", "test")}
    conversation_sets = {
        split: {row["conversation_id"] for row in data[0]} for split, data in loaded.items()
    }
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = conversation_sets[first] & conversation_sets[second]
        if overlap:
            raise AssertionError(f"Conversation leakage between {first} and {second}: {sorted(overlap)[:3]}")

    train_rows, train_source, train_target = loaded["train"]
    train_indices, train_matrices, feature_names, pca = _observable_features(
        train_rows, train_source, train_target, fit_pca=True,
        components=args.embedding_components, seed=args.seed,
    )
    # Build held-out matrices with the PCA fitted exclusively on training embeddings.
    split_features = {"train": (train_indices, train_matrices)}
    for split in ("valid", "test"):
        rows, source, target = loaded[split]
        indices, matrices, _, _ = _observable_features(
            rows, source, target, pca=pca, fit_pca=False,
            components=args.embedding_components, seed=args.seed,
        )
        split_features[split] = (indices, matrices)
    labels = {}
    for split, (rows, _, _) in loaded.items():
        indices = split_features[split][0]
        labels[split] = np.asarray([
            as_float(rows[index], "signed_impact") < -args.impact_eps for index in indices
        ], dtype=int)
    models = _fit_models(train_matrices, labels["train"], args.max_iter, args.seed)
    scores = {
        split: {name: model.predict_proba(split_features[split][1][name])[:, 1]
                for name, model in models.items()}
        for split in ("valid", "test")
    }
    valid_rows = loaded["valid"][0]
    thresholds = {
        name: _select_threshold(valid_rows, split_features["valid"][0],
                                scores["valid"][name])
        for name in models
    }
    threshold = thresholds["learned_full"]
    predictor_metrics = []
    for split in ("valid", "test"):
        for name in models:
            predictor_metrics.append(_predictor_metric(
                name, split, labels[split], scores[split][name], thresholds[name]
            ))

    output = root / "phase_c"
    supervision = []
    for split in ("train", "valid"):
        rows = loaded[split][0]
        for index in split_features[split][0]:
            row = rows[index]
            supervision.append({
                "split": split, "conversation_id": row["conversation_id"],
                "source_idx": row["source_idx"], "target_idx": row["target_idx"],
                "signed_impact": row["signed_impact"],
                "help_to_mask": int(as_float(row, "signed_impact") < -args.impact_eps),
                "impact_eps": args.impact_eps,
            })
    write_rows(output / "train_val_labels.csv", supervision)
    write_rows(output / "predictor_metrics.csv", predictor_metrics)

    test_rows = loaded["test"][0]
    test_indices = split_features["test"][0]
    policies = _policy_masks(
        test_rows, test_indices, scores["test"]["learned_full"],
        scores["test"]["uncertainty_only"], scores["test"]["structure_only"],
        threshold, args.seed,
    )
    decisions = []
    eligible_score = {int(index): float(score) for index, score in
                      zip(test_indices, scores["test"]["learned_full"])}
    learned_mask = policies["learned_policy"]
    for index, row in enumerate(test_rows):
        action = bool(learned_mask[index])
        decisions.append({
            "conversation_id": row["conversation_id"], "target_idx": row["target_idx"],
            "eligible": row["eligible"], "score_help_to_mask": eligible_score.get(index, ""),
            "threshold_selected_on_validation": threshold,
            "decision": "mask" if action else "keep",
            "pred_full": row["pred_full"], "pred_mask": row["pred_mask"],
            "final_prediction": row["pred_mask"] if action else row["pred_full"],
            "decision_uses_test_label_or_impact": 0,
        })
    write_rows(output / "policy_decisions.csv", decisions)
    write_rows(output / "mitigation_metrics.csv",
               _mitigation_rows(test_rows, policies, args.bootstrap_reps, args.seed))
    artifact = {
        "models": models, "embedding_pca": pca, "feature_names": feature_names,
        "threshold": threshold, "validation_thresholds": thresholds,
        "impact_eps": args.impact_eps,
        "training_split": "train", "threshold_split": "valid",
        "test_labels_or_impacts_used_for_fit_or_threshold": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "reliability_predictor.tmp.pkl"
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output / "reliability_predictor.pkl")
    print(f"Phase C complete; validation-selected threshold={threshold:.8g}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.impact_eps < 0 or args.bootstrap_reps < 0 or args.embedding_components < 1:
        raise ValueError("Invalid nonnegative count/threshold argument")
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
