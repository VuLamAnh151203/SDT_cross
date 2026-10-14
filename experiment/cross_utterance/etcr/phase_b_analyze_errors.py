"""Phase B: target-level errors, adjusted transition effects, and oracle bounds."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from etcr_common import (
    EPSILON, as_bool, as_float, classification_metrics, cluster_bootstrap,
    read_rows, validate_target_rows, write_rows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--impact-eps", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args(argv)


def _score(rows: Sequence[Mapping[str, str]], prediction: str, nll: str) -> dict[str, float]:
    labels = [int(row["target_label"]) for row in rows]
    predictions = [int(row[prediction]) for row in rows]
    result = classification_metrics(labels, predictions)
    result["nll"] = float(np.mean([as_float(row, nll) for row in rows]))
    return result


def _metric_rows(strategy: str, rows: Sequence[Mapping[str, str]], reps: int,
                 seed: int) -> list[dict[str, object]]:
    subsets = {
        "continuity": [row for row in rows if as_bool(row, "continuity")],
        "transition": [row for row in rows if as_bool(row, "transition")],
        "all_eligible": [row for row in rows if as_bool(row, "eligible")],
        "all_test": list(rows),
    }
    output = []
    for offset, (subset, selected) in enumerate(subsets.items()):
        full, masked = _score(selected, "pred_full", "nll_full"), _score(
            selected, "pred_mask", "nll_mask"
        )
        row: dict[str, object] = {"replacement_strategy": strategy,
                                  "subset": subset, "count": len(selected)}
        for metric in ("accuracy", "weighted_f1", "macro_f1", "nll"):
            delta = masked[metric] - full[metric]
            low, high = cluster_bootstrap(
                selected,
                lambda sample, name=metric: (
                    _score(sample, "pred_mask", "nll_mask")[name]
                    - _score(sample, "pred_full", "nll_full")[name]
                ), reps, seed + offset * 31,
            )
            row.update({f"full_{metric}": full[metric], f"masked_{metric}": masked[metric],
                        f"delta_{metric}": delta, f"delta_{metric}_ci95_low": low,
                        f"delta_{metric}_ci95_high": high})
        output.append(row)
    return output


def _error_row(strategy: str, rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    transitions = [row for row in rows if as_bool(row, "transition")]
    persistence = [row for row in transitions
                   if int(row["pred_full"]) == int(row["source_label"])
                   and int(row["source_label"]) != int(row["target_label"])]
    corrected = [row for row in persistence
                 if int(row["pred_mask"]) == int(row["target_label"])]
    remains = [row for row in persistence
               if int(row["pred_mask"]) == int(row["source_label"])]
    baseline_correct = [row for row in transitions if as_bool(row, "correct_full")]
    damaged = [row for row in baseline_correct if not as_bool(row, "correct_mask")]
    return {
        "replacement_strategy": strategy,
        "transition_targets": len(transitions),
        "persistence_error_count": len(persistence),
        "persistence_error_rate": len(persistence) / len(transitions),
        "corrected_count": len(corrected),
        "correction_rate_among_persistence_errors": (
            len(corrected) / len(persistence) if persistence else 0.0
        ),
        "remains_count": len(remains),
        "remains_rate_among_persistence_errors": (
            len(remains) / len(persistence) if persistence else 0.0
        ),
        "baseline_correct_transition_targets": len(baseline_correct),
        "damaged_count": len(damaged),
        "damage_rate_among_baseline_correct": (
            len(damaged) / len(baseline_correct) if baseline_correct else 0.0
        ),
    }


def _design(rows: Sequence[Mapping[str, str]], means=None, scales=None):
    continuous = np.asarray([
        [as_float(row, "baseline_margin"), math.log1p(int(row["source_distance"]))]
        for row in rows
    ], dtype=np.float64)
    if means is None:
        means = continuous.mean(axis=0)
        scales = continuous.std(axis=0)
        scales[scales == 0] = 1.0
    continuous = (continuous - means) / scales
    matrix = []
    for index, row in enumerate(rows):
        source, target = int(row["source_label"]), int(row["target_label"])
        matrix.append([
            int(row["transition"]), continuous[index, 0], continuous[index, 1],
            int(row["correct_full"]),
            *[int(source == value) for value in range(1, 6)],
            *[int(target == value) for value in range(1, 6)],
        ])
    return np.asarray(matrix, dtype=np.float64), means, scales


def _fit_adjusted(rows: Sequence[Mapping[str, str]], epsilon: float, max_iter: int):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError("Phase B adjusted model requires scikit-learn") from error
    x, means, scales = _design(rows)
    y = np.asarray([as_float(row, "signed_impact") < -epsilon for row in rows], dtype=int)
    if np.unique(y).size != 2:
        raise ValueError("Harmful outcome has only one class")
    model = LogisticRegression(C=1000.0, max_iter=max_iter, solver="lbfgs").fit(x, y)
    x0, x1 = x.copy(), x.copy()
    x0[:, 0], x1[:, 0] = 0.0, 1.0
    risk_difference = float(np.mean(model.predict_proba(x1)[:, 1]
                                    - model.predict_proba(x0)[:, 1]))
    return model, means, scales, float(model.coef_[0, 0]), risk_difference


def _adjusted_effects(rows: Sequence[Mapping[str, str]], epsilon: float,
                      repetitions: int, seed: int, max_iter: int) -> list[dict[str, object]]:
    eligible = [row for row in rows if as_bool(row, "eligible")]
    model, means, scales, coefficient, risk_difference = _fit_adjusted(
        eligible, epsilon, max_iter
    )
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in eligible:
        grouped.setdefault(row["conversation_id"], []).append(row)
    conversations = sorted(grouped)
    rng = np.random.default_rng(seed)
    coefficients, differences = [], []
    for _ in range(repetitions):
        selected = rng.choice(conversations, len(conversations), replace=True)
        sample = [row for conversation in selected for row in grouped[str(conversation)]]
        try:
            _, _, _, coef, difference = _fit_adjusted(sample, epsilon, max_iter)
            coefficients.append(coef)
            differences.append(difference)
        except ValueError:
            continue
    coef_ci = np.quantile(coefficients, [0.025, 0.975]) if coefficients else [math.nan] * 2
    diff_ci = np.quantile(differences, [0.025, 0.975]) if differences else [math.nan] * 2
    return [
        {"effect": "transition_log_odds_coefficient", "estimate": coefficient,
         "ci95_low": float(coef_ci[0]), "ci95_high": float(coef_ci[1]),
         "odds_ratio": math.exp(coefficient), "bootstrap_successes": len(coefficients)},
        {"effect": "transition_average_marginal_risk_difference", "estimate": risk_difference,
         "ci95_low": float(diff_ci[0]), "ci95_high": float(diff_ci[1]),
         "odds_ratio": "", "bootstrap_successes": len(differences)},
    ]


def _robustness(train_mean, zero, epsilon: float) -> list[dict[str, object]]:
    mean_map = {(row["conversation_id"], row["target_idx"]): row for row in train_mean}
    zero_map = {(row["conversation_id"], row["target_idx"]): row for row in zero}
    if mean_map.keys() != zero_map.keys():
        raise ValueError("train_mean and zero targets are not paired")
    selected = [(mean_map[key], zero_map[key]) for key in mean_map
                if as_bool(mean_map[key], "eligible")]
    first = np.asarray([as_float(a, "signed_impact") for a, _ in selected])
    second = np.asarray([as_float(b, "signed_impact") for _, b in selected])
    first_harm = first < -epsilon
    second_harm = second < -epsilon
    return [{
        "count": len(selected), "pearson_signed_impact": float(np.corrcoef(first, second)[0, 1]),
        "mean_absolute_strategy_difference": float(np.mean(np.abs(first - second))),
        "harmful_agreement": float(np.mean(first_harm == second_harm)),
        "both_harmful": int(np.sum(first_harm & second_harm)),
        "train_mean_only_harmful": int(np.sum(first_harm & ~second_harm)),
        "zero_only_harmful": int(np.sum(~first_harm & second_harm)),
    }]


def _oracle_rows(strategy: str, rows: Sequence[Mapping[str, str]], reps: int,
                 seed: int) -> list[dict[str, object]]:
    prepared = []
    for row in rows:
        copy = dict(row)
        use_mask = as_bool(row, "eligible") and as_float(row, "nll_mask") < as_float(row, "nll_full")
        copy["pred_oracle"] = row["pred_mask"] if use_mask else row["pred_full"]
        copy["nll_oracle"] = row["nll_mask"] if use_mask else row["nll_full"]
        copy["oracle_mask"] = int(use_mask)
        prepared.append(copy)
    output = []
    for subset, selected in {
        "all_test": prepared,
        "all_eligible": [row for row in prepared if as_bool(row, "eligible")],
        "continuity": [row for row in prepared if as_bool(row, "continuity")],
        "transition": [row for row in prepared if as_bool(row, "transition")],
    }.items():
        baseline = _score(selected, "pred_full", "nll_full")
        oracle = _score(selected, "pred_oracle", "nll_oracle")
        corrected = sum(
            int(int(row["pred_full"]) != int(row["target_label"])
                and int(row["pred_oracle"]) == int(row["target_label"]))
            for row in selected
        )
        damaged = sum(
            int(int(row["pred_full"]) == int(row["target_label"])
                and int(row["pred_oracle"]) != int(row["target_label"]))
            for row in selected
        )
        output.append({
            "replacement_strategy": strategy, "subset": subset, "count": len(selected),
            "oracle_mask_count": sum(int(row["oracle_mask"]) for row in selected),
            "corrected_count": corrected, "damaged_count": damaged,
            **{f"baseline_{name}": value for name, value in baseline.items()},
            **{f"oracle_{name}": value for name, value in oracle.items()},
            **{f"delta_{name}": oracle[name] - baseline[name] for name in baseline},
            "uses_ground_truth": 1,
        })
    return output


def analyze(args: argparse.Namespace) -> None:
    root = args.run_dir.expanduser().resolve() / "etcr"
    datasets = {}
    for strategy in ("train_mean", "zero"):
        path = root / "raw" / f"test_{strategy}.csv"
        if path.is_file():
            rows = read_rows(path)
            validate_target_rows(rows, strategy)
            datasets[strategy] = rows
    if "train_mean" not in datasets:
        raise FileNotFoundError(root / "raw" / "test_train_mean.csv")
    output = root / "phase_b"
    write_rows(output / "persistence_error_transitions.csv",
               [_error_row(strategy, rows) for strategy, rows in datasets.items()])
    metric_rows = []
    oracle_rows = []
    for offset, (strategy, rows) in enumerate(datasets.items()):
        metric_rows.extend(_metric_rows(strategy, rows, args.bootstrap_reps,
                                        args.seed + offset * 1000))
        oracle_rows.extend(_oracle_rows(strategy, rows, args.bootstrap_reps,
                                        args.seed + offset * 1000))
    write_rows(output / "target_level_metrics.csv", metric_rows)
    write_rows(output / "oracle_metrics.csv", oracle_rows)
    adjusted = _adjusted_effects(datasets["train_mean"], args.impact_eps,
                                 args.bootstrap_reps, args.seed, args.max_iter)
    for row in adjusted:
        row["replacement_strategy"] = "train_mean"
        row["controls"] = "margin+log_distance+source_emotion+target_emotion+baseline_correctness"
    write_rows(output / "adjusted_transition_effects.csv", adjusted)
    if "zero" in datasets:
        write_rows(output / "strategy_robustness.csv",
                   _robustness(datasets["train_mean"], datasets["zero"], args.impact_eps))
    print(f"Phase B complete: {output}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.impact_eps < 0 or args.bootstrap_reps < 0:
        raise ValueError("impact-eps and bootstrap-reps must be nonnegative")
    analyze(args)


if __name__ == "__main__":
    main()
