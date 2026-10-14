"""Experiment 3: target influence profiles and singleton-removal oracle."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from experiment_analysis_common import (
    MODALITIES,
    STRATEGIES,
    binary_int,
    classification_scores,
    finite_float,
    require_columns,
    resolve_impact_paths,
    write_rows,
)


REQUIRED = {
    "conversation_id", "source_idx", "target_idx", "modality",
    "replacement_strategy", "target_emotion", "loss_full", "loss_mask",
    "signed_impact", "pred_full", "pred_mask", "correct_full", "correct_mask",
}
TargetKey = Tuple[str, int]


@dataclass(frozen=True)
class Baseline:
    label: int
    prediction: int
    loss: float
    correct: int


@dataclass
class Profile:
    count: int = 0
    signed_sum: float = 0.0
    absolute_sum: float = 0.0
    harmful_count: int = 0
    helpful_count: int = 0
    near_zero_count: int = 0

    def add(self, impact: float, epsilon: float) -> None:
        self.count += 1
        self.signed_sum += impact
        self.absolute_sum += abs(impact)
        self.harmful_count += int(impact < -epsilon)
        self.helpful_count += int(impact > epsilon)
        self.near_zero_count += int(abs(impact) <= epsilon)


@dataclass
class OracleChoice:
    loss: float
    prediction: int
    choice: str = "baseline"
    strategy: str = "baseline"
    modality: str = "none"
    source_idx: int | None = None
    signed_impact: float = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    return parser.parse_args(argv)


def _same_baseline(first: Baseline, second: Baseline) -> bool:
    return (
        first.label == second.label
        and first.prediction == second.prediction
        and first.correct == second.correct
        and math.isclose(first.loss, second.loss, rel_tol=2e-6, abs_tol=2e-6)
    )


def analyze(run_dir: Path, impact_eps: float = 1e-4) -> Path:
    if impact_eps < 0:
        raise ValueError("impact_eps must be nonnegative")
    paths = resolve_impact_paths(run_dir)
    result_dir = Path(run_dir).expanduser().resolve() / "results"

    baselines: Dict[TargetKey, Baseline] = {}
    profiles: Dict[Tuple[str, str, TargetKey], Profile] = defaultdict(Profile)
    correcting_counts = defaultdict(int)
    correcting_best_gain = defaultdict(float)
    # Candidates are retained separately by strategy and modality scope.
    candidates: Dict[Tuple[str, str, TargetKey], OracleChoice] = {}

    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require_columns(path, reader.fieldnames or [], REQUIRED)
            for line_number, row in enumerate(reader, start=2):
                strategy = row["replacement_strategy"]
                modality = row["modality"]
                if strategy not in STRATEGIES or modality not in MODALITIES:
                    raise ValueError(f"{path}:{line_number}: invalid strategy/modality")
                target_key = (row["conversation_id"], int(row["target_idx"]))
                baseline = Baseline(
                    label=int(row["target_emotion"]),
                    prediction=int(row["pred_full"]),
                    loss=finite_float(row, "loss_full"),
                    correct=binary_int(row, "correct_full"),
                )
                previous = baselines.setdefault(target_key, baseline)
                if not _same_baseline(previous, baseline):
                    raise ValueError(
                        f"{path}:{line_number}: inconsistent baseline for {target_key}"
                    )

                impact = finite_float(row, "signed_impact")
                loss_mask = finite_float(row, "loss_mask")
                if not math.isclose(
                    impact, loss_mask - baseline.loss, rel_tol=2e-6, abs_tol=2e-6
                ):
                    raise ValueError(f"{path}:{line_number}: inconsistent signed impact")
                pred_mask = int(row["pred_mask"])
                correct_mask = binary_int(row, "correct_mask")
                for scope in (modality, "all_modalities"):
                    profiles[(strategy, scope, target_key)].add(impact, impact_eps)
                if not baseline.correct and correct_mask:
                    gain = baseline.loss - loss_mask
                    for scope_strategy, scope_modality in (
                        (strategy, modality),
                        (strategy, "all_modalities"),
                        ("all_strategies", modality),
                        ("all_strategies", "all_modalities"),
                    ):
                        correction_key = (scope_strategy, scope_modality, target_key)
                        correcting_counts[correction_key] += 1
                        correcting_best_gain[correction_key] = max(
                            correcting_best_gain[correction_key], gain
                        )

                for scope_strategy, scope_modality in (
                    (strategy, modality),
                    (strategy, "all_modalities"),
                    ("all_strategies", modality),
                    ("all_strategies", "all_modalities"),
                ):
                    key = (scope_strategy, scope_modality, target_key)
                    choice = candidates.get(key)
                    if choice is None:
                        choice = OracleChoice(baseline.loss, baseline.prediction)
                        candidates[key] = choice
                    # Strict improvement keeps ties deterministic and favors baseline.
                    if loss_mask < choice.loss:
                        candidates[key] = OracleChoice(
                            loss=loss_mask,
                            prediction=pred_mask,
                            choice="singleton_intervention",
                            strategy=strategy,
                            modality=modality,
                            source_idx=int(row["source_idx"]),
                            signed_impact=impact,
                        )

    if not baselines:
        raise ValueError("No impact rows found")

    profile_rows = []
    for (strategy, scope, target_key), item in sorted(profiles.items()):
        baseline = baselines[target_key]
        profile_rows.append({
            "replacement_strategy": strategy,
            "modality_scope": scope,
            "conversation_id": target_key[0],
            "target_idx": target_key[1],
            "target_emotion": baseline.label,
            "pred_full": baseline.prediction,
            "correct_full": baseline.correct,
            "source_modality_count": item.count,
            "impact_eps": impact_eps,
            "mean_signed_impact": item.signed_sum / item.count,
            "mean_absolute_impact": item.absolute_sum / item.count,
            "harmful_count": item.harmful_count,
            "harmful_rate": item.harmful_count / item.count,
            "helpful_count": item.helpful_count,
            "helpful_rate": item.helpful_count / item.count,
            "near_zero_count": item.near_zero_count,
            "near_zero_rate": item.near_zero_count / item.count,
        })

    wrong_targets = {key for key, value in baselines.items() if not value.correct}
    correction_summary_rows = []
    correction_detail_rows = []
    scopes = [
        *((strategy, modality) for strategy in STRATEGIES for modality in MODALITIES),
        *((strategy, "all_modalities") for strategy in STRATEGIES),
        *(("all_strategies", modality) for modality in MODALITIES),
        ("all_strategies", "all_modalities"),
    ]
    for strategy, modality_scope in scopes:
        corrected = sorted(
            target_key for target_key in wrong_targets
            if correcting_counts[(strategy, modality_scope, target_key)] > 0
        )
        correction_summary_rows.append({
            "replacement_strategy_scope": strategy,
            "modality_scope": modality_scope,
            "baseline_wrong_target_count": len(wrong_targets),
            "unique_wrong_targets_corrected": len(corrected),
            "corrected_target_fraction": len(corrected) / len(wrong_targets),
            "correcting_intervention_count": sum(
                correcting_counts[(strategy, modality_scope, key)] for key in corrected
            ),
        })
        for target_key in corrected:
            baseline = baselines[target_key]
            correction_detail_rows.append({
                "replacement_strategy_scope": strategy,
                "modality_scope": modality_scope,
                "conversation_id": target_key[0],
                "target_idx": target_key[1],
                "target_emotion": baseline.label,
                "pred_full": baseline.prediction,
                "correcting_intervention_count": correcting_counts[
                    (strategy, modality_scope, target_key)
                ],
                "best_true_class_loss_reduction": correcting_best_gain[
                    (strategy, modality_scope, target_key)
                ],
            })

    oracle_metric_rows = []
    oracle_selection_rows = []
    baseline_scores = classification_scores(
        (value.label, value.prediction) for value in baselines.values()
    )
    for strategy, modality_scope in scopes:
        selected = {
            target_key: candidates[(strategy, modality_scope, target_key)]
            for target_key in baselines
        }
        oracle_scores = classification_scores(
            (baselines[key].label, choice.prediction)
            for key, choice in selected.items()
        )
        intervention_count = sum(
            choice.choice == "singleton_intervention" for choice in selected.values()
        )
        oracle_metric_rows.append({
            "replacement_strategy_scope": strategy,
            "modality_scope": modality_scope,
            "target_count": len(baselines),
            "selected_intervention_target_count": intervention_count,
            "selected_intervention_target_fraction": intervention_count / len(baselines),
            "baseline_accuracy": baseline_scores[0],
            "oracle_accuracy": oracle_scores[0],
            "delta_accuracy": oracle_scores[0] - baseline_scores[0],
            "baseline_weighted_f1": baseline_scores[1],
            "oracle_weighted_f1": oracle_scores[1],
            "delta_weighted_f1": oracle_scores[1] - baseline_scores[1],
            "baseline_macro_f1": baseline_scores[2],
            "oracle_macro_f1": oracle_scores[2],
            "delta_macro_f1": oracle_scores[2] - baseline_scores[2],
            "mean_true_class_loss_reduction": sum(
                baselines[key].loss - choice.loss for key, choice in selected.items()
            ) / len(baselines),
        })
        for target_key, choice in sorted(selected.items()):
            baseline = baselines[target_key]
            oracle_selection_rows.append({
                "replacement_strategy_scope": strategy,
                "modality_scope": modality_scope,
                "conversation_id": target_key[0],
                "target_idx": target_key[1],
                "target_emotion": baseline.label,
                "pred_full": baseline.prediction,
                "correct_full": baseline.correct,
                "loss_full": baseline.loss,
                "choice": choice.choice,
                "chosen_strategy": choice.strategy,
                "chosen_modality": choice.modality,
                "chosen_source_idx": choice.source_idx,
                "chosen_prediction": choice.prediction,
                "chosen_loss": choice.loss,
                "true_class_loss_reduction": baseline.loss - choice.loss,
                "chosen_signed_impact": choice.signed_impact,
                "correct_after_oracle": int(choice.prediction == baseline.label),
            })

    write_rows(result_dir / "experiment3_target_influence_profiles.csv", profile_rows)
    write_rows(
        result_dir / "experiment3_unique_wrong_targets_corrected.csv",
        correction_summary_rows,
    )
    # A valid model can theoretically have no corrected wrong target. Keep the
    # summary useful and omit the empty detail file in that unusual case.
    if correction_detail_rows:
        write_rows(
            result_dir / "experiment3_corrected_target_details.csv",
            correction_detail_rows,
        )
    write_rows(result_dir / "experiment3_oracle_metrics.csv", oracle_metric_rows)
    write_rows(
        result_dir / "experiment3_oracle_target_selections.csv", oracle_selection_rows
    )
    print(
        f"Experiment 3 complete: {len(baselines)} unique targets in {result_dir}",
        flush=True,
    )
    return result_dir


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return analyze(args.run_dir, args.impact_eps)


if __name__ == "__main__":
    main()
