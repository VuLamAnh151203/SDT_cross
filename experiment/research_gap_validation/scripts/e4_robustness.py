"""E4: robustness, ranking, and shortcut/falsification analyses."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (EPS_VALUES, classification_metrics, read_csv, session_id,
                    write_csv, write_json)  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eps-values", type=float, nargs="+", default=EPS_VALUES)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def _safe_corr(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def _ranking_rows(audit_paths, eps_values):
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise RuntimeError("E4 ranking metrics require scikit-learn") from error
    output = []
    for path in audit_paths:
        rows = read_csv(path)
        model, seed = rows[0]["model"], int(rows[0]["seed"])
        for epsilon in eps_values:
            selected = [row for row in rows if abs(float(row["signed_impact"])) > epsilon]
            if not selected:
                continue
            labels = np.asarray([float(row["signed_impact"]) > epsilon for row in selected],
                                dtype=int)
            scores = np.asarray([float(row["gate_value"]) for row in selected])
            target_scores = []
            grouped = defaultdict(list)
            for row in selected:
                grouped[(row["conversation_id"], row["target_idx"], row["modality"])].append(row)
            for group in grouped.values():
                group_labels = np.asarray([float(row["signed_impact"]) > epsilon for row in group],
                                          dtype=int)
                if np.unique(group_labels).size < 2:
                    continue
                group_scores = np.asarray([float(row["gate_value"]) for row in group])
                target_scores.append(roc_auc_score(group_labels, group_scores))
            output.append({
                "model": model, "seed": seed, "impact_eps": epsilon,
                "pair_count": len(selected), "positive_beneficial_count": int(labels.sum()),
                "auroc_gate_ranks_beneficial": (
                    float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else ""
                ),
                "auprc_gate_ranks_beneficial": float(average_precision_score(labels, scores)),
                "per_target_auc_mean": float(np.mean(target_scores)) if target_scores else "",
                "per_target_auc_target_count": len(target_scores),
                "gate_impact_pearson": _safe_corr(
                    scores, [float(row["signed_impact"]) for row in selected]
                ),
            })
    return output


def analyze(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    e1_pairs = read_csv(root / "E1" / "source_impact_pairs.csv")
    e2 = read_csv(root / "E2" / "stage_intervention_pairs.csv")
    model_comparison = read_csv(root / "E3" / "model_comparison.csv")
    predictions = read_csv(root / "E3" / "test_predictions.csv")
    audit_paths = sorted((root / "E3" / "audits").glob("*.csv"))
    robustness = []
    for epsilon in args.eps_values:
        for strategy in sorted({row["replacement_strategy"] for row in e1_pairs}):
            for modality in sorted({row["modality"] for row in e1_pairs}):
                selected = [row for row in e1_pairs
                            if row["replacement_strategy"] == strategy
                            and row["modality"] == modality]
                if not selected:
                    continue
                impacts = np.asarray([float(row["signed_impact"]) for row in selected])
                robustness.append({
                    "analysis": "E1_perturbation_sensitivity", "model": "frozen_SDT",
                    "strategy_or_policy": strategy, "modality": modality,
                    "impact_eps": epsilon, "count": len(selected),
                    "harmful_rate": float(np.mean(impacts < -epsilon)),
                    "beneficial_rate": float(np.mean(impacts > epsilon)),
                    "near_zero_rate": float(np.mean(np.abs(impacts) <= epsilon)),
                    "metric": "", "value": "",
                })
    for location in ("propagation_edge", "late_modality"):
        for gate in sorted({row["gate"] for row in e2
                            if row["intervention_location"] == location}, key=float):
            selected = [row for row in e2 if row["intervention_location"] == location
                        and row["gate"] == gate and row["path_scope"] != "one_source_branch"]
            if selected:
                robustness.append({
                    "analysis": "E2_stage_cost", "model": "frozen_SDT",
                    "strategy_or_policy": location, "modality": "all",
                    "impact_eps": "", "count": len(selected),
                    "harmful_rate": "", "beneficial_rate": "", "near_zero_rate": "",
                    "metric": f"mean_delta_nll_gate_{gate}",
                    "value": float(np.mean([float(row["delta_nll"]) for row in selected])),
                })
    for row in model_comparison:
        for metric in ("accuracy", "weighted_f1", "macro_f1", "nll"):
            robustness.append({
                "analysis": "E3_seed_stability", "model": row["model"],
                "strategy_or_policy": f"seed_{row['seed']}", "modality": "all",
                "impact_eps": "", "count": row["test_utterances"],
                "harmful_rate": "", "beneficial_rate": "", "near_zero_rate": "",
                "metric": metric, "value": row[metric],
            })
    conversation_lengths = defaultdict(int)
    for row in predictions:
        key = (row["model"], row["seed"], row["conversation_id"])
        conversation_lengths[key] = max(conversation_lengths[key], int(row["target_idx"]) + 1)
    for model in sorted({row["model"] for row in predictions}):
        for run_seed in sorted({row["seed"] for row in predictions if row["model"] == model}):
            model_rows = [row for row in predictions
                          if row["model"] == model and row["seed"] == run_seed]
            strata = {
                "short_<=30": [row for row in model_rows
                                if conversation_lengths[(model, run_seed, row["conversation_id"])] <= 30],
                "middle_31-60": [row for row in model_rows
                                  if 30 < conversation_lengths[(model, run_seed, row["conversation_id"])] <= 60],
                "long_61+": [row for row in model_rows
                              if conversation_lengths[(model, run_seed, row["conversation_id"])] > 60],
            }
            for session in sorted({session_id(row["conversation_id"]) for row in model_rows}):
                strata[f"session_{session}"] = [
                    row for row in model_rows if session_id(row["conversation_id"]) == session
                ]
            for stratum, selected in strata.items():
                if not selected:
                    continue
                metrics = classification_metrics(
                    [int(row["target_label"]) for row in selected],
                    [int(row["prediction"]) for row in selected],
                )
                robustness.append({
                    "analysis": "E4_length_session_shift", "model": model,
                    "strategy_or_policy": f"seed_{run_seed}_{stratum}",
                    "modality": "all", "impact_eps": "", "count": len(selected),
                    "harmful_rate": "", "beneficial_rate": "", "near_zero_rate": "",
                    "metric": "weighted_f1", "value": metrics["weighted_f1"],
                })
    write_csv(root / "E4" / "robustness_matrix.csv", robustness)
    ranking = _ranking_rows(audit_paths, args.eps_values)
    write_csv(root / "E4" / "ranking_metrics.csv", ranking)

    margin_lookup = {(row["model"], row["seed"], row["conversation_id"], row["target_idx"]):
                     float(row["margin"]) for row in predictions}
    controls = []
    for path in audit_paths:
        rows = read_csv(path)
        model, seed = rows[0]["model"], rows[0]["seed"]
        gates = [float(row["gate_value"]) for row in rows]
        distances = [float(row["absolute_distance"]) for row in rows]
        same = [float(row["same_speaker"]) for row in rows]
        margins = [margin_lookup.get((model, seed, row["conversation_id"], row["target_idx"]),
                                     math.nan) for row in rows]
        finite_margin = [(g, m) for g, m in zip(gates, margins) if math.isfinite(m)]
        controls.extend([
            {"model": model, "seed": seed, "control": "distance_shortcut",
             "count": len(rows), "correlation": _safe_corr(gates, distances),
             "interpretation": "large magnitude suggests gate may mainly encode distance"},
            {"model": model, "seed": seed, "control": "same_speaker_shortcut",
             "count": len(rows), "correlation": _safe_corr(gates, same),
             "interpretation": "large magnitude suggests gate may mainly encode speaker relation"},
            {"model": model, "seed": seed, "control": "margin_shortcut",
             "count": len(finite_margin),
             "correlation": _safe_corr([item[0] for item in finite_margin],
                                        [item[1] for item in finite_margin]),
             "interpretation": "large magnitude suggests gate may mainly encode target uncertainty"},
            {"model": model, "seed": seed, "control": "gate_collapse",
             "count": len(rows), "correlation": float(np.mean(np.asarray(gates) < 0.1)),
             "interpretation": "reported value is fraction of nearly closed gates"},
        ])
    fixed_path = root / "E4" / "fixed_control_metrics.csv"
    if fixed_path.is_file():
        for row in read_csv(fixed_path):
            controls.append({
                "model": row["model"], "seed": row["seed"],
                "control": f"same_budget_{row['control']}",
                "count": row["action_count"], "correlation": "",
                "interpretation": (
                    f"ACC={row['accuracy']}; WF1={row['weighted_f1']}; "
                    f"NLL={row['nll']}; action_rate={row['action_rate']}"
                ),
            })
    write_csv(root / "E4" / "negative_controls.csv", controls)
    write_json(root / "E4" / "manifest.json", {
        "eps_values": args.eps_values, "seed": args.seed,
        "audit_files": [str(path) for path in audit_paths],
        "test_labels_used_only_for_post_hoc_scoring": True,
    })
    print(f"E4 complete: {len(robustness)} robustness rows, {len(ranking)} ranking rows")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if any(value < 0 for value in args.eps_values):
        raise ValueError("eps values must be nonnegative")
    analyze(args)


if __name__ == "__main__":
    main()
