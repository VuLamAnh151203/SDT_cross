"""E1: source-specific mixed beneficial/harmful contributions within target-modality."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (EPS_VALUES, cluster_bootstrap, distance_bin, finite, margin_bin,
                    read_csv, require_columns, sha256, write_csv, write_json)


REQUIRED = {
    "conversation_id", "conversation_length", "source_idx", "target_idx",
    "source_speaker", "target_speaker", "absolute_distance", "source_before_target",
    "modality", "replacement_strategy", "source_emotion", "target_emotion",
    "signed_impact", "correct_full",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impact-files", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--edge-file", type=Path,
                        help="Optional E2 stage_intervention_pairs.csv edge-gate control")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--eps-values", type=float, nargs="+", default=EPS_VALUES)
    parser.add_argument("--minimum-source-counts", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--effect-thresholds", type=float, nargs="+",
                        default=(1e-4, 1e-3, 1e-2))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def _baseline(path: Path):
    rows = read_csv(path)
    probabilities = sorted(
        [name for name in rows[0] if name.startswith("prob_") and name != "prob_true"],
        key=lambda name: int(name.split("_")[1]),
    )
    result = {}
    for row in rows:
        values = sorted(float(row[name]) for name in probabilities)
        key = (row["conversation_id"], int(row["utterance_index"]))
        result[key] = {
            "baseline_margin": values[-1] - values[-2],
            "baseline_correct": int(row["correct"]),
            "target_emotion": int(row["label"]),
        }
    return result


def _sign_diversity(positive: int, negative: int, influential: int) -> float:
    if not influential:
        return 0.0
    p = positive / influential
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2.0))


def _rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _correlation(first, second):
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def _strategy_robustness(pairs, epsilon):
    output = []
    for modality in sorted({row["modality"] for row in pairs}):
        strategies = sorted({row["replacement_strategy"] for row in pairs
                             if row["modality"] == modality})
        maps = {}
        for strategy in strategies:
            maps[strategy] = {
                (row["conversation_id"], row["source_idx"], row["target_idx"]):
                float(row["signed_impact"])
                for row in pairs if row["modality"] == modality
                and row["replacement_strategy"] == strategy
            }
        for first_index, first_name in enumerate(strategies):
            for second_name in strategies[first_index + 1:]:
                keys = sorted(maps[first_name].keys() & maps[second_name].keys())
                if not keys:
                    continue
                first = np.asarray([maps[first_name][key] for key in keys])
                second = np.asarray([maps[second_name][key] for key in keys])
                influential = (np.abs(first) > epsilon) & (np.abs(second) > epsilon)
                by_target = defaultdict(list)
                for index, key in enumerate(keys):
                    by_target[(key[0], key[2])].append(index)
                target_rank = []
                for indices in by_target.values():
                    if len(indices) >= 2:
                        value = _correlation(
                            _rank(first[indices]), _rank(second[indices])
                        )
                        if math.isfinite(value):
                            target_rank.append(value)
                output.append({
                    "modality": modality, "strategy_a": first_name,
                    "strategy_b": second_name, "paired_count": len(keys),
                    "pearson": _correlation(first, second),
                    "spearman_rank": _correlation(_rank(first), _rank(second)),
                    "sign_agreement_both_influential": (
                        float(np.mean(np.sign(first[influential]) == np.sign(second[influential])))
                        if influential.any() else ""
                    ),
                    "mean_within_target_rank_correlation": float(np.mean(target_rank))
                    if target_rank else "",
                    "within_target_count": len(target_rank), "impact_eps": epsilon,
                })
    return output


def _structural_summary(pairs, epsilon):
    grouped = defaultdict(list)
    for row in pairs:
        relation = "same_emotion" if row["source_emotion"] == row["target_emotion"] else "different_emotion"
        key = (row["modality"], row["replacement_strategy"], row["speaker_relation"],
               row["distance_bin"], row["direction"], relation)
        grouped[key].append(float(row["signed_impact"]))
    output = []
    for key, values in sorted(grouped.items()):
        impacts = np.asarray(values)
        output.append({
            "modality": key[0], "replacement_strategy": key[1],
            "speaker_relation": key[2], "distance_bin": key[3],
            "direction": key[4], "emotion_relation": key[5],
            "count": len(values), "mean_signed_impact": float(np.mean(impacts)),
            "mean_absolute_impact": float(np.mean(np.abs(impacts))),
            "beneficial_rate": float(np.mean(impacts > epsilon)),
            "harmful_rate": float(np.mean(impacts < -epsilon)),
            "near_zero_rate": float(np.mean(np.abs(impacts) <= epsilon)),
        })
    return output


def aggregate_pairs(pair_rows, baseline, epsilon: float):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row["conversation_id"], int(row["target_idx"]), row["modality"],
                 row["replacement_strategy"])].append(row)
    output = []
    for (conversation, target, modality, strategy), rows in sorted(grouped.items()):
        impacts = np.asarray([finite(row, "signed_impact") for row in rows])
        beneficial = int(np.sum(impacts > epsilon))
        harmful = int(np.sum(impacts < -epsilon))
        near_zero = int(len(impacts) - beneficial - harmful)
        influential = beneficial + harmful
        metadata = baseline[(conversation, target)]
        output.append({
            "conversation_id": conversation, "target_idx": target,
            "modality": modality, "replacement_strategy": strategy,
            "n_sources": len(rows), "n_beneficial": beneficial,
            "n_harmful": harmful, "n_near_zero": near_zero,
            "mixed": int(beneficial > 0 and harmful > 0),
            "max_benefit": float(max(0.0, impacts.max())),
            "max_harm": float(max(0.0, -impacts.min())),
            "mixed_span": float(max(0.0, impacts.max()) + max(0.0, -impacts.min())),
            "mean_abs_impact": float(np.mean(np.abs(impacts))),
            "harmful_fraction_among_influential": (
                harmful / influential if influential else ""
            ),
            "source_impact_sign_diversity": _sign_diversity(
                beneficial, harmful, influential
            ),
            "baseline_margin": metadata["baseline_margin"],
            "margin_bin": margin_bin(metadata["baseline_margin"]),
            "baseline_correct": metadata["baseline_correct"],
            "target_emotion": metadata["target_emotion"],
            "conversation_length": int(rows[0]["conversation_length"]),
            "impact_eps": epsilon,
        })
    return output


def _mixed_summary(rows, args):
    output = []
    for epsilon in args.eps_values:
        # Recompute flags directly from counts is invalid across epsilon, so the
        # caller provides only primary rows here; sensitivity is rebuilt outside.
        if not math.isclose(epsilon, args.eps):
            continue
        for modality in sorted({row["modality"] for row in rows}):
            for strategy in sorted({row["replacement_strategy"] for row in rows}):
                for minimum in args.minimum_source_counts:
                    for effect_threshold in args.effect_thresholds:
                        for margin in ("all", "low_<0.05", "middle_0.05-0.20", "high_>=0.20"):
                            selected = [row for row in rows
                                        if row["modality"] == modality
                                        and row["replacement_strategy"] == strategy
                                        and int(row["n_sources"]) >= minimum
                                        and (margin == "all" or row["margin_bin"] == margin)]
                            if not selected:
                                continue
                            def strong_mixed(row):
                                return (float(row["max_benefit"]) >= effect_threshold
                                        and float(row["max_harm"]) >= effect_threshold)
                            estimate = float(np.mean([strong_mixed(row) for row in selected]))
                            low, high = cluster_bootstrap(
                                selected,
                                lambda sample: float(np.mean([strong_mixed(row) for row in sample])),
                                args.bootstrap_reps, args.seed + minimum,
                            )
                            output.append({
                                "modality": modality, "replacement_strategy": strategy,
                                "impact_eps": epsilon, "effect_threshold": effect_threshold,
                                "minimum_source_count": minimum,
                                "margin_group": margin, "target_count": len(selected),
                                "mixed_count": sum(strong_mixed(row) for row in selected),
                                "mixed_rate": estimate, "ci95_low": low, "ci95_high": high,
                                "mean_mixed_span": float(np.mean([
                                    float(row["mixed_span"]) for row in selected
                                ])),
                            })
    return output


def analyze(args: argparse.Namespace) -> None:
    baseline = _baseline(args.baseline_file)
    pairs = []
    source_files = []
    seen = set()
    for path in args.impact_files:
        rows = read_csv(path)
        require_columns(rows, REQUIRED)
        source_files.append(str(path.resolve()))
        for row in rows:
            key = (row["conversation_id"], row["source_idx"], row["target_idx"],
                   row["modality"], row["replacement_strategy"])
            if key in seen:
                raise ValueError(f"Duplicate intervention pair: {key}")
            if row["source_idx"] == row["target_idx"]:
                raise ValueError(f"Self pair: {key}")
            if (row["conversation_id"], int(row["target_idx"])) not in baseline:
                raise ValueError(f"Missing baseline target for {key}")
            seen.add(key)
            copy = dict(row)
            impact = finite(row, "signed_impact")
            copy["impact_sign"] = (
                "beneficial" if impact > args.eps
                else "harmful" if impact < -args.eps else "near_zero"
            )
            copy["speaker_relation"] = (
                "same" if row["source_speaker"] == row["target_speaker"] else "cross"
            )
            copy["distance_bin"] = distance_bin(int(row["absolute_distance"]))
            copy["direction"] = (
                "before" if int(row["source_before_target"]) else "after"
            )
            pairs.append(copy)
    if args.edge_file is not None:
        edge_rows = read_csv(args.edge_file)
        source_files.append(str(args.edge_file.resolve()))
        for row in edge_rows:
            if (row["intervention_location"] != "propagation_edge"
                    or row["path_scope"] != "all_source_branches"
                    or float(row["gate"]) != 0.0):
                continue
            converted = {
                "conversation_id": row["conversation_id"],
                "conversation_length": row["conversation_length"],
                "source_idx": row["source_idx"], "target_idx": row["target_idx"],
                "source_speaker": row["source_speaker"],
                "target_speaker": row["target_speaker"],
                "absolute_distance": row["absolute_distance"],
                "source_before_target": row["source_before_target"],
                "modality": row["modality"], "replacement_strategy": "edge_gate_0",
                "source_emotion": row["source_emotion"],
                "target_emotion": row["target_emotion"],
                "signed_impact": row["delta_nll"],
                "correct_full": row["baseline_correct"],
                "impact_sign": ("beneficial" if float(row["delta_nll"]) > args.eps
                                else "harmful" if float(row["delta_nll"]) < -args.eps
                                else "near_zero"),
                "speaker_relation": "same" if row["same_speaker"] == "1" else "cross",
                "distance_bin": distance_bin(int(row["absolute_distance"])),
                "direction": "before" if row["source_before_target"] == "1" else "after",
            }
            key = (converted["conversation_id"], converted["source_idx"],
                   converted["target_idx"], converted["modality"],
                   converted["replacement_strategy"])
            if key in seen:
                raise ValueError(f"Duplicate edge control pair: {key}")
            seen.add(key)
            pairs.append(converted)
    target_rows = aggregate_pairs(pairs, baseline, args.eps)
    output = args.output_dir.resolve() / "E1"
    write_csv(output / "source_impact_pairs.csv", pairs)
    write_csv(output / "target_modality_mixed.csv", target_rows)
    summaries = _mixed_summary(target_rows, args)
    # Full epsilon sensitivity is recomputed from the original pair rows.
    for epsilon in args.eps_values:
        if math.isclose(epsilon, args.eps):
            continue
        epsilon_rows = aggregate_pairs(pairs, baseline, epsilon)
        shadow = argparse.Namespace(**vars(args))
        shadow.eps = epsilon
        shadow.eps_values = (epsilon,)
        summaries.extend(_mixed_summary(epsilon_rows, shadow))
    write_csv(output / "mixed_summary.csv", summaries)
    robustness = _strategy_robustness(pairs, args.eps)
    if robustness:
        write_csv(output / "intervention_robustness.csv", robustness)
    write_csv(output / "heterogeneity_by_structure.csv",
              _structural_summary(pairs, args.eps))
    controls = []
    rng = np.random.default_rng(args.seed)
    for modality in sorted({row["modality"] for row in target_rows}):
        for strategy in sorted({row["replacement_strategy"] for row in target_rows}):
            selected_groups = [row for row in target_rows if row["modality"] == modality
                               and row["replacement_strategy"] == strategy]
            selected_pairs = [row for row in pairs if row["modality"] == modality
                              and row["replacement_strategy"] == strategy]
            if not selected_groups:
                continue
            signs = np.asarray([
                1 if float(row["signed_impact"]) > args.eps
                else -1 if float(row["signed_impact"]) < -args.eps else 0
                for row in selected_pairs
            ])
            sizes = [int(row["n_sources"]) for row in selected_groups]
            shuffled_rates = []
            for _ in range(200):
                permuted = rng.permutation(signs)
                offset, flags = 0, []
                for size in sizes:
                    group = permuted[offset:offset + size]
                    flags.append(np.any(group > 0) and np.any(group < 0))
                    offset += size
                shuffled_rates.append(float(np.mean(flags)))
            controls.append({
                "control": "impact_to_target_reshuffle_descriptive_only",
                "modality": modality, "replacement_strategy": strategy,
                "target_count": len(selected_groups),
                "observed_mixed_rate": float(np.mean([
                    int(row["mixed"]) for row in selected_groups
                ])),
                "reshuffled_mixed_rate_mean": float(np.mean(shuffled_rates)),
                "reshuffled_mixed_rate_std": float(np.std(shuffled_rates)),
                "reshuffle_repetitions": len(shuffled_rates), "causal_null": 0,
            })
    write_csv(output / "negative_controls.csv", controls)
    write_json(output / "manifest.json", {
        "source_files": source_files, "baseline_file": str(args.baseline_file.resolve()),
        "source_sha256": {str(path.resolve()): sha256(path.resolve())
                          for path in args.impact_files},
        "edge_file": str(args.edge_file.resolve()) if args.edge_file else None,
        "edge_sha256": sha256(args.edge_file.resolve()) if args.edge_file else None,
        "baseline_sha256": sha256(args.baseline_file.resolve()),
        "primary_eps": args.eps, "eps_values": list(args.eps_values),
        "pair_rows": len(pairs), "target_modality_rows": len(target_rows),
        "conversation_count": len({row["conversation_id"] for row in pairs}),
        "unique_target_count": len({(row["conversation_id"], row["target_idx"])
                                    for row in pairs}),
        "bootstrap_unit": "conversation", "seed": args.seed,
    })
    print(f"E1 complete: {len(pairs)} pairs, {len(target_rows)} target-modality groups")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.eps < 0 or args.bootstrap_reps < 0:
        raise ValueError("eps/bootstrap-reps must be nonnegative")
    analyze(args)


if __name__ == "__main__":
    main()
