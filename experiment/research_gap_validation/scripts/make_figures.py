"""Create the pre-registered static figures from E1--E4 result tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common import read_csv  # noqa: E402


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    _plt().close(fig)


def make(output_dir: Path) -> None:
    plt = _plt()
    root, figures = output_dir.resolve(), output_dir.resolve() / "figures"
    e1 = read_csv(root / "E1" / "target_modality_mixed.csv")
    text = [row for row in e1 if row["modality"] == "text"
            and row["replacement_strategy"] == "train_mean"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([float(row["max_benefit"]) for row in text],
               [float(row["max_harm"]) for row in text], s=9, alpha=.35)
    ax.set(xlabel="Strongest beneficial impact", ylabel="Strongest harmful magnitude")
    _save(fig, figures / "E1_mixed_source_impacts.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([float(row["baseline_margin"]) for row in text],
               [float(row["mixed_span"]) for row in text], s=9, alpha=.35)
    ax.set(xlabel="Baseline margin", ylabel="Beneficial + harmful span")
    _save(fig, figures / "E1_mixed_vs_margin.png")

    e2 = read_csv(root / "E2" / "stage_intervention_pairs.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for location in ("propagation_edge", "late_modality"):
        rows = [row for row in e2 if row["intervention_location"] == location
                and row["path_scope"] != "one_source_branch"]
        ax.scatter([float(row["normalized_cost"]) for row in rows],
                   [-float(row["delta_nll"]) for row in rows], s=7, alpha=.2,
                   label=location)
    ax.set(xlabel="Normalized intervention cost", ylabel="NLL reduction")
    ax.legend()
    _save(fig, figures / "E2_stage_pareto.png")
    best = sorted(e2, key=lambda row: float(row["delta_nll"]))[:30]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(best)), [-float(row["delta_nll"]) for row in best])
    ax.set(xlabel="Top frozen intervention cases", ylabel="NLL reduction")
    _save(fig, figures / "E2_case_studies.png")

    comparison_path = root / "E3" / "model_comparison.csv"
    if comparison_path.is_file():
        comparison = read_csv(comparison_path)
        models = sorted({row["model"] for row in comparison})
        means = [np.mean([float(row["weighted_f1"]) for row in comparison
                         if row["model"] == model]) for model in models]
        stds = [np.std([float(row["weighted_f1"]) for row in comparison
                       if row["model"] == model]) for model in models]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(models, means, yerr=stds, capsize=3)
        ax.set(ylabel="Test weighted F1", xlabel="Model")
        _save(fig, figures / "E3_performance.png")
        subsets = read_csv(root / "E3" / "transition_subset_metrics.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        width = .35
        x = np.arange(len(models))
        for offset, subset in ((-.5, "continuity"), (.5, "transition")):
            values = [np.mean([float(row["weighted_f1"]) for row in subsets
                              if row["model"] == model and row["subset"] == subset])
                      for model in models]
            ax.bar(x + offset * width, values, width, label=subset)
        ax.set_xticks(x, models)
        ax.set_ylabel("Weighted F1")
        ax.legend()
        _save(fig, figures / "E3_continuity_transition.png")
        predictions = read_csv(root / "E3" / "test_predictions.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([[float(row["gate_mean"]) for row in predictions if row["model"] == model]
                    for model in models], labels=models, showfliers=False)
        ax.set(ylabel="Mean gate", xlabel="Model")
        _save(fig, figures / "E3_gate_behavior.png")

    robustness_path = root / "E4" / "ranking_metrics.csv"
    if robustness_path.is_file():
        ranking = read_csv(robustness_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        selected = [row for row in ranking if row["impact_eps"] == "0.0001"]
        labels = [f"{row['model']}-{row['seed']}" for row in selected]
        values = [float(row["auroc_gate_ranks_beneficial"] or "nan") for row in selected]
        ax.bar(labels, values)
        ax.axhline(.5, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=60)
        ax.set(ylabel="Gate ranking AUROC")
        _save(fig, figures / "E4_pareto_robustness.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    make(parser.parse_args(argv).output_dir)


if __name__ == "__main__":
    main()
