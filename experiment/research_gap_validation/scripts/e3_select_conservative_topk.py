"""Select strong-impact teacher filtering and conservative top-K on validation."""

from __future__ import annotations

import argparse
import math
import sys
import time
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

from common import classification_metrics, write_csv, write_json  # noqa: E402
from dataloader import make_loaders  # noqa: E402
from impact_core import load_experiment, seed_everything  # noqa: E402
from e3_posttraining_audit import _load_wrapper  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}
METRICS = ("accuracy", "weighted_f1", "macro_f1", "nll")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--trained-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-k", nargs="+", type=int, default=(0, 1, 3, 5),
        help="0 is the original dense learned gate; positive values keep only K edges.",
    )
    parser.add_argument("--selection-metric", choices=("weighted_f1", "accuracy", "nll"),
                        default="weighted_f1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _configuration(artifact: dict) -> tuple:
    config = artifact["configuration"]
    modalities = config.get("gate_modalities", ("text", "audio", "visual"))
    if isinstance(modalities, str):
        modalities = modalities.replace("+", " ").split()
    return (
        artifact["model_id"],
        float(config.get("initial_gate", 0.99)),
        float(config.get("lambda_impact", 0.0)) if artifact["model_id"] == "B4" else 0.0,
        float(config.get("impact_min_margin", 0.0)) if artifact["model_id"] == "B4" else 0.0,
        "+".join(modalities),
    )


def _metadata(key: tuple) -> dict[str, object]:
    model, initial_gate, lambda_impact, min_margin, modalities = key
    return {
        "model": model, "initial_gate": initial_gate,
        "lambda_impact": lambda_impact, "impact_min_margin": min_margin,
        "gate_modalities": modalities,
    }


def _evaluate(wrapper, loader, experiment, top_k: int,
              max_batches: int) -> dict[str, float]:
    wrapper.eval()
    labels_all, predictions_all = [], []
    nll_sum = 0.0
    utterances = 0
    applied_gate_sum = applied_gate_items = 0
    nonidentity_edges = valid_edges = 0
    strong_targets = targets = 0
    active_indices = list(wrapper.active_gate_indices)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            if max_batches and batch_number >= max_batches:
                break
            text, visual, audio, speakers_lbs, mask, labels = [
                item.to(experiment.device) for item in batch[:6]
            ]
            speakers = speakers_lbs.transpose(0, 1)
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            output = wrapper(
                text, visual, audio, mask, speakers, lengths,
                pair_gate_top_k=top_k,
            )
            prediction = output["logits"].argmax(dim=-1)
            log_prob = output["log_prob"][valid]
            count = int(valid.sum())
            nll_sum += float(-log_prob.gather(
                1, labels[valid].unsqueeze(-1)
            ).sum().cpu())
            utterances += count
            labels_all.extend(labels[valid].cpu().tolist())
            predictions_all.extend(prediction[valid].cpu().tolist())

            edge_gate = output["edge_gate_applied"]
            if edge_gate is not None and active_indices:
                length = edge_gate.size(1)
                diagonal = torch.eye(length, dtype=torch.bool,
                                     device=experiment.device)
                pair_mask = (valid[:, :, None] & valid[:, None, :]
                             & ~diagonal[None])
                selected = edge_gate[..., active_indices]
                expanded = pair_mask.unsqueeze(-1).expand_as(selected)
                values = selected[expanded]
                applied_gate_sum += float(values.sum().cpu())
                applied_gate_items += int(values.numel())
                nonidentity_edges += int((values < 1.0 - 1e-7).sum().cpu())
                valid_edges += int(values.numel())
                for batch_index, length_value in enumerate(lengths):
                    for target in range(length_value):
                        candidates = selected[batch_index, target, :length_value]
                        nonself = torch.arange(length_value, device=experiment.device) != target
                        strong_targets += int((candidates[nonself] < 0.5).any().cpu())
                        targets += 1
    if not utterances:
        raise ValueError("Empty validation/test loader in conservative top-K evaluation")
    metrics = classification_metrics(labels_all, predictions_all)
    metrics.update({
        "nll": nll_sum / utterances,
        "utterances": utterances,
        "mean_applied_gate": (
            applied_gate_sum / applied_gate_items if applied_gate_items else 1.0
        ),
        "nonidentity_edge_rate": (
            nonidentity_edges / valid_edges if valid_edges else 0.0
        ),
        "strong_suppression_target_rate": (
            strong_targets / targets if targets else 0.0
        ),
        "seconds": time.perf_counter() - started,
    })
    return metrics


def _score(row: dict[str, object], metric: str) -> tuple[float, float, float]:
    accuracy = float(row["valid_accuracy_mean"])
    weighted_f1 = float(row["valid_weighted_f1_mean"])
    negative_nll = -float(row["valid_nll_mean"])
    if metric == "accuracy":
        return accuracy, weighted_f1, negative_nll
    if metric == "nll":
        return negative_nll, weighted_f1, accuracy
    return weighted_f1, accuracy, negative_nll


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def run(args: argparse.Namespace) -> None:
    if not args.top_k or any(value < 0 for value in args.top_k):
        raise ValueError("--top-k must contain nonnegative integers")
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    if not experiment.split_ids.get("valid"):
        raise ValueError("Conservative top-K selection requires a validation split")

    artifacts = []
    seen = set()
    loader_cache = {}
    validation_rows = []
    for path in sorted(args.trained_checkpoints, key=lambda value: value.name):
        wrapper, artifact = _load_wrapper(experiment, path)
        seed = int(artifact["seed"])
        key = _configuration(artifact)
        identity = (key, seed)
        if identity in seen:
            raise ValueError(f"Duplicate checkpoint for configuration/seed: {path}")
        seen.add(identity)
        artifacts.append((path, artifact, key))
        if seed not in loader_cache:
            loader_cache[seed] = make_loaders(
                experiment.dataset, experiment.split_ids, args.batch_size, seed,
                args.num_workers, experiment.device.type == "cuda",
            )
        for top_k in sorted(set(args.top_k)):
            seed_everything(seed)
            metrics = _evaluate(
                wrapper, loader_cache[seed]["valid"], experiment, top_k,
                args.max_batches,
            )
            validation_rows.append({
                **_metadata(key), "seed": seed, "top_k": top_k,
                "split": "valid", **metrics, "checkpoint": str(path.resolve()),
            })
            print(
                f"top-K valid model={artifact['model_id']} seed={seed} "
                f"margin={key[3]:g} K={top_k} WF1={metrics['weighted_f1']:.3f}",
                flush=True,
            )

    grouped = defaultdict(list)
    for row in validation_rows:
        group = (
            row["model"], float(row["initial_gate"]), float(row["lambda_impact"]),
            float(row["impact_min_margin"]), row["gate_modalities"], int(row["top_k"]),
        )
        grouped[group].append(row)
    selection_rows = []
    for group, rows in sorted(grouped.items()):
        key, top_k = group[:5], group[5]
        summary = {**_metadata(key), "top_k": top_k, "seeds": len(rows)}
        for metric in METRICS + ("mean_applied_gate", "nonidentity_edge_rate",
                                 "strong_suppression_target_rate"):
            mean, std = _mean_std([float(row[metric]) for row in rows])
            summary[f"valid_{metric}_mean"] = mean
            summary[f"valid_{metric}_std"] = std
        selection_rows.append(summary)

    winners = {}
    winner_rows = []
    for model in sorted({row["model"] for row in selection_rows}):
        candidates = [row for row in selection_rows if row["model"] == model]
        winner = max(candidates, key=lambda row: _score(row, args.selection_metric))
        winner = {**winner, "selected_on": f"validation_{args.selection_metric}"}
        winner_rows.append(winner)
        winners[model] = (
            model, float(winner["initial_gate"]), float(winner["lambda_impact"]),
            float(winner["impact_min_margin"]), winner["gate_modalities"],
        )

    test_rows = []
    for path, artifact, key in artifacts:
        model = artifact["model_id"]
        if key != winners[model]:
            continue
        seed = int(artifact["seed"])
        top_k = int(next(row["top_k"] for row in winner_rows
                         if row["model"] == model))
        wrapper, _ = _load_wrapper(experiment, path)
        seed_everything(seed)
        metrics = _evaluate(
            wrapper, loader_cache[seed]["test"], experiment, top_k,
            args.max_batches,
        )
        test_rows.append({
            **_metadata(key), "seed": seed, "top_k": top_k,
            "split": "test", **metrics, "checkpoint": str(path.resolve()),
        })

    test_summary = []
    for model in sorted({row["model"] for row in test_rows}):
        rows = [row for row in test_rows if row["model"] == model]
        first = rows[0]
        summary = {
            "model": model, "initial_gate": first["initial_gate"],
            "lambda_impact": first["lambda_impact"],
            "impact_min_margin": first["impact_min_margin"],
            "gate_modalities": first["gate_modalities"],
            "top_k": first["top_k"], "seeds": len(rows),
        }
        for metric in METRICS + ("mean_applied_gate", "nonidentity_edge_rate",
                                 "strong_suppression_target_rate"):
            mean, std = _mean_std([float(row[metric]) for row in rows])
            summary[f"test_{metric}_mean"] = mean
            summary[f"test_{metric}_std"] = std
        test_summary.append(summary)

    paired_rows = []
    by_model_seed = {(row["model"], int(row["seed"])): row for row in test_rows}
    if "B3" in winners and "B4" in winners:
        seeds = sorted(
            {seed for model, seed in by_model_seed if model == "B3"}
            & {seed for model, seed in by_model_seed if model == "B4"}
        )
        for metric in METRICS:
            deltas = [
                float(by_model_seed[("B4", seed)][metric])
                - float(by_model_seed[("B3", seed)][metric])
                for seed in seeds
            ]
            mean, std = _mean_std(deltas)
            sem = std / math.sqrt(len(deltas)) if deltas else float("nan")
            paired_rows.append({
                "comparison": "B4-B3", "metric": metric, "seeds": len(seeds),
                "delta_mean": mean, "delta_std": std,
                "normal_approx_ci95_low": mean - 1.96 * sem,
                "normal_approx_ci95_high": mean + 1.96 * sem,
            })

    output = args.output_dir.resolve()
    write_csv(output / "topk_validation_runs.csv", validation_rows)
    write_csv(output / "topk_validation_summary.csv", selection_rows)
    write_csv(output / "topk_selected_config.csv", winner_rows)
    write_csv(output / "topk_selected_test_runs.csv", test_rows)
    write_csv(output / "topk_selected_test_summary.csv", test_summary)
    if paired_rows:
        write_csv(output / "topk_selected_paired_deltas.csv", paired_rows)
    write_json(output / "topk_manifest.json", {
        "base_checkpoint_id": experiment.checkpoint_id,
        "selection_split": "valid", "selection_metric": args.selection_metric,
        "top_k_values": sorted(set(args.top_k)),
        "checkpoint_count": len(artifacts),
        "test_evaluated_only_for_validation_selected_configuration": True,
        "smoke_limited": bool(args.max_batches),
    })
    print(f"Conservative top-K results written to: {output}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
