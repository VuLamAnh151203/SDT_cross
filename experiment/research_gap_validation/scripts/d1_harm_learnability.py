"""D1: test whether frozen-SDT harmful edge impacts are learnable.

Three supervision schemes share the same observable pair features and MLP
capacity: binary sign classification, three-class impact classification, and
magnitude-aware signed-impact regression.  Strong-impact thresholds and all
training decisions are selected on validation conversations only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import sha256, write_csv, write_json  # noqa: E402
from d1_prepare_edge_dataset import FEATURE_NAMES  # noqa: E402


SUPERVISIONS = ("binary", "three_class", "magnitude")
REQUIRED_COLUMNS = {
    "predictor_split", "conversation_id", "source_idx", "target_idx",
    "modality", "signed_impact", *FEATURE_NAMES,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--supervisions", nargs="+", choices=SUPERVISIONS,
                        default=SUPERVISIONS)
    parser.add_argument("--strong-eps-candidates", nargs="+", type=float,
                        default=(1e-4, 1e-3, 1e-2))
    parser.add_argument("--base-impact-eps", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=("auroc", "auprc", "spearman"),
                        default="auroc")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=(2024, 2025, 2026))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--precision-k", nargs="+", type=int, default=(1, 3, 5))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Deterministic smoke-only limit; 0 uses every train row")
    return parser.parse_args(argv)


class ImpactMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _device(name: str, gpu_id: int) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("D1 requested CUDA but torch.cuda.is_available() is false")
        torch.cuda.set_device(gpu_id)
        return torch.device("cuda", gpu_id)
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_dataset(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    features: list[list[float]] = []
    impacts: list[float] = []
    splits: list[str] = []
    conversations: list[str] = []
    sources: list[int] = []
    targets: list[int] = []
    modalities: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"D1 dataset missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            split = row["predictor_split"]
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"{path}:{line_number}: invalid predictor split {split!r}")
            vector = [float(row[name]) for name in FEATURE_NAMES]
            impact = float(row["signed_impact"])
            if not all(math.isfinite(value) for value in (*vector, impact)):
                raise ValueError(f"{path}:{line_number}: non-finite D1 value")
            features.append(vector)
            impacts.append(impact)
            splits.append(split)
            conversations.append(row["conversation_id"])
            sources.append(int(row["source_idx"]))
            targets.append(int(row["target_idx"]))
            modalities.append(row["modality"])
    if not features:
        raise ValueError(f"Empty D1 dataset: {path}")
    split_array = np.asarray(splits)
    conversation_array = np.asarray(conversations)
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = set(conversation_array[split_array == left]) & set(
            conversation_array[split_array == right]
        )
        if overlap:
            raise ValueError(f"D1 conversation leakage between {left}/{right}: {sorted(overlap)[:5]}")
    for name in ("train", "valid", "test"):
        if not np.any(split_array == name):
            raise ValueError(f"D1 dataset has no {name} rows")
    return {
        "x": np.asarray(features, dtype=np.float32),
        "impact": np.asarray(impacts, dtype=np.float32),
        "split": split_array,
        "conversation": conversation_array,
        "source": np.asarray(sources, dtype=np.int32),
        "target": np.asarray(targets, dtype=np.int32),
        "modality": np.asarray(modalities),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 2:
        return math.nan
    left = _average_ranks(np.asarray(predicted, dtype=np.float64))
    right = _average_ranks(np.asarray(actual, dtype=np.float64))
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _safe_binary_metrics(labels: np.ndarray, harm_scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) != 2:
        return math.nan, math.nan
    from sklearn.metrics import average_precision_score, roc_auc_score
    return (
        float(roc_auc_score(labels, harm_scores)),
        float(average_precision_score(labels, harm_scores)),
    )


def _three_class_labels(impact: np.ndarray, epsilon: float) -> np.ndarray:
    # 0=harmful, 1=neutral, 2=beneficial.
    return np.where(impact < -epsilon, 0, np.where(impact > epsilon, 2, 1)).astype(np.int64)


def _metric_row(supervision: str, seed: int, epsilon: float, split: str,
                scope: str, impact: np.ndarray, harm_score: np.ndarray,
                signed_prediction: np.ndarray,
                class_prediction: np.ndarray | None = None) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, f1_score
    mask = np.ones(len(impact), dtype=bool)
    if scope == "strong":
        mask = np.abs(impact) >= epsilon
    selected_impact = impact[mask]
    selected_harm_score = harm_score[mask]
    selected_signed = signed_prediction[mask]
    harmful = selected_impact < -epsilon
    auroc, auprc = _safe_binary_metrics(harmful.astype(np.int8), selected_harm_score)
    row: dict[str, object] = {
        "supervision": supervision,
        "seed": seed,
        "selected_strong_eps": epsilon,
        "split": split,
        "scope": scope,
        "count": int(mask.sum()),
        "harmful_count": int(harmful.sum()),
        "beneficial_count": int(np.sum(selected_impact > epsilon)),
        "neutral_count": int(np.sum(np.abs(selected_impact) < epsilon)),
        "harmful_prevalence": float(harmful.mean()) if len(harmful) else math.nan,
        "auroc": auroc,
        "auprc": auprc,
        "spearman_signed_impact": spearman(selected_signed, selected_impact),
        "three_class_accuracy": "",
        "three_class_macro_f1": "",
    }
    if class_prediction is not None and len(selected_impact):
        truth = _three_class_labels(selected_impact, epsilon)
        prediction = class_prediction[mask]
        row["three_class_accuracy"] = float(accuracy_score(truth, prediction))
        row["three_class_macro_f1"] = float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        )
    return row


def _loss_inputs(supervision: str, impact: np.ndarray, epsilon: float):
    if supervision == "binary":
        mask = np.abs(impact) >= epsilon
        target = (impact[mask] < 0.0).astype(np.int64)
        return mask, target, 1.0
    if supervision == "three_class":
        return np.ones(len(impact), dtype=bool), _three_class_labels(impact, epsilon), 1.0
    mask = np.abs(impact) >= epsilon
    scale = max(float(np.quantile(np.abs(impact[mask]), 0.90)), epsilon, 1e-8)
    return mask, (impact[mask] / scale).astype(np.float32), scale


def _balanced_class_weights(target: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(target, minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"D1 training target is missing a class: counts={counts.tolist()}")
    weights = counts.sum() / (classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _batch_loss(supervision: str, prediction: torch.Tensor, target: torch.Tensor,
                impact_scale: float, class_weights: torch.Tensor | None) -> torch.Tensor:
    del impact_scale
    if supervision == "binary":
        assert class_weights is not None
        pos_weight = class_weights[1] / class_weights[0]
        return F.binary_cross_entropy_with_logits(
            prediction.squeeze(-1), target.float(), pos_weight=pos_weight,
        )
    if supervision == "three_class":
        return F.cross_entropy(prediction, target.long(), weight=class_weights)
    errors = F.smooth_l1_loss(prediction.squeeze(-1), target.float(), reduction="none")
    magnitude = target.abs().clamp_min(1e-3)
    weights = torch.sqrt(magnitude).clamp(max=4.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    return (errors * weights).mean()


def _predict(model: nn.Module, x: np.ndarray, supervision: str,
             impact_scale: float, device: torch.device,
             batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(tensor).cpu().numpy())
    raw = np.concatenate(outputs, axis=0)
    if supervision == "binary":
        harmful = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 0], -30.0, 30.0)))
        return harmful, 1.0 - 2.0 * harmful, None
    if supervision == "three_class":
        shifted = raw - raw.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        return probability[:, 0], probability[:, 2] - probability[:, 0], probability.argmax(1)
    signed = raw[:, 0] * impact_scale
    return -signed, signed, None


def _train_one(supervision: str, epsilon: float, seed: int,
               x_train: np.ndarray, impact_train: np.ndarray,
               x_valid: np.ndarray, impact_valid: np.ndarray,
               args: argparse.Namespace, device: torch.device):
    _seed_everything(seed)
    train_mask, train_target, impact_scale = _loss_inputs(
        supervision, impact_train, epsilon,
    )
    valid_mask, valid_target, _ = _loss_inputs(supervision, impact_valid, epsilon)
    if supervision == "magnitude":
        valid_target = (impact_valid[valid_mask] / impact_scale).astype(np.float32)
    if int(train_mask.sum()) < 32 or int(valid_mask.sum()) < 16:
        raise ValueError(
            f"Too few strong D1 rows for {supervision} eps={epsilon:g}: "
            f"train={int(train_mask.sum())}, valid={int(valid_mask.sum())}"
        )
    classes = 3 if supervision == "three_class" else 2
    class_weights = None
    if supervision in {"binary", "three_class"}:
        class_weights = _balanced_class_weights(train_target, classes).to(device)
    output_dim = 3 if supervision == "three_class" else 1
    model = ImpactMLP(
        x_train.shape[1], args.hidden_dim, output_dim, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    train_x = torch.from_numpy(x_train[train_mask])
    train_y = torch.from_numpy(train_target)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            bx = train_x[indices].to(device)
            by = train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                supervision, model(bx), by, impact_scale, class_weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite D1 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        selected_x = x_valid[valid_mask]
        selected_y = torch.from_numpy(valid_target)
        with torch.inference_mode():
            for start in range(0, len(selected_x), args.batch_size):
                bx = torch.from_numpy(selected_x[start:start + args.batch_size]).to(device)
                by = selected_y[start:start + args.batch_size].to(device)
                losses.append(float(_batch_loss(
                    supervision, model(bx), by, impact_scale, class_weights,
                ).cpu()) * len(bx))
        validation_loss = sum(losses) / len(selected_x)
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("D1 failed to select a checkpoint")
    model.load_state_dict(best_state)
    harm_score, signed, classes_pred = _predict(
        model, x_valid, supervision, impact_scale, device, args.batch_size,
    )
    validation = _metric_row(
        supervision, seed, epsilon, "valid", "strong", impact_valid,
        harm_score, signed, classes_pred,
    )
    return {
        "model": model,
        "state": best_state,
        "impact_scale": impact_scale,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": time.perf_counter() - started,
        "train_count": int(train_mask.sum()),
        "valid_count": int(valid_mask.sum()),
        "validation": validation,
    }


def _selection_key(row: Mapping[str, object], metric: str) -> tuple[float, ...]:
    primary_name = {
        "auroc": "auroc", "auprc": "auprc", "spearman": "spearman_signed_impact",
    }[metric]
    values = [
        float(row[primary_name]), float(row["auroc"]), float(row["auprc"]),
        float(row["spearman_signed_impact"]), -float(row["selected_strong_eps"]),
    ]
    return tuple(-math.inf if not math.isfinite(value) else value for value in values)


def precision_at_k(
    impact: np.ndarray,
    harm_score: np.ndarray,
    conversations: np.ndarray,
    targets: np.ndarray,
    modalities: np.ndarray,
    epsilon: float,
    k_values: Sequence[int],
    scope: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(conversations, targets, modalities)):
        if scope == "strong" and abs(float(impact[index])) < epsilon:
            continue
        groups[(str(key[0]), int(key[1]), str(key[2]))].append(index)
    output = []
    candidate_edge_count = sum(len(indices) for indices in groups.values())
    candidate_harmful_count = sum(
        int(impact[index] < -epsilon)
        for indices in groups.values() for index in indices
    )
    micro_random_precision = (
        candidate_harmful_count / candidate_edge_count
        if candidate_edge_count else math.nan
    )
    group_prevalences = [
        float(np.mean([impact[index] < -epsilon for index in indices]))
        for indices in groups.values() if indices
    ]
    macro_random_precision = (
        float(np.mean(group_prevalences)) if group_prevalences else math.nan
    )
    for k in k_values:
        precisions, hits = [], []
        selected_edges = harmful_edges = 0
        for indices in groups.values():
            ranked = sorted(indices, key=lambda index: (-harm_score[index], index))
            chosen = ranked[:min(k, len(ranked))]
            labels = [impact[index] < -epsilon for index in chosen]
            precisions.append(float(np.mean(labels)))
            hits.append(float(any(labels)))
            selected_edges += len(chosen)
            harmful_edges += sum(labels)
        output.append({
            "scope": scope,
            "k": k,
            "target_groups": len(groups),
            "candidate_edge_count": candidate_edge_count,
            "candidate_harmful_count": candidate_harmful_count,
            "macro_random_precision": macro_random_precision,
            "micro_random_precision": micro_random_precision,
            "selected_edges": selected_edges,
            "harmful_edges": harmful_edges,
            "macro_precision_at_k": float(np.mean(precisions)) if precisions else math.nan,
            "micro_precision_at_k": harmful_edges / selected_edges if selected_edges else math.nan,
            "macro_precision_lift": (
                float(np.mean(precisions)) / macro_random_precision
                if precisions and macro_random_precision > 0.0 else math.nan
            ),
            "micro_precision_lift": (
                (harmful_edges / selected_edges) / micro_random_precision
                if selected_edges and micro_random_precision > 0.0 else math.nan
            ),
            "harmful_hit_rate_at_k": float(np.mean(hits)) if hits else math.nan,
        })
    return output


def _normalize(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0, dtype=np.float64)
    std = x[train_mask].std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    normalized = ((x - mean) / std).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def _write_summary(path: Path, metrics: Sequence[Mapping[str, object]],
                   precision_rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# D1 — Harmful influence learnability",
        "",
        "Thresholds and checkpoints were selected on validation conversations; "
        "test impacts were used only for final scoring.",
        "",
        "| Supervision | Scope | AUROC | AUPRC | Harm prevalence | Spearman | Precision@1 | P@1 lift |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for supervision in SUPERVISIONS:
        if not any(row["supervision"] == supervision for row in metrics):
            continue
        for scope in ("all", "strong"):
            selected = [row for row in metrics
                        if row["supervision"] == supervision and row["scope"] == scope]
            p1 = [row for row in precision_rows
                  if row["supervision"] == supervision and row["scope"] == scope
                  and int(row["k"]) == 1]
            def mean(name: str, rows=selected) -> float:
                values = [float(row[name]) for row in rows
                          if math.isfinite(float(row[name]))]
                return float(np.mean(values)) if values else math.nan
            precision = (
                float(np.mean([float(row["macro_precision_at_k"]) for row in p1]))
                if p1 else math.nan
            )
            precision_lift = (
                float(np.mean([float(row["macro_precision_lift"]) for row in p1
                               if math.isfinite(float(row["macro_precision_lift"]))]))
                if any(math.isfinite(float(row["macro_precision_lift"])) for row in p1)
                else math.nan
            )
            lines.append(
                f"| {supervision} | {scope} | {mean('auroc'):.4f} | "
                f"{mean('auprc'):.4f} | {mean('harmful_prevalence'):.4f} | "
                f"{mean('spearman_signed_impact'):.4f} | {precision:.4f} | "
                f"{precision_lift:.3f}x |"
            )
    lines.extend([
        "",
        "Interpretation: a clear AUROC increase from `all` to `strong` supports "
        "the noisy-supervision hypothesis. AUROC near 0.5 on `strong` pairs is "
        "a negative finding for the tested observable features and predictor, "
        "not proof that harmful influence is fundamentally unlearnable. AUPRC "
        "must be compared with harmful prevalence, not with 0.5.",
        "",
    ])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    candidates = sorted(set(float(value) for value in args.strong_eps_candidates))
    if (not candidates or any(value <= 0.0 for value in candidates)
            or args.base_impact_eps < 0.0):
        raise ValueError("D1 impact thresholds must be positive")
    if args.hidden_dim < 4 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Invalid D1 training configuration")
    if any(value < 1 for value in args.precision_k):
        raise ValueError("Precision@K values must be positive")
    device = _device(args.device, args.gpu_id)
    data = _load_dataset(args.dataset.resolve())
    split = data["split"]
    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"
    if args.max_train_rows:
        indices = np.flatnonzero(train_mask)
        if args.max_train_rows >= len(indices):
            raise ValueError("--max-train-rows must be smaller than the full train set")
        train_mask = np.zeros(len(split), dtype=bool)
        train_mask[indices[:args.max_train_rows]] = True
    normalized, feature_mean, feature_std = _normalize(data["x"], train_mask)
    x_train, impact_train = normalized[train_mask], data["impact"][train_mask]
    x_valid, impact_valid = normalized[valid_mask], data["impact"][valid_mask]
    x_test, impact_test = normalized[test_mask], data["impact"][test_mask]
    root = args.output_dir.resolve() / "D1"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    precision_rows: list[dict[str, object]] = []
    prediction_paths: list[str] = []

    for seed in args.seeds:
        for supervision in args.supervisions:
            candidates_run = []
            for epsilon in candidates:
                try:
                    artifact = _train_one(
                        supervision, epsilon, seed, x_train, impact_train,
                        x_valid, impact_valid, args, device,
                    )
                except ValueError as error:
                    selection_rows.append({
                        "supervision": supervision, "seed": seed,
                        "selected_strong_eps": epsilon, "status": "skipped",
                        "reason": str(error),
                    })
                    continue
                validation = dict(artifact["validation"])
                validation.update({
                    "status": "candidate", "reason": "",
                    "best_epoch": artifact["best_epoch"],
                    "validation_loss": artifact["best_validation_loss"],
                    "train_count": artifact["train_count"],
                    "valid_count_for_loss": artifact["valid_count"],
                    "training_seconds": artifact["training_seconds"],
                })
                selection_rows.append(validation)
                candidates_run.append(artifact)
            if not candidates_run:
                raise RuntimeError(f"No valid D1 candidates for {supervision} seed={seed}")
            selected = max(
                candidates_run,
                key=lambda item: _selection_key(item["validation"], args.selection_metric),
            )
            epsilon = float(selected["validation"]["selected_strong_eps"])
            for row in selection_rows:
                if (row.get("supervision") == supervision and row.get("seed") == seed
                        and row.get("status") == "candidate"):
                    row["selected"] = int(float(row["selected_strong_eps"]) == epsilon)
            model = selected["model"]
            harm_score, signed_prediction, class_prediction = _predict(
                model, x_test, supervision, selected["impact_scale"],
                device, args.batch_size,
            )
            for scope in ("all", "strong"):
                metric_rows.append(_metric_row(
                    supervision, seed, epsilon, "test", scope, impact_test,
                    harm_score, signed_prediction, class_prediction,
                ))
                for row in precision_at_k(
                    impact_test, harm_score,
                    data["conversation"][test_mask], data["target"][test_mask],
                    data["modality"][test_mask], epsilon, args.precision_k, scope,
                ):
                    precision_rows.append({
                        "supervision": supervision, "seed": seed,
                        "selected_strong_eps": epsilon, **row,
                    })
            prediction_rows: list[dict[str, object]] = []
            test_indices = np.flatnonzero(test_mask)
            for local, global_index in enumerate(test_indices):
                prediction_rows.append({
                    "supervision": supervision,
                    "seed": seed,
                    "selected_strong_eps": epsilon,
                    "conversation_id": data["conversation"][global_index],
                    "source_idx": int(data["source"][global_index]),
                    "target_idx": int(data["target"][global_index]),
                    "modality": data["modality"][global_index],
                    "actual_signed_impact": float(impact_test[local]),
                    "harm_score": float(harm_score[local]),
                    "predicted_signed_impact": float(signed_prediction[local]),
                    "actual_harmful_base_eps": int(
                        impact_test[local] < -args.base_impact_eps
                    ),
                    "actual_strong_for_selected_eps": int(
                        abs(float(impact_test[local])) >= epsilon
                    ),
                })
            prediction_path = (
                root / "test_predictions" / f"{supervision}_seed{seed}.csv"
            )
            write_csv(prediction_path, prediction_rows)
            prediction_paths.append(str(prediction_path))
            checkpoint_path = checkpoints / f"{supervision}_seed{seed}.pt"
            torch.save({
                "supervision": supervision,
                "seed": seed,
                "selected_strong_eps": epsilon,
                "impact_scale": selected["impact_scale"],
                "best_epoch": selected["best_epoch"],
                "feature_names": FEATURE_NAMES,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "model_config": {
                    "input_dim": len(FEATURE_NAMES),
                    "hidden_dim": args.hidden_dim,
                    "output_dim": 3 if supervision == "three_class" else 1,
                    "dropout": args.dropout,
                },
                "model_state_dict": selected["state"],
            }, checkpoint_path)
            print(
                f"D1 {supervision} seed={seed}: selected eps={epsilon:g}; "
                f"valid AUROC={float(selected['validation']['auroc']):.4f}",
                flush=True,
            )

    write_csv(root / "threshold_selection.csv", selection_rows)
    write_csv(root / "model_metrics.csv", metric_rows)
    write_csv(root / "precision_at_k.csv", precision_rows)
    _write_summary(root / "summary.md", metric_rows, precision_rows)
    write_json(root / "experiment_manifest.json", {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "supervisions": list(args.supervisions),
        "strong_eps_candidates": candidates,
        "base_impact_eps": args.base_impact_eps,
        "selection_metric": args.selection_metric,
        "selection_split": "valid",
        "test_used_for_training_or_selection": False,
        "seeds": list(args.seeds),
        "predictor": {
            "type": "shared_capacity_mlp",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "row_counts": {
            name: int(np.sum(split == name)) for name in ("train", "valid", "test")
        },
        "conversation_counts": {
            name: len(set(data["conversation"][split == name]))
            for name in ("train", "valid", "test")
        },
        "forbidden_inference_inputs": [
            "source/target true emotion", "signed impact", "ground-truth correctness",
            "suppressed prediction", "suppressed NLL",
        ],
        "smoke_limited": bool(args.max_train_rows),
        "test_prediction_files": prediction_paths,
    })
    print(f"D1 learnability complete: {root}", flush=True)
    return root


def main(argv: Sequence[str] | None = None) -> Path:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
