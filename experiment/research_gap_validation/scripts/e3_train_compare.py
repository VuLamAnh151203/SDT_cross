"""E3: controlled B0--B6 learned comparison with frozen SDT contextual backbone."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
SDT_ROOT = ROOT.parents[1]
for path in (ROOT, CROSS, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (classification_metrics, cluster_bootstrap, read_csv, session_id,
                    write_csv, write_json)  # noqa: E402
from dataloader import make_loaders  # noqa: E402
from impact_core import load_experiment, seed_everything  # noqa: E402
from model import Transformer_Based_Model  # noqa: E402
from reliability_model import ReliabilitySDT  # noqa: E402


MODEL_IDS = tuple(f"B{index}" for index in range(7))
MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--impact-teacher", type=Path,
                        help="Train-split raw impact pairs required by B4")
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=MODEL_IDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(2024, 2025, 2026))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--gate-lr", type=float,
        help="Optional learning rate for gate/control parameters; defaults to --lr",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--gate-hidden", type=int, default=64)
    parser.add_argument("--capacity-hidden", type=int, default=0,
                        help="0 chooses the closest B5 residual width to B3 gate parameters")
    parser.add_argument("--initial-gate", type=float, default=0.99)
    parser.add_argument(
        "--gate-modalities", nargs="+", choices=tuple(MODALITY_INDEX),
        default=tuple(MODALITY_INDEX),
        help=("Modalities whose reliability gates may change. Unselected modalities "
              "are fixed exactly at gate=1."),
    )
    parser.add_argument("--lambda-identity", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-impact", type=float, default=0.02,
        help="Weight of B4 edge-teacher supervision; kept below emotion loss by default",
    )
    parser.add_argument("--lambda-relevance", type=float, default=0.2)
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    parser.add_argument(
        "--impact-min-margin", type=float, default=0.0,
        help=("Keep only teacher edges whose propagation decision margin is at "
              "least this value. This implements strong-impact filtering."),
    )
    parser.add_argument(
        "--impact-loss", choices=("soft_mse", "bce"), default="soft_mse",
        help=("B4 teacher objective. soft_mse is recommended for continuous "
              "edge-gate teachers; bce is retained for compatibility."),
    )
    parser.add_argument(
        "--impact-gate-floor", type=float, default=0.75,
        help=("Lowest soft gate target assigned to a harmful legacy input-"
              "replacement teacher row."),
    )
    parser.add_argument(
        "--impact-scale", type=float, default=1e-2,
        help=("Impact magnitude that receives unit teacher weight when a legacy "
              "signed-impact file is used."),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--defer-test-evaluation", action="store_true",
        help=("Train and save validation-selected checkpoints without evaluating "
              "the test split. Intended for outer validation-only sweeps."),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--renormalize-attention", action="store_true")
    parser.add_argument("--fold-name", default="checkpoint_split")
    parser.add_argument(
        "--selection-split", choices=("valid", "test"), default="valid",
        help=("Split used for early stopping and best-epoch selection. Using test "
              "is exploratory and makes the reported test metrics optimistically biased."),
    )
    parser.add_argument(
        "--selection-metric", choices=("nll", "weighted_f1", "accuracy"),
        default="nll",
        help=("Primary checkpoint-selection metric. Weighted F1 uses accuracy and NLL "
              "as deterministic tie-breakers; accuracy uses weighted F1 and NLL."),
    )
    parser.add_argument("--require-session-disjoint", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="Secondary protocol; applies identically to every selected model")
    parser.add_argument(
        "--freeze-classifier", action="store_true",
        help="Strict diagnostic: freeze the SDT classifier and train only added gates/controls",
    )
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


class ImpactTeacher:
    """Train-only edge-gate targets.

    New propagation-edge teachers provide ``teacher_gate`` and
    ``teacher_weight`` directly. Legacy singleton input-replacement teachers are
    still accepted, but are converted to conservative magnitude-aware targets
    instead of the former hard 0/1 labels.
    """

    def __init__(self, path: Path | None, train_ids: set[str], epsilon: float,
                 gate_floor: float, impact_scale: float,
                 min_margin: float = 0.0):
        self.values: dict[tuple[str, int, int, int], tuple[float, float]] = {}
        self.teacher_kind = "none"
        self.target_counts = {"suppressed": 0, "identity": 0, "intermediate": 0}
        self.raw_pair_count = 0
        self.filtered_weak_count = 0
        self.min_margin = float(min_margin)
        if path is None:
            return
        for row in read_csv(path):
            self.raw_pair_count += 1
            conversation = row["conversation_id"]
            if conversation not in train_ids:
                raise ValueError(
                    f"Impact teacher contains non-train conversation: {conversation}"
                )
            modality = MODALITY_INDEX[row["modality"]]
            key = (conversation, int(row["target_idx"]), int(row["source_idx"]), modality)
            if key in self.values:
                raise ValueError(f"Duplicate impact-teacher key: {key}")
            if row.get("teacher_gate", "") != "":
                self.teacher_kind = "propagation_edge_grid"
                target = float(row["teacher_gate"])
                weight = float(row.get("teacher_weight", 1.0) or 1.0)
                margin = abs(float(row.get("teacher_decision_margin", 0.0) or 0.0))
            else:
                if self.teacher_kind == "none":
                    self.teacher_kind = "legacy_input_replacement"
                impact = float(row["signed_impact"])
                margin = abs(impact)
                if abs(impact) <= epsilon:
                    continue
                strength = min(abs(impact) / impact_scale, 1.0)
                target = 1.0 if impact > epsilon else 1.0 - (
                    (1.0 - gate_floor) * strength
                )
                weight = strength
            if margin < self.min_margin:
                self.filtered_weak_count += 1
                continue
            if not 0.0 <= target <= 1.0:
                raise ValueError(f"Teacher gate outside [0,1] for {key}: {target}")
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Invalid teacher weight for {key}: {weight}")
            if weight == 0.0:
                continue
            self.values[key] = (target, weight)
            category = ("suppressed" if target <= 1e-6 else "identity"
                        if target >= 0.999999 else "intermediate")
            self.target_counts[category] += 1

    def batch(self, dialogue_ids, lengths, max_length, device):
        targets = torch.ones(len(dialogue_ids), max_length, max_length, 3,
                             device=device)
        weights = torch.zeros_like(targets)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for batch_index, (conversation, length) in enumerate(zip(dialogue_ids, lengths)):
            for target in range(length):
                for source in range(length):
                    if source == target:
                        continue
                    for modality in range(3):
                        key = (conversation, target, source, modality)
                        if key in self.values:
                            value, weight = self.values[key]
                            targets[batch_index, target, source, modality] = value
                            weights[batch_index, target, source, modality] = weight
                            mask[batch_index, target, source, modality] = True
        return targets, weights, mask

    def summary(self) -> dict[str, object]:
        weights = [value[1] for value in self.values.values()]
        targets = [value[0] for value in self.values.values()]
        return {
            "teacher_kind": self.teacher_kind,
            "teacher_pair_count": len(self.values),
            "teacher_raw_pair_count": self.raw_pair_count,
            "teacher_filtered_weak_count": self.filtered_weak_count,
            "teacher_min_margin": self.min_margin,
            "teacher_mean_target": float(np.mean(targets)) if targets else None,
            "teacher_mean_weight": float(np.mean(weights)) if weights else None,
            **{f"teacher_{key}_count": value
               for key, value in self.target_counts.items()},
        }


def _class_weights(experiment):
    args = experiment.checkpoint.get("args", {})
    if experiment.dataset.dataset == "IEMOCAP" and not args.get("no_class_weight", False):
        return torch.tensor([1 / p for p in
                             (0.086747, 0.144406, 0.227883, 0.160585, 0.127711, 0.252668)],
                            device=experiment.device)
    return None


def _valid_pair_mask(valid: torch.Tensor) -> torch.Tensor:
    length = valid.size(1)
    pairs = valid[:, :, None] & valid[:, None, :]
    diagonal = torch.eye(length, dtype=torch.bool, device=valid.device)
    return pairs & ~diagonal[None]


def _relevance_proxy(speakers: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length = valid.shape
    position = torch.arange(length, device=valid.device)
    distance = (position[:, None] - position[None, :]).abs().float()
    same = speakers.argmax(dim=-1)[:, :, None].eq(
        speakers.argmax(dim=-1)[:, None, :]
    ).float()
    relevance = torch.exp(-distance[None] / 10.0) * (0.5 + 0.5 * same)
    relevance = relevance.unsqueeze(-1).expand(batch, -1, -1, 3)
    return relevance, _valid_pair_mask(valid).unsqueeze(-1).expand_as(relevance)


def _loss(model_id, output, labels, valid, class_weights, args, teacher,
          dialogue_ids, lengths, speakers):
    emotion = F.nll_loss(output["log_prob"][valid], labels[valid], weight=class_weights)
    auxiliary = emotion.new_zeros(())
    identity = emotion.new_zeros(())
    gates = output["reliability_pair_gate"]
    modality_mask = torch.zeros(3, dtype=torch.bool, device=valid.device)
    modality_mask[[MODALITY_INDEX[name] for name in args.gate_modalities]] = True
    if gates is not None:
        pair_mask = (_valid_pair_mask(valid).unsqueeze(-1).expand_as(gates)
                     & modality_mask.view(1, 1, 1, 3))
        identity = ((1.0 - gates[pair_mask]) ** 2).mean()
        if model_id == "B4":
            targets, teacher_weights, teacher_mask = teacher.batch(
                dialogue_ids, lengths, valid.size(1), valid.device
            )
            if teacher_mask.any():
                predicted = gates[teacher_mask]
                expected = targets[teacher_mask]
                weights = teacher_weights[teacher_mask]
                if args.impact_loss == "bce":
                    errors = F.binary_cross_entropy(
                        predicted, expected, reduction="none"
                    )
                else:
                    errors = (predicted - expected).square()
                auxiliary = (errors * weights).sum() / weights.sum().clamp_min(1e-12)
        elif model_id == "B6":
            targets, relevance_mask = _relevance_proxy(speakers, valid)
            relevance_mask = (relevance_mask
                              & modality_mask.view(1, 1, 1, 3))
            auxiliary = F.binary_cross_entropy(gates[relevance_mask],
                                               targets[relevance_mask])
    modality_gates = output["reliability_modality_gate"]
    if modality_gates is not None:
        expanded = (valid.unsqueeze(-1).expand_as(modality_gates)
                    & modality_mask.view(1, 1, 3))
        identity = ((1.0 - modality_gates[expanded]) ** 2).mean()
    auxiliary_weight = (
        args.lambda_impact if model_id == "B4"
        else args.lambda_relevance if model_id == "B6" else 0.0
    )
    total = emotion + args.lambda_identity * identity + auxiliary_weight * auxiliary
    return total, emotion, identity, auxiliary


def _batch_to_device(batch, device):
    text, visual, audio, speakers, mask, labels = [item.to(device) for item in batch[:6]]
    return text, visual, audio, speakers, mask, labels, batch[6]


def run_epoch(model, model_id, loader, experiment, args, teacher,
              optimizer=None, collect=False):
    training = optimizer is not None
    model.train(training)
    labels_all, predictions_all, rows = [], [], []
    total_nll = total_loss = total_identity = total_auxiliary = 0.0
    gate_sum = gate_items = 0
    gate_minimum = 1.0
    utterances = batches = 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, batch in enumerate(loader):
            if args.max_batches and batch_number >= args.max_batches:
                break
            text, visual, audio, speakers_lbs, mask, labels, ids = _batch_to_device(
                batch, experiment.device
            )
            speakers = speakers_lbs.transpose(0, 1)
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(text, visual, audio, mask, speakers, lengths)
            pair_gate = output["reliability_pair_gate"]
            modality_gate = output["reliability_modality_gate"]
            if pair_gate is not None:
                modality_mask = torch.zeros(3, dtype=torch.bool, device=valid.device)
                modality_mask[list(model.active_gate_indices)] = True
                active_mask = (_valid_pair_mask(valid).unsqueeze(-1).expand_as(pair_gate)
                               & modality_mask.view(1, 1, 1, 3))
                active_gates = pair_gate[active_mask].detach()
            elif modality_gate is not None:
                modality_mask = torch.zeros(3, dtype=torch.bool, device=valid.device)
                modality_mask[list(model.active_gate_indices)] = True
                active_mask = (valid.unsqueeze(-1).expand_as(modality_gate)
                               & modality_mask.view(1, 1, 3))
                active_gates = modality_gate[active_mask].detach()
            else:
                active_gates = None
            if active_gates is not None and active_gates.numel():
                gate_sum += float(active_gates.sum().cpu())
                gate_items += int(active_gates.numel())
                gate_minimum = min(gate_minimum, float(active_gates.min().cpu()))
            valid_log_prob = output["log_prob"][valid]
            unweighted_nll = -valid_log_prob.gather(
                1, labels[valid].unsqueeze(-1)
            ).mean()
            loss, emotion, identity, auxiliary = _loss(
                model_id, output, labels, valid, _class_weights(experiment), args,
                teacher, ids, lengths, speakers, 
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite E3 training loss")
            if training:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip
                    )
                optimizer.step()
            prediction = output["logits"].argmax(dim=-1)
            count = int(valid.sum().item())
            utterances += count
            batches += 1
            total_loss += float(loss.detach()) * count
            total_nll += float(unweighted_nll.detach()) * count
            total_identity += float(identity.detach()) * count
            total_auxiliary += float(auxiliary.detach()) * count
            labels_all.extend(labels[valid].detach().cpu().tolist())
            predictions_all.extend(prediction[valid].detach().cpu().tolist())
            if collect:
                probabilities = output["prob"].detach().cpu()
                pair_gate = output["reliability_pair_gate"]
                modality_gate = output["reliability_modality_gate"]
                if pair_gate is not None:
                    pair_gate = pair_gate.detach().cpu()
                if modality_gate is not None:
                    modality_gate = modality_gate.detach().cpu()
                labels_cpu = labels.detach().cpu()
                speakers_cpu = speakers.argmax(dim=-1).detach().cpu()
                for batch_index, (conversation, length) in enumerate(zip(ids, lengths)):
                    previous = {}
                    for target in range(length):
                        speaker = int(speakers_cpu[batch_index, target])
                        source = previous.get(speaker, -1)
                        previous[speaker] = target
                        label = int(labels_cpu[batch_index, target])
                        pred = int(prediction[batch_index, target].detach().cpu())
                        probability = probabilities[batch_index, target]
                        sorted_prob = torch.topk(probability, 2).values
                        row = {
                            "conversation_id": conversation, "target_idx": target,
                            "target_label": label, "prediction": pred,
                            "correct": int(label == pred),
                            "nll": float(-torch.log(probability[label].clamp_min(1e-12))),
                            "margin": float(sorted_prob[0] - sorted_prob[1]),
                            "previous_same_speaker_idx": source,
                            "transition": int(source >= 0 and
                                              int(labels_cpu[batch_index, source]) != label),
                            "continuity": int(source >= 0 and
                                              int(labels_cpu[batch_index, source]) == label),
                        }
                        if pair_gate is not None:
                            active = pair_gate[batch_index, target, :length]
                            nonself = torch.arange(length) != target
                            selected = active[nonself][:, list(model.active_gate_indices)]
                            row["gate_mean"] = float(selected.mean())
                            row["gate_min"] = float(selected.min())
                            for name, index in MODALITY_INDEX.items():
                                values = active[nonself, index]
                                row[f"gate_mean_{name}"] = float(values.mean())
                                row[f"gate_min_{name}"] = float(values.min())
                        elif modality_gate is not None:
                            active = modality_gate[batch_index, target]
                            selected = active[list(model.active_gate_indices)]
                            row["gate_mean"] = float(selected.mean())
                            row["gate_min"] = float(selected.min())
                            for name, index in MODALITY_INDEX.items():
                                row[f"gate_mean_{name}"] = float(active[index])
                                row[f"gate_min_{name}"] = float(active[index])
                        else:
                            row["gate_mean"] = 1.0
                            row["gate_min"] = 1.0
                            for name in MODALITY_INDEX:
                                row[f"gate_mean_{name}"] = 1.0
                                row[f"gate_min_{name}"] = 1.0
                        row.update({f"prob_{index}": float(value)
                                    for index, value in enumerate(probability)})
                        rows.append(row)
    if not utterances:
        raise ValueError("Empty E3 epoch")
    metrics = classification_metrics(labels_all, predictions_all)
    metrics.update({
        "nll": total_nll / utterances, "training_objective": total_loss / utterances,
        "identity_loss": total_identity / utterances,
        "auxiliary_loss": total_auxiliary / utterances,
        "gate_mean": gate_sum / gate_items if gate_items else 1.0,
        "gate_min": gate_minimum,
        "utterances": utterances, "seconds": time.perf_counter() - started,
        "batches": batches,
    })
    return metrics, rows


def _new_model(experiment, model_id, args):
    base = Transformer_Based_Model(**experiment.checkpoint["model_config"]).to(experiment.device)
    base.load_state_dict(experiment.checkpoint["model_state_dict"], strict=True)
    if hasattr(base, "set_tical_epoch"):
        base.set_tical_epoch(int(experiment.checkpoint.get("epoch", 0)))
    capacity_hidden = args.capacity_hidden
    if model_id == "B5" and capacity_hidden == 0:
        hidden = int(base.all_output_layer.in_features)
        # Exact parameter count of PairGateNetwork: three query and key
        # projections, three 4->1 structural heads, scale and bias vectors.
        target = 12 * hidden * args.gate_hidden + 6 * args.gate_hidden + 21
        capacity_hidden = max(1, round((target - hidden) / (2 * hidden + 1)))
    elif capacity_hidden == 0:
        capacity_hidden = 1
    wrapper = ReliabilitySDT(
        base, model_id, gate_hidden=args.gate_hidden,
        initial_gate=args.initial_gate, capacity_hidden=capacity_hidden,
        renormalize_attention=args.renormalize_attention,
        gate_modalities=args.gate_modalities,
    ).to(experiment.device)
    wrapper.freeze_backbone(train_classifier=not args.freeze_classifier)
    if args.unfreeze_backbone:
        wrapper.unfreeze_backbone()
    wrapper.effective_capacity_hidden = capacity_hidden
    return wrapper


def _make_optimizer(model, args):
    classifier_parameters = [
        parameter for parameter in model.base.all_output_layer.parameters()
        if parameter.requires_grad
    ]
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    gate_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    groups = []
    if classifier_parameters:
        groups.append({"params": classifier_parameters, "lr": args.lr})
    if gate_parameters:
        groups.append({
            "params": gate_parameters,
            "lr": args.lr if args.gate_lr is None else args.gate_lr,
        })
    if not groups:
        return None
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _transition_summaries(rows, metadata):
    output = []
    subsets = {
        "all": rows,
        "eligible": [row for row in rows if int(row["previous_same_speaker_idx"]) >= 0],
        "continuity": [row for row in rows if int(row["continuity"]) == 1],
        "transition": [row for row in rows if int(row["transition"]) == 1],
        "low_margin": [row for row in rows if float(row["margin"]) < 0.05],
        "high_margin": [row for row in rows if float(row["margin"]) >= 0.20],
    }
    for subset, selected in subsets.items():
        if not selected:
            continue
        metrics = classification_metrics(
            [int(row["target_label"]) for row in selected],
            [int(row["prediction"]) for row in selected],
        )
        output.append({
            **metadata, "subset": subset, "count": len(selected), **metrics,
            "nll": float(np.mean([float(row["nll"]) for row in selected])),
            "gate_mean": float(np.mean([float(row["gate_mean"]) for row in selected])),
        })
    return output


def _paired_deltas(prediction_groups, repetitions, seed):
    comparisons = (("B2", "B1", "placement"), ("B3", "B2", "granularity"),
                   ("B4", "B3", "impact_supervision"),
                   ("B4", "B6", "predictive_vs_relevance"),
                   ("B3", "B5", "pair_vs_capacity"),
                   ("B4", "B5", "impact_pair_vs_capacity"))
    output = []
    for better, reference, claim in comparisons:
        keys = sorted(set(key[:2] for key in prediction_groups if key[2] == better)
                      & set(key[:2] for key in prediction_groups if key[2] == reference))
        for fold, run_seed in keys:
            left = prediction_groups[(fold, run_seed, better)]
            right = prediction_groups[(fold, run_seed, reference)]
            left_map = {(row["conversation_id"], row["target_idx"]): row for row in left}
            right_map = {(row["conversation_id"], row["target_idx"]): row for row in right}
            if left_map.keys() != right_map.keys():
                raise ValueError("Paired model predictions have different target sets")
            paired = []
            for target in left_map:
                a, b = left_map[target], right_map[target]
                paired.append({
                    "conversation_id": a["conversation_id"],
                    "label": int(a["target_label"]), "a": int(a["prediction"]),
                    "b": int(b["prediction"]), "nll_a": float(a["nll"]),
                    "nll_b": float(b["nll"]),
                })
            for metric in ("accuracy", "weighted_f1", "macro_f1", "nll"):
                def statistic(sample, name=metric):
                    if name == "nll":
                        return float(np.mean([row["nll_a"] - row["nll_b"] for row in sample]))
                    labels = [row["label"] for row in sample]
                    return (classification_metrics(labels, [row["a"] for row in sample])[name]
                            - classification_metrics(labels, [row["b"] for row in sample])[name])
                estimate = statistic(paired)
                low, high = cluster_bootstrap(paired, statistic, repetitions, seed)
                output.append({
                    "fold": fold, "seed": run_seed, "comparison": f"{better}-{reference}",
                    "claim": claim, "metric": metric, "delta": estimate,
                    "ci95_low": low, "ci95_high": high,
                    "cluster_unit": "conversation", "target_count": len(paired),
                })
    return output


def _checkpoint_selection_score(metrics, metric):
    """Return a maximized score with deterministic secondary criteria."""
    accuracy = float(metrics["accuracy"])
    weighted_f1 = float(metrics["weighted_f1"])
    negative_nll = -float(metrics["nll"])
    if metric == "weighted_f1":
        return weighted_f1, accuracy, negative_nll
    if metric == "accuracy":
        return accuracy, weighted_f1, negative_nll
    return negative_nll, weighted_f1, accuracy


def train(args: argparse.Namespace) -> None:
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    split_ids = {name: list(values) for name, values in experiment.split_ids.items()}
    if args.selection_split == "valid" and not split_ids.get("valid"):
        raise ValueError(
            "E3 requires a non-empty validation split for validation-based model "
            "selection; pass --selection-split test only for exploratory test-selected runs"
        )
    if args.selection_split == "test":
        if not split_ids.get("test"):
            raise ValueError("E3 test-based model selection requires a non-empty test split")
        print(
            "WARNING: selecting E3 epochs on the test split; resulting test metrics are "
            "not held-out estimates.",
            flush=True,
        )
    sets = {name: set(values) for name, values in split_ids.items()}
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]:
        raise ValueError("Conversation leakage across checkpoint splits")
    sessions = {name: {session_id(item) for item in values}
                for name, values in split_ids.items()}
    if args.require_session_disjoint:
        for first, second in (("train", "valid"), ("train", "test"),
                              ("valid", "test")):
            overlap = sessions[first] & sessions[second]
            if overlap:
                raise ValueError(
                    f"Session leakage between {first}/{second}: {sorted(overlap)}"
                )
    if "B4" in args.models and args.impact_teacher is None:
        raise ValueError("B4 requires --impact-teacher generated only from train conversations")
    teacher = ImpactTeacher(
        args.impact_teacher, sets["train"], args.impact_eps,
        args.impact_gate_floor, args.impact_scale, args.impact_min_margin,
    )
    teacher_summary = teacher.summary()
    if "B4" in args.models:
        if not teacher.values:
            raise ValueError("B4 impact teacher contains no usable weighted edge targets")
        active_modalities = {MODALITY_INDEX[name] for name in args.gate_modalities}
        teacher_modalities = {key[3] for key in teacher.values}
        inactive_teacher_modalities = teacher_modalities.difference(active_modalities)
        if inactive_teacher_modalities:
            names = [name for name, index in MODALITY_INDEX.items()
                     if index in inactive_teacher_modalities]
            raise ValueError(
                "B4 teacher targets modalities fixed at identity: " + ", ".join(names)
            )
        print(
            "B4 teacher: "
            f"kind={teacher_summary['teacher_kind']} "
            f"pairs={teacher_summary['teacher_pair_count']}/"
            f"{teacher_summary['teacher_raw_pair_count']} "
            f"filtered_weak={teacher_summary['teacher_filtered_weak_count']} "
            f"min_margin={teacher_summary['teacher_min_margin']:.6g} "
            f"mean_target={teacher_summary['teacher_mean_target']:.4f} "
            f"mean_weight={teacher_summary['teacher_mean_weight']:.4f}",
            flush=True,
        )
    output = args.output_dir.resolve() / "E3"
    model_rows, transition_rows, classwise_rows, epoch_rows = [], [], [], []
    all_predictions, prediction_groups = [], {}
    run_manifest = []
    for run_seed in args.seeds:
        loaders = make_loaders(
            experiment.dataset, split_ids, args.batch_size, run_seed,
            args.num_workers, experiment.device.type == "cuda",
        )
        for model_id in args.models:
            seed_everything(run_seed)
            model = _new_model(experiment, model_id, args)
            optimizer = _make_optimizer(model, args)
            initial_selection_metrics, _ = run_epoch(
                model, model_id, loaders[args.selection_split], experiment, args, teacher,
            )
            history_metadata = {
                "fold": args.fold_name, "seed": run_seed, "model": model_id,
            }
            epoch_rows.append({
                **history_metadata, "epoch": 0, "phase": "selection",
                "split": args.selection_split, **initial_selection_metrics,
            })
            best_state = copy.deepcopy(model.trainable_state_dict())
            best_score = _checkpoint_selection_score(
                initial_selection_metrics, args.selection_metric
            )
            best_epoch, stale = 0, 0
            best_selection_metrics = dict(initial_selection_metrics)
            train_seconds = 0.0
            if experiment.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(experiment.device)
            epoch_iterator = range(1, args.epochs + 1) if optimizer is not None else ()
            for epoch in epoch_iterator:
                train_metrics, _ = run_epoch(
                    model, model_id, loaders["train"], experiment, args, teacher,
                    optimizer=optimizer,
                )
                epoch_rows.append({
                    **history_metadata, "epoch": epoch, "phase": "train",
                    "split": "train", **train_metrics,
                })
                selection_metrics, _ = run_epoch(
                    model, model_id, loaders[args.selection_split], experiment, args, teacher,
                )
                epoch_rows.append({
                    **history_metadata, "epoch": epoch, "phase": "selection",
                    "split": args.selection_split, **selection_metrics,
                })
                train_seconds += train_metrics["seconds"]
                selection_score = _checkpoint_selection_score(
                    selection_metrics, args.selection_metric
                )
                if selection_score > best_score:
                    best_score = selection_score
                    best_epoch = epoch
                    best_selection_metrics = dict(selection_metrics)
                    best_state = copy.deepcopy(model.trainable_state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            model.load_trainable_state_dict(best_state)
            fold = args.fold_name
            checkpoint_path = output / "checkpoints" / f"{fold}_{model_id}_seed{run_seed}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            saved_configuration = {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            }
            saved_configuration["capacity_hidden"] = model.effective_capacity_hidden
            torch.save({
                "trainable_state_dict": best_state, "model_id": model_id,
                "seed": run_seed, "best_epoch": best_epoch,
                "base_checkpoint_id": experiment.checkpoint_id,
                "configuration": saved_configuration,
            }, checkpoint_path)
            run_manifest.append(str(checkpoint_path))
            if args.defer_test_evaluation:
                model_rows.append({
                    **history_metadata, "best_epoch": best_epoch,
                    "selection_split": args.selection_split,
                    "selection_metric": args.selection_metric,
                    "epoch0_selection_accuracy": initial_selection_metrics["accuracy"],
                    "epoch0_selection_weighted_f1": initial_selection_metrics["weighted_f1"],
                    "epoch0_selection_nll": initial_selection_metrics["nll"],
                    "best_selection_accuracy": best_selection_metrics["accuracy"],
                    "best_selection_weighted_f1": best_selection_metrics["weighted_f1"],
                    "best_selection_nll": best_selection_metrics["nll"],
                    "test_evaluation_deferred": 1,
                    "gate_modalities": "+".join(args.gate_modalities),
                    "initial_gate": args.initial_gate,
                    "lambda_identity": args.lambda_identity,
                    "lambda_impact": args.lambda_impact if model_id == "B4" else 0.0,
                    "impact_min_margin": (
                        args.impact_min_margin if model_id == "B4" else 0.0
                    ),
                    "impact_teacher_pairs": (
                        teacher_summary["teacher_pair_count"] if model_id == "B4" else 0
                    ),
                })
                print(
                    f"E3 {model_id} seed={run_seed}: best_epoch={best_epoch} "
                    f"best_valid_WF1={best_selection_metrics['weighted_f1']:.3f}; "
                    "test deferred",
                    flush=True,
                )
                continue
            test_metrics, predictions = run_epoch(
                model, model_id, loaders["test"], experiment, args, teacher,
                collect=True,
            )
            prediction_keys = {(row["conversation_id"], row["target_idx"])
                               for row in predictions}
            if len(prediction_keys) != len(predictions):
                raise AssertionError("E3 violated one-prediction-per-target invariant")
            metadata = {"fold": fold, "seed": run_seed, "model": model_id}
            for row in predictions:
                row.update(metadata)
            all_predictions.extend(predictions)
            prediction_groups[(fold, run_seed, model_id)] = predictions
            transition_rows.extend(_transition_summaries(predictions, metadata))
            for label in range(experiment.dataset.n_classes):
                tp = sum(int(int(row["target_label"]) == label
                             and int(row["prediction"]) == label) for row in predictions)
                fp = sum(int(int(row["target_label"]) != label
                             and int(row["prediction"]) == label) for row in predictions)
                fn = sum(int(int(row["target_label"]) == label
                             and int(row["prediction"]) != label) for row in predictions)
                denominator = 2 * tp + fp + fn
                classwise_rows.append({
                    **metadata, "class": label,
                    "support": sum(int(int(row["target_label"]) == label)
                                   for row in predictions),
                    "f1": 0.0 if denominator == 0 else 2 * tp / denominator,
                })
            peak_memory = (torch.cuda.max_memory_allocated(experiment.device)
                           if experiment.device.type == "cuda" else 0)
            lengths_by_conversation = defaultdict(int)
            for row in predictions:
                lengths_by_conversation[row["conversation_id"]] = max(
                    lengths_by_conversation[row["conversation_id"]],
                    int(row["target_idx"]) + 1,
                )
            mean_length = int(round(np.mean(list(lengths_by_conversation.values()))))
            model_rows.append({
                **metadata, "best_epoch": best_epoch,
                "selection_split": args.selection_split,
                "selection_metric": args.selection_metric,
                "epoch0_selection_accuracy": initial_selection_metrics["accuracy"],
                "epoch0_selection_weighted_f1": initial_selection_metrics["weighted_f1"],
                "epoch0_selection_nll": initial_selection_metrics["nll"],
                "best_selection_accuracy": best_selection_metrics["accuracy"],
                "best_selection_weighted_f1": best_selection_metrics["weighted_f1"],
                "best_selection_nll": best_selection_metrics["nll"],
                "accuracy": test_metrics["accuracy"],
                "weighted_f1": test_metrics["weighted_f1"],
                "macro_f1": test_metrics["macro_f1"], "nll": test_metrics["nll"],
                "test_utterances": test_metrics["utterances"],
                "trainable_parameters": model.trainable_parameter_count(),
                "mean_test_conversation_length": mean_length,
                "estimated_added_flops_at_mean_length": model.estimated_gate_flops(mean_length),
                "train_seconds": train_seconds, "test_seconds": test_metrics["seconds"],
                "test_utterances_per_second": (
                    test_metrics["utterances"] / test_metrics["seconds"]
                ),
                "peak_gpu_memory_bytes": peak_memory,
                "mean_gate": float(np.mean([float(row["gate_mean"])
                                            for row in predictions])),
                "strong_suppression_target_rate": float(np.mean([
                    float(row["gate_min"]) < 0.5 for row in predictions
                ])),
                **{
                    f"mean_gate_{name}": float(np.mean([
                        float(row[f"gate_mean_{name}"]) for row in predictions
                    ]))
                    for name in MODALITY_INDEX
                },
                **{
                    f"strong_suppression_target_rate_{name}": float(np.mean([
                        float(row[f"gate_min_{name}"]) < 0.5 for row in predictions
                    ]))
                    for name in MODALITY_INDEX
                },
                "gate_modalities": "+".join(args.gate_modalities),
                "initial_gate": args.initial_gate,
                "lambda_identity": args.lambda_identity,
                "lambda_impact": args.lambda_impact if model_id == "B4" else 0.0,
                "impact_min_margin": (
                    args.impact_min_margin if model_id == "B4" else 0.0
                ),
                "renormalize_attention": int(args.renormalize_attention),
                "backbone_frozen": int(not args.unfreeze_backbone),
                "classifier_frozen": int(args.freeze_classifier),
                "classifier_lr": 0.0 if args.freeze_classifier else args.lr,
                "gate_lr": args.lr if args.gate_lr is None else args.gate_lr,
                "impact_loss": args.impact_loss if model_id == "B4" else "",
                "impact_teacher_kind": (
                    teacher_summary["teacher_kind"] if model_id == "B4" else ""
                ),
                "impact_teacher_pairs": (
                    teacher_summary["teacher_pair_count"] if model_id == "B4" else 0
                ),
            })
            print(
                f"E3 {model_id} seed={run_seed}: best_epoch={best_epoch} "
                f"epoch0_WF1={initial_selection_metrics['weighted_f1']:.3f} "
                f"best_WF1={test_metrics['weighted_f1']:.3f}",
                flush=True,
            )
    write_csv(output / "model_comparison.csv", model_rows)
    write_csv(output / "epoch_history.csv", epoch_rows)
    if transition_rows:
        write_csv(output / "transition_subset_metrics.csv", transition_rows)
    if classwise_rows:
        write_csv(output / "classwise_f1.csv", classwise_rows)
    if all_predictions:
        write_csv(output / "test_predictions.csv", all_predictions)
    paired_delta_rows = _paired_deltas(
        prediction_groups, args.bootstrap_reps, args.seeds[0]
    )
    paired_delta_path = output / "paired_deltas.csv"
    if paired_delta_rows:
        write_csv(paired_delta_path, paired_delta_rows)
    else:
        # A single-model run is valid (for example one B4 sweep point), but it
        # has no within-run model pair from which a paired delta can be formed.
        if paired_delta_path.exists():
            paired_delta_path.unlink()
        print("E3 paired deltas skipped: fewer than two comparable models", flush=True)
    write_csv(output / "ablation_results.csv", [{
        "model": model_id,
        "placement": ("none" if model_id in ("B0", "B5") else
                      "late" if model_id == "B1" else "propagation"),
        "granularity": ("none" if model_id in ("B0", "B5") else
                        "target_modality" if model_id in ("B1", "B2")
                        else "source_target_modality"),
        "impact_supervision": int(model_id == "B4"),
        "impact_teacher_kind": (
            teacher_summary["teacher_kind"] if model_id == "B4" else ""
        ),
        "impact_loss": args.impact_loss if model_id == "B4" else "",
        "relevance_supervision": int(model_id == "B6"),
        "capacity_control": int(model_id == "B5"),
    } for model_id in args.models])
    write_json(output / "manifest.json", {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "split_ids": split_ids,
        "sessions": {name: sorted(values) for name, values in sessions.items()},
        "session_disjoint_required": args.require_session_disjoint,
        "models": args.models, "seeds": args.seeds,
        "training_protocol": (
            "joint end-to-end fine-tuning" if args.unfreeze_backbone
            else "strict frozen SDT; train added gates/controls only"
            if args.freeze_classifier
            else "frozen contextual backbone; trainable gate/control and classifier"
        ),
        "selection": (
            f"best {args.selection_split} {args.selection_metric}; deterministic "
            "tie-break by the other reported selection metrics"
        ),
        "selection_includes_epoch_zero": True,
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "test_evaluation_deferred": args.defer_test_evaluation,
        "impact_teacher": str(args.impact_teacher.resolve()) if args.impact_teacher else None,
        "impact_teacher_summary": teacher_summary,
        "impact_loss": args.impact_loss,
        "impact_gate_floor": args.impact_gate_floor,
        "impact_scale": args.impact_scale,
        "test_labels_or_impacts_used_for_training_or_selection": (
            args.selection_split == "test"
        ),
        "checkpoints": run_manifest,
        "paired_delta_rows": len(paired_delta_rows),
        "smoke_limited": bool(args.max_batches),
    })


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.gate_hidden) < 1 or args.capacity_hidden < 0:
        raise ValueError("Positive training dimensions and nonnegative capacity width required")
    if (args.lr <= 0 or (args.gate_lr is not None and args.gate_lr <= 0)
            or args.impact_eps < 0 or args.impact_scale <= 0
            or args.impact_min_margin < 0
            or not 0.0 <= args.impact_gate_floor <= 1.0
            or not 0.5 < args.initial_gate < 1):
        raise ValueError("Invalid learning rate, epsilon, or initial gate")
    if args.freeze_classifier and args.unfreeze_backbone:
        raise ValueError("--freeze-classifier and --unfreeze-backbone are incompatible")
    train(args)


if __name__ == "__main__":
    main()
