"""Generate train-only B4 supervision with the exact propagation-edge action.

For every source-target-modality edge, this script evaluates a fixed gate grid
at the same attention hooks used by B3/B4.  The best train-label NLL gate is the
teacher target.  Confidence is the NLL advantage of the best suppress/keep
decision and is stored separately so weak, nearly tied interventions do not
dominate B4 training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
for path in (ROOT, CROSS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import write_csv, write_json  # noqa: E402
from impact_core import (dataset_lookup, load_conversation, load_experiment,
                         seed_everything)  # noqa: E402
from reliability_model import ReliabilitySDT, source_branches  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITY_INDEX),
                        default=("text",))
    parser.add_argument("--gates", nargs="+", type=float,
                        default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--intervention-batch-size", type=int, default=64)
    parser.add_argument("--teacher-weight-scale", type=float, default=1e-2)
    parser.add_argument("--teacher-margin-eps", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--renormalize-attention", action="store_true")
    return parser.parse_args(argv)


def _repeat_batch(batch, count: int):
    return (
        batch.text.repeat(1, count, 1),
        batch.visual.repeat(1, count, 1),
        batch.audio.repeat(1, count, 1),
        batch.mask.repeat(count, 1),
        batch.speakers.repeat(count, 1, 1),
        [batch.length] * count,
    )


def _evaluate_actions(wrapper, batch, modality: int, gates: Sequence[float],
                      batch_size: int) -> dict[tuple[int, int], dict[float, float]]:
    length = batch.length
    labels = batch.labels[0, :length]
    actions = [
        (target, source, gate)
        for target in range(length)
        for source in range(length)
        if source != target
        for gate in gates
    ]
    values: dict[tuple[int, int], dict[float, float]] = {}
    for start in range(0, len(actions), batch_size):
        chunk = actions[start:start + batch_size]
        count = len(chunk)
        edge = torch.ones(count, length, length, 3, device=batch.text.device)
        for index, (target, source, gate) in enumerate(chunk):
            edge[index, target, source, modality] = gate
        repeated = _repeat_batch(batch, count)
        with torch.inference_mode():
            output = wrapper(
                *repeated, fixed_edge_gate=edge,
                active_branches=source_branches(modality),
            )
        for index, (target, source, gate) in enumerate(chunk):
            label = int(labels[target].item())
            nll = float(-output["log_prob"][index, target, label].detach().cpu())
            values.setdefault((target, source), {})[float(gate)] = nll
    return values


def select_teacher_action(values: dict[float, float], weight_scale: float,
                          margin_epsilon: float) -> tuple[float, float, str, float]:
    """Return target gate, confidence weight, decision, and decision margin."""
    if 1.0 not in values or len(values) < 2:
        raise ValueError("Teacher action values require identity and suppression gates")
    action_gates = [gate for gate in values if gate < 1.0]
    if not action_gates:
        raise ValueError("Teacher action values contain no suppression gate")
    ranked = sorted(values.items(), key=lambda item: (item[1], -item[0]))
    best_gate, best_nll = ranked[0]
    baseline_nll = values[1.0]
    best_suppressed_nll = min(values[gate] for gate in action_gates)
    if best_gate < 1.0:
        decision_margin = baseline_nll - best_nll
        decision = "suppress"
    else:
        decision_margin = best_suppressed_nll - baseline_nll
        decision = "keep"
    weight = float(np.clip(
        (decision_margin - margin_epsilon) / weight_scale, 0.0, 1.0
    ))
    return best_gate, weight, decision, decision_margin


def generate(args: argparse.Namespace) -> None:
    gates = sorted(set(float(value) for value in args.gates))
    if (not gates or any(value < 0.0 or value > 1.0 for value in gates)
            or 1.0 not in gates):
        raise ValueError("--gates must be in [0,1] and include identity gate 1")
    action_gates = [value for value in gates if value < 1.0]
    if not action_gates:
        raise ValueError("At least one non-identity teacher gate is required")
    if (args.intervention_batch_size < 1 or args.teacher_weight_scale <= 0.0
            or args.teacher_margin_eps < 0.0):
        raise ValueError("Invalid batch size or teacher confidence parameters")

    seed_everything(args.seed)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    wrapper = ReliabilitySDT(
        experiment.model, "B0", renormalize_attention=args.renormalize_attention
    ).to(experiment.device).eval()
    ids = list(experiment.split_ids["train"])
    if args.max_conversations:
        ids = ids[:args.max_conversations]
    lookup = dataset_lookup(experiment.dataset)
    rows = []
    decision_counts = {"keep": 0, "suppress": 0, "zero_weight": 0}

    for number, conversation in enumerate(ids, start=1):
        batch = load_conversation(
            experiment.dataset, lookup, conversation, experiment.device
        )
        length = batch.length
        labels = batch.labels[0, :length]
        identity = torch.ones(1, length, length, 3, device=experiment.device)
        with torch.inference_mode():
            baseline = wrapper(
                batch.text, batch.visual, batch.audio, batch.mask,
                batch.speakers, batch.lengths, fixed_edge_gate=identity,
            )
        baseline_nll = [
            float(-baseline["log_prob"][0, target, int(labels[target])].cpu())
            for target in range(length)
        ]

        for modality_name in args.modalities:
            modality = MODALITY_INDEX[modality_name]
            action_values = _evaluate_actions(
                wrapper, batch, modality, action_gates,
                args.intervention_batch_size,
            )
            for (target, source), values in action_values.items():
                values[1.0] = baseline_nll[target]
                best_gate, weight, decision, decision_margin = select_teacher_action(
                    values, args.teacher_weight_scale, args.teacher_margin_eps
                )
                best_nll = values[best_gate]
                decision_counts[decision] += 1
                if weight == 0.0:
                    decision_counts["zero_weight"] += 1
                source_speaker = int(batch.speakers[0, source].argmax().item())
                target_speaker = int(batch.speakers[0, target].argmax().item())
                rows.append({
                    "split": "train",
                    "checkpoint_id": experiment.checkpoint_id,
                    "conversation_id": conversation,
                    "source_idx": source,
                    "target_idx": target,
                    "modality": modality_name,
                    "teacher_kind": "propagation_edge_grid",
                    "path_scope": "all_source_branches",
                    "teacher_gate": best_gate,
                    "teacher_weight": weight,
                    "teacher_decision": decision,
                    "teacher_decision_margin": decision_margin,
                    "baseline_nll": baseline_nll[target],
                    "best_nll": best_nll,
                    "best_delta_nll": best_nll - baseline_nll[target],
                    "full_suppression_delta_nll": (
                        values[0.0] - baseline_nll[target]
                        if 0.0 in values else ""
                    ),
                    "same_speaker": int(source_speaker == target_speaker),
                    "absolute_distance": abs(source - target),
                    "source_before_target": int(source < target),
                    **{f"nll_gate_{gate:g}": values[gate] for gate in gates},
                })
        print(f"edge teacher {number}/{len(ids)} {conversation}", flush=True)

    output = args.output_dir.resolve() / "manifests"
    path = write_csv(output / "train_edge_impact_teacher.csv", rows)
    effective = [float(row["teacher_weight"]) for row in rows
                 if float(row["teacher_weight"]) > 0.0]
    write_json(output / "train_edge_impact_teacher_manifest.json", {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "split": "train",
        "conversation_ids": ids,
        "modalities": args.modalities,
        "teacher_kind": "propagation_edge_grid",
        "path_scope": "all_source_branches",
        "gates": gates,
        "teacher_weight_scale": args.teacher_weight_scale,
        "teacher_margin_eps": args.teacher_margin_eps,
        "rows": len(rows),
        "effective_rows": len(effective),
        "mean_effective_weight": float(np.mean(effective)) if effective else None,
        "decision_counts": decision_counts,
        "renormalize_attention": bool(args.renormalize_attention),
        "test_conversations_present": bool(
            set(ids) & set(experiment.split_ids.get("test", []))
        ),
        "output": str(path),
    })


def main(argv: Sequence[str] | None = None) -> None:
    generate(parse_args(argv))


if __name__ == "__main__":
    main()
