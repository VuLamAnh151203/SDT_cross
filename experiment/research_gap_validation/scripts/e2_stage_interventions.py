"""E2: frozen input, propagation-edge, and late-modality interventions."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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

from common import (classification_metrics, cluster_bootstrap, read_csv, write_csv,
                    write_json)  # noqa: E402
from impact_core import (apply_replacement, compute_training_means, dataset_lookup,
                         load_conversation, load_experiment, seed_everything)  # noqa: E402
from reliability_model import ReliabilitySDT, source_branches  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--input-impact-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITY_INDEX),
                        default=("text",))
    parser.add_argument("--gates", nargs="+", type=float,
                        default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--intervention-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--renormalize-attention", action="store_true")
    return parser.parse_args(argv)


def _repeat_batch(batch, count: int):
    return (
        batch.text.repeat(1, count, 1), batch.visual.repeat(1, count, 1),
        batch.audio.repeat(1, count, 1), batch.mask.repeat(count, 1),
        batch.speakers.repeat(count, 1, 1), [batch.length] * count,
    )


def _base_row(experiment, split, batch, baseline, target: int):
    label = int(batch.labels[0, target].item())
    logits = baseline["logits"][0, target]
    probs = baseline["prob"][0, target]
    pred = int(logits.argmax().item())
    target_speaker = int(batch.speakers[0, target].argmax().item())
    previous_same = next(
        (index for index in range(target - 1, -1, -1)
         if int(batch.speakers[0, index].argmax().item()) == target_speaker), -1
    )
    row = {
        "dataset": experiment.dataset.dataset, "split": split,
        "checkpoint_id": experiment.checkpoint_id,
        "conversation_id": batch.conversation_id,
        "conversation_length": batch.length, "target_idx": target,
        "target_label": label,
        "target_speaker": target_speaker,
        "previous_same_speaker_idx": previous_same,
        "continuity": int(previous_same >= 0
                          and int(batch.labels[0, previous_same].item()) == label),
        "transition": int(previous_same >= 0
                          and int(batch.labels[0, previous_same].item()) != label),
        "baseline_prediction": pred,
        "baseline_correct": int(pred == label),
        "baseline_nll": float(-baseline["log_prob"][0, target, label].cpu()),
    }
    row.update({f"baseline_prob_{index}": float(value.cpu())
                for index, value in enumerate(probs)})
    row.update({f"baseline_logit_{index}": float(value.cpu())
                for index, value in enumerate(logits)})
    return row


def _action_row(base, output, batch_index: int, target: int, location: str,
                modality: str, gate: float, source: int, path_scope: str,
                cost: float, n_classes: int, batch=None):
    label = int(base["target_label"])
    logits = output["logits"][batch_index, target]
    probs = output["prob"][batch_index, target]
    log_prob = output["log_prob"][batch_index, target]
    prediction = int(logits.argmax().item())
    nll = float(-log_prob[label].detach().cpu())
    row = dict(base)
    row.update({
        "intervention_location": location, "path_scope": path_scope,
        "action_kind": ("train_mean_replacement" if location == "input_source"
                        else "continuous_gate"),
        "source_idx": source, "modality": modality, "gate": gate,
        "action_budget": 1, "normalized_cost": cost,
        "prediction": prediction, "correct": int(prediction == label),
        "nll": nll, "delta_nll": nll - float(base["baseline_nll"]),
        "changed_prediction": int(prediction != int(base["baseline_prediction"])),
    })
    if source >= 0 and batch is not None:
        source_speaker = int(batch.speakers[0, source].argmax().item())
        row.update({
            "source_speaker": source_speaker,
            "same_speaker": int(source_speaker == int(base["target_speaker"])),
            "absolute_distance": abs(source - target),
            "source_before_target": int(source < target),
            "source_emotion": int(batch.labels[0, source].item()),
            "target_emotion": base["target_label"],
        })
    else:
        row.update({"source_speaker": "", "same_speaker": "",
                    "absolute_distance": "", "source_before_target": "",
                    "source_emotion": "", "target_emotion": base["target_label"]})
    row.update({f"prob_{index}": float(value.detach().cpu())
                for index, value in enumerate(probs[:n_classes])})
    row.update({f"logit_{index}": float(value.detach().cpu())
                for index, value in enumerate(logits[:n_classes])})
    return row


def _identity_action(base, location: str, modality: str, source: int,
                     path_scope: str, n_classes: int, batch=None):
    row = dict(base)
    row.update({
        "intervention_location": location, "path_scope": path_scope,
        "action_kind": "continuous_gate",
        "source_idx": source, "modality": modality, "gate": 1.0,
        "action_budget": 0, "normalized_cost": 0.0,
        "prediction": base["baseline_prediction"],
        "correct": base["baseline_correct"], "nll": base["baseline_nll"],
        "delta_nll": 0.0, "changed_prediction": 0,
    })
    for index in range(n_classes):
        row[f"prob_{index}"] = base[f"baseline_prob_{index}"]
        row[f"logit_{index}"] = base[f"baseline_logit_{index}"]
    if source >= 0 and batch is not None:
        source_speaker = int(batch.speakers[0, source].argmax().item())
        row.update({
            "source_speaker": source_speaker,
            "same_speaker": int(source_speaker == int(base["target_speaker"])),
            "absolute_distance": abs(source - int(base["target_idx"])),
            "source_before_target": int(source < int(base["target_idx"])),
            "source_emotion": int(batch.labels[0, source].item()),
            "target_emotion": base["target_label"],
        })
    else:
        row.update({"source_speaker": "", "same_speaker": "",
                    "absolute_distance": "", "source_before_target": "",
                    "source_emotion": "", "target_emotion": base["target_label"]})
    return row


def _run_edge_actions(wrapper, batch, baseline_rows, actions, batch_size,
                      baseline_attention, n_classes):
    rows = []
    length = batch.length
    for start in range(0, len(actions), batch_size):
        chunk = actions[start:start + batch_size]
        count = len(chunk)
        edge = torch.ones(count, length, length, 3, device=batch.text.device)
        branch_scopes = []
        for index, (target, source, modality, gate, scope) in enumerate(chunk):
            edge[index, target, source, modality] = gate
            branch_scopes.append(
                source_branches(modality) if scope == "all_source_branches"
                else (source_branches(modality)[0],)
            )
        # Different one-branch selections cannot share hook scope. All one-branch
        # actions for one source modality use the same first branch by design.
        scopes = {tuple(value) for value in branch_scopes}
        if len(scopes) != 1:
            raise AssertionError("Intervention chunk mixes incompatible path scopes")
        repeated = _repeat_batch(batch, count)
        with torch.inference_mode():
            output = wrapper(*repeated, fixed_edge_gate=edge,
                             active_branches=next(iter(scopes)))
        for index, (target, source, modality, gate, scope) in enumerate(chunk):
            branches = branch_scopes[index]
            masses = [float(baseline_attention[name][:, target, source].mean().cpu())
                      for name in branches]
            cost = (1.0 - gate) * float(np.mean(masses))
            rows.append(_action_row(
                baseline_rows[target], output, index, target, "propagation_edge",
                ("text", "audio", "visual")[modality], gate, source, scope,
                cost, n_classes,
                batch,
            ))
    return rows


def _run_late_actions(wrapper, batch, baseline_rows, actions, batch_size,
                      enhanced, n_classes):
    rows = []
    length = batch.length
    for start in range(0, len(actions), batch_size):
        chunk = actions[start:start + batch_size]
        count = len(chunk)
        gate_tensor = torch.ones(count, length, 3, device=batch.text.device)
        for index, (target, modality, gate) in enumerate(chunk):
            gate_tensor[index, target, modality] = gate
        repeated = _repeat_batch(batch, count)
        with torch.inference_mode():
            output = wrapper(*repeated, fixed_late_gate=gate_tensor)
        for index, (target, modality, gate) in enumerate(chunk):
            feature = enhanced[("t", "a", "v")[modality]][0, target]
            cost = (1.0 - gate) * float(feature.norm().cpu())
            rows.append(_action_row(
                baseline_rows[target], output, index, target, "late_modality",
                ("text", "audio", "visual")[modality], gate, -1, "target_modality",
                cost, n_classes,
                batch,
            ))
    return rows


def _oracle(stage_rows, baseline_rows, repetitions: int, seed: int,
            input_impacts=None, impact_eps: float = 1e-4):
    actions = defaultdict(list)
    for row in stage_rows:
        if row["path_scope"] in ("all_source_branches", "target_modality"):
            actions[(row["conversation_id"], int(row["target_idx"]),
                     row["intervention_location"])].append(row)
    targets = []
    for base in baseline_rows:
        key = (base["conversation_id"], int(base["target_idx"]))
        choices = {}
        for location in ("propagation_edge", "late_modality"):
            candidates = actions.get((*key, location), [])
            best = min(candidates, key=lambda row: float(row["nll"])) if candidates else None
            if best is None or float(best["nll"]) >= float(base["baseline_nll"]):
                choices[location] = {
                    "nll": float(base["baseline_nll"]),
                    "prediction": int(base["baseline_prediction"]),
                    "source_idx": -1, "modality": "none", "gate": 1.0,
                    "cost": 0.0,
                }
            else:
                choices[location] = {
                    "nll": float(best["nll"]), "prediction": int(best["prediction"]),
                    "source_idx": int(best["source_idx"]), "modality": best["modality"],
                    "gate": float(best["gate"]), "cost": float(best["normalized_cost"]),
                }
        label = int(base["target_label"])
        beneficial_keys = []
        if input_impacts is not None:
            beneficial_keys = [item_key for item_key, impact in input_impacts.items()
                               if item_key[0] == key[0] and item_key[2] == key[1]
                               and impact > impact_eps]
        selected_key = (key[0], choices["propagation_edge"]["source_idx"], key[1],
                        choices["propagation_edge"]["modality"])
        suppressed_beneficial = int(
            choices["propagation_edge"]["source_idx"] >= 0
            and choices["propagation_edge"]["gate"] < 1.0
            and selected_key in beneficial_keys
        )
        targets.append({
            "conversation_id": base["conversation_id"], "target_idx": base["target_idx"],
            "target_label": label, "baseline_prediction": base["baseline_prediction"],
            "baseline_nll": base["baseline_nll"],
            "early_prediction": choices["propagation_edge"]["prediction"],
            "early_nll": choices["propagation_edge"]["nll"],
            "early_delta_nll": choices["propagation_edge"]["nll"] - float(base["baseline_nll"]),
            "early_source_idx": choices["propagation_edge"]["source_idx"],
            "early_modality": choices["propagation_edge"]["modality"],
            "early_gate": choices["propagation_edge"]["gate"],
            "early_cost": choices["propagation_edge"]["cost"],
            "late_prediction": choices["late_modality"]["prediction"],
            "late_nll": choices["late_modality"]["nll"],
            "late_delta_nll": choices["late_modality"]["nll"] - float(base["baseline_nll"]),
            "late_modality": choices["late_modality"]["modality"],
            "late_gate": choices["late_modality"]["gate"],
            "late_cost": choices["late_modality"]["cost"],
            "early_advantage_nll": (choices["late_modality"]["nll"]
                                    - choices["propagation_edge"]["nll"]),
            "early_corrected": int(int(base["baseline_prediction"]) != label
                                   and choices["propagation_edge"]["prediction"] == label),
            "early_damaged": int(int(base["baseline_prediction"]) == label
                                 and choices["propagation_edge"]["prediction"] != label),
            "late_corrected": int(int(base["baseline_prediction"]) != label
                                  and choices["late_modality"]["prediction"] == label),
            "late_damaged": int(int(base["baseline_prediction"]) == label
                                and choices["late_modality"]["prediction"] != label),
            "baseline_beneficial_source_count": len(beneficial_keys),
            "retained_beneficial_source_count": len(beneficial_keys) - suppressed_beneficial,
            "retained_beneficial_source_fraction": (
                (len(beneficial_keys) - suppressed_beneficial) / len(beneficial_keys)
                if beneficial_keys else ""
            ),
            "selected_edge_was_beneficial_under_input_intervention": suppressed_beneficial,
            "oracle_uses_true_label": 1,
        })
    differences = [float(row["early_advantage_nll"]) for row in targets]
    low, high = cluster_bootstrap(
        targets, lambda sample: float(np.mean([
            float(row["early_advantage_nll"]) for row in sample
        ])), repetitions, seed,
    )
    for row in targets:
        row["aggregate_mean_early_advantage_nll"] = float(np.mean(differences))
        row["aggregate_ci95_low"] = low
        row["aggregate_ci95_high"] = high
    return targets


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    wrapper = ReliabilitySDT(
        experiment.model, "B0", renormalize_attention=args.renormalize_attention
    ).to(experiment.device).eval()
    ids = list(experiment.split_ids[args.split])
    if args.max_conversations:
        ids = ids[:args.max_conversations]
    lookup = dataset_lookup(experiment.dataset)
    train_means, _ = compute_training_means(
        experiment.dataset, experiment.split_ids["train"]
    )
    replacements = {name: torch.from_numpy(train_means[name]).to(experiment.device)
                    for name in args.modalities}
    all_rows, all_baselines, path_rows = [], [], []
    gates = sorted(set(args.gates))
    if any(not 0 <= value <= 1 for value in gates) or 1.0 not in gates:
        raise ValueError("gates must be in [0,1] and include identity gate 1")
    action_gates = [value for value in gates if value < 1.0]
    for number, conversation_id in enumerate(ids, start=1):
        batch = load_conversation(experiment.dataset, lookup, conversation_id,
                                  experiment.device)
        length = batch.length
        identity_edge = torch.ones(1, length, length, 3, device=experiment.device)
        identity_late = torch.ones(1, length, 3, device=experiment.device)
        with torch.inference_mode():
            baseline = wrapper(
                batch.text, batch.visual, batch.audio, batch.mask, batch.speakers,
                batch.lengths, fixed_edge_gate=identity_edge,
                fixed_late_gate=identity_late, capture_attention=True,
            )
        baseline_attention = {name: values[0][0] for name, values in
                              baseline["attention_audit"].items()}
        enhanced = baseline["late_feature_audit"]
        baseline_rows = [_base_row(experiment, args.split, batch, baseline, target)
                         for target in range(length)]
        all_baselines.extend(baseline_rows)
        for modality_name in args.modalities:
            for source in range(length):
                modified = apply_replacement(
                    batch.features[modality_name], source, replacements[modality_name]
                )
                features = {"text": batch.text, "audio": batch.audio,
                            "visual": batch.visual}
                features[modality_name] = modified
                with torch.inference_mode():
                    input_output = wrapper(
                        features["text"], features["visual"], features["audio"],
                        batch.mask, batch.speakers, batch.lengths,
                    )
                cost = float((batch.features[modality_name][source, 0]
                              - replacements[modality_name]).norm().cpu())
                for target in range(length):
                    if source != target:
                        all_rows.append(_action_row(
                            baseline_rows[target], input_output, 0, target,
                            "input_source", modality_name, 0.0, source,
                            "all_propagation_paths", cost,
                            experiment.dataset.n_classes, batch,
                        ))
        for modality_name in args.modalities:
            for target in range(length):
                all_rows.append(_identity_action(
                    baseline_rows[target], "late_modality", modality_name, -1,
                    "target_modality", experiment.dataset.n_classes,
                    batch,
                ))
                for source in range(length):
                    if source != target:
                        all_rows.append(_identity_action(
                            baseline_rows[target], "propagation_edge", modality_name,
                            source, "all_source_branches", experiment.dataset.n_classes,
                            batch,
                        ))
        edge_actions = []
        one_branch_actions = []
        for modality_name in args.modalities:
            modality = MODALITY_INDEX[modality_name]
            for target in range(length):
                for source in range(length):
                    if source == target:
                        continue
                    for gate in action_gates:
                        edge_actions.append((target, source, modality, gate,
                                             "all_source_branches"))
                    one_branch_actions.append((target, source, modality, 0.0,
                                               "one_source_branch"))
        # Chunk by modality/scope so every batch uses one compatible hook set.
        for modality_name in args.modalities:
            modality = MODALITY_INDEX[modality_name]
            selected = [action for action in edge_actions if action[2] == modality]
            all_rows.extend(_run_edge_actions(
                wrapper, batch, baseline_rows, selected,
                args.intervention_batch_size, baseline_attention,
                experiment.dataset.n_classes,
            ))
            selected_one = [action for action in one_branch_actions if action[2] == modality]
            one_rows = _run_edge_actions(
                wrapper, batch, baseline_rows, selected_one,
                args.intervention_batch_size, baseline_attention,
                experiment.dataset.n_classes,
            )
            all_rows.extend(one_rows)
            path_rows.extend(one_rows)
        late_actions = [
            (target, MODALITY_INDEX[modality], gate)
            for modality in args.modalities for target in range(length)
            for gate in action_gates
        ]
        all_rows.extend(_run_late_actions(
            wrapper, batch, baseline_rows, late_actions,
            args.intervention_batch_size, enhanced, experiment.dataset.n_classes,
        ))
        print(f"E2 {number}/{len(ids)} {conversation_id}", flush=True)
    output = args.output_dir.resolve() / "E2"
    input_impact_rows = (read_csv(args.input_impact_file)
                         if args.input_impact_file and args.input_impact_file.is_file()
                         else [])
    input_impact_lookup = {
        (row["conversation_id"], int(row["source_idx"]), int(row["target_idx"]),
         row["modality"]): float(row["delta_nll"])
        for row in all_rows if row["intervention_location"] == "input_source"
    }
    saved_input_lookup = {}
    if input_impact_rows:
        for row in input_impact_rows:
            if row["modality"] not in args.modalities:
                continue
            saved_input_lookup[(row["conversation_id"], int(row["source_idx"]),
                                int(row["target_idx"]), row["modality"])] = float(
                                    row["signed_impact"])
    common_input_keys = input_impact_lookup.keys() & saved_input_lookup.keys()
    maximum_saved_input_difference = (
        max(abs(input_impact_lookup[key] - saved_input_lookup[key])
            for key in common_input_keys) if common_input_keys else None
    )
    write_csv(output / "stage_intervention_pairs.csv", all_rows)
    oracle = _oracle(all_rows, all_baselines, args.bootstrap_reps, args.seed,
                     input_impact_lookup, args.impact_eps)
    write_csv(output / "oracle_stage_comparison.csv", oracle)
    oracle_summary = []
    labels = [int(row["target_label"]) for row in oracle]
    baseline_predictions = [int(row["baseline_prediction"]) for row in oracle]
    baseline_metrics = classification_metrics(labels, baseline_predictions)
    for location, prefix in (("propagation_edge", "early"),
                             ("late_modality", "late")):
        predictions = [int(row[f"{prefix}_prediction"]) for row in oracle]
        metrics = classification_metrics(labels, predictions)
        delta_low, delta_high = cluster_bootstrap(
            oracle,
            lambda sample, name=prefix: float(np.mean([
                float(row[f"{name}_delta_nll"]) for row in sample
            ])),
            args.bootstrap_reps, args.seed + (0 if prefix == "early" else 1),
        )
        oracle_summary.append({
            "location": location, "target_count": len(oracle),
            "baseline_accuracy": baseline_metrics["accuracy"],
            "oracle_accuracy": metrics["accuracy"],
            "delta_accuracy": metrics["accuracy"] - baseline_metrics["accuracy"],
            "baseline_weighted_f1": baseline_metrics["weighted_f1"],
            "oracle_weighted_f1": metrics["weighted_f1"],
            "delta_weighted_f1": metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "oracle_macro_f1": metrics["macro_f1"],
            "delta_macro_f1": metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "mean_delta_nll": float(np.mean([
                float(row[f"{prefix}_delta_nll"]) for row in oracle
            ])),
            "mean_delta_nll_ci95_low": delta_low,
            "mean_delta_nll_ci95_high": delta_high,
            "corrected_count": sum(int(row[f"{prefix}_corrected"]) for row in oracle),
            "damaged_count": sum(int(row[f"{prefix}_damaged"]) for row in oracle),
            "oracle_uses_true_label": 1,
        })
    write_csv(output / "oracle_stage_summary.csv", oracle_summary)
    all_branch = {
        (row["conversation_id"], row["source_idx"], row["target_idx"], row["modality"]): row
        for row in all_rows if row["intervention_location"] == "propagation_edge"
        and row["path_scope"] == "all_source_branches" and float(row["gate"]) == 0.0
    }
    sensitivity = []
    for row in path_rows:
        key = (row["conversation_id"], row["source_idx"], row["target_idx"], row["modality"])
        complete = all_branch[key]
        sensitivity.append({
            "conversation_id": row["conversation_id"], "source_idx": row["source_idx"],
            "target_idx": row["target_idx"], "modality": row["modality"],
            "one_branch_delta_nll": row["delta_nll"],
            "all_source_branches_delta_nll": complete["delta_nll"],
            "remaining_fraction": (
                float(row["delta_nll"]) / float(complete["delta_nll"])
                if abs(float(complete["delta_nll"])) > 1e-12 else ""
            ),
            "transformer_layers_per_branch": 1,
        })
    write_csv(output / "layer_path_sensitivity.csv", sensitivity)
    write_json(output / "manifest.json", {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash, "split": args.split,
        "conversation_ids": ids, "seed": args.seed,
        "conversations": len(ids), "targets": len(all_baselines),
        "modalities": args.modalities, "gates": gates,
        "edge_gate_location": "post_softmax",
        "renormalize_attention": args.renormalize_attention,
        "early_budget": "at most one source-target-modality edge",
        "late_budget": "at most one target-modality representation",
        "granularity_warning": "early and late action spaces have different granularity",
        "oracle_uses_true_test_labels": True,
        "saved_input_impact_common_pairs": len(common_input_keys),
        "maximum_saved_vs_recomputed_input_impact_difference": (
            maximum_saved_input_difference
        ),
    })
    print(f"E2 complete: {len(all_rows)} stage actions")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.intervention_batch_size < 1 or args.bootstrap_reps < 0:
        raise ValueError("Invalid batch/bootstrap count")
    run(args)


if __name__ == "__main__":
    main()
