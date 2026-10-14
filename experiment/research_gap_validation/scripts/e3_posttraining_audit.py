"""Re-run singleton source interventions on trained B0--B6 models."""

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
SDT_ROOT = ROOT.parents[1]
for path in (ROOT, CROSS, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import write_csv, write_json  # noqa: E402
from impact_core import (apply_replacement, compute_training_means, dataset_lookup,
                         forward_conversation, load_conversation, load_experiment,
                         seed_everything)  # noqa: E402
from model import Transformer_Based_Model  # noqa: E402
from reliability_model import ReliabilitySDT  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--trained-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITY_INDEX),
                        default=("text",))
    parser.add_argument("--strategies", nargs="+", choices=("train_mean", "zero"),
                        default=("train_mean",))
    parser.add_argument("--impact-eps", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _load_wrapper(experiment, artifact_path: Path):
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact["base_checkpoint_id"] != experiment.checkpoint_id:
        raise ValueError(f"Base checkpoint mismatch: {artifact_path}")
    config = artifact["configuration"]
    base = Transformer_Based_Model(**experiment.checkpoint["model_config"]).to(experiment.device)
    base.load_state_dict(experiment.checkpoint["model_state_dict"], strict=True)
    if hasattr(base, "set_tical_epoch"):
        base.set_tical_epoch(int(experiment.checkpoint.get("epoch", 0)))
    wrapper = ReliabilitySDT(
        base, artifact["model_id"], gate_hidden=int(config["gate_hidden"]),
        initial_gate=float(config["initial_gate"]),
        capacity_hidden=int(config["capacity_hidden"]),
        renormalize_attention=bool(config["renormalize_attention"]),
    ).to(experiment.device)
    freeze_classifier = bool(config.get("freeze_classifier", False))
    wrapper.freeze_backbone(train_classifier=not freeze_classifier)
    if bool(config.get("unfreeze_backbone", False)):
        wrapper.unfreeze_backbone()
    wrapper.load_trainable_state_dict(artifact["trainable_state_dict"])
    wrapper.eval()
    return wrapper, artifact


def _forward(wrapper, batch, override=None):
    with torch.inference_mode():
        return wrapper(
            batch.text, batch.visual, batch.audio, batch.mask, batch.speakers,
            batch.lengths, **({} if override is None else override),
        )


def audit(args: argparse.Namespace) -> None:
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    means, _ = compute_training_means(experiment.dataset, experiment.split_ids["train"])
    replacements = {
        "train_mean": {name: torch.from_numpy(means[name]).to(experiment.device)
                       for name in MODALITY_INDEX},
        "zero": {name: torch.zeros_like(torch.from_numpy(means[name])).to(experiment.device)
                 for name in MODALITY_INDEX},
    }
    ids = list(experiment.split_ids[args.split])
    if args.max_conversations:
        ids = ids[:args.max_conversations]
    lookup = dataset_lookup(experiment.dataset)
    all_summary = []
    audit_paths = []
    baseline_pair_maps = {}
    for checkpoint_path in sorted(args.trained_checkpoints, key=lambda path: path.name):
        wrapper, artifact = _load_wrapper(experiment, checkpoint_path)
        seed_everything(int(artifact["seed"]))
        rows = []
        for number, conversation in enumerate(ids, start=1):
            batch = load_conversation(experiment.dataset, lookup, conversation,
                                      experiment.device)
            length = batch.length
            labels = batch.labels[0, :length]
            full = _forward(wrapper, batch)
            pair_gates = full["reliability_pair_gate"]
            modality_gates = full["reliability_modality_gate"]
            for strategy in args.strategies:
                for modality in args.modalities:
                    original = batch.features[modality]
                    for source in range(length):
                        modified = apply_replacement(
                            original, source, replacements[strategy][modality]
                        )
                        # The wrapper accepts raw tensors rather than feature override;
                        # replace the selected input explicitly.
                        features = {"text": batch.text, "audio": batch.audio,
                                    "visual": batch.visual}
                        features[modality] = modified
                        with torch.inference_mode():
                            masked = wrapper(
                                features["text"], features["visual"], features["audio"],
                                batch.mask, batch.speakers, batch.lengths,
                            )
                        for target in range(length):
                            if source == target:
                                continue
                            label = int(labels[target].item())
                            source_label = int(labels[source].item())
                            full_nll = float(-full["log_prob"][0, target, label].cpu())
                            mask_nll = float(-masked["log_prob"][0, target, label].cpu())
                            if pair_gates is not None:
                                gate_value = float(pair_gates[
                                    0, target, source, MODALITY_INDEX[modality]
                                ].cpu())
                            elif modality_gates is not None:
                                gate_value = float(modality_gates[
                                    0, target, MODALITY_INDEX[modality]
                                ].cpu())
                            else:
                                gate_value = 1.0
                            rows.append({
                                "model": artifact["model_id"], "seed": artifact["seed"],
                                "split": args.split, "conversation_id": conversation,
                                "source_idx": source, "target_idx": target,
                                "modality": modality, "replacement_strategy": strategy,
                                "source_label": source_label, "target_label": label,
                                "emotion_transition": int(source_label != label),
                                "loss_full": full_nll,
                                "loss_mask": mask_nll, "signed_impact": mask_nll - full_nll,
                                "pred_full": int(full["logits"][0, target].argmax().item()),
                                "pred_mask": int(masked["logits"][0, target].argmax().item()),
                                "gate_value": gate_value,
                                "old_emotion_probability_bias_B": float(
                                    full["prob"][0, target, source_label]
                                    - masked["prob"][0, target, source_label]
                                ),
                                "old_vs_current_logit_shift_R": float(
                                    (full["logits"][0, target, source_label]
                                     - full["logits"][0, target, label])
                                    - (masked["logits"][0, target, source_label]
                                       - masked["logits"][0, target, label])
                                ),
                                "same_speaker": int(
                                    (batch.speakers[0, source].argmax()
                                     == batch.speakers[0, target].argmax()).item()
                                ),
                                "absolute_distance": abs(source - target),
                            })
            print(f"audit {artifact['model_id']} {number}/{len(ids)}", flush=True)
        audit_path = (args.output_dir.resolve() / "E3" / "audits"
                      / f"{artifact['model_id']}_seed{artifact['seed']}.csv")
        write_csv(audit_path, rows)
        audit_paths.append(str(audit_path))
        pair_map = {(row["conversation_id"], row["source_idx"], row["target_idx"],
                     row["modality"], row["replacement_strategy"]): row for row in rows}
        if artifact["model_id"] == "B0":
            baseline_pair_maps[int(artifact["seed"])] = pair_map
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["modality"], row["replacement_strategy"])].append(row)
        for (modality, strategy), selected in grouped.items():
            impacts = np.asarray([float(row["signed_impact"]) for row in selected])
            target_groups = defaultdict(list)
            for row in selected:
                target_groups[(row["conversation_id"], row["target_idx"])].append(
                    float(row["signed_impact"])
                )
            mixed = [any(value > args.impact_eps for value in values)
                     and any(value < -args.impact_eps for value in values)
                     for values in target_groups.values()]
            all_summary.append({
                "model": artifact["model_id"], "seed": artifact["seed"],
                "modality": modality, "replacement_strategy": strategy,
                "pair_count": len(selected),
                "harmful_rate": float(np.mean(impacts < -args.impact_eps)),
                "beneficial_rate": float(np.mean(impacts > args.impact_eps)),
                "mean_absolute_impact": float(np.mean(np.abs(impacts))),
                "mixed_target_fraction": float(np.mean(mixed)),
                "mean_gate": float(np.mean([float(row["gate_value"]) for row in selected])),
                "transition_old_emotion_R_mean": float(np.mean([
                    float(row["old_vs_current_logit_shift_R"]) for row in selected
                    if int(row["emotion_transition"]) == 1
                ])) if any(int(row["emotion_transition"]) == 1 for row in selected) else "",
                "helpful_retention": "", "harmful_reduction": "",
            })
        baseline_pair_map = baseline_pair_maps.get(int(artifact["seed"]))
        if baseline_pair_map is not None and artifact["model_id"] != "B0":
            for summary in all_summary:
                if summary["model"] != artifact["model_id"] or summary["seed"] != artifact["seed"]:
                    continue
                keys = [key for key in pair_map if key[3] == summary["modality"]
                        and key[4] == summary["replacement_strategy"] and key in baseline_pair_map]
                helpful = [key for key in keys
                           if float(baseline_pair_map[key]["signed_impact"]) > args.impact_eps]
                harmful = [key for key in keys
                           if float(baseline_pair_map[key]["signed_impact"]) < -args.impact_eps]
                summary["helpful_retention"] = (
                    float(np.mean([float(pair_map[key]["signed_impact"]) > args.impact_eps
                                   for key in helpful])) if helpful else ""
                )
                summary["harmful_reduction"] = (
                    float(np.mean([float(pair_map[key]["signed_impact"]) >= -args.impact_eps
                                   for key in harmful])) if harmful else ""
                )
    write_csv(args.output_dir.resolve() / "E3" / "posttraining_influence_audit.csv",
              all_summary)
    write_json(args.output_dir.resolve() / "E3" / "posttraining_audit_manifest.json", {
        "base_checkpoint_id": experiment.checkpoint_id, "split": args.split,
        "impact_eps": args.impact_eps, "audit_files": audit_paths,
        "warning": "Impact is model-relative predictive sensitivity, not causal ground truth.",
    })


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.impact_eps < 0:
        raise ValueError("impact-eps must be nonnegative")
    audit(args)


if __name__ == "__main__":
    main()
