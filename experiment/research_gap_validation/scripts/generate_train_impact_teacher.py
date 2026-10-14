"""Generate train-only singleton impact supervision for E3 B4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
for path in (ROOT, CROSS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import write_csv, write_json  # noqa: E402
from impact_core import (apply_replacement, compute_training_means, dataset_lookup,
                         forward_conversation, load_conversation, load_experiment,
                         seed_everything)  # noqa: E402


MODALITIES = ("text", "audio", "visual")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=("text",))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def generate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    means, mean_metadata = compute_training_means(
        experiment.dataset, experiment.split_ids["train"]
    )
    replacements = {name: torch.from_numpy(means[name]).to(experiment.device)
                    for name in args.modalities}
    ids = list(experiment.split_ids["train"])
    if args.max_conversations:
        ids = ids[:args.max_conversations]
    lookup = dataset_lookup(experiment.dataset)
    rows = []
    for number, conversation in enumerate(ids, start=1):
        batch = load_conversation(experiment.dataset, lookup, conversation,
                                  experiment.device)
        labels = batch.labels[0, :batch.length]
        full = forward_conversation(experiment.model, batch)
        for modality in args.modalities:
            original = batch.features[modality]
            for source in range(batch.length):
                modified = apply_replacement(original, source, replacements[modality])
                masked = forward_conversation(experiment.model, batch, {modality: modified})
                for target in range(batch.length):
                    if source == target:
                        continue
                    label = int(labels[target].item())
                    full_nll = float(-full["log_prob"][0, target, label].cpu())
                    mask_nll = float(-masked["log_prob"][0, target, label].cpu())
                    rows.append({
                        "split": "train", "checkpoint_id": experiment.checkpoint_id,
                        "conversation_id": conversation, "source_idx": source,
                        "target_idx": target, "modality": modality,
                        "replacement_strategy": "train_mean",
                        "loss_full": full_nll, "loss_mask": mask_nll,
                        "signed_impact": mask_nll - full_nll,
                    })
        print(f"teacher {number}/{len(ids)} {conversation}", flush=True)
    output = args.output_dir.resolve() / "manifests"
    path = write_csv(output / "train_impact_teacher.csv", rows)
    write_json(output / "train_impact_teacher_manifest.json", {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash, "split": "train",
        "conversation_ids": ids, "modalities": args.modalities,
        "replacement_strategy": "train_mean", "rows": len(rows),
        "training_mean": mean_metadata, "output": str(path),
        "test_conversations_present": bool(
            set(ids) & set(experiment.split_ids.get("test", []))
        ),
    })


def main(argv: Sequence[str] | None = None) -> None:
    generate(parse_args(argv))


if __name__ == "__main__":
    main()
